"""train.py - vòng huấn luyện cho mọi thí nghiệm (B, T, F).

PSEUDO-CODE: chỉ có khung (cấu hình và quy ước đặt tên file); bạn tự hoàn thiện mọi hàm có
`raise NotImplementedError` và các bước TODO trong `run()`. Dùng MỘT hàm `run(cfg)` cho mọi cấu hình
(RUBRIC mục H): đổi thí nghiệm chỉ bằng cách đổi `Config`.

Chạy một thí nghiệm từ dòng lệnh:
    python train.py --set exp_id=B01 backbone=resnet50 seed=0
Chỉ số dùng để chọn checkpoint (macro-F1 val) phải tính bằng eval.compute_metrics của repo gốc,
để cùng định nghĩa với lúc chấm:
    sys.path.insert(0, "<thư mục chứa eval.py>");  from eval import compute_metrics
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
import json
import math
import time
from dataclasses import asdict, fields
from typing import get_args, get_type_hints

# Allow `python starter/train.py` to import eval.py from the project root.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Ghi file dự đoán đúng định dạng bằng hàm có sẵn trong eval.py (repo gốc):
#     from eval import save_predictions, compute_metrics
# Log theo epoch (history.csv) và config.json bạn tự ghi bằng pandas/json.


@dataclass
class Config:
    # --- định danh ---
    exp_id: str = "T00"
    seed: int = 0
    fold: int = 0
    # --- mô hình ---
    backbone: str = "resnet50"
    init: str = "finetune"            # scratch | frozen | finetune
    drop_rate: float = 0.0
    # --- dữ liệu / augmentation ---
    img_size: int = 224
    aug: str = "basic"                # basic | color | trivial | randaug ...
    sampler: str | None = None        # None | balanced
    mix: str | None = None            # None | mixup | cutmix
    mix_alpha: float = 1.0
    # --- loss ---
    loss: str = "ce"                  # ce | ls | focal | ce_weighted
    label_smoothing: float = 0.0
    focal_gamma: float = 2.0
    class_weight_beta: float | None = None
    # --- tối ưu (công thức nền, GUIDE.md mục 1.4) ---
    epochs: int = 12
    batch_size: int = 64
    lr_backbone: float = 1e-4
    lr_head: float = 1e-3
    weight_decay: float = 0.05
    warmup_epochs: float = 1.0
    ema_decay: float | None = None
    amp: bool = True
    num_workers: int = 2
    # --- đường dẫn ---
    images_dir: str = "data/images"
    labels_dir: str = "data/labels"
    out_dir: str = "runs"             # config.json, history.csv, checkpoint, logit của từng lần chạy
    pred_dir: str = "predictions"     # file dự đoán đúng định dạng eval.py (nộp cùng bài)
    curve_dir: str = "curves"
    # --- chỉ bật ở Bước 4 (chung kết): ghi predictions trên TEST. Mặc định TẮT (quy tắc S4). ---
    save_test_predictions: bool = False
    resume: bool = False


def run_dir(cfg: Config) -> Path:
    """Thư mục kết quả của một lần chạy: <out_dir>/<exp_id>/seed<k>/ ."""
    return Path(cfg.out_dir) / cfg.exp_id / f"seed{cfg.seed}"


def pred_path(cfg: Config, split: str) -> Path:
    """Đường dẫn chuẩn của file dự đoán: <pred_dir>/<exp_id>_seed<k>_<split>.csv (split = val | test)."""
    return Path(cfg.pred_dir) / f"{cfg.exp_id}_seed{cfg.seed}_{split}.csv"


def set_seed(seed: int) -> None:
    """Cố định mọi nguồn ngẫu nhiên.

    TODO: random, numpy, torch (CPU và CUDA); cân nhắc cudnn.deterministic/benchmark và
    seed cho worker của DataLoader. Ghi lại trong báo cáo mức độ tái lập bạn đạt được.
    """
    import random, numpy as np, torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_optimizer(model, cfg: Config):
    """AdamW với 3 nhóm tham số (xem model.param_groups). TODO."""
    try:
        from . import model as mm
    except ImportError:
        import model as mm
    import torch
    return torch.optim.AdamW(mm.param_groups(model,cfg.lr_backbone,cfg.lr_head,cfg.weight_decay))


def build_scheduler(optimizer, cfg: Config, steps_per_epoch: int):
    """Warmup tuyến tính rồi cosine về ~0 (slide trang 55). TODO.

    Cập nhật theo bước (iteration) hoặc theo epoch đều được; ghi rõ bạn chọn gì.
    Gợi ý kiểm tra: vẽ đường LR theo bước để thấy đúng hình warmup + cosine.
    """
    import torch
    total=max(1,cfg.epochs*steps_per_epoch); warm=max(0,int(cfg.warmup_epochs*steps_per_epoch))
    def f(step):
        if step < warm: return float(step+1)/max(1,warm)
        q=min(1.0,(step-warm)/max(1,total-warm)); return 0.5*(1+math.cos(math.pi*q))
    return torch.optim.lr_scheduler.LambdaLR(optimizer,f)


class EMA:
    """Trung bình động trọng số: W_ema <- d * W_ema + (1 - d) * W  (slide trang 56).

    TODO:
      - __init__(model, decay): sao chép trọng số
      - update(model): sau mỗi bước tối ưu
      - copy_to(model) hoặc dùng bản sao riêng để đánh giá bằng trọng số EMA
      - lưu ý BatchNorm: buffer (running_mean/var) cũng phải được xử lý hợp lý
    """

    def __init__(self, model, decay: float):
        import copy
        if not 0 < decay < 1:
            raise ValueError("ema_decay phải nằm trong (0, 1)")
        self.decay=float(decay); self.shadow=copy.deepcopy(model).eval()
        for p in self.shadow.parameters(): p.requires_grad=False

    def update(self, model) -> None:
        with __import__('torch').no_grad():
            for a,b in zip(self.shadow.parameters(), model.parameters()): a.mul_(self.decay).add_(b.detach(), alpha=1-self.decay)
            for a,b in zip(self.shadow.buffers(), model.buffers()): a.copy_(b)
    def copy_to(self, model):
        model.load_state_dict(self.shadow.state_dict())


def train_one_epoch(model, loader, criterion, optimizer, scheduler, scaler, cfg: Config,
                    device, ema: EMA | None = None) -> dict:
    """Một epoch huấn luyện. Trả về dict, ví dụ {"train_loss": ..., "lr": ...}.

    TODO:
      - model.train() (nếu init == "frozen": giữ phần backbone ở eval, xem model.freeze_backbone)
      - nếu cfg.mix: mix_batch rồi mixed_loss (losses.py)
      - AMP (autocast + GradScaler), clip gradient nếu cần, optimizer.step(), scheduler.step()
      - nếu có EMA: ema.update(model)
    """
    import torch
    try:
        from .losses import mix_batch, mixed_loss
    except ImportError:
        from losses import mix_batch, mixed_loss
    model.train(); total=0; n=0
    if cfg.init == "frozen":
        model.eval()
        classifier=model.get_classifier() if hasattr(model,"get_classifier") else None
        if classifier is not None:
            classifier.train()
    for x,y,_ in loader:
        x=x.to(device); y=y.to(device); optimizer.zero_grad(set_to_none=True)
        mixed=None
        if cfg.mix: x,mixed=mix_batch(x,y,cfg.mix_alpha,cfg.mix)
        with torch.autocast(device_type=device.type, enabled=bool(cfg.amp and device.type=="cuda")):
            logits=model(x); loss=mixed_loss(criterion,logits,mixed) if mixed else criterion(logits,y)
        if not torch.isfinite(loss):
            raise ValueError(f"train loss không hữu hạn: {loss.item()}")
        if scaler is not None and scaler.is_enabled():
            scale_before=scaler.get_scale()
            scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
            stepped=scaler.get_scale() >= scale_before
        else:
            loss.backward(); optimizer.step(); stepped=True
        if stepped:
            if scheduler is not None: scheduler.step()
            if ema: ema.update(model)
        total += float(loss.detach())*len(y); n += len(y)
    return {"train_loss":total/max(1,n),"lr":optimizer.param_groups[0]["lr"]}


def evaluate(model, loader, criterion, device):
    """Chạy model trên một loader ở chế độ eval, KHÔNG tính gradient.

    Trả về (filenames: list[str], y_true: ndarray[N], logits: ndarray[N, 9], loss: float).
    Giữ đúng thứ tự của loader để ghép logit với tên file.

    TODO: model.eval(), torch.inference_mode(), gom kết quả. Softmax khi cần xác suất.
    """
    import numpy as np, torch
    model.eval(); names=[]; ys=[]; zs=[]; total=0; n=0
    with torch.inference_mode():
        for x,y,fn in loader:
            z=model(x.to(device)); loss=criterion(z,y.to(device))
            total += float(loss)*len(y); n += len(y); names.extend(fn); ys.extend(y.numpy()); zs.append(z.cpu().numpy())
    return names,np.asarray(ys),np.concatenate(zs),total/max(1,n)


def plot_curves(history: list[dict], path: str | Path, title: str) -> None:
    """Vẽ đường cong training của một thí nghiệm -> curves/<exp_id>_<mota>.png (GUIDE.md mục 6.2).

    TODO: tối thiểu loss train/val và macro-F1 val theo epoch; có tiêu đề, nhãn trục, chú thích;
    khuyến khích thêm LR theo bước. Lưu bằng matplotlib với dpi đủ nét để đọc số.
    """
    import matplotlib.pyplot as plt
    df=__import__('pandas').DataFrame(history); fig,ax=plt.subplots(1,2,figsize=(11,4));
    ax[0].plot(df.epoch,df.train_loss,label="train"); ax[0].plot(df.epoch,df.val_loss,label="val"); ax[0].set_title(title); ax[0].set_xlabel("epoch"); ax[0].legend(); ax[1].plot(df.epoch,df.val_macro_f1,label="val macro-F1"); ax[1].legend(); fig.tight_layout(); Path(path).parent.mkdir(parents=True,exist_ok=True); fig.savefig(path,dpi=150); plt.close(fig)


def run(cfg: Config) -> dict:
    """Huấn luyện một cấu hình và lưu mọi thứ cần thiết. Trả về dict kết quả tóm tắt.

    TODO theo thứ tự:
      1. set_seed; tạo thư mục run_dir(cfg); ghi config.json (dataclasses.asdict(cfg))
      2. dataset.load_split + dataset.check_split (dừng nếu vi phạm S1-S6)
      3. dựng train/val loader (test loader chỉ tạo khi cfg.save_test_predictions)
      4. model.build_model, criterion (losses.build_criterion), optimizer, scheduler, scaler, EMA
      5. với mỗi epoch: train_one_epoch -> evaluate(val) -> ghi history (loss, macro-F1 val, lr...)
         và lưu checkpoint tốt nhất theo MACRO-F1 VAL (hòa thì lấy epoch sớm hơn)
      6. cuối: nạp checkpoint tốt nhất, lưu val logits và eval.save_predictions(pred_path(cfg, "val"), ...)
      7. NẾU cfg.save_test_predictions (chỉ ở Bước 4): đánh giá test đúng MỘT lần,
         lưu logits và eval.save_predictions(pred_path(cfg, "test"), ...)
      8. ghi history.csv, plot_curves(...), trả về dict tóm tắt
         (best_epoch, macro-F1 val, thời gian train mỗi epoch, số tham số, GMAC)
    Quy tắc: KHÔNG dùng test để chọn checkpoint hay bất kỳ quyết định nào (README.md, S4).
    """
    import numpy as np
    import torch, pandas as pd, timm, torchvision
    try:
        from . import dataset as ds, model as mm, losses
    except ImportError:
        import dataset as ds, model as mm, losses
    from eval import compute_metrics, save_predictions
    if cfg.epochs < 1 or cfg.batch_size < 1:
        raise ValueError("epochs và batch_size phải dương")
    if cfg.save_test_predictions and pred_path(cfg,"test").exists():
        raise FileExistsError(f"Test prediction đã tồn tại: {pred_path(cfg, 'test')}")
    rd=run_dir(cfg); rd.mkdir(parents=True,exist_ok=True)
    config_path=rd/"config.json"
    if cfg.resume:
        if not config_path.exists() or not (rd/"last.pt").exists():
            raise FileNotFoundError(f"Không có checkpoint để resume trong {rd}")
        previous=json.loads(config_path.read_text()); previous.pop("resume",None)
        current=asdict(cfg); current.pop("resume",None)
        if previous != current:
            raise ValueError("Cấu hình resume khác cấu hình đã lưu")
    set_seed(cfg.seed)
    config_path.write_text(json.dumps(asdict(cfg),indent=2))
    (rd/"versions.json").write_text(json.dumps({"torch":torch.__version__,"torchvision":torchvision.__version__,"timm":timm.__version__},indent=2))
    print(f"[{cfg.exp_id} seed={cfg.seed}] loading data and model {cfg.backbone}", flush=True)
    train_df,val_df,test_df=ds.load_split(cfg.labels_dir,cfg.fold)
    split_stats=ds.check_split(train_df,val_df,test_df,cfg.images_dir)
    (rd/"split_stats.json").write_text(json.dumps(split_stats,indent=2))
    tr=ds.make_loader(train_df,cfg.images_dir,ds.build_transforms(True,cfg.img_size,cfg.aug),cfg.batch_size,True,cfg.sampler,cfg.num_workers)
    eval_transform=ds.build_transforms(False,cfg.img_size,cfg.aug)
    va=ds.make_loader(val_df,cfg.images_dir,eval_transform,cfg.batch_size,False,None,cfg.num_workers)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"); net=mm.build_model(cfg.backbone,True,9,cfg.drop_rate,cfg.init).to(device)
    (rd/"pretrained_tag.json").write_text(json.dumps(getattr(net,"pretrained_cfg",{}),default=str,indent=2))
    weight=None
    if cfg.loss=="ce_weighted":
        counts=train_df["Label"].value_counts().reindex(range(9),fill_value=0).to_numpy()
        weight=losses.class_weights(counts,cfg.class_weight_beta or 0).to(device)
    criterion=losses.build_criterion(cfg.loss,smoothing=cfg.label_smoothing,gamma=cfg.focal_gamma,weight=weight)
    opt=build_optimizer(net,cfg); sch=build_scheduler(opt,cfg,len(tr))
    scaler=torch.amp.GradScaler("cuda",enabled=cfg.amp and device.type=="cuda")
    ema=EMA(net,cfg.ema_decay) if cfg.ema_decay is not None else None
    best=-1; best_epoch=0; hist=[]; start_epoch=0; t0=time.perf_counter()
    if cfg.resume:
        saved=torch.load(rd/"last.pt",map_location=device,weights_only=False)
        net.load_state_dict(saved["model"]); opt.load_state_dict(saved["optimizer"])
        sch.load_state_dict(saved["scheduler"]); scaler.load_state_dict(saved["scaler"])
        if ema: ema.shadow.load_state_dict(saved["ema"])
        best=saved["best"]; best_epoch=saved["best_epoch"]
        hist=saved["history"]; start_epoch=saved["epoch"]
    print(f"[{cfg.exp_id} seed={cfg.seed}] start training: {cfg.epochs} epochs, {len(tr)} batches/epoch, device={device}, from epoch={start_epoch+1}", flush=True)
    for epoch in range(start_epoch,cfg.epochs):
        epoch_start=time.perf_counter()
        a=train_one_epoch(net,tr,criterion,opt,sch,scaler,cfg,device,ema)
        eval_model=ema.shadow if ema else net
        _,y,z,vl=evaluate(eval_model,va,criterion,device)
        shifted=z-z.max(1,keepdims=True); p=np.exp(shifted); p/=p.sum(1,keepdims=True)
        m=compute_metrics(y,p.argmax(1),p)
        rec={"epoch":epoch+1,"val_loss":vl,"val_macro_f1":m["macro_f1"],"val_top1":m["top1"],**a}; hist.append(rec)
        if m["macro_f1"]>best:
            best=m["macro_f1"]; best_epoch=epoch+1
            torch.save(eval_model.state_dict(),rd/"best.pt")
        rec["epoch_seconds"]=time.perf_counter()-epoch_start
        torch.save({"model":net.state_dict(),"optimizer":opt.state_dict(),"scheduler":sch.state_dict(),
                    "scaler":scaler.state_dict(),"ema":ema.shadow.state_dict() if ema else None,
                    "epoch":epoch+1,"best":best,"best_epoch":best_epoch,"history":hist},rd/"last.pt")
        pd.DataFrame(hist).to_csv(rd/"history.csv",index=False)
        plot_curves(hist,Path(cfg.curve_dir)/f"{cfg.exp_id}_seed{cfg.seed}_{cfg.backbone}.png",f"{cfg.exp_id} seed {cfg.seed} - {cfg.backbone}")
        print(f"[{cfg.exp_id} seed={cfg.seed}] epoch {epoch+1}/{cfg.epochs}: train_loss={a['train_loss']:.4f} val_loss={vl:.4f} val_macro_f1={m['macro_f1']:.4f} best={best:.4f} time={rec['epoch_seconds']:.1f}s", flush=True)
    net.load_state_dict(torch.load(rd/"best.pt",map_location=device,weights_only=True))
    names,y,z,_=evaluate(net,va,criterion,device)
    np.save(rd/"val_logits.npy",z)
    shifted=z-z.max(1,keepdims=True); p=np.exp(shifted); p/=p.sum(1,keepdims=True)
    save_predictions(pred_path(cfg,"val"),names,y,p)
    if cfg.save_test_predictions:
        te=ds.make_loader(test_df,cfg.images_dir,eval_transform,cfg.batch_size,False,None,cfg.num_workers)
        names,y,z,_=evaluate(net,te,criterion,device)
        np.save(rd/"test_logits.npy",z)
        shifted=z-z.max(1,keepdims=True); p=np.exp(shifted); p/=p.sum(1,keepdims=True)
        save_predictions(pred_path(cfg,"test"),names,y,p)
    try:
        gmac=mm.count_gmacs(net,cfg.img_size)
    except Exception as exc:
        gmac=None
        print(f"GMAC chưa đo được: {exc}",flush=True)
    result={"exp_id":cfg.exp_id,"seed":cfg.seed,"best_epoch":best_epoch,
            "macro_f1_val":best,"train_seconds":time.perf_counter()-t0,
            "params_m":mm.count_params(net),"gmac":gmac}
    (rd/"summary.json").write_text(json.dumps(result,indent=2))
    return result


def parse_overrides(pairs: list[str]) -> dict:
    """Biến ['seed=1', 'loss=focal', 'ema_decay=none'] thành dict, ép kiểu theo field của Config.

    TODO: tách key/value, báo lỗi rõ nếu key không có trong Config, ép int/float/bool/None theo kiểu field.
    """
    types=get_type_hints(Config)
    allowed={f.name for f in fields(Config)}
    out={}
    for item in pairs:
        if "=" not in item: raise ValueError(f"override không hợp lệ: {item}")
        k,v=item.split("=",1)
        if k not in allowed: raise ValueError(f"Config không có trường {k}")
        possible=get_args(types[k]) or (types[k],)
        if v.lower()=="none":
            if type(None) not in possible: raise ValueError(f"{k} không nhận None")
            out[k]=None
        elif bool in possible:
            if v.lower() not in {"true","false"}: raise ValueError(f"{k} phải là true hoặc false")
            out[k]=v.lower()=="true"
        elif int in possible: out[k]=int(v)
        elif float in possible: out[k]=float(v)
        else: out[k]=v
    return out


def main() -> None:
    """Điểm vào dòng lệnh: `python train.py --set exp_id=B01 backbone=resnet50 seed=0`.

    TODO: argparse nhận `--set KEY=VALUE ...`, dựng Config qua parse_overrides, gọi run(cfg), in kết quả.
    """
    import argparse
    ap=argparse.ArgumentParser(); ap.add_argument("--set",nargs="*",default=[]); args=ap.parse_args(); cfg=Config(**parse_overrides(args.set)); print(run(cfg))


if __name__ == "__main__":
    main()
