"""model.py - tạo backbone, đóng băng, nhóm tham số, đếm params/GMAC.

PSEUDO-CODE: bạn tự hoàn thiện mọi hàm có `raise NotImplementedError`.

Giao diện bạn phải giữ:
    build_model(name, pretrained, num_classes, drop_rate, init) -> nn.Module
    freeze_backbone(model)                                        -> None
    param_groups(model, lr_backbone, lr_head, weight_decay)       -> list[dict] cho optimizer
    count_params(model) -> float (triệu)     count_gmacs(model, img_size) -> float
"""
from __future__ import annotations

# Gợi ý backbone (GUIDE.md mục 2.1). Tag trọng số của timm có thể đổi theo phiên bản:
# dùng timm.list_pretrained("resnet50*") để xem, và GHI LẠI tag bạn dùng trong results.xlsx.
SUGGESTED_BACKBONES = {
    "resnet50": "resnet50",
    "resnext50": "resnext50_32x4d",
    "convnext_tiny": "convnext_tiny",
    "deit_small": "deit_small_patch16_224",      # hoặc vit_small_patch16_224
    "swin_tiny": "swin_tiny_patch4_window7_224",
    "efficientnet_b0": "efficientnet_b0",        # mạng nhẹ
    "mobilenetv3": "mobilenetv3_large_100",      # mạng nhẹ
}


def build_model(name: str, pretrained: bool = True, num_classes: int = 9,
                drop_rate: float = 0.0, init: str = "finetune"):
    """Tạo model phân loại 9 lớp.

    `init` (trục A của GUIDE.md mục 3):
      - "scratch"  : pretrained=False, huấn luyện toàn bộ
      - "frozen"   : pretrained=True, đóng băng backbone, chỉ train head
      - "finetune" : pretrained=True, train toàn bộ

    TODO:
      - timm.create_model(name, pretrained=..., num_classes=num_classes, drop_rate=...)
        (timm tự thay head mới; head khởi tạo ngẫu nhiên)
      - nếu init == "frozen": gọi freeze_backbone(model)
      - ghi lại tên tag trọng số thực sự được tải (model.pretrained_cfg)
    """
    try:
        import timm
    except ImportError as exc:
        raise ImportError("Cần cài torch và timm trên Colab/Kaggle") from exc
    if init not in {"scratch", "frozen", "finetune"}:
        raise ValueError("init phải là scratch, frozen hoặc finetune")
    model = timm.create_model(name, pretrained=(pretrained and init != "scratch"), num_classes=num_classes,
                              drop_rate=drop_rate)
    if init == "frozen":
        freeze_backbone(model)
    return model


def freeze_backbone(model) -> None:
    """Đóng băng mọi tham số trừ head.

    TODO:
      - requires_grad = False cho tham số backbone; head (model.get_classifier()) vẫn train
      - lưu ý (GUIDE.md mục 3.2): backbone đóng băng thì BatchNorm cũng phải ở chế độ eval.
        Hãy nghĩ nơi nào trong train loop phải gọi lại model.train() mà vẫn giữ BN ở eval.
    """
    head = model.get_classifier() if hasattr(model, "get_classifier") else None
    head_ids = {id(p) for p in head.parameters()} if head is not None else set()
    for p in model.parameters():
        p.requires_grad = id(p) in head_ids
    model._backbone_frozen = True


def param_groups(model, lr_backbone: float, lr_head: float, weight_decay: float):
    """Chia tham số thành 3 nhóm như slide Day 2, trang 52.

    - backbone có ndim > 1: lr = lr_backbone, weight_decay = weight_decay
    - norm và bias của backbone (ndim <= 1): lr = lr_backbone, weight_decay = 0
    - head mới: lr = lr_head (thường gấp 10 lần backbone), weight_decay = weight_decay

    TODO:
      - bỏ qua tham số requires_grad == False
      - trả về list[dict] dạng {"params": [...], "lr": ..., "weight_decay": ...}
      - (trục E) mở rộng: LR theo tầng nếu bạn muốn thử
    """
    head = model.get_classifier() if hasattr(model, "get_classifier") else None
    head_ids = {id(p) for p in head.parameters()} if head is not None else set()
    groups = {("bb_decay", lr_backbone, weight_decay): [], ("bb_nodecay", lr_backbone, 0.0): [],
              ("head_decay", lr_head, weight_decay): [], ("head_nodecay", lr_head, 0.0): []}
    for p in model.parameters():
        if not p.requires_grad: continue
        is_head = id(p) in head_ids
        key = (("head_decay", lr_head, weight_decay) if p.ndim > 1 else ("head_nodecay", lr_head, 0.0)) if is_head else (("bb_decay", lr_backbone, weight_decay) if p.ndim > 1 else ("bb_nodecay", lr_backbone, 0.0))
        groups[key].append(p)
    return [{"params": ps, "lr": lr, "weight_decay": wd} for (_, lr, wd), ps in groups.items() if ps]


def count_params(model) -> float:
    """Số tham số (triệu), đếm cả tham số bị đóng băng. TODO."""
    return sum(p.numel() for p in model.parameters()) / 1e6


def count_gmacs(model, img_size: int = 224) -> float:
    """GMAC cho một ảnh 3 x img_size x img_size (slide tính MAC, không phải FLOPs 2x).

    TODO: dùng thư viện đếm (fvcore, ptflops, thop...) hoặc tự đếm bằng hook.
    Ghi rõ công cụ đã dùng; số có thể lệch vài phần trăm giữa các công cụ.
    """
    import torch
    device = next(model.parameters()).device
    was_training = model.training
    model.eval()
    try:
        with torch.inference_mode():
            x = torch.zeros(1, 3, img_size, img_size, device=device)
            try:
                from thop import profile
                macs, _ = profile(model, inputs=(x,), verbose=False)
            except (ImportError, RuntimeError):
                from fvcore.nn import FlopCountAnalysis
                macs = FlopCountAnalysis(model, x).total()
        return float(macs / 1e9)
    finally:
        model.train(was_training)
