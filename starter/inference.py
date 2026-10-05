"""inference.py - các phương pháp suy luận (Bước 3 của GUIDE.md).

PSEUDO-CODE: bạn tự hoàn thiện mọi hàm có `raise NotImplementedError`.
Liên hệ slide Day 2: TTA (trang 62-66, 75), ensemble/EMA/soup (trang 67), độ phân giải kiểm tra
(trang 68), temperature scaling (trang 69), gộp BatchNorm (trang 71).

Mọi hàm phải chạy ở chế độ eval, không gradient. Chọn phương pháp CHỈ dựa trên val;
nhiệt độ T khớp trên VAL rồi áp dụng sang test (README.md, S2 và S4).

Giao diện bạn nên giữ:
    predict_logits(model, loader, device, view=None) -> (filenames, y_true, logits[N, 9])
    aggregate_views(list_of_logits, space)           -> probs[N, 9]
    fit_temperature(val_logits, val_labels)          -> float T
    apply_temperature(logits, T)                     -> probs
    ensemble_probs(list_of_probs)                    -> probs
    fuse_conv_bn(model)                              -> model (BN đã gộp vào conv)
"""
from __future__ import annotations


def predict_logits(model, loader, device, view=None):
    """Chạy model trên loader và gom logit theo đúng thứ tự file.

    `view` là hàm biến đổi batch ảnh trước khi đưa vào model (ví dụ lật ngang), hoặc None.
    TODO: model.eval(), torch.inference_mode(), (tuỳ chọn) autocast. Trả về numpy.
    """
    import numpy as np
    import torch
    model.eval(); names=[]; ys=[]; outs=[]
    with torch.inference_mode():
        for x, y, fn in loader:
            x = x.to(device, non_blocking=True)
            if view is not None: x = view(x)
            outs.append(model(x).detach().cpu().numpy()); ys.extend(y.numpy().tolist()); names.extend(list(fn))
    return names, np.asarray(ys), np.concatenate(outs, axis=0)


def view_identity(x):
    return x


def view_hflip(x):
    """Lật ngang batch (N, C, H, W). TODO: dùng torch.flip trên chiều rộng (slide trang 75)."""
    import torch
    return torch.flip(x, dims=(-1,))


def views_multicrop(x, crop: int):
    """5 crop (4 góc + giữa) kích thước `crop`, và tuỳ chọn thêm bản lật. Trả về list các batch. TODO."""
    h, w = x.shape[-2:]
    if crop > h or crop > w: raise ValueError("crop lớn hơn ảnh")
    boxes = [(0,0),(0,w-crop),(h-crop,0),(h-crop,w-crop),((h-crop)//2,(w-crop)//2)]
    return [x[..., y:y+crop, z:z+crop] for y,z in boxes]


def views_multiscale(x, sizes):
    """Resize batch về từng kích thước trong `sizes`, trả về list các batch. TODO.

    Lưu ý: model phải chấp nhận ảnh khác kích thước lúc train (CNN có global pooling thì được;
    ViT/Swin cần xử lý riêng vị trí/cửa sổ). Ghi rõ giới hạn bạn gặp.
    """
    import torch.nn.functional as F
    return [F.interpolate(x, size=(s,s), mode="bilinear", align_corners=False) for s in sizes]


def aggregate_views(logits_per_view, space: str = "prob"):
    """Gộp K lượt chạy của TTA thành một dự đoán (slide trang 62).

      - space="prob":  trung bình softmax của từng view
      - space="logit": trung bình logit rồi softmax
    Slide chưa kết luận cách nào luôn tốt hơn: chọn một và ghi rõ, hoặc so sánh cả hai (I03).
    TODO: trả về xác suất (N, 9) đã chuẩn hoá.
    """
    import numpy as np
    a = np.asarray(logits_per_view)
    if a.ndim != 3: raise ValueError("cần mảng (K,N,C)")
    if space == "logit": z = a.mean(0); z = z - z.max(1, keepdims=True); p=np.exp(z); return p/p.sum(1,keepdims=True)
    if space == "prob":
        z=a-a.max(2,keepdims=True); p=np.exp(z); p/=p.sum(2,keepdims=True); return p.mean(0)
    raise ValueError("space phải là prob hoặc logit")


def ensemble_probs(list_of_probs):
    """Trung bình xác suất của nhiều mô hình (khác backbone hoặc khác seed). TODO.

    Chi phí suy luận = số mô hình. Chỉ ghép các mô hình trên CÙNG tập ảnh và cùng thứ tự file.
    """
    import numpy as np
    a=np.asarray(list_of_probs, dtype=float)
    if a.ndim != 3 or a.shape[2] != 9: raise ValueError("cần list xác suất (M,N,9)")
    p=a.mean(0); return p/p.sum(1,keepdims=True)


def fit_temperature(val_logits, val_labels) -> float:
    """Tìm nhiệt độ T > 0 cực tiểu NLL trên VAL: p = softmax(logit / T)  (slide trang 69).

    TODO: tối ưu hoá một tham số (LBFGS trên log T, hoặc tìm lưới thô rồi tinh).
    Accuracy không đổi vì thứ tự lớp không đổi. KHÔNG khớp T trên test.
    """
    import numpy as np
    from scipy.optimize import minimize_scalar
    z=np.asarray(val_logits,float); y=np.asarray(val_labels,int)
    def nll(logt):
        t=np.exp(logt); q=z/t; q=q-q.max(1,keepdims=True); lp=q-np.log(np.exp(q).sum(1,keepdims=True)); return -lp[np.arange(len(y)),y].mean()
    r=minimize_scalar(nll, bounds=(-4,4), method="bounded")
    return float(np.exp(r.x))


def apply_temperature(logits, T: float):
    """Trả về softmax(logits / T). TODO."""
    import numpy as np
    if T <= 0:
        raise ValueError("temperature phải > 0")
    z=np.asarray(logits,float)/float(T); z-=z.max(1,keepdims=True); p=np.exp(z); return p/p.sum(1,keepdims=True)


def fuse_conv_bn(model):
    """Gộp BatchNorm vào tích chập liền trước, chính xác lúc suy luận (slide trang 71, 75):

        w' = gamma * w / sqrt(var + eps)        b' = beta + gamma * (b - mean) / sqrt(var + eps)

    TODO:
      - model.eval() trước
      - với từng cặp (Conv2d, BatchNorm2d) liền kề: tạo conv mới (có bias) và thay BN bằng Identity
      - kiểm tra: đầu ra trước/sau gộp lệch nhau cỡ 1e-5 trở xuống (in ra sai số lớn nhất)
    Với kiến trúc không có BN (ViT, Swin, ConvNeXt dùng LayerNorm), mục này không áp dụng; ghi rõ.
    """
    import torch.nn as nn
    model.eval()
    def rec(parent):
        items=list(parent.named_children())
        for i in range(len(items)-1):
            name, conv=items[i]
            bn_name, bn=items[i+1]
            if isinstance(conv, nn.Conv2d) and isinstance(bn, nn.BatchNorm2d):
                parent._modules[name]=nn.utils.fusion.fuse_conv_bn_eval(conv,bn)
                parent._modules[bn_name]=nn.Identity()
        for child in parent.children():
            rec(child)
    rec(model); return model
