"""benchmark.py - đo độ trễ suy luận đúng cách (slide Day 2, trang 73 và 75; GUIDE.md mục 4.1).

PSEUDO-CODE: bạn tự hoàn thiện mọi hàm có `raise NotImplementedError`.

Quy tắc đo (vi phạm bị trừ điểm, RUBRIC mục 3):
  - warmup: bỏ >= 10 lần chạy đầu
  - đồng bộ GPU: torch.cuda.synchronize() (hoặc CUDA event) TRƯỚC và SAU đoạn cần đo
  - >= 50 lần đo, báo cáo p50, p95, p99 (không chỉ trung bình)
  - ghi rõ GPU, dtype (FP32/AMP/FP16), batch, độ phân giải, có/không gộp BN, phiên bản torch
  - chọn và ghi rõ có tính tiền xử lý hay không
"""
from __future__ import annotations


def bench(fn, warmup: int = 10, iters: int = 100, sync=None) -> dict:
    """Đo thời gian một hàm `fn()` (không tham số), trả về mili-giây.

    `sync` là hàm đồng bộ (ví dụ torch.cuda.synchronize) hoặc None trên CPU.

    TODO:
      - chạy warmup lần đầu rồi bỏ
      - với mỗi lần đo: sync(); t0 = time.perf_counter(); fn(); sync(); lấy hiệu * 1000
      - trả về {"p50": ..., "p95": ..., "p99": ..., "mean": ..., "n": iters}
    Gợi ý: dùng numpy.percentile hoặc torch.quantile.
    """
    import time, numpy as np
    for _ in range(warmup): fn()
    vals=[]
    for _ in range(iters):
        if sync: sync()
        t=time.perf_counter(); fn()
        if sync: sync()
        vals.append((time.perf_counter()-t)*1000)
    a=np.asarray(vals)
    return {"p50":float(np.percentile(a,50)),"p95":float(np.percentile(a,95)),"p99":float(np.percentile(a,99)),"mean":float(a.mean()),"n":iters}


def latency_report(model, batch_size: int, img_size: int, dtype: str = "fp32", device: str = "cuda",
                   warmup: int = 10, iters: int = 100) -> dict:
    """Đo độ trễ forward của `model` với đầu vào ngẫu nhiên (batch_size, 3, img_size, img_size).

    Trả về dict có thể ghi thẳng vào sheet `Latency` của results.xlsx:
        {"gpu": ..., "dtype": ..., "batch": ..., "img_size": ..., "p50": ..., "p95": ..., "p99": ...,
         "images_per_s": batch_size / (p50 / 1000), "torch": torch.__version__}

    TODO:
      - model.eval(), torch.inference_mode()
      - dtype: "fp32" | "amp" (autocast) | "fp16" (model.half())
      - gọi bench(...) với sync phù hợp; lấy tên GPU bằng torch.cuda.get_device_name
      - Nhớ: ở batch 1, AMP có thể CHẬM hơn FP32 (slide trang 73): đo thật, đừng giả định
    """
    import torch
    dev=torch.device(device if device == "cpu" or torch.cuda.is_available() else "cpu")
    model=model.to(dev).eval(); x=torch.randn(batch_size,3,img_size,img_size,device=dev)
    sync=torch.cuda.synchronize if dev.type == "cuda" else None
    use_amp=dtype == "amp" and dev.type == "cuda"
    if dtype == "fp16": model=model.half(); x=x.half()
    def fn():
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp): model(x)
    out=bench(fn,warmup,iters,sync); out.update({"gpu":torch.cuda.get_device_name(dev) if dev.type=="cuda" else "cpu","dtype":dtype,"batch":batch_size,"img_size":img_size,"images_per_s":batch_size/(out["p50"]/1000),"torch":torch.__version__}); return out


def tta_latency(model, k_views: int, **kw) -> dict:
    """Measure K complete forward passes; the reported latency is measured, not extrapolated."""
    import torch
    if k_views < 1:
        raise ValueError("k_views phải >= 1")
    device = torch.device(kw.get("device", "cuda") if torch.cuda.is_available() else "cpu")
    batch = kw.get("batch_size", 1)
    size = kw.get("img_size", 224)
    dtype = kw.get("dtype", "fp32")
    model = model.to(device).eval()
    images = torch.randn(batch, 3, size, size, device=device)
    if dtype == "fp16":
        model = model.half()
        images = images.half()
    elif dtype not in {"fp32", "amp"}:
        raise ValueError(f"dtype không hợp lệ: {dtype}")
    sync = torch.cuda.synchronize if device.type == "cuda" else None
    def forward():
        with torch.inference_mode(), torch.autocast(device_type=device.type, enabled=dtype == "amp" and device.type == "cuda"):
            for view in range(k_views):
                model(images if view % 2 == 0 else torch.flip(images, (-1,)))
    out = bench(forward, kw.get("warmup", 10), kw.get("iters", 100), sync)
    out.update({"gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
                "dtype": dtype, "batch": batch, "img_size": size, "k_views": k_views,
                "images_per_s": batch / (out["p50"] / 1000), "torch": torch.__version__})
    return out
