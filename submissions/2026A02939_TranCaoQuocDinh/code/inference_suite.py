"""Validation inference comparison and one-time final prediction export."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def load_run(exp_id: str, seed: int, runs_dir: str):
    import torch
    try:
        from .model import build_model
    except ImportError:
        from model import build_model

    run_dir = Path(runs_dir) / exp_id / f"seed{seed}"
    cfg = json.loads((run_dir / "config.json").read_text())
    model = build_model(cfg["backbone"], pretrained=False, num_classes=9,
                        drop_rate=cfg.get("drop_rate", 0.0), init="scratch")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(run_dir / "best.pt", map_location=device, weights_only=True))
    return model.to(device).eval(), cfg, device, run_dir


def make_eval_loader(cfg, split: str, full_image: bool = False):
    import torchvision.transforms as transforms
    try:
        from .dataset import IMAGENET_MEAN, IMAGENET_STD, build_transforms, load_split, make_loader
    except ImportError:
        from dataset import IMAGENET_MEAN, IMAGENET_STD, build_transforms, load_split, make_loader

    frames = dict(zip(("train", "val", "test"), load_split(cfg["labels_dir"], cfg.get("fold", 0))))
    if split not in {"val", "test"}:
        raise ValueError("split phải là val hoặc test")
    transform = (transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor(),
                  transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)]) if full_image
                 else build_transforms(False, cfg["img_size"], cfg["aug"]))
    return make_loader(frames[split], cfg["images_dir"], transform,
                       cfg["batch_size"], False, None, cfg["num_workers"])


def logits_for_views(model, loader, device, views):
    import numpy as np
    import torch
    names, labels = [], []
    collected = [[] for _ in views]
    with torch.inference_mode():
        for images, target, filenames in loader:
            images = images.to(device, non_blocking=True)
            names.extend(filenames)
            labels.append(target.numpy())
            for index, view in enumerate(views):
                collected[index].append(model(view(images)).float().cpu().numpy())
    return names, np.concatenate(labels), [np.concatenate(parts) for parts in collected]


def _metrics(labels, probs):
    from eval import compute_metrics
    result = compute_metrics(labels, probs.argmax(1), probs)
    return {key: float(result[key]) for key in ("top1", "macro_f1", "balanced_acc", "ece", "nll")}


def compare_on_val(args):
    import numpy as np
    import torch
    from eval import save_predictions
    try:
        from .benchmark import latency_report, tta_latency
        from .inference import aggregate_views, apply_temperature, fit_temperature, views_multicrop
    except ImportError:
        from benchmark import latency_report, tta_latency
        from inference import aggregate_views, apply_temperature, fit_temperature, views_multicrop

    model, cfg, device, run_dir = load_run(args.exp_id, args.seed, args.runs_dir)
    base_loader = make_eval_loader(cfg, "val")
    names, labels, base_views = logits_for_views(model, base_loader, device,
        [lambda x: x, lambda x: torch.flip(x, (-1,))])
    base_logits, flip_logits = base_views
    crop_loader = make_eval_loader(cfg, "val", full_image=True)
    crop_views = [lambda x, index=i: views_multicrop(x, cfg["img_size"])[index]
                  for i in range(5)]
    crop_names, crop_labels, crop_logits = logits_for_views(model, crop_loader, device, crop_views)
    if names != crop_names or not np.array_equal(labels, crop_labels):
        raise ValueError("Thứ tự ảnh giữa 1-view và 5-crop khác nhau")
    temperature = fit_temperature(base_logits, labels)
    methods = {
        "I00_one_view": aggregate_views([base_logits]),
        "I01_hflip": aggregate_views([base_logits, flip_logits]),
        "I02_five_crop_prob": aggregate_views(crop_logits, "prob"),
        "I03_five_crop_logit": aggregate_views(crop_logits, "logit"),
        "I04_temperature": apply_temperature(base_logits, temperature),
    }
    output = Path(args.out_dir)
    output.mkdir(parents=True, exist_ok=True)
    report = {"exp_id": args.exp_id, "seed": args.seed, "temperature": temperature,
              "split": "val", "methods": {}}
    latency = {}
    if args.measure_latency:
        kw = {"batch_size": 1, "img_size": cfg["img_size"], "device": str(device),
              "warmup": 10, "iters": 50}
        latency["I00_one_view"] = latency_report(model, **kw)
        latency["I01_hflip"] = tta_latency(model, k_views=2, **kw)
        latency["I02_five_crop_prob"] = tta_latency(model, k_views=5, **kw)
        latency["I03_five_crop_logit"] = latency["I02_five_crop_prob"]
        latency["I04_temperature"] = latency["I00_one_view"]
    for method, probs in methods.items():
        path = output / f"{args.exp_id}_{method}_seed{args.seed}_val.csv"
        save_predictions(path, names, labels, probs)
        report["methods"][method] = {"metrics": _metrics(labels, probs),
                                      "latency": latency.get(method), "prediction": str(path)}
        print(method, report["methods"][method]["metrics"], flush=True)
    (output / f"{args.exp_id}_seed{args.seed}_val_inference.json").write_text(json.dumps(report, indent=2))
    (run_dir / "temperature.json").write_text(json.dumps({"T": temperature, "fit_split": "val"}, indent=2))


def export_final(args):
    import numpy as np
    import torch
    from eval import save_predictions
    try:
        from .inference import aggregate_views, apply_temperature, fit_temperature, views_multicrop
    except ImportError:
        from inference import aggregate_views, apply_temperature, fit_temperature, views_multicrop

    if args.method not in {"I00_one_view", "I01_hflip", "I02_five_crop_prob",
                           "I03_five_crop_logit", "I04_temperature"}:
        raise ValueError("Phương pháp chưa được hỗ trợ")
    out = Path(args.pred_dir) / f"{args.exp_id}_seed{args.seed}_test.csv"
    if out.exists():
        raise FileExistsError(f"Test đã được xuất: {out}")
    model, cfg, device, run_dir = load_run(args.exp_id, args.seed, args.runs_dir)
    if cfg.get("fold", 0) != 0:
        raise ValueError("Final phải dùng fold 0")
    if args.method == "I04_temperature":
        val_logits = np.load(run_dir / "val_logits.npy")
        val_loader = make_eval_loader(cfg, "val")
        val_names, val_labels, _ = logits_for_views(model, val_loader, device, [lambda x: x])
        temperature = fit_temperature(val_logits, val_labels)
        (run_dir / "temperature.json").write_text(json.dumps({"T": temperature, "fit_split": "val"}, indent=2))
    if args.method.startswith(("I02", "I03")):
        loader = make_eval_loader(cfg, "test", full_image=True)
        views = [lambda x, index=i: views_multicrop(x, cfg["img_size"])[index]
                 for i in range(5)]
    elif args.method == "I01_hflip":
        loader = make_eval_loader(cfg, "test")
        views = [lambda x: x, lambda x: torch.flip(x, (-1,))]
    else:
        loader = make_eval_loader(cfg, "test")
        views = [lambda x: x]
    names, labels, logits = logits_for_views(model, loader, device, views)
    probs = (apply_temperature(logits[0], temperature) if args.method == "I04_temperature"
             else aggregate_views(logits, "logit" if args.method == "I03_five_crop_logit" else "prob"))
    save_predictions(out, names, labels, probs)
    (run_dir / "final_inference.json").write_text(json.dumps({"method": args.method,
        "temperature": temperature if args.method == "I04_temperature" else None,
        "test_prediction": str(out)}, indent=2))
    print(f"Saved {out}: {_metrics(labels, probs)}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("compare-val", "export-final"))
    parser.add_argument("--exp-id", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument("--pred-dir", default="predictions")
    parser.add_argument("--out-dir", default="inference_out")
    parser.add_argument("--measure-latency", action="store_true")
    parser.add_argument("--method", default="I00_one_view")
    args = parser.parse_args()
    if args.command == "compare-val":
        compare_on_val(args)
    else:
        export_final(args)


if __name__ == "__main__":
    main()
