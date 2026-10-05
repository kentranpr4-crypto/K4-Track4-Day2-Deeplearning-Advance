"""Run controlled backbone ablations and a validation-selected combination."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

try:
    from .train import Config, run
except ImportError:
    from train import Config, run

from eval import compute_metrics, read_pred


EXPERIMENTS = {
    "T01": {"axis": "initialization", "init": "frozen"},
    "T02": {"axis": "augmentation", "aug": "color"},
    "T03": {"axis": "augmentation", "aug": "randaug"},
    "T04": {"axis": "loss", "loss": "ls", "label_smoothing": 0.1},
    "T05": {"axis": "loss", "loss": "focal", "focal_gamma": 2.0},
    "T06": {"axis": "loss", "loss": "ce_weighted", "class_weight_beta": 0.0},
}


def score(path):
    pred = read_pred(str(path))
    metrics = compute_metrics(pred.y_true, pred.y_pred, pred.probs)
    return {"macro_f1_val": metrics["macro_f1"], "top1_val": metrics["top1"],
            "ece_val": metrics["ece"]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images-dir", required=True)
    parser.add_argument("--labels-dir", default="data/labels")
    parser.add_argument("--backbone", default="convnext_tiny")
    parser.add_argument("--baseline-id", default="B03")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--only", nargs="*", choices=[*EXPERIMENTS, "T07"])
    args = parser.parse_args()
    baseline_path = Path(f"predictions/{args.baseline_id}_seed{args.seed}_val.csv")
    if not baseline_path.exists():
        raise FileNotFoundError(f"Cần baseline cùng seed: {baseline_path}")
    baseline_config = json.loads((Path("runs") / args.baseline_id / f"seed{args.seed}" / "config.json").read_text())
    if baseline_config["backbone"] != args.backbone:
        raise ValueError("Backbone ablation phải trùng với backbone baseline")
    selected = args.only or [*EXPERIMENTS, "T07"]
    common = dict(backbone=args.backbone, seed=args.seed, epochs=args.epochs,
                  batch_size=args.batch_size, num_workers=args.num_workers,
                  images_dir=args.images_dir, labels_dir=args.labels_dir)
    records = [{"exp_id": args.baseline_id, "axis": "baseline", **score(baseline_path)}]
    for exp_id, change in EXPERIMENTS.items():
        path = Path(f"predictions/{exp_id}_seed{args.seed}_val.csv")
        if exp_id in selected and not path.exists():
            options = {key: value for key, value in change.items() if key != "axis"}
            print(f"START {exp_id}: {change}", flush=True)
            resume = (Path("runs") / exp_id / f"seed{args.seed}" / "last.pt").exists()
            run(Config(exp_id=exp_id, resume=resume, **common, **options))
        if path.exists():
            records.append({"exp_id": exp_id, "axis": change["axis"],
                            **score(path)})
            print(records[-1], flush=True)
    if "T07" in selected:
        needed = {"T02", "T03", "T04", "T05", "T06"}
        available = {row["exp_id"] for row in records}
        if not needed <= available:
            raise ValueError(f"T07 cần hoàn thành {sorted(needed - available)}")
        ranked = {row["exp_id"]: row["macro_f1_val"] for row in records}
        aug_id = max(("T02", "T03"), key=ranked.get)
        loss_id = max(("T04", "T05", "T06"), key=ranked.get)
        combination = {**{k: v for k, v in EXPERIMENTS[aug_id].items() if k != "axis"},
                       **{k: v for k, v in EXPERIMENTS[loss_id].items() if k != "axis"}}
        Path("ablation_combination.json").write_text(json.dumps({"augmentation": aug_id,
            "loss": loss_id, "config": combination}, indent=2))
        path = Path(f"predictions/T07_seed{args.seed}_val.csv")
        if not path.exists():
            print(f"START T07: {combination}", flush=True)
            resume = (Path("runs") / "T07" / f"seed{args.seed}" / "last.pt").exists()
            run(Config(exp_id="T07", resume=resume, **common, **combination))
        records.append({"exp_id": "T07", "axis": "combination", **score(path)})
    pd.DataFrame(records).to_csv("ablation_summary.csv", index=False)
    print(pd.DataFrame(records).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
