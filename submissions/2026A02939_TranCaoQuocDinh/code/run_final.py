"""Run validation-selected final recipe and baseline over three seeds."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import fields
from pathlib import Path

try:
    from .inference_suite import export_final
    from .train import Config, run
except ImportError:
    from inference_suite import export_final
    from train import Config, run


def from_record(path: Path, exp_id: str, seed: int, images_dir: str, labels_dir: str):
    saved = json.loads(path.read_text())
    allowed = {field.name for field in fields(Config)}
    cfg = {key: value for key, value in saved.items() if key in allowed}
    cfg.update(exp_id=exp_id, seed=seed, images_dir=images_dir,
               labels_dir=labels_dir, save_test_predictions=False, resume=False)
    return Config(**cfg)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--recipe", help="B04 or selected T01..T07 based on val")
    parser.add_argument("--method",
                        choices=("I00_one_view", "I01_hflip", "I02_five_crop_prob",
                                 "I03_five_crop_logit", "I04_temperature"))
    parser.add_argument("--images-dir", required=True)
    parser.add_argument("--labels-dir", default="data/labels")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    args = parser.parse_args()
    if args.recipe is None or args.method is None:
        chosen = json.loads(Path("selection_val.json").read_text())
        args.recipe = args.recipe or chosen["recipe"]
        args.method = args.method or chosen["method"]
    if len(set(args.seeds)) < 3:
        raise ValueError("Final cần >= 3 seed khác nhau")
    recipe_path = Path("runs") / args.recipe / "seed0" / "config.json"
    if not recipe_path.exists():
        raise FileNotFoundError(f"Không thấy công thức đã chọn: {recipe_path}")
    baseline_path = Path("runs/T00/seed0/config.json")
    if not baseline_path.exists():
        print("Không có T00 cũ; dùng công thức nền ResNet50", flush=True)
    selected = {"recipe": args.recipe, "method": args.method, "seeds": args.seeds}
    selection_path = Path("final_selection.json")
    if selection_path.exists() and json.loads(selection_path.read_text()) != selected:
        raise ValueError("final_selection.json đã chốt cấu hình khác")
    selection_path.write_text(json.dumps(selected, indent=2))
    for seed in args.seeds:
        for exp_id, source, method in (("F01", recipe_path, args.method),
                                       ("T00", baseline_path, "I00_one_view")):
            val_pred = Path(f"predictions/{exp_id}_seed{seed}_val.csv")
            test_pred = Path(f"predictions/{exp_id}_seed{seed}_test.csv")
            if test_pred.exists():
                print(f"{test_pred} đã tồn tại, bỏ qua", flush=True)
                continue
            checkpoint = Path("runs") / exp_id / f"seed{seed}" / "best.pt"
            if not val_pred.exists() or not checkpoint.exists():
                if source.exists():
                    cfg = from_record(source, exp_id, seed, args.images_dir, args.labels_dir)
                else:
                    cfg = Config(exp_id=exp_id, seed=seed, backbone="resnet50",
                                 batch_size=16, num_workers=0, images_dir=args.images_dir,
                                 labels_dir=args.labels_dir)
                cfg.resume = (Path("runs") / exp_id / f"seed{seed}" / "last.pt").exists()
                print(f"TRAIN {exp_id} seed={seed} resume={cfg.resume}", flush=True)
                run(cfg)
            print(f"TEST {exp_id} seed={seed} method={method}", flush=True)
            export_final(argparse.Namespace(exp_id=exp_id, seed=seed, method=method,
                         runs_dir="runs", pred_dir="predictions"))
    for exp_id in ("F01", "T00"):
        subprocess.run([sys.executable, "eval.py", "score", "--pred",
                        f"predictions/{exp_id}_seed*_test.csv", "--test-csv",
                        f"{args.labels_dir}/test_subset0.csv", "--labels",
                        f"{args.labels_dir}/labels.csv", "--tag", exp_id,
                        "--out", "eval_out"], check=True)
    subprocess.run([sys.executable, "eval.py", "grade", "--final",
                    "predictions/F01_seed*_test.csv", "--baseline",
                    "predictions/T00_seed*_test.csv", "--test-csv",
                    f"{args.labels_dir}/test_subset0.csv", "--labels",
                    f"{args.labels_dir}/labels.csv"], check=True)


if __name__ == "__main__":
    main()
