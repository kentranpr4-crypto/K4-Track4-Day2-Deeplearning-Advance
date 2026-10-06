"""Freeze the final recipe and inference method using validation only."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from inference_suite import compare_on_val
from run_ablation import score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--baseline-id", default="B03")
    parser.add_argument("--measure-latency", action="store_true")
    args = parser.parse_args()
    recipes = (args.baseline_id, "T01", "T02", "T03", "T04", "T05", "T06", "T07")
    missing = [exp for exp in recipes if not Path(f"predictions/{exp}_seed{args.seed}_val.csv").exists()]
    if missing:
        raise FileNotFoundError(f"Chưa đủ ablation validation: {missing}")
    ranked = sorted(((score(Path(f"predictions/{exp}_seed{args.seed}_val.csv"))["macro_f1_val"], exp)
                     for exp in recipes), reverse=True)
    best_score, recipe = ranked[0]
    compare_on_val(argparse.Namespace(exp_id=recipe, seed=args.seed, runs_dir="runs",
                                   out_dir="inference_out", measure_latency=args.measure_latency))
    report = json.loads(Path(f"inference_out/{recipe}_seed{args.seed}_val_inference.json").read_text())
    def order(item):
        method, record = item
        metrics = record["metrics"]
        timing = record.get("latency") or {}
        return (metrics["macro_f1"], -timing.get("p95", float("inf")), -metrics["ece"])
    method, record = max(report["methods"].items(), key=order)
    choice = {"recipe": recipe, "method": method, "recipe_macro_f1_val": best_score,
              "method_macro_f1_val": record["metrics"]["macro_f1"],
              "seed_used_for_selection": args.seed, "ranked_recipes": ranked}
    Path("selection_val.json").write_text(json.dumps(choice, indent=2))
    print(json.dumps(choice, indent=2), flush=True)


if __name__ == "__main__":
    main()
