"""Backfill parameter, GMAC and batch-1 latency data for completed backbone runs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from .benchmark import latency_report
    from .inference_suite import load_run
    from .model import count_gmacs, count_params
except ImportError:
    from benchmark import latency_report
    from inference_suite import load_run
    from model import count_gmacs, count_params


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiments", nargs="+", default=["T00", "B02", "B03", "B04", "B05"])
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    for exp_id in args.experiments:
        run_dir = Path("runs") / exp_id / f"seed{args.seed}"
        if not (run_dir / "best.pt").exists():
            print(f"SKIP {exp_id}: no checkpoint", flush=True)
            continue
        model, cfg, device, _ = load_run(exp_id, args.seed, "runs")
        tag_path = run_dir / "pretrained_tag.json"
        if not tag_path.exists():
            tag_path.write_text(json.dumps(getattr(model, "pretrained_cfg", {}), default=str, indent=2))
        profile = {"exp_id": exp_id, "seed": args.seed,
                   "params_m": count_params(model), "gmac_tool": "thop/fvcore"}
        try:
            profile["gmac"] = count_gmacs(model, cfg["img_size"])
        except Exception as exc:
            profile["gmac"] = None
            profile["gmac_error"] = str(exc)
        profile["latency"] = latency_report(model, batch_size=1,
            img_size=cfg["img_size"], device=str(device), warmup=10, iters=50)
        (run_dir / "profile.json").write_text(json.dumps(profile, indent=2))
        print(json.dumps(profile, indent=2), flush=True)


if __name__ == "__main__":
    main()
