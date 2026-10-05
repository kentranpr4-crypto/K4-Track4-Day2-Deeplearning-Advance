"""Build submission tables from recorded runs and predictions; never invent metrics."""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval import CLASS_NAMES, compute_metrics, mean_std, read_pred


def _prediction_metrics(path):
    prediction = read_pred(str(path))
    return compute_metrics(prediction.y_true, prediction.y_pred, prediction.probs)


def _sort_id(path):
    match = re.search(r"seed(\d+)", str(path))
    return int(match.group(1)) if match else -1


def make_test_figures(root: Path):
    prediction_path = root / "predictions/F01_seed0_test.csv"
    if not prediction_path.exists():
        return
    import matplotlib.pyplot as plt
    import numpy as np
    from PIL import Image
    from eval import confusion_matrix

    prediction = read_pred(str(prediction_path))
    figures = root / "figures"
    figures.mkdir(exist_ok=True)
    cm = confusion_matrix(prediction.y_true, prediction.y_pred)
    fig, ax = plt.subplots(figsize=(9, 8))
    image = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(9), CLASS_NAMES, rotation=45, ha="right")
    ax.set_yticks(range(9), CLASS_NAMES)
    ax.set(xlabel="Predicted", ylabel="True", title="F01 seed 0 - test confusion matrix")
    for row in range(9):
        for col in range(9):
            ax.text(col, row, str(cm[row, col]), ha="center", va="center", fontsize=8)
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(figures / "confusion_matrix_F01.png", dpi=160)
    plt.close(fig)

    config_path = root / "runs/F01/seed0/config.json"
    if not config_path.exists():
        return
    images_dir = Path(json.loads(config_path.read_text())["images_dir"])
    if not images_dir.is_absolute():
        images_dir = root / images_dir
    errors = np.flatnonzero(prediction.y_true != prediction.y_pred)
    # Show the two difficult classes first, then other errors.
    order = sorted(errors, key=lambda index: (prediction.y_true[index] not in (0, 7), index))[:9]
    fig, axes = plt.subplots(3, 3, figsize=(10, 10))
    for ax, index in zip(axes.flat, order):
        filename = prediction.filenames[index]
        with Image.open(images_dir / filename) as image:
            ax.imshow(image.convert("RGB"))
        ax.set_title(f"{CLASS_NAMES[prediction.y_true[index]]} -> {CLASS_NAMES[prediction.y_pred[index]]}", fontsize=8)
        ax.axis("off")
    for ax in list(axes.flat)[len(order):]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(figures / "error_examples_F01.png", dpi=140)
    plt.close(fig)


def collect(root: Path):
    backbones, training, final, per_class, latency, inference, summary = [], [], [], [], [], [], []
    runs = root / "runs"
    predictions = root / "predictions"
    for config_file in sorted(runs.glob("*/seed*/config.json")):
        run_dir = config_file.parent
        cfg = json.loads(config_file.read_text())
        exp_id, seed = cfg["exp_id"], cfg["seed"]
        history_path = run_dir / "history.csv"
        history = pd.read_csv(history_path) if history_path.exists() else pd.DataFrame()
        if not history.empty:
            curve_path = root / "curves" / f"{exp_id}_seed{seed}_{cfg['backbone']}.png"
            if not curve_path.exists():
                try:
                    from .train import plot_curves
                except ImportError:
                    from train import plot_curves
                plot_curves(history.to_dict("records"), curve_path,
                            f"{exp_id} seed {seed} - {cfg['backbone']}")
        val_path = predictions / f"{exp_id}_seed{seed}_val.csv"
        if not val_path.exists():
            continue
        val = _prediction_metrics(val_path)
        info_path = run_dir / "summary.json"
        info = json.loads(info_path.read_text()) if info_path.exists() else {}
        profile_path = run_dir / "profile.json"
        profile = json.loads(profile_path.read_text()) if profile_path.exists() else {}
        if profile.get("latency"):
            latency.append({"exp_id": exp_id, "seed": seed,
                            "method": "I00_one_view", **profile["latency"]})
        tag_path = run_dir / "pretrained_tag.json"
        tag = json.loads(tag_path.read_text()).get("hf_hub_id", "") if tag_path.exists() else ""
        common = {"exp_id": exp_id, "backbone": cfg["backbone"], "seed": seed,
                  "macro_f1_val": val["macro_f1"], "top1_val": val["top1"],
                  "best_epoch": info.get("best_epoch", int(history.loc[history.val_macro_f1.idxmax(), "epoch"]) if not history.empty else None),
                  "params_m": info.get("params_m", profile.get("params_m")),
                  "gmac": info.get("gmac") or profile.get("gmac"),
                  "latency_batch1_ms": (profile.get("latency") or {}).get("p50"),
                  "train_seconds_per_epoch": history.epoch_seconds.mean() if "epoch_seconds" in history else None}
        summary.append({**common, "ece_val": val["ece"]})
        if exp_id.startswith("B") or exp_id == "T00":
            backbones.append({**common, "pretrained_tag": tag, "img_size": cfg["img_size"],
                              "epochs": cfg["epochs"]})
        if exp_id.startswith("T"):
            training.append({**common, "init": cfg["init"], "aug": cfg["aug"],
                             "loss": cfg["loss"], "mix": cfg.get("mix"),
                             "sampler": cfg.get("sampler"), "ema_decay": cfg.get("ema_decay")})
        test_path = predictions / f"{exp_id}_seed{seed}_test.csv"
        if test_path.exists():
            test = _prediction_metrics(test_path)
            final.append({**common, "macro_f1_test": test["macro_f1"],
                          "top1_test": test["top1"], "ece_test": test["ece"]})
            for index, name in enumerate(CLASS_NAMES):
                per_class.append({"exp_id": exp_id, "seed": seed, "class": name,
                    "support": int(test["support"][index]), "precision": test["precision"][index],
                    "recall": test["recall"][index], "f1": test["f1"][index]})
    for path in sorted((root / "inference_out").glob("*_val_inference.json")):
        data = json.loads(path.read_text())
        for method, record in data["methods"].items():
            metrics = record["metrics"]
            timing = record.get("latency") or {}
            inference.append({"exp_id": data["exp_id"], "seed": data["seed"],
                "method": method, "macro_f1_val": metrics["macro_f1"],
                "top1_val": metrics["top1"], "ece_val": metrics["ece"],
                "k": timing.get("k_views", 1), "p50_ms": timing.get("p50"),
                "p95_ms": timing.get("p95"), "p99_ms": timing.get("p99"),
                "images_per_s": timing.get("images_per_s")})
            if timing:
                latency.append({"exp_id": data["exp_id"], "seed": data["seed"],
                                "method": method, **timing})
    baseline = next((row["macro_f1_val"] for row in summary
                     if row["exp_id"] == "B04" and row["seed"] == 0), None)
    for row in training:
        row["delta_macro_f1_vs_B04"] = (row["macro_f1_val"] - baseline
                                        if baseline is not None else None)
    if final:
        final_frame = pd.DataFrame(final)
        for exp_id, group in final_frame.groupby("exp_id"):
            if len(group) < 2:
                continue
            aggregate = {"exp_id": exp_id, "seed": "mean +/- std"}
            for field in ("macro_f1_val", "macro_f1_test", "top1_test", "ece_test"):
                avg, std = mean_std(group[field].tolist())
                aggregate[field] = f"{avg:.4f} +/- {std:.4f}"
            final.append(aggregate)
    return {"Backbones": backbones, "Training": training, "Inference": inference,
            "Final": final, "PerClass": per_class, "Latency": latency,
            "Summary": sorted(summary, key=lambda row: row["macro_f1_val"], reverse=True)[:10]}


def write_report(path: Path, sheets: dict):
    def markdown_table(frame):
        columns = list(frame.columns)
        def cell(value):
            return str(value).replace("|", "\\|").replace("\n", " ")
        rows = ["| " + " | ".join(map(cell, columns)) + " |",
                "| " + " | ".join("---" for _ in columns) + " |"]
        rows.extend("| " + " | ".join(cell(value) for value in row) + " |"
                    for row in frame.itertuples(index=False, name=None))
        return "\n".join(rows)

    lines = ["# DeepWeeds Lab Day 2 - Bao cao ket qua", "",
             "## Trang thai", "", "Bao cao duoc tao tu log va prediction that; cac muc chua co so lieu can duoc hoan thien sau khi chay Kaggle.", ""]
    backbones = pd.DataFrame(sheets["Backbones"])
    if not backbones.empty:
        lines += ["## Backbone tren validation", "", markdown_table(backbones), ""]
    for title, key in (("Ablation huan luyen", "Training"),
                       ("Phuong phap suy luan", "Inference"),
                       ("Ket qua test", "Final"),
                       ("Chi so tung lop", "PerClass")):
        frame = pd.DataFrame(sheets[key])
        lines += [f"## {title}", "", markdown_table(frame) if not frame.empty else "Chua co ket qua.", ""]
    final = pd.DataFrame([row for row in sheets["Final"] if isinstance(row.get("seed"), int)])
    if not final.empty:
        lines += ["## Mean va std theo seed", ""]
        for exp_id, group in final.groupby("exp_id"):
            lines.append(f"### {exp_id} ({len(group)} seed)")
            for field in ("macro_f1_test", "top1_test", "ece_test"):
                avg, std = mean_std(group[field].tolist())
                lines.append(f"- {field}: {avg:.4f} +/- {std:.4f}")
            lines.append("")
    if {"F01", "T00"} <= set(final.get("exp_id", [])):
        means = final.groupby("exp_id")["macro_f1_test"].mean()
        lines += ["## Cai thien so voi moc", "",
                  f"Delta macro-F1 test F01 - T00: {means['F01'] - means['T00']:+.4f}.", ""]
    if (path.parent / "figures" / "confusion_matrix_F01.png").exists():
        lines += ["## Phan tich loi", "", "![Confusion matrix](figures/confusion_matrix_F01.png)", "",
                  "![Error examples](figures/error_examples_F01.png)", ""]
    lines += ["## Phan tich va han che can bo sung", "",
              "- Giai thich moi thay doi dua tren ket qua validation va do nhieu qua seed.",
              "- Them ma tran nham lan, anh du doan sai va khuyen nghi latency cho robot.",
              "- Neu gioi han: mot fold, chia ngau nhien theo anh thay vi dia diem, co the lac quan tren dia diem moi.", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--xlsx", type=Path, default=Path("results.xlsx"))
    parser.add_argument("--report", type=Path, default=Path("report_draft.md"))
    args = parser.parse_args()
    sheets = collect(args.root)
    make_test_figures(args.root)
    with pd.ExcelWriter(args.xlsx, engine="openpyxl") as writer:
        for name, rows in sheets.items():
            frame = pd.DataFrame(rows)
            frame.to_excel(writer, sheet_name=name, index=False)
            ws = writer.sheets[name]
            ws.freeze_panes = "A2"
            ws.auto_filter.ref = ws.dimensions
            for column in ws.columns:
                ws.column_dimensions[column[0].column_letter].width = min(38, max(12,
                    max(len(str(cell.value or "")) for cell in column) + 2))
    write_report(args.report, sheets)
    print(f"Saved {args.xlsx} and {args.report}")


if __name__ == "__main__":
    main()
