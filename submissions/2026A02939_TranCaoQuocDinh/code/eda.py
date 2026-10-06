"""Generate DeepWeeds split evidence and sample images from the original CSVs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

try:
    from .dataset import CLASS_NAMES, check_split, load_split
except ImportError:
    from dataset import CLASS_NAMES, check_split, load_split


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images-dir", type=Path, default=Path("data/images"))
    parser.add_argument("--labels-dir", type=Path, default=Path("data/labels"))
    parser.add_argument("--out-dir", type=Path, default=Path("eda"))
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    train, val, test = load_split(args.labels_dir, 0)
    stats = check_split(train, val, test, args.images_dir)
    (args.out_dir / "split_stats.json").write_text(json.dumps(stats, indent=2))

    counts = np.array([[stats["per_class"][split].get(i, 0) for i in range(9)]
                       for split in ("train", "val", "test")])
    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(9)
    for offset, label, values in zip((-0.25, 0, 0.25), ("train", "val", "test"), counts):
        ax.bar(x + offset, values, width=0.25, label=label)
    ax.set_xticks(x, CLASS_NAMES, rotation=35, ha="right")
    ax.set(ylabel="Images", title="DeepWeeds fold 0 class distribution")
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.out_dir / "class_distribution.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(9, 3, figsize=(9, 22))
    for label in range(9):
        rows = train[train["Label"] == label].head(3)
        for column, (_, row) in enumerate(rows.iterrows()):
            with Image.open(args.images_dir / row["Filename"]) as image:
                axes[label, column].imshow(image.convert("RGB"))
            axes[label, column].set_title(CLASS_NAMES[label], fontsize=9)
            axes[label, column].axis("off")
    fig.tight_layout()
    fig.savefig(args.out_dir / "samples_three_per_class.png", dpi=130)
    plt.close(fig)
    print(f"Saved EDA artifacts in {args.out_dir}")


if __name__ == "__main__":
    main()
