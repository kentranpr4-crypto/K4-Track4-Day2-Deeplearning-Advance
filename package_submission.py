"""Package a completed Kaggle run into the required submission layout."""
from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path

from eval import check_against_csv, read_pred


def package(source: Path, destination: Path, notebook_url: str) -> Path:
    source = source.resolve()
    if not re.fullmatch(r"[A-Za-z0-9]+_[a-z0-9_]+", destination.name):
        raise ValueError("Tên thư mục phải là <mssv>_<ho_ten_khong_dau>")
    if not notebook_url.startswith("https://www.kaggle.com/") and not notebook_url.startswith(
        "https://colab.research.google.com/"
    ):
        raise ValueError("Cần URL notebook Kaggle hoặc Colab")
    required = [source / "results.xlsx", source / "report.md", source / "starter",
                source / "curves", source / "predictions", source / "eval.py"]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Thiếu sản phẩm: {missing}")
    test_csv = source / "data/labels/test_subset0.csv"
    if not test_csv.exists():
        raise FileNotFoundError(f"Cần CSV gốc để kiểm tra prediction: {test_csv}")
    for group in ("F01", "T00"):
        files = sorted((source / "predictions").glob(f"{group}_seed*_test.csv"))
        seeds = set()
        for path in files:
            prediction = read_pred(str(path))
            check_against_csv(prediction, str(test_csv))
            seeds.add(prediction.seed)
        if len(seeds) < 3:
            raise ValueError(f"{group} cần >= 3 seed test hợp lệ, hiện có {sorted(seeds)}")
    run_configs = sorted((source / "runs").glob("*/seed*/config.json"))
    missing_curves = []
    for path in run_configs:
        config = json.loads(path.read_text())
        pattern = f"{config['exp_id']}_seed{config['seed']}_*.png"
        if not list((source / "curves").glob(pattern)):
            missing_curves.append(pattern)
    if missing_curves:
        raise ValueError(f"Thiếu biểu đồ training: {missing_curves}")
    if destination.exists():
        raise FileExistsError(f"Đích đã tồn tại: {destination}")

    destination.mkdir(parents=True)
    shutil.copy2(source / "results.xlsx", destination / "results.xlsx")
    shutil.copy2(source / "report.md", destination / "report.md")
    shutil.copytree(source / "curves", destination / "curves")
    shutil.copytree(source / "predictions", destination / "predictions")
    code = destination / "code"
    code.mkdir()
    for path in (source / "starter").glob("*.py"):
        shutil.copy2(path, code / path.name)
    for path in (source / "starter").glob("*.ipynb"):
        shutil.copy2(path, code / path.name)
    shutil.copy2(source / "eval.py", code / "eval.py")
    versions = sorted((source / "runs").glob("*/seed*/versions.json"))
    version_text = versions[0].read_text() if versions else "Chua co versions.json"
    (destination / "README.md").write_text(
        "# DeepWeeds Lab Day 2\n\n"
        f"Notebook: {notebook_url}\n\n"
        "## Cach chay lai\n\n"
        "1. Tai DeepWeeds images.zip va labels goc theo README repo; dung fold 0.\n"
        "2. Cai torch, torchvision, timm, thop, fvcore, scipy, pandas, openpyxl.\n"
        "3. Chay notebook theo thu tu EDA, backbone, ablation, inference, final.\n"
        "4. Kiem tra prediction: `python code/eval.py score --pred 'predictions/F01_seed*_test.csv' "
        "--test-csv data/labels/test_subset0.csv --labels data/labels/labels.csv`.\n\n"
        "## Phien ban thu vien da ghi nhan\n\n```json\n"
        f"{version_text}\n```\n",
        encoding="utf-8",
    )
    return destination


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=Path("."))
    parser.add_argument("--student", required=True, help="<mssv>_<ho_ten_khong_dau>")
    parser.add_argument("--notebook-url", required=True)
    parser.add_argument("--destination-root", type=Path, default=Path("submissions"))
    args = parser.parse_args()
    destination = package(args.source, args.destination_root / args.student, args.notebook_url)
    print(f"Submission ready: {destination}")


if __name__ == "__main__":
    main()
