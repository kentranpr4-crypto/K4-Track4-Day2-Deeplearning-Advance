# Kaggle: chay lab theo 4 giai doan co luu trang thai

Moi khoi `python` duoi day la **mot cell rieng**, dan nguyen khoi. Khong dung `%%bash`. Bat GPU va Internet trong Notebook Settings. Cho moi giai doan, dat `STAGE` va `PREVIOUS_STATE`, sau do chon **Save Version -> Save & Run All**; doi version hoan thanh va kiem tra Output. `Quick Save` chi luu notebook source, khong luu file trong `/kaggle/working`.

## Cell 1 - Chon giai doan va trang thai cu

Lan dau dung `STAGE = "backbones"`, `PREVIOUS_STATE = None`. Tu giai doan sau, attach file `lab_state.tar.gz` tu Output giai doan truoc thanh Kaggle Dataset, roi dien duong dan file do vao `PREVIOUS_STATE`.

```python
STAGE = "backbones"  # backbones | ablation | selection | final
PREVIOUS_STATE = None  # vi du: "/kaggle/input/datasets/<owner>/<dataset>/lab_state.tar.gz"
ASSETS = "/kaggle/input/datasets/anhtrangram/deepweeds-lab-assets"
```

## Cell 2 - Setup tu dau, khong phu thuoc session cu

```python
from pathlib import Path
import shutil, subprocess, sys, tarfile, zipfile

work = Path("/kaggle/working")
repo = work / "K4-Track4-Day2-Deeplearning-Advance"
if not repo.exists():
    subprocess.run(["git", "clone", "https://github.com/kentranpr4-crypto/K4-Track4-Day2-Deeplearning-Advance.git", str(repo)], check=True)

assets = Path(ASSETS)
code_root = assets / "starter_implemented_v3"
if not code_root.exists():
    matches = list(Path("/kaggle/input").rglob("starter_implemented_v3"))
    matches += list(Path("/kaggle/input").rglob("starter_implemented_v3.zip"))
    if not matches:
        raise FileNotFoundError("Upload starter_implemented_v3.zip vao Kaggle Dataset va attach version moi")
    code_root = matches[0]
if code_root.is_file():
    with zipfile.ZipFile(code_root) as archive:
        archive.extractall(repo)
else:
    for path in (code_root / "starter").glob("*.py"):
        shutil.copy2(path, repo / "starter" / path.name)
    for path in (code_root / "tests").glob("*.py"):
        shutil.copy2(path, repo / "tests" / path.name)
    for name in ("package_submission.py", "KAGGLE_RUN.md"):
        if (code_root / name).exists():
            shutil.copy2(code_root / name, repo / name)

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "timm", "thop", "fvcore", "scikit-learn", "openpyxl", "scipy"], check=True)
data = repo / "data"
data.mkdir(exist_ok=True)
image_dir = assets / "images"
if not image_dir.exists():
    raise FileNotFoundError(image_dir)
link = data / "images"
if not link.exists():
    link.symlink_to(image_dir, target_is_directory=True)
labels = data / "labels"
if not (labels / "train_subset0.csv").exists():
    upstream = work / "DeepWeeds_labels"
    if not upstream.exists():
        subprocess.run(["git", "clone", "-q", "https://github.com/AlexOlsen/DeepWeeds.git", str(upstream)], check=True)
    labels.mkdir(exist_ok=True)
    for path in (upstream / "labels").glob("*.csv"):
        shutil.copy2(path, labels / path.name)

if PREVIOUS_STATE:
    with tarfile.open(PREVIOUS_STATE, "r:gz") as archive:
        archive.extractall(repo)
print("Repo:", repo)
print("Images:", len(list(image_dir.glob("*.jpg"))))
print("Previous state:", PREVIOUS_STATE)
```

`starter_implemented_v3` co the la thu muc Kaggle tu giai nen zip hoac file `.zip`; Cell 2 xu ly ca hai. Cell setup se bao ro neu chua attach code/dataset.

## Cell 3 - Test code va du lieu

```python
import os, subprocess, sys
env = os.environ.copy()
env["PYTHONPATH"] = str(repo)
env["PYTHONUNBUFFERED"] = "1"
subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"], cwd=repo, env=env, check=True)
subprocess.run([sys.executable, "starter/eda.py", "--images-dir", "data/images", "--labels-dir", "data/labels", "--out-dir", "eda"], cwd=repo, env=env, check=True)
```

## Cell 4 - Chay giai doan da chon, in log theo epoch

```python
import json, os, subprocess, sys
from pathlib import Path

repo = Path("/kaggle/working/K4-Track4-Day2-Deeplearning-Advance")
env = os.environ.copy()
env["PYTHONPATH"] = str(repo)
env["PYTHONUNBUFFERED"] = "1"

def run_command(*args):
    print("RUN:", " ".join(map(str, args)), flush=True)
    subprocess.run([sys.executable, "-u", *map(str, args)], cwd=repo, env=env, check=True)

if STAGE == "backbones":
    experiments = [
        ("B01", "resnet50"),
        ("B02", "resnext50_32x4d"),
        ("B03", "convnext_tiny"),
        ("B04", "vit_small_patch16_224"),
        ("B05", "efficientnet_b0"),
    ]
    for exp_id, backbone in experiments:
        pred = repo / "predictions" / f"{exp_id}_seed0_val.csv"
        if pred.exists():
            print("SKIP completed", exp_id, flush=True)
            continue
        last = repo / "runs" / exp_id / "seed0" / "last.pt"
        run_command("starter/train.py", "--set", f"exp_id={exp_id}", f"backbone={backbone}",
                    "seed=0", "epochs=12", "batch_size=16", "num_workers=0", "amp=true",
                    "images_dir=data/images", "labels_dir=data/labels",
                    "out_dir=runs", "pred_dir=predictions", f"resume={str(last.exists()).lower()}")
    run_command("starter/profile_backbones.py", "--experiments", "B01", "B02", "B03", "B04", "B05")

elif STAGE == "ablation":
    run_command("starter/run_ablation.py", "--images-dir", "data/images", "--labels-dir", "data/labels",
                "--backbone", "vit_small_patch16_224", "--seed", "0", "--epochs", "12",
                "--batch-size", "16", "--num-workers", "0")

elif STAGE == "selection":
    run_command("starter/select_final.py", "--seed", "0", "--measure-latency")
    print((repo / "selection_val.json").read_text())

elif STAGE == "final":
    selection = repo / "selection_val.json"
    if not selection.exists():
        raise FileNotFoundError("Thieu selection_val.json; chua duoc dung test")
    print("Frozen validation choice:", selection.read_text(), flush=True)
    run_command("starter/run_final.py", "--images-dir", "data/images", "--labels-dir", "data/labels",
                "--seeds", "0", "1", "2")
    run_command("starter/build_results.py", "--root", ".", "--xlsx", "results.xlsx", "--report", "report_draft.md")

else:
    raise ValueError(f"STAGE khong hop le: {STAGE}")
```

Khong chay `final` truoc khi `selection` da hoan tat. Test chi dung o giai doan `final`.

## Cell 5 - Dong goi trang thai de chuyen sang version sau

```python
import tarfile
state = Path("/kaggle/working/lab_state.tar.gz")
with tarfile.open(state, "w:gz") as archive:
    for name in ("runs", "predictions", "curves", "eval_out", "inference_out", "eda", "figures"):
        path = repo / name
        if path.exists():
            for file in path.rglob("*"):
                if not file.is_file():
                    continue
                if file.name == "last.pt":
                    exp_id, seed_dir = file.parent.parent.name, file.parent.name
                    val_pred = repo / "predictions" / f"{exp_id}_{seed_dir}_val.csv"
                    if val_pred.exists():
                        continue  # completed run can use best.pt; no need duplicate optimizer state
                archive.add(file, arcname=str(file.relative_to(repo)))
    for name in ("ablation_summary.csv", "ablation_combination.json", "selection_val.json",
                 "final_selection.json", "results.xlsx", "report_draft.md", "report.md"):
        path = repo / name
        if path.exists():
            archive.add(path, arcname=name)
print(state, round(state.stat().st_size / 1024**2, 1), "MB")
```

Khi version hoan thanh, xem tab **Output** co `lab_state.tar.gz`, tai file va upload lai lam Kaggle Dataset/attach vao notebook cua giai doan sau. Sau do doi `STAGE` va `PREVIOUS_STATE` o Cell 1, lai chon **Save & Run All**. Moi stage la mot version. Neu stage bi loi giua chung, phai phuc hoi tu version Output gan nhat; `Quick Save` khong thay the buoc nay.

## Cell 6 - Chi sau final: hoan thien bao cao va dong goi bai nop

**Khong them Cell 6 vao notebook chay Save & Run All cac stage.** Sau khi version `final` hoan thanh, khoi phuc `lab_state.tar.gz` cua final trong mot session moi. `report_draft.md` la nhap tu so lieu that; bo sung nhan xet, ket luan, ma tran nham lan, anh loi va han che. Doi thanh `report.md`, dien MSSV, ten va link notebook that, roi moi chay cell:

```python
STUDENT = "<mssv>_<ho_ten_khong_dau>"
NOTEBOOK_URL = "https://www.kaggle.com/code/<tai-khoan>/<notebook>"
subprocess.run([sys.executable, "package_submission.py", "--source", str(repo),
                "--student", STUDENT, "--notebook-url", NOTEBOOK_URL], cwd=repo, env=env, check=True)
print("Submission:", repo / "submissions" / STUDENT)
```

Khong chay Cell 6 voi placeholder. Script se tu choi bai nop thieu prediction test, curves hoac report.
