# Chay phan con lai cua DeepWeeds lab tren Kaggle

Repo local da co EDA va code; `results.xlsx` va bao cao cuoi phai duoc tao tu ket qua GPU that. Khong chay test de chon model.

## 1. Dua code v3 vao Kaggle

File `starter_implemented_v3.zip` nam o thu muc goc project local. Upload file nay vao Kaggle Dataset `deepweeds-lab-assets`, roi gan version moi cua dataset vao notebook. Kaggle co the hien zip thanh thu muc `starter_implemented_v3/`.

Trong notebook, kiem tra:

```python
!find /kaggle/input -name train.py -path '*starter_implemented_v3*' | head
```

Neu project trong `/kaggle/working` da mat, clone lai. Neu no van con, giu nguyen `runs/` va `predictions/`:

```python
%cd /kaggle/working
!test -d K4-Track4-Day2-Deeplearning-Advance || git clone https://github.com/kentranpr4-crypto/K4-Track4-Day2-Deeplearning-Advance.git
%cd /kaggle/working/K4-Track4-Day2-Deeplearning-Advance
```

Copy code. Duong dan duoi day dung voi cau truc dataset hien tai; neu `find` in duong dan khac thi thay bang duong dan do:

```python
!cp /kaggle/input/datasets/anhtrangram/deepweeds-lab-assets/starter_implemented_v3/starter/*.py starter/
!cp /kaggle/input/datasets/anhtrangram/deepweeds-lab-assets/starter_implemented_v3/tests/test_starter.py tests/
!cp /kaggle/input/datasets/anhtrangram/deepweeds-lab-assets/starter_implemented_v3/tests/test_implementation.py tests/
!python -m pip install -q timm thop fvcore scikit-learn openpyxl scipy
!python -m unittest discover -s tests -v
```

Neu Kaggle giu zip la file, thay ba lenh `cp` bang `!unzip -o <duong_dan_zip> -d .`.

## 2. Du lieu va ket qua backbone cu

```python
!mkdir -p data
!test -e data/images || ln -s /kaggle/input/datasets/anhtrangram/deepweeds-lab-assets/images data/images
!test -f data/labels/train_subset0.csv || git clone -q https://github.com/AlexOlsen/DeepWeeds.git /tmp/DeepWeeds
!mkdir -p data/labels
!test -f data/labels/train_subset0.csv || cp /tmp/DeepWeeds/labels/*.csv data/labels/
!ls runs/B03/seed0/best.pt predictions/B03_seed0_val.csv
```

Neu dong cuoi bao thieu file, can gan lai Output cua Kaggle version backbone vao notebook va copy `runs/`, `predictions/` tu Output do. Kaggle session moi khong tu giu `/kaggle/working` cua session cu.

## 3. EDA va profile backbone

```python
!python starter/eda.py --images-dir data/images --labels-dir data/labels --out-dir eda
!python -u starter/profile_backbones.py --experiments T00 B02 B03 B04 B05
```

`profile_backbones.py` do GMAC va latency batch 1 thuc te. Neu ViT khong dem duoc GMAC, `profile.json` ghi loi va bang se de trong, khong dien so uoc doan.

## 4. Ablation huan luyen

Chay tren B03 (ConvNeXt Tiny, seed 0) voi cung epoch/batch/split. Script thu `T01` frozen, `T02` color, `T03` RandAugment, `T04` label smoothing, `T05` focal, `T06` class-weighted CE. Sau do `T07` ket hop augmentation va loss co macro-F1 validation cao nhat trong tung truc. Nen chia thanh 4 Kaggle version nhu `KAGGLE_CELLS.md` de luu state sau moi 1-2 run.

```python
!PYTHONPATH=. python -u starter/run_ablation.py --images-dir data/images --labels-dir data/labels --baseline-id B03 --backbone convnext_tiny --seed 0 --epochs 12 --batch-size 16 --num-workers 0 --only T01 T02
```

Log in sau moi epoch va luu `history.csv`, `last.pt`, `best.pt`, `curves/`. Neu session ngat, khoi phuc Output vao cung duong dan va chay lai lenh; script bo qua prediction da xong va resume run dang do.

## 5. Chot recipe va inference chi tren validation

```python
!PYTHONPATH=. python -u starter/select_final.py --seed 0 --measure-latency
!cat selection_val.json
```

Script chon recipe macro-F1 validation cao nhat trong B03/T01..T07, roi so sanh I00 1-view, I01 horizontal flip, I02 5-crop gop probability, I03 5-crop gop logit, I04 temperature scaling. Latency p50/p95/p99 do batch 1, 10 warmup, 50 lan. T chi khop tren val. `selection_val.json` la quyet dinh chot truoc khi dung test.

## 6. Chung ket 3 seed, test mot lan/seed

Chi chay sau khi da xem `selection_val.json`. Script huan luyen F01 va baseline T00 voi seed 0, 1, 2, xuat prediction test moi seed mot lan, roi chay `eval.py score` va `grade`.

```python
!PYTHONPATH=. python -u starter/run_final.py --images-dir data/images --labels-dir data/labels --seeds 0 1 2
```

Khong doi `selection_val.json` sau khi da xem test. Neu test prediction da ton tai, script bo qua no.

## 7. Bang, bieu do va bao cao

```python
!PYTHONPATH=. python starter/build_results.py --root . --xlsx results.xlsx --report report_draft.md
!ls -lh results.xlsx report_draft.md figures/ curves/ predictions/F01_seed*_test.csv predictions/T00_seed*_test.csv
```

`report_draft.md` chi chua bang va so lieu duoc lay tu file that. Can bo sung phan nhan xet bang loi van cua ban: so sanh voi std, anh du doan sai, trade-off latency, rui ro chia ngau nhien khong theo dia diem, va ly do chon cau hinh. Doi ten thanh `report.md` khi da hoan thanh. Khong ghi so chua do duoc.

## 8. Dong goi dung cau truc nop bai

Sau khi co `results.xlsx`, `report.md`, curves va prediction test cua F01/T00 moi seed, copy `package_submission.py` tu project local vao Kaggle hoac chay tren may local sau khi tai output Kaggle ve:

```bash
python package_submission.py --source . --student <mssv>_<ho_ten_khong_dau> --notebook-url https://www.kaggle.com/code/<tai_khoan>/<notebook>
```

Script tao `submissions/<mssv>_<ten>/` gom README rieng, xlsx, report, curves, predictions va code. No dung neu thieu file, thieu curve, hoac prediction test khong khop CSV goc. Dataset va checkpoint lon khong duoc copy vao bai nop.

## 9. Luu output

Sau moi giai doan, luu Kaggle version hoac tai artifacts xuong. `Quick Save` chi luu notebook source; de luu output cua mot lan chay, dung `Save Version` va kiem tra tab Output. Khi quay lai session moi, can attach/copy Output truoc khi resume.
