# DeepWeeds Lab Day 2 - 2026A02939 Tran Cao Quoc Dinh

## Notebook va ket qua

Kaggle notebook da chay: https://www.kaggle.com/code/anhtrangram/10-39
Notebook tai lap tu dau nam o `code/reproduce.ipynb`; can GPU va Internet, co the chia cac giai doan theo `code/KAGGLE_CELLS.md` de tranh gioi han phien Kaggle.

`results.xlsx` co 7 sheet: Backbones, Training, Inference, Final, PerClass, Latency va Summary. `report.md` trinh bay ket luan; `predictions/` co test/validation cho F01, T00 (seed 0/1/2) va validation cua cac backbone/ablation. `curves/` co bieu do cho moi lan train. `logs/` giu config, history, tag pretrained va version de truy nguoc so lieu. Dataset va checkpoint lon khong nam trong Git.

## Du lieu va moi truong

Dung DeepWeeds 17.509 anh RGB, fold 0 goc cua tac gia: 10.501 train, 3.501 val, 3.507 test. Khong chia lai, khong gop val vao train. Anh tu Zenodo (MD5 `b7b30f96d466fba86016aa5a26606e0f`); CSV nhan tu repo AlexOlsen/DeepWeeds. Kaggle train tren Tesla T4; run final ghi torch `2.11.0+cu128`, torchvision `0.26.0+cu128`, timm `1.0.29` trong `logs/<exp_id>/seed<k>/versions.json`. Can scipy, pandas, Pillow, matplotlib, thop/fvcore va openpyxl.

## Cach chay lai

1. Clone repo goc va copy cac file `code/*.py` vao thu muc `starter/` cua repo; giu `eval.py` goc o thu muc root. Dung Kaggle notebook o link tren de tai dataset va chay theo dung thu tu. Cung cap anh o `data/images/` va CSV goc o `data/labels/`; chay `python starter/eda.py --images-dir data/images --labels-dir data/labels`.
2. Chay backbone va ablation theo thu tu trong `code/KAGGLE_CELLS.md`. Cac thiet lap tung run nam o `logs/`; seed validation la 0.
3. Chot recipe va inference tren val: `python starter/select_final.py --seed 0 --baseline-id B03 --measure-latency`. Quyet dinh da luu trong `selection_val.json`.
4. Chay final dung ba seed 0/1/2 theo `starter/run_final.py`; khong dung test de chon lai cau hinh.
5. Kiem tra so lieu tu file du doan bang `code/eval.py`:

```bash
python eval.py score --pred 'predictions/F01_seed*_test.csv' --test-csv data/labels/test_subset0.csv --labels data/labels/labels.csv --tag F01
python eval.py grade --final 'predictions/F01_seed*_test.csv' --baseline 'predictions/T00_seed*_test.csv' --test-csv data/labels/test_subset0.csv --labels data/labels/labels.csv --latency-p95-ms 32.6162 --latency-method proper
```

`eval_out/` luu ket qua chinh thuc da tinh lai. `inference_out/` luu nam phuong phap validation va latency. Do tre chi tinh forward tren GPU, khong tinh doc anh/tien xu ly. Bao cao ghi ro gioi han va mot luot final tuong tac da bi gian doan truoc version hoan chinh.

## Gioi han da khai bao

Chua co bang chung loss ban dau / overfit mot batch / anh sau augmentation truoc training, latency batch lon, va validation 5-crop du ba seed. Khong dien so lieu cho cac muc nay. Ba seed final va tat ca prediction test da duoc doi chieu voi CSV goc; cac gioi han va luot test bi gian doan duoc trinh bay trong `report.md` de giang vien danh gia.
