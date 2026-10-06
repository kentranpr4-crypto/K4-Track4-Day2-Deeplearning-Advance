# DeepWeeds Lab Day 2 - 2026A02939 Tran Cao Quoc Dinh

## Trang thai

Day la ban ket qua tam thoi. Da co 5 backbone, 7 ablation, 12 file du doan validation, EDA, bieu do va log. Chua co du doan test 3 seed, ket qua suy luan cuoi, `report.md` va link notebook; **chua du dieu kien nop bai**. So trong `results.xlsx` va `report_draft.md` chi lay tu cac file du doan/log da chay that.

## Notebook

Link Kaggle notebook: https://www.kaggle.com/code/anhtrangram/10-39/edit
Link version chay final se duoc bo sung sau khi hoan thanh.

## Du lieu va cach chay lai

Dung DeepWeeds 17.509 anh, fold 0 goc. Anh va checkpoint lon khong duoc commit. Chi so duoc tinh bang `code/eval.py`; validation: 3.501 anh. Cac script chay theo thu tu trong `code/KAGGLE_CELLS.md`: EDA, backbone, ablation, chon suy luan tren validation, final 3 seed, tong hop bang/bao cao.

Code can PyTorch, torchvision, timm, scipy, pandas, matplotlib, Pillow, thop/fvcore va openpyxl. Phien ban da dung duoc ghi trong `logs/<exp_id>/seed0/versions.json`. Cac run validation hien tai dung seed 0; final phai dung seed 0, 1, 2. `logs/` chua config, history va profile cho tung run de doi chieu `results.xlsx`.

## Viec con thieu truoc khi nop

- Lay Output cua Kaggle final va bo sung `predictions/F01_seed{0,1,2}_test.csv` va `predictions/T00_seed{0,1,2}_test.csv`.
- Bo sung ket qua inference validation, latency va final vao `results.xlsx`; kiem tra bang `eval.py score` va `eval.py grade`.
- Hoan thien `report.md` tu `report_draft.md`, ghi ket luan dua tren so lieu test that, phan tich loi va han che.
- Ghi link version notebook Kaggle chay final va xac nhan du 3 seed.
