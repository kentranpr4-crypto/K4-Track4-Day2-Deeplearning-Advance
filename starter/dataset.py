"""dataset.py - đọc DeepWeeds, kiểm tra chia dữ liệu, transform, DataLoader.

PSEUDO-CODE: bạn tự hoàn thiện mọi hàm có `raise NotImplementedError`.
Quy tắc chia dữ liệu bắt buộc (S1-S6) nằm ở README.md, mục 2.1. Đọc trước khi viết.

Giao diện bạn phải giữ (để notebook, train.py và eval.py ghép được với nhau):
    load_split(labels_dir, fold=0)            -> (train_df, val_df, test_df)
    check_split(train_df, val_df, test_df, images_dir) -> dict  (số liệu để ghi báo cáo)
    build_transforms(train, img_size, aug)    -> torchvision transform
    DeepWeedsDataset[i]                       -> (image_tensor, label:int, filename:str)
    make_loader(df, images_dir, transform, batch_size, train, sampler, num_workers)
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

NUM_CLASSES = 9
# Thứ tự lớp theo cột `Label` của labels.csv (0 = Chinee Apple ... 7 = Snake Weed, 8 = Negatives).
CLASS_NAMES = [
    "Chinee Apple", "Lantana", "Parkinsonia", "Parthenium", "Prickly Acacia",
    "Rubber Vine", "Siam Weed", "Snake Weed", "Negatives",
]
IMAGENET_MEAN = (0.485, 0.456, 0.406)  # đổi nếu trọng số timm bạn dùng yêu cầu mean/std khác
IMAGENET_STD = (0.229, 0.224, 0.225)


def load_split(labels_dir: str | Path, fold: int = 0):
    """Đọc train_subset{fold}.csv, val_subset{fold}.csv, test_subset{fold}.csv (S1).

    Mỗi file có cột `Filename, Label, Species`. Trả về ba DataFrame.
    KHÔNG sửa, lọc hay chia lại dữ liệu.

    TODO:
      - đọc ba file CSV bằng pandas
      - trả về (train_df, val_df, test_df)
    """
    root = Path(labels_dir)
    out = tuple(pd.read_csv(root / f"{name}_subset{fold}.csv") for name in ("train", "val", "test"))
    for df in out:
        if not {"Filename", "Label"}.issubset(df.columns):
            raise ValueError("CSV phải có cột Filename và Label")
    return out


def check_split(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame,
                images_dir: str | Path) -> dict:
    """Kiểm tra bắt buộc trước khi train (README.md, mục 2.1). In ra và trả về dict số liệu.

    TODO kiểm tra, mỗi ý lỗi thì `assert` / raise để dừng ngay:
      1. số ảnh mỗi tập và số ảnh mỗi lớp trong từng tập (kỳ vọng xấp xỉ 60/20/20)
      2. giao của từng cặp tập theo Filename phải RỖNG (train∩val, train∩test, val∩test)
      3. hợp ba tập phải bằng đúng 17.509 ảnh
      4. mọi Filename đều tồn tại trong `images_dir`
    Trả về dict, ví dụ {"n": {...}, "per_class": {...}, "overlap": {...}} để dán vào báo cáo.
    """
    frames = (train_df, val_df, test_df)
    names = [set(df["Filename"].astype(str)) for df in frames]
    for label, df, unique in zip(("train", "val", "test"), frames, names):
        if len(df) != len(unique):
            raise ValueError(f"{label} có Filename trùng trong cùng tập")
        values = pd.to_numeric(df["Label"], errors="coerce")
        if values.isna().any() or not values.between(0, NUM_CLASSES - 1).all():
            raise ValueError(f"{label} có Label ngoài 0..{NUM_CLASSES - 1}")
    overlap = {"train_val": len(names[0] & names[1]), "train_test": len(names[0] & names[2]),
               "val_test": len(names[1] & names[2])}
    if any(overlap.values()):
        raise ValueError(f"ảnh bị trùng giữa các tập: {overlap}")
    union = set().union(*names)
    if len(union) != 17509:
        raise ValueError(f"hợp ba tập phải có 17509 ảnh, nhận {len(union)}")
    image_root = Path(images_dir)
    missing = sorted(f for f in union if not (image_root / f).is_file())
    if missing:
        raise FileNotFoundError(f"thiếu {len(missing)} ảnh, ví dụ {missing[:3]}")
    per_class = {"train": train_df["Label"].value_counts().sort_index().to_dict(),
                 "val": val_df["Label"].value_counts().sort_index().to_dict(),
                 "test": test_df["Label"].value_counts().sort_index().to_dict()}
    result = {"n": {"train": len(train_df), "val": len(val_df), "test": len(test_df)},
              "per_class": per_class, "overlap": overlap, "union": len(union)}
    print(result)
    return result


def build_transforms(train: bool, img_size: int = 224, aug: str = "basic"):
    """Tạo transform. `aug` chọn mức augmentation; bạn tự định nghĩa các giá trị.

    Gợi ý các giá trị `aug` (trục B của GUIDE.md mục 3): "basic", "color", "trivial", "randaug".
    Mixup/CutMix trộn theo batch nên nằm ở losses.py, không ở đây.

    Train (basic): RandomResizedCrop(img_size) + lật ngang + ToTensor + Normalize.
    Val/test: ảnh gốc 256x256 -> CenterCrop(img_size) (hoặc giữ nguyên 256; ghi rõ bạn chọn gì)
              + ToTensor + Normalize. KHÔNG augmentation ngẫu nhiên khi đánh giá.

    TODO: dùng torchvision.transforms (hoặc v2). Lưu ý: lật dọc có hợp lệ với ảnh cỏ dại không?
    """
    try:
        from torchvision import transforms
    except ImportError as exc:
        raise ImportError("Cần cài torch và torchvision để tạo transform") from exc
    if not train:
        ops = [transforms.Resize(int(img_size * 256 / 224)), transforms.CenterCrop(img_size)]
    else:
        ops = [transforms.RandomResizedCrop(img_size, scale=(0.7, 1.0)), transforms.RandomHorizontalFlip()]
        if aug not in {"basic", "color", "trivial", "randaug"}:
            raise ValueError(f"augmentation không hợp lệ: {aug}")
        if aug == "color":
            ops.append(transforms.ColorJitter(0.25, 0.25, 0.25, 0.05))
        if aug == "trivial":
            ops.append(transforms.TrivialAugmentWide())
        if aug == "randaug":
            ops.append(transforms.RandAugment(num_ops=2, magnitude=7))
    ops.extend([transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
    return transforms.Compose(ops)


class DeepWeedsDataset:
    """Dataset đọc ảnh từ `images_dir` theo DataFrame (Filename, Label).

    __getitem__(i) phải trả về (ảnh đã transform, nhãn int, tên file str).
    Tên file cần có để ghi `predictions/*.csv` đúng định dạng của eval.py.

    TODO:
      - __init__(self, df, images_dir, transform): giữ df, mở ảnh bằng PIL, chuyển sang RGB
      - __len__
      - __getitem__ -> (tensor, int(label), filename)
      - (tuỳ chọn) nạp trước ảnh vào RAM nếu bị nghẽn đọc đĩa trên Colab
    """

    def __init__(self, df: pd.DataFrame, images_dir: str | Path, transform=None):
        self.df = df.reset_index(drop=True)
        self.images_dir = Path(images_dir)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, i: int):
        from PIL import Image
        row = self.df.iloc[i]
        filename = str(row["Filename"])
        with Image.open(self.images_dir / filename) as im:
            image = im.convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, int(row["Label"]), filename


def make_loader(df: pd.DataFrame, images_dir: str | Path, transform, batch_size: int,
                train: bool, sampler: str | None = None, num_workers: int = 2):
    """Tạo DataLoader.

    TODO:
      - train=True: shuffle (hoặc dùng sampler); train=False: không shuffle, giữ thứ tự df
        (thứ tự phải ổn định để ghép logit với Filename)
      - sampler=None | "balanced": "balanced" dùng WeightedRandomSampler với trọng số
        1/(số ảnh của lớp) (trục D của GUIDE.md mục 3)
      - drop_last=True khi train nếu batch cuối quá nhỏ làm BatchNorm không ổn định
      - pin_memory=True, num_workers hợp lý; seed cho worker (worker_init_fn) để tái lập
    """
    try:
        import torch
        from torch.utils.data import DataLoader, WeightedRandomSampler
    except ImportError as exc:
        raise ImportError("Cần cài torch để tạo DataLoader") from exc
    ds = DeepWeedsDataset(df, images_dir, transform)
    def seed_worker(worker_id):
        import random
        import numpy as np
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)
    kwargs = dict(batch_size=batch_size, num_workers=num_workers,
                  pin_memory=torch.cuda.is_available(), worker_init_fn=seed_worker if num_workers else None)
    if sampler == "balanced":
        counts = df["Label"].value_counts().to_dict()
        weights = torch.as_tensor([1.0 / counts[int(y)] for y in df["Label"]], dtype=torch.double)
        kwargs["sampler"] = WeightedRandomSampler(weights, len(weights), replacement=True)
    else:
        kwargs["shuffle"] = bool(train)
    kwargs["drop_last"] = bool(train)
    return DataLoader(ds, **kwargs)
