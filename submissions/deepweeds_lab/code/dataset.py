"""DeepWeeds fold loader and deterministic evaluation preprocessing."""
from __future__ import annotations
from pathlib import Path
import random
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

NUM_CLASSES = 9
CLASS_NAMES = ["Chinee Apple", "Lantana", "Parkinsonia", "Parthenium", "Prickly Acacia", "Rubber Vine", "Siam Weed", "Snake Weed", "Negatives"]
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def load_split(labels_dir: str | Path, fold: int = 0):
    root = Path(labels_dir)
    if fold not in range(5):
        raise ValueError("fold must be 0..4")
    return tuple(pd.read_csv(root / f"{part}_subset{fold}.csv") for part in ("train", "val", "test"))


def check_split(train_df, val_df, test_df, images_dir):
    frames = dict(train=train_df, val=val_df, test=test_df)
    sets = {}
    for part, frame in frames.items():
        if not {"Filename", "Label"}.issubset(frame.columns):
            raise ValueError(f"{part}: missing columns")
        if frame[["Filename", "Label"]].isna().any().any() or frame.Filename.duplicated().any():
            raise ValueError(f"{part}: null or duplicate Filename")
        if not frame.Label.isin(range(NUM_CLASSES)).all():
            raise ValueError(f"{part}: invalid labels")
        sets[part] = set(frame.Filename)
    overlap = {f"{a}_{b}": len(sets[a] & sets[b]) for a, b in (("train", "val"), ("train", "test"), ("val", "test"))}
    if any(overlap.values()):
        raise ValueError(f"overlapping splits: {overlap}")
    total = len(set.union(*sets.values()))
    if total != 17509:
        raise ValueError(f"expected 17509 unique files, got {total}")
    n = {part: len(frame) for part, frame in frames.items()}
    if any(abs(n[part] / total - expected) > 0.01 for part, expected in (("train", .6), ("val", .2), ("test", .2))):
        raise ValueError(f"split ratio differs >1 percentage point: {n}")
    root = Path(images_dir)
    missing = [name for name in set.union(*sets.values()) if not (root / name).is_file()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} images missing, e.g. {missing[:5]}")
    result = {"n": n, "per_class": {part: {int(k): int(v) for k, v in frame.Label.value_counts().sort_index().items()} for part, frame in frames.items()}, "overlap": overlap, "union": total}
    print(result)
    return result


def build_transforms(train: bool, img_size: int = 224, aug: str = "basic", mean=None, std=None):
    from torchvision import transforms as T
    if img_size <= 0:
        raise ValueError("img_size must be positive")
    norm = [T.ToTensor(), T.Normalize(IMAGENET_MEAN if mean is None else mean, IMAGENET_STD if std is None else std)]
    if not train:
        return T.Compose([T.Resize(256), T.CenterCrop(img_size), *norm])
    a = [T.RandomResizedCrop(img_size), T.RandomHorizontalFlip()]
    if aug == "color":
        a.append(T.ColorJitter(.2, .2, .2, .1))
    elif aug == "trivial":
        a.append(T.TrivialAugmentWide())
    elif aug == "randaug":
        a.append(T.RandAugment(num_ops=2, magnitude=9))
    elif aug != "basic":
        raise ValueError(f"unknown aug: {aug}")
    return T.Compose([*a, *norm])


class DeepWeedsDataset(Dataset):
    def __init__(self, df, images_dir, transform=None):
        self.df = df.reset_index(drop=True)
        self.images_dir = Path(images_dir)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        from PIL import Image
        row = self.df.iloc[i]
        name = str(row.Filename)
        with Image.open(self.images_dir / name) as im:
            image = im.convert("RGB")
        return (self.transform(image) if self.transform else image, int(row.Label), name)


def _worker_seed(worker_id):
    import torch
    seed = torch.initial_seed() % (2**32)
    np.random.seed(seed)
    random.seed(seed)


def make_loader(df, images_dir, transform, batch_size, train, sampler=None, num_workers=2):
    import torch
    ds = DeepWeedsDataset(df, images_dir, transform)
    if sampler not in (None, "balanced"):
        raise ValueError(f"unknown sampler: {sampler}")
    weighted = None
    if train and sampler == "balanced":
        counts = df.Label.value_counts()
        weights = df.Label.map(lambda label: 1.0 / counts[label]).to_numpy(dtype=np.float64)
        weighted = torch.utils.data.WeightedRandomSampler(torch.as_tensor(weights), len(df), replacement=True)
    return torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=bool(train and weighted is None), sampler=weighted, drop_last=bool(train), num_workers=num_workers, pin_memory=torch.cuda.is_available(), worker_init_fn=_worker_seed if num_workers else None)
