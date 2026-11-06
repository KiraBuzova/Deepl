from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

import config as C
import split as S


def train_transform():
    return transforms.Compose([
        transforms.RandomResizedCrop(C.IMG_SIZE, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(C.MEAN, C.STD),
    ])


def eval_transform():
    return transforms.Compose([
        transforms.Resize((C.IMG_SIZE, C.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(C.MEAN, C.STD),
    ])


class SplitDataset(Dataset):

    def __init__(self, rows, data_root, transform):
        self.rows = rows
        self.transform = transform
        self.train_dir, self.test_dir = S.resolve_dirs(data_root)

    def path(self, i):
        r = self.rows[i]
        base = self.train_dir if r["part"] == "train" else self.test_dir
        return base / r["label"] / r["file"]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        img = Image.open(self.path(i)).convert("RGB")
        return self.transform(img), C.CLASSES.index(self.rows[i]["label"])


def part_rows(all_rows, part):
    return [r for r in all_rows if r["part"] == part]


def make_train_loader(all_rows, data_root):
    ds = SplitDataset(part_rows(all_rows, "train"), data_root, train_transform())
    return DataLoader(ds, batch_size=C.BATCH_TRAIN, shuffle=True, drop_last=False, num_workers=0)


def make_eval_loader(all_rows, data_root, part):
    ds = SplitDataset(part_rows(all_rows, part), data_root, eval_transform())
    return DataLoader(ds, batch_size=C.BATCH_EVAL, shuffle=False, num_workers=0)


def labels_of(all_rows, part):
    import numpy as np
    return np.array([C.CLASSES.index(r["label"]) for r in part_rows(all_rows, part)], dtype=np.int64)
