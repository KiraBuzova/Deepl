"""Поділ, контрольні числа, калібрувальна вибірка, перетворення та завантаження зображень."""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from PIL import Image
from torchvision import transforms

CLASSES = ["buildings", "forest", "glacier", "mountain", "sea", "street"]
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)

# частина -> (кількість, сума числових імен файлів)
SPLIT_EXPECTED = {"train": (1200, 11_993_084), "test": (1502, 33_351_444)}
CALIB_EXPECTED_N = 200
CALIB_EXPECTED_SUM = 1_921_470
CALIB_EXPECTED_CLASSES = {
    "buildings": 40, "forest": 35, "glacier": 36,
    "mountain": 33, "sea": 29, "street": 27,
}
PART_DIR = {"train": "seg_train", "test": "seg_test"}


def numeric_name(file: str) -> int:
    return int(Path(file).stem)


def read_split(path: str | Path) -> list[dict]:
    """Рядки split.csv у вихідному порядку: file, label (назва класу), part."""
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            label = r["label"].strip()
            if label.isdigit():  # на випадок, якщо в ЛР3 мітки записано індексами
                label = CLASSES[int(label)]
            rows.append({"file": Path(r["file"].strip()).name, "label": label, "part": r["part"].strip()})
    return rows


def part_rows(rows: list[dict], part: str) -> list[dict]:
    return [r for r in rows if r["part"] == part]


def check_split(rows: list[dict]) -> dict:
    """Контроль train/test за кількістю та сумою числових імен. Кидає AssertionError при розбіжності."""
    out = {}
    for part, (n_exp, s_exp) in SPLIT_EXPECTED.items():
        sel = part_rows(rows, part)
        n, s = len(sel), sum(numeric_name(r["file"]) for r in sel)
        out[part] = {"count": n, "sum": s, "expected_count": n_exp, "expected_sum": s_exp,
                     "ok": (n, s) == (n_exp, s_exp)}
        assert (n, s) == (n_exp, s_exp), (
            f"Контроль поділу '{part}' не збігся: {n}/{s} проти {n_exp}/{s_exp}. Виправте ЛР3.")
    return out


def calibration_rows(rows: list[dict]) -> list[dict]:
    """200 зображень train: відсортувати як рядки за іменем файлу, взяти permutation(1200)[:200]."""
    train = sorted(part_rows(rows, "train"), key=lambda r: r["file"])
    assert len(train) == 1200
    idx = np.random.default_rng(0).permutation(1200)[:200]
    return [train[i] for i in idx]


def check_calibration(cal: list[dict]) -> dict:
    n = len(cal)
    s = sum(numeric_name(r["file"]) for r in cal)
    per_class = {c: sum(1 for r in cal if r["label"] == c) for c in CLASSES}
    ok = (n == CALIB_EXPECTED_N and s == CALIB_EXPECTED_SUM and per_class == CALIB_EXPECTED_CLASSES)
    out = {"count": n, "sum": s, "per_class": per_class, "ok": ok}
    assert ok, f"Калібрувальна вибірка не збіглася з контролем: {out}"
    return out


def make_transform(img_size: int):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])


def image_path(root: str | Path, row: dict) -> Path:
    d = PART_DIR[row["part"]]
    return Path(root) / d / d / row["label"] / row["file"]


def load_array(root: str | Path, rows: list[dict], img_size: int) -> np.ndarray:
    """Усі зображення як float32 [N, 3, S, S] у порядку rows."""
    tf = make_transform(img_size)
    out = np.empty((len(rows), 3, img_size, img_size), dtype=np.float32)
    for i, r in enumerate(rows):
        with Image.open(image_path(root, r)) as im:
            out[i] = tf(im.convert("RGB")).numpy()
    return out


def labels_of(rows: list[dict]) -> np.ndarray:
    return np.array([CLASSES.index(r["label"]) for r in rows], dtype=np.int64)
