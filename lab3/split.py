import csv
from pathlib import Path

import numpy as np

import config as C


def resolve_dirs(data_root):
    root = Path(data_root)

    def pick(name):
        nested = root / name / name
        if nested.is_dir():
            return nested
        flat = root / name
        if flat.is_dir() and (flat / C.CLASSES[0]).is_dir():
            return flat
        raise FileNotFoundError(f"Не знайдено теку {nested} (шлях до набору: {root})")

    return pick("seg_train"), pick("seg_test")


def _sorted_files(folder):
    return sorted(p.name for p in Path(folder).iterdir() if p.suffix.lower() == ".jpg")


def build_split(data_root):
    train_dir, test_dir = resolve_dirs(data_root)
    rows = []

    rng = np.random.default_rng(C.SPLIT_SEED)
    train_rows = []
    for label in C.CLASSES:
        files = _sorted_files(train_dir / label)
        idx = np.arange(len(files))
        rng.shuffle(idx)
        chosen = sorted(files[i] for i in idx[:C.N_TRAIN_PER_CLASS])
        train_rows += [(f, label, "train") for f in chosen]

    rng = np.random.default_rng(C.SPLIT_SEED)
    val_rows, test_rows = [], []
    for label in C.CLASSES:
        files = _sorted_files(test_dir / label)
        idx = np.arange(len(files))
        rng.shuffle(idx)
        cut = len(files) // 2
        val_rows += [(f, label, "validation") for f in sorted(files[i] for i in idx[:cut])]
        test_rows += [(f, label, "test") for f in sorted(files[i] for i in idx[cut:])]

    rows = train_rows + val_rows + test_rows
    return rows


def write_split(rows, path):
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["file", "label", "part"])
        w.writerows(rows)


def read_split(path):
    with open(path, newline="", encoding="utf-8") as fh:
        return [dict(r) for r in csv.DictReader(fh)]


def control_table(rows):
    table = {}
    for r in rows:
        f, label, part = (r["file"], r["label"], r["part"]) if isinstance(r, dict) else r
        n, s = table.get((label, part), (0, 0))
        table[(label, part)] = (n + 1, s + int(Path(f).stem))
    return table


def check_split(rows):
    table = control_table(rows)
    parts = ["train", "validation", "test"]
    ok_all = True
    lines = ["| Клас | train | validation | test |", "|---|---|---|---|"]
    for label in C.CLASSES:
        cells = []
        for part in parts:
            got = table.get((label, part), (0, 0))
            exp = C.EXPECTED_SPLIT[(label, part)]
            same = got == exp
            ok_all &= same
            cells.append(f"{got[0]} / {got[1]:,}".replace(",", " ") + (" ✓" if same else f" ✗ (очікувано {exp[0]} / {exp[1]})"))
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    cells = []
    for part in parts:
        n = sum(table.get((l, part), (0, 0))[0] for l in C.CLASSES)
        s = sum(table.get((l, part), (0, 0))[1] for l in C.CLASSES)
        same = (n, s) == C.EXPECTED_TOTALS[part]
        ok_all &= same
        cells.append(f"{n} / {s:,}".replace(",", " ") + (" ✓" if same else " ✗"))
    lines.append("| **Разом** | " + " | ".join(cells) + " |")

    sets = {p: set() for p in parts}
    for r in rows:
        f, label, part = (r["file"], r["label"], r["part"]) if isinstance(r, dict) else r
        sets[part].add(f)
    overlap = (sets["train"] & sets["validation"]) | (sets["train"] & sets["test"]) | (sets["validation"] & sets["test"])
    disjoint = len(overlap) == 0
    ok_all &= disjoint
    md = "\n".join(lines)
    md += f"\n\nУсі 18 клітинок і підсумки збігаються з контрольною таблицею: **{'так' if ok_all and disjoint else 'НІ'}**."
    md += f"\n\nФайлів, що входять у дві частини одночасно: **{len(overlap)}** " \
          f"({'частини не перетинаються' if disjoint else 'ПОМИЛКА'})."
    return ok_all, md, {"disjoint": disjoint, "overlap": sorted(overlap)[:10]}


def check_raw_counts(data_root):
    train_dir, test_dir = resolve_dirs(data_root)
    n_tr = sum(len(_sorted_files(train_dir / l)) for l in C.CLASSES)
    n_te = sum(len(_sorted_files(test_dir / l)) for l in C.CLASSES)
    return (n_tr, n_te) == (C.EXPECTED_RAW_COUNTS["train"], C.EXPECTED_RAW_COUNTS["test"]), n_tr, n_te
