"""Підрахунок матриці плутанини, вартості C = 10*FN + FP і пошук порога t*.

Правило прогнозу: шахрайство, якщо s >= t.
Дві незалежні реалізації:
  - fast_table: сортування + кумулятивні суми, O(n log n);
  - brute_table: прямий перебір для кожного кандидата (лише для малих перевірок).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

COLUMNS = ["threshold", "TP", "FP", "FN", "TN", "C"]
FN_COST = 10
FP_COST = 1


def cost(fn: int, fp: int) -> int:
    return FN_COST * int(fn) + FP_COST * int(fp)


def confusion(y: np.ndarray, s: np.ndarray, t: float) -> dict[str, int]:
    """Пряме застосування правила s >= t для одного порога."""
    tp = fp = fn = tn = 0
    for yi, si in zip(y, s):
        pred = si >= t
        if pred and yi == 1:
            tp += 1
        elif pred and yi == 0:
            fp += 1
        elif (not pred) and yi == 1:
            fn += 1
        else:
            tn += 1
    return {"TP": tp, "FP": fp, "FN": fn, "TN": tn, "C": cost(fn, fp)}


def confusion_fast(y: np.ndarray, s: np.ndarray, t: float) -> dict[str, int]:
    """Те саме, але векторизовано (для довільних порогів на великих вибірках)."""
    pred = s >= t
    tp = int(np.count_nonzero(pred & (y == 1)))
    fp = int(np.count_nonzero(pred & (y == 0)))
    fn = int(np.count_nonzero(~pred & (y == 1)))
    tn = int(np.count_nonzero(~pred & (y == 0)))
    return {"TP": tp, "FP": fp, "FN": fn, "TN": tn, "C": cost(fn, fp)}


def fast_table(y: np.ndarray, s: np.ndarray) -> pd.DataFrame:
    """Усі кандидати (унікальні оцінки та +inf) через сортування й кумулятивні суми."""
    y = np.asarray(y, dtype=np.int64)
    s = np.asarray(s, dtype=np.float64)
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    order = np.argsort(-s, kind="stable")          # за спаданням оцінки
    s_sorted = s[order]
    y_sorted = y[order]
    cum_tp = np.cumsum(y_sorted)
    cum_fp = np.cumsum(1 - y_sorted)
    # кінець кожної групи рівних оцінок: усі рівні оцінки отримують однаковий прогноз
    ends = np.flatnonzero(np.append(s_sorted[:-1] != s_sorted[1:], True))
    thr = s_sorted[ends]
    tp = cum_tp[ends]
    fp = cum_fp[ends]
    # додаємо поріг +inf: усі негативні
    thr = np.append(thr, np.inf)
    tp = np.append(tp, 0)
    fp = np.append(fp, 0)
    fn = n_pos - tp
    tn = n_neg - fp
    table = pd.DataFrame(
        {"threshold": thr, "TP": tp, "FP": fp, "FN": fn, "TN": tn, "C": FN_COST * fn + FP_COST * fp}
    )
    return table.sort_values("threshold").reset_index(drop=True)[COLUMNS]


def brute_table(y: np.ndarray, s: np.ndarray) -> pd.DataFrame:
    """Незалежний перебір: для кожного кандидата правило застосовується до кожного об'єкта."""
    cands = sorted(set(float(v) for v in s) | {float("inf")})
    rows = []
    for t in cands:
        r = confusion(y, s, t)
        rows.append({"threshold": t, **r})
    return pd.DataFrame(rows)[COLUMNS]


def best_threshold(table: pd.DataFrame) -> float:
    """Мінімум C; за рівної вартості - найбільший поріг."""
    c_min = table["C"].min()
    return float(table.loc[table["C"] == c_min, "threshold"].max())
