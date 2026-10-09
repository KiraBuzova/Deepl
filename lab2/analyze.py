"""ЛР2, пункти 2-4: аналіз порогів зі збережених прогнозів (без навчання).

Запуск:
    uv run python analyze.py --saved results --out analysis
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_auc_score, roc_curve

from lab2_starter import load_predictions
from thresholds import (
    COLUMNS,
    best_threshold,
    brute_table,
    confusion_fast,
    fast_table,
)

# Набори з умови завдання (п. 3) та очікуваний поріг, розрахований вручну
SETS = {
    "A": (
        np.array([0.9, 0.7, 0.7, 0.4, 0.2, 0.1]),
        np.array([0, 1, 0, 1, 0, 0]),
        0.4,
    ),
    "B": (
        np.array([0.9, 0.6, 0.3]),
        np.array([0, 0, 0]),
        math.inf,
    ),
    "C": (
        np.array([0.8] + [0.5] * 11 + [0.2]),
        np.array([1] + [1] + [0] * 10 + [0]),
        0.8,
    ),
}

_log_lines: list[str] = []


def log(*parts: object) -> None:
    line = " ".join(str(p) for p in parts)
    print(line)
    _log_lines.append(line)


def jnum(x: float) -> float | str | None:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return None
    if isinstance(x, float) and math.isinf(x):
        return "inf"
    return x


def safe_div(a: float, b: float) -> float:
    return a / b if b else float("nan")


def run_checks(out: Path) -> bool:
    """П. 3: швидкий алгоритм проти прямого перебору на наборах A/B/C."""
    ok_all = True
    summary = []
    for name, (s, y, expected) in SETS.items():
        fast = fast_table(y, s)
        slow = brute_table(y, s)
        fast.to_csv(out / f"check_{name}_fast.csv", index=False)
        slow.to_csv(out / f"check_{name}.csv", index=False)
        same_table = fast.equals(slow) or (
            np.array_equal(fast[COLUMNS[1:]].to_numpy(), slow[COLUMNS[1:]].to_numpy())
            and np.array_equal(fast["threshold"].to_numpy(), slow["threshold"].to_numpy())
        )
        t_fast, t_slow = best_threshold(fast), best_threshold(slow)
        ok = same_table and t_fast == t_slow == expected
        ok_all &= ok
        log(f"\n=== Набір {name} (прямий перебір) ===")
        log(slow.to_string(index=False))
        log(f"t* (швидкий) = {t_fast}, t* (перебір) = {t_slow}, очікувано = {expected}, "
            f"таблиці збігаються: {same_table} -> {'OK' if ok else 'ПОМИЛКА'}")
        summary.append({
            "set": name,
            "n_candidates": len(slow),
            "tables_equal": bool(same_table),
            "t_star_fast": str(t_fast),
            "t_star_brute": str(t_slow),
            "t_star_expected": str(expected),
            "passed": bool(ok),
        })
    pd.DataFrame(summary).to_csv(out / "checks_summary.csv", index=False)
    return ok_all


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--saved", type=Path, required=True, help="Тека з predictions.npz")
    ap.add_argument("--out", type=Path, default=Path("analysis"))
    args = ap.parse_args()
    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)

    data = load_predictions(args.saved)

    # ---- П. 3: перевірки (до будь-якої роботи з test) ----
    log("##### Перевірка пошуку порога на наборах A/B/C #####")
    if not run_checks(out):
        log("\nПеревірки НЕ пройдено - тестову оцінку не виконуємо.")
        (out / "analysis_log.txt").write_text("\n".join(_log_lines) + "\n", encoding="utf-8")
        sys.exit(1)
    log("\nУсі три перевірки пройдено.")

    # ---- П. 2: validation ----
    yv = data["validation_y"].astype(np.int64)
    sv = data["validation_scores"].astype(np.float64)
    log("\n##### Validation #####")
    log(f"n={len(yv)}, позитивних={int(yv.sum())}")

    table = fast_table(yv, sv)
    table.to_csv(out / "validation_thresholds.csv", index=False, float_format="%.17g")
    t_star = best_threshold(table)
    log(f"Кандидатів: {len(table)}; t* = {t_star!r}")

    rows = []
    for label, t in (("Стандартний поріг 0.5", 0.5), ("Мінімум вартості t*", t_star), ("Усі негативні +inf", math.inf)):
        r = confusion_fast(yv, sv, t)
        rows.append({"rule": label, "threshold": t, **r})
    summary = pd.DataFrame(rows)
    summary.to_csv(out / "validation_summary.csv", index=False, float_format="%.17g")
    log(summary.to_string(index=False))

    # ROC-AUC та криві
    auc = float(roc_auc_score(yv, sv))
    log(f"ROC-AUC (validation) = {auc:.6f}")
    prec, rec, _ = precision_recall_curve(yv, sv)
    fpr, tpr, _ = roc_curve(yv, sv)
    plt.figure(figsize=(5, 4))
    plt.plot(rec, prec)
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("PR-крива (validation)")
    plt.grid(alpha=0.3); plt.tight_layout(); plt.savefig(out / "pr_curve.png", dpi=150); plt.close()
    plt.figure(figsize=(5, 4))
    plt.plot(fpr, tpr, label=f"AUC = {auc:.4f}")
    plt.plot([0, 1], [0, 1], "--", color="gray")
    plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title("ROC-крива (validation)")
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout(); plt.savefig(out / "roc_curve.png", dpi=150); plt.close()

    # ---- П. 4: test (поріг t* уже зафіксовано) ----
    yt = data["test_y"].astype(np.int64)
    st = data["test_scores"].astype(np.float64)
    amount = data["test_amount"]
    rowids = data["test_row_ids"]
    log("\n##### Test (поріг t* зафіксовано) #####")
    cm = confusion_fast(yt, st, t_star)
    precision = safe_div(cm["TP"], cm["TP"] + cm["FP"])
    recall = safe_div(cm["TP"], cm["TP"] + cm["FN"])
    log(f"t* = {t_star!r}")
    log(cm)
    log(f"precision = {precision}  (TP+FP = {cm['TP'] + cm['FP']})")
    log(f"recall    = {recall}  (TP+FN = {cm['TP'] + cm['FN']})")
    for label, t in (("0.5", 0.5), ("+inf", math.inf)):
        log(f"[довідково] test при t={label}: {confusion_fast(yt, st, t)}")

    pred = st >= t_star
    err_rows = []
    for kind, mask in (("FP", pred & (yt == 0)), ("FN", ~pred & (yt == 1))):
        idx = np.flatnonzero(mask)
        idx = idx[np.argsort(rowids[idx], kind="stable")][:5]   # перші 5 за номером рядка CSV
        for i in idx:
            err_rows.append({
                "csv_row": int(rowids[i]),
                "error_type": kind,
                "label": int(yt[i]),
                "score": float(st[i]),
                "amount": float(amount[i]),
                "score_minus_t": (float(st[i] - t_star) if math.isfinite(t_star) else float("nan")),
            })
    errors = pd.DataFrame(err_rows, columns=["csv_row", "error_type", "label", "score", "amount", "score_minus_t"])
    errors.to_csv(out / "test_errors.csv", index=False, float_format="%.10g")
    log("\nПерші помилки (за номером рядка CSV):")
    log(errors.to_string(index=False) if len(errors) else "(помилок немає)")

    results = {
        "t_star": jnum(float(t_star)),
        "roc_auc_validation": auc,
        "validation": {r["rule"]: {k: jnum(v) if k == "threshold" else int(v) for k, v in r.items() if k != "rule"}
                       for r in rows},
        "test": {"confusion": cm, "precision": jnum(precision), "recall": jnum(recall)},
    }
    (out / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (out / "analysis_log.txt").write_text("\n".join(_log_lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
