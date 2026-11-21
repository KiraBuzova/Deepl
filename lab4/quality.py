"""Метрики якості та парне порівняння FP32 / INT8."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import f1_score


def basic_metrics(logits: np.ndarray, y: np.ndarray) -> dict:
    pred = logits.argmax(1)
    return {"macro_f1": float(f1_score(y, pred, average="macro")),
            "accuracy": float((pred == y).mean())}


def export_check(logits_torch: np.ndarray, logits_onnx: np.ndarray, y: np.ndarray) -> dict:
    pt, ox = logits_torch.argmax(1), logits_onnx.argmax(1)
    return {
        "max_abs_logit_diff": float(np.abs(logits_onnx - logits_torch).max()),
        "class_match": float((pt == ox).mean()),
        "macro_f1_pytorch": float(f1_score(y, pt, average="macro")),
        "macro_f1_onnx_fp32": float(f1_score(y, ox, average="macro")),
    }


def compare_to_fp32(logits_fp32: np.ndarray, logits_q: np.ndarray, y: np.ndarray) -> dict:
    """Рядок таблиці п. 3 + парне порівняння: b — INT8 правильно, FP32 помиляється; c — навпаки."""
    pf, pq = logits_fp32.argmax(1), logits_q.argmax(1)
    cf, cq = pf == y, pq == y
    n = len(y)
    b = int((cq & ~cf).sum())
    c = int((~cq & cf).sum())
    delta = (b - c) / n
    se = float(np.sqrt(b + c) / n)
    threshold = 2 * np.sqrt(b + c)
    if abs(b - c) <= threshold:
        verdict = "різниця якості не відрізняється від шуму"
    else:
        verdict = "різниця відрізняється від шуму: " + ("INT8 краща" if b > c else "INT8 гірша")
    p = float(cf.mean())
    return {
        **basic_metrics(logits_q, y),
        "agreement_with_fp32": float((pf == pq).mean()),
        "b": b, "c": c, "n": n,
        "delta_acc": float(delta), "se_paired": se,
        "abs_b_minus_c": abs(b - c), "threshold_2sqrt_b_plus_c": float(threshold),
        "verdict": verdict,
        "se_single_model_fp32": float(np.sqrt(p * (1 - p) / n)),
    }
