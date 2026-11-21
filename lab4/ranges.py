"""Діапазони ваг по вихідних каналах для кожної згортки FP32-графа."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper


def conv_ranges(path: str | Path) -> dict:
    g = onnx.load(str(path)).graph
    inits = {i.name: i for i in g.initializer}
    rows = []
    for node in g.node:
        if node.op_type != "Conv":
            continue
        wname = node.input[1]
        if wname not in inits:
            raise KeyError(f"Ваги {wname} вузла {node.name} не знайдено в graph.initializer")
        w = numpy_helper.to_array(inits[wname])  # (Cout, Cin, kH, kW)
        alpha = np.abs(w).reshape(w.shape[0], -1).max(axis=1)
        r = float(alpha.max() / np.median(alpha))
        rows.append({"node": node.name, "weight": wname, "shape": list(w.shape),
                     "alpha_max": float(alpha.max()), "alpha_median": float(np.median(alpha)),
                     "r": r, "log2_r": math.log2(r)})
    top3 = sorted(rows, key=lambda x: x["r"], reverse=True)[:3]
    med_r = float(np.median([x["r"] for x in rows]))
    return {"n_conv": len(rows), "per_conv": rows, "top3": top3,
            "median_r": med_r, "median_log2_r_bits": math.log2(med_r)}
