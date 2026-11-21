"""Вимірювання затримки за протоколом п. 5."""
from __future__ import annotations

import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROUNDS, WARMUP, MEASURES = 3, 20, 100


def _timed(sess, feed) -> list[float]:
    for _ in range(WARMUP):
        sess.run(None, feed)
    out = []
    for _ in range(MEASURES):
        t0 = time.perf_counter()
        sess.run(None, feed)
        t1 = time.perf_counter()
        out.append((t1 - t0) * 1000.0)
    return out


def measure(sess_fp32, sess_int8, x0: np.ndarray, fp32_path, int8_path) -> dict:
    """x0 — перше зображення test, [1, 3, S, S] float32."""
    feed = {"input": np.ascontiguousarray(x0, dtype=np.float32)}
    runs = {"fp32": [], "int8": []}
    for _ in range(ROUNDS):
        runs["fp32"].append(_timed(sess_fp32, feed))   # спершу FP32
        runs["int8"].append(_timed(sess_int8, feed))   # потім INT8
    res = {}
    paths = {"fp32": fp32_path, "int8": int8_path}
    for k in ("fp32", "int8"):
        allv = np.concatenate(runs[k])
        rp50 = [float(np.percentile(r, 50)) for r in runs[k]]
        res[k] = {
            "file_mb": os.path.getsize(paths[k]) / 1e6,
            "p50_ms": float(np.percentile(allv, 50)),
            "p95_ms": float(np.percentile(allv, 95)),
            "p50_per_round_ms": rp50,
            "round_p50_range_ms": [min(rp50), max(rp50)],
            "all_measurements_ms": [list(map(float, r)) for r in runs[k]],
        }
    f, i = res["fp32"], res["int8"]
    disjoint = (f["round_p50_range_ms"][1] < i["round_p50_range_ms"][0]
                or i["round_p50_range_ms"][1] < f["round_p50_range_ms"][0])
    res["speedup_p50"] = f["p50_ms"] / i["p50_ms"]
    res["size_ratio"] = f["file_mb"] / i["file_mb"]
    res["p95_ratio"] = f["p95_ms"] / i["p95_ms"]
    res["latency_change_measured"] = bool(disjoint)
    res["rule"] = "виміряно, якщо відрізки p50 за раундами не перетинаються"
    return res


def _cpu_info() -> dict:
    name, flags = platform.processor(), ""
    try:
        if sys.platform.startswith("linux"):
            txt = Path("/proc/cpuinfo").read_text()
            for line in txt.splitlines():
                if line.startswith("model name"):
                    name = line.split(":", 1)[1].strip()
                if line.startswith("flags") and not flags:
                    flags = line.split(":", 1)[1]
        elif sys.platform == "darwin":
            name = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
        elif sys.platform == "win32":
            out = subprocess.check_output(
                ["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_Processor).Name"],
                text=True).strip()
            name = out or name
    except Exception:
        pass
    keys = ("avx2", "avx512f", "avx512_vnni", "avx_vnni", "avx512bw")
    return {"cpu_model": name, "cpu_flags_of_interest": {k: (k in flags.split()) for k in keys} if flags else None}


def environment() -> dict:
    import numpy, onnx, onnxruntime, torch, torchvision
    import sklearn, PIL
    return {
        **_cpu_info(),
        "os": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__, "torchvision": torchvision.__version__,
        "onnx": onnx.__version__, "onnxruntime": onnxruntime.__version__,
        "numpy": numpy.__version__, "scikit_learn": sklearn.__version__, "pillow": PIL.__version__,
        "intra_op_num_threads": 1, "inter_op_num_threads": 1, "batch_size_latency": 1,
        "cpu_count_logical": os.cpu_count(),
        "note": "ноутбук має бути підключений до живлення, важкі програми закриті",
    }
