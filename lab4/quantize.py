"""Статична квантизація INT8 (QDQ) із калібруванням на 200 зображеннях train."""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
from onnxruntime.quantization import (CalibrationDataReader, CalibrationMethod, QuantFormat,
                                      QuantType, quantize_static)
from onnxruntime.quantization.shape_inference import quant_pre_process


class ArrayReader(CalibrationDataReader):
    """Подає калібрувальні зображення по одному, форма [1, 3, S, S]."""

    def __init__(self, calib: np.ndarray, input_name: str = "input"):
        self.calib = calib
        self.name = input_name
        self.it = iter(range(len(calib)))

    def get_next(self):
        i = next(self.it, None)
        if i is None:
            return None
        return {self.name: np.ascontiguousarray(self.calib[i:i + 1])}

    def rewind(self):
        self.it = iter(range(len(self.calib)))


def preprocess(fp32: str | Path, prep: str | Path) -> float:
    t0 = time.perf_counter()
    quant_pre_process(str(fp32), str(prep), skip_symbolic_shape=True)
    return time.perf_counter() - t0


def quantize(prep: str | Path, out: str | Path, calib: np.ndarray, per_channel: bool) -> float:
    reader = ArrayReader(calib)
    t0 = time.perf_counter()
    quantize_static(
        str(prep), str(out), reader,
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
        per_channel=per_channel,
        calibrate_method=CalibrationMethod.MinMax,
    )
    return time.perf_counter() - t0
