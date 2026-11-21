"""Завантаження моделі ЛР3, експорт в ONNX, сесії ORT, логіти PyTorch/ONNX, підрахунок вузлів."""
from __future__ import annotations

import time
from collections import Counter
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from torch import nn
from torchvision.models import resnet18

from data import CLASSES


def load_model(path: str | Path):
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    classes = list(ckpt["classes"])
    assert classes == CLASSES, f"Порядок класів не збігається з ЛР3: {classes}"
    model = resnet18(weights=None)
    model.fc = nn.Linear(512, 6)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    meta = {"regime": str(ckpt["regime"]), "seed": int(ckpt["seed"]),
            "classes": classes, "img_size": int(ckpt["img_size"])}
    return model, meta


def export_onnx(model: nn.Module, path: str | Path, img_size: int) -> float:
    """Експорт за фіксованими налаштуваннями; повертає час у секундах."""
    model.eval()
    dummy = torch.zeros(1, 3, img_size, img_size)
    t0 = time.perf_counter()
    torch.onnx.export(
        model, dummy, str(path),
        dynamo=False, opset_version=17,
        input_names=["input"], output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
    )
    return time.perf_counter() - t0


def count_ops(path: str | Path) -> dict:
    g = onnx.load(str(path)).graph
    c = Counter(n.op_type for n in g.node)
    return {"Conv": c.get("Conv", 0), "BatchNormalization": c.get("BatchNormalization", 0),
            "total_nodes": len(g.node)}


def make_session(path: str | Path) -> ort.InferenceSession:
    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    so.inter_op_num_threads = 1
    return ort.InferenceSession(str(path), sess_options=so, providers=["CPUExecutionProvider"])


def torch_logits(model: nn.Module, x: np.ndarray, bs: int = 64) -> np.ndarray:
    model.eval()
    outs = []
    with torch.no_grad():
        for i in range(0, len(x), bs):
            outs.append(model(torch.from_numpy(x[i:i + bs])).numpy())
    return np.concatenate(outs)


def onnx_logits(sess: ort.InferenceSession, x: np.ndarray, bs: int = 64) -> np.ndarray:
    outs = []
    for i in range(0, len(x), bs):
        outs.append(sess.run(None, {"input": np.ascontiguousarray(x[i:i + bs])})[0])
    return np.concatenate(outs)
