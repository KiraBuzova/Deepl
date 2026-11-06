from pathlib import Path

import numpy as np
import torch
import torchvision

import config as C
import data as D
import engine as E


def load_checkpoint(path):
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    for k in ("state_dict", "regime", "seed", "classes", "img_size"):
        if k not in ckpt:
            raise ValueError(f"У {path} немає ключа {k}")
    return ckpt


def evaluate_test(model_path, all_rows, data_root):
    ckpt = load_checkpoint(model_path)
    device = torch.device("cpu")
    model = torchvision.models.resnet18(weights=None)
    model.fc = torch.nn.Linear(512, len(C.CLASSES))
    model.load_state_dict(ckpt["state_dict"])
    model = model.to(device).eval()
    loader = D.make_eval_loader(all_rows, data_root, "test")
    logits = E.predict_logits(model, loader, device)
    labels = D.labels_of(all_rows, "test")
    files = [r["file"] for r in D.part_rows(all_rows, "test")]
    return logits, labels, files
