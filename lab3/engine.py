import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score

import config as C
import data as D
import model as M
import utils as U


def macro_f1(y_true, logits):
    pred = np.asarray(logits).argmax(axis=1)
    return float(f1_score(y_true, pred, average="macro", labels=list(range(len(C.CLASSES))), zero_division=0))


def train_one_epoch(model, loader, optimizer, device, regime, criterion):
    M.set_train_mode(model, regime)
    total, n = 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
        total += loss.item() * x.size(0)
        n += x.size(0)
    return total / n


@torch.no_grad()
def predict_logits(model, loader, device):
    model.eval()
    out = []
    for x, _ in loader:
        out.append(model(x.to(device)).float().cpu())
    return torch.cat(out).numpy().astype(np.float32)


def run_training(regime, seed, all_rows, data_root, device, run_dir, split_hash):
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    model, optimizer = M.build_model(regime, seed, device)
    train_loader = D.make_train_loader(all_rows, data_root)
    val_loader = D.make_eval_loader(all_rows, data_root, "validation")
    y_val = D.labels_of(all_rows, "validation")
    criterion = torch.nn.CrossEntropyLoss()

    train_losses, val_f1s = [], []
    train_time = 0.0
    eval_time = 0.0
    logits = None
    for epoch in range(1, C.EPOCHS + 1):
        U.sync(device)
        t0 = time.perf_counter()
        loss = train_one_epoch(model, train_loader, optimizer, device, regime, criterion)
        U.sync(device)
        train_time += time.perf_counter() - t0

        t0 = time.perf_counter()
        logits = predict_logits(model, val_loader, device)
        eval_time += time.perf_counter() - t0
        f1 = macro_f1(y_val, logits)
        train_losses.append(loss)
        val_f1s.append(f1)
        print(f"  [{regime} seed{seed}] epoch {epoch}/{C.EPOCHS}  train_loss={loss:.4f}  val_macro_F1={f1:.4f}", flush=True)

    assert logits.shape == (len(y_val), len(C.CLASSES))
    np.save(run_dir / "val_logits.npy", logits)
    torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, run_dir / "model.pt")

    log = {
        "regime": regime, "seed": seed,
        "settings": {
            "transforms": {
                "train": "RandomResizedCrop(160, scale=(0.7, 1.0)) -> RandomHorizontalFlip() -> ToTensor() -> Normalize(mean, std)",
                "eval": "Resize((160, 160)) -> ToTensor() -> Normalize(mean, std)",
                "mean": list(C.MEAN), "std": list(C.STD), "img_size": C.IMG_SIZE,
                "RandomResizedCrop_ratio_default": [3 / 4, 4 / 3], "interpolation_default": "bilinear",
                "RandomHorizontalFlip_p_default": 0.5,
            },
            "model": {
                "arch": "torchvision resnet18",
                "weights": None if regime == "scratch" else "ResNet18_Weights.IMAGENET1K_V1",
                "fc": "Linear(512, 6), стандартна ініціалізація",
                "bn_momentum": C.BN_MOMENTUM, "bn_eps": C.BN_EPS,
                "total_params": M.count_params(model.parameters()),
                "trainable_params": M.count_params(p for p in model.parameters() if p.requires_grad),
                "eval_modules_during_training": C.EVAL_MODULES[regime],
            },
            "optimizer": {
                "name": "torch.optim.AdamW", "weight_decay": C.WEIGHT_DECAY,
                "param_groups": M.group_info(model, optimizer),
                "defaults": {k: (list(v) if isinstance(v, tuple) else v)
                             for k, v in optimizer.defaults.items() if k in ("betas", "eps", "amsgrad", "maximize")},
            },
            "loss": "CrossEntropyLoss(), mean за пакетом",
            "loader": {
                "train": {"batch_size": C.BATCH_TRAIN, "shuffle": True, "drop_last": False, "num_workers": 0,
                          "generator": None, "sampler": None},
                "eval": {"batch_size": C.BATCH_EVAL, "shuffle": False, "num_workers": 0},
            },
            "budget": {"epochs": C.EPOCHS, "steps_per_epoch": len(train_loader),
                       "total_steps": C.EPOCHS * len(train_loader), "last_batch_size": 1200 % C.BATCH_TRAIN or C.BATCH_TRAIN},
            "not_used": "розклади lr, обрізання градієнта, AMP, label smoothing, ваги класів, рання зупинка, вибір епохи за validation",
            "weights_used": "після 8-ї епохи",
        },
        "git": U.git_info(),
        "split_sha256": split_hash,
        "versions": U.versions(),
        "device": U.device_name(device),
        "train_loss_per_epoch": train_losses,
        "val_macro_f1_per_epoch": val_f1s,
        "final_train_loss": train_losses[-1],
        "final_val_macro_f1": val_f1s[-1],
        "train_time_seconds": train_time,
        "val_eval_time_seconds": eval_time,
    }
    U.write_json(log, run_dir / "run.json")
    return log
