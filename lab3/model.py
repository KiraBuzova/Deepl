import numpy as np
import torch
import torchvision

import config as C


def _set_requires_grad(model, regime):
    for name, p in model.named_parameters():
        if regime == "scratch":
            p.requires_grad = True
        elif regime == "frozen":
            p.requires_grad = name.startswith("fc.")
        elif regime == "unfreeze":
            p.requires_grad = name.startswith("fc.") or name.startswith("layer4.")
        else:
            raise ValueError(regime)


def make_optimizer(model, regime):
    if regime == "scratch":
        groups = [{"name": "all", "params": list(model.parameters()), "lr": C.LR_HEAD}]
    elif regime == "frozen":
        groups = [{"name": "fc", "params": list(model.fc.parameters()), "lr": C.LR_HEAD}]
    elif regime == "unfreeze":
        groups = [{"name": "fc", "params": list(model.fc.parameters()), "lr": C.LR_HEAD},
                  {"name": "layer4", "params": list(model.layer4.parameters()), "lr": C.LR_LAYER4}]
    else:
        raise ValueError(regime)
    return torch.optim.AdamW(groups, weight_decay=C.WEIGHT_DECAY)


def build_model(regime, seed, device):
    torch.manual_seed(seed)
    np.random.seed(seed)
    weights = None if regime == "scratch" else torchvision.models.ResNet18_Weights.IMAGENET1K_V1
    model = torchvision.models.resnet18(weights=weights)
    model.fc = torch.nn.Linear(512, len(C.CLASSES))
    _set_requires_grad(model, regime)
    model = model.to(device)
    optimizer = make_optimizer(model, regime)
    return model, optimizer


def set_train_mode(model, regime):
    model.train()
    for name in C.EVAL_MODULES[regime]:
        getattr(model, name).eval()


def count_params(params):
    return sum(p.numel() for p in params)


def group_info(model, optimizer):
    return [{"name": g.get("name", "?"), "numel": count_params(g["params"]), "lr": g["lr"]}
            for g in optimizer.param_groups]
