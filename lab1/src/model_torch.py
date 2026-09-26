"""
Еталонна реалізація мережі в PyTorch (лише для звірки, autograd не використовується
у власній NumPy-реалізації).

torch.nn.Linear зберігає вагу як (out_features, in_features), тобто
транспоновано відносно NumPy-масивів W1 (4,8) і W2 (8,3):
  linear1.weight.shape == (8, 4)  == W1.T
  linear2.weight.shape == (3, 8)  == W2.T
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class TorchNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(4, 8)
        self.linear2 = nn.Linear(8, 3)

    def forward(self, x):
        z1 = self.linear1(x)
        a1 = F.relu(z1)
        logits = self.linear2(a1)
        return logits


def build_torch_model(numpy_params: dict) -> TorchNet:
    """Створює TorchNet і копіює в нього параметри NumPy-мережі (з транспонуванням ваг)."""
    model = TorchNet().double()  # обчислення у float64

    with torch.no_grad():
        model.linear1.weight.copy_(torch.from_numpy(numpy_params["W1"].T.copy()))
        model.linear1.bias.copy_(torch.from_numpy(numpy_params["b1"].copy()))
        model.linear2.weight.copy_(torch.from_numpy(numpy_params["W2"].T.copy()))
        model.linear2.bias.copy_(torch.from_numpy(numpy_params["b2"].copy()))

    return model


def torch_forward_backward(model: TorchNet, X: np.ndarray, y: np.ndarray):
    """Прямий+зворотний прохід у PyTorch. Повертає loss (float) і градієнти у форматі NumPy-мережі."""
    X_t = torch.from_numpy(X).double()
    y_t = torch.from_numpy(y).long()

    for p in model.parameters():
        if p.grad is not None:
            p.grad = None

    logits = model(X_t)
    loss = F.cross_entropy(logits, y_t)  # усереднення за обʼєктами за замовчуванням
    loss.backward()

    grads = {
        "W1": model.linear1.weight.grad.numpy().T.copy(),   # (8,4)->(4,8)
        "b1": model.linear1.bias.grad.numpy().copy(),
        "W2": model.linear2.weight.grad.numpy().T.copy(),   # (3,8)->(8,3)
        "b2": model.linear2.bias.grad.numpy().copy(),
    }
    return float(loss.item()), grads


if __name__ == "__main__":
    from data import load_and_split
    from model_numpy import init_params, forward, backward

    Xtr, ytr, _, _ = load_and_split()
    params = init_params()

    loss_np, _, cache = forward(params, Xtr, ytr)
    grads_np = backward(params, cache)

    model = build_torch_model(params)
    loss_torch, grads_torch = torch_forward_backward(model, Xtr, ytr)

    print("NumPy loss :", loss_np)
    print("Torch loss :", loss_torch)
    print("|diff| loss:", abs(loss_np - loss_torch))

    for k in ("W1", "b1", "W2", "b2"):
        diff = np.max(np.abs(grads_np[k] - grads_torch[k]))
        print(f"grad {k}: max abs diff = {diff:.3e}")
