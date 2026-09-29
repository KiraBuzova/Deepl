"""
Ручна NumPy-реалізація мережі X -> Linear(4,8) -> ReLU -> Linear(8,3).

Формули зворотного проходу (N — кількість обʼєктів у батчі):

  Прямий прохід:
    z1 = X @ W1 + b1                (N,8)
    a1 = ReLU(z1)                   (N,8)
    logits = a1 @ W2 + b2           (N,3)
    p = softmax(logits)             (N,3)  (обчислюється стабільно через log-softmax)
    L = -(1/N) * sum_i log p[i, y_i]

  Зворотний прохід (bug_mode=False -> коректна реалізація):
    dlogits = (p - onehot(y)) / N          (N,3)
    dW2 = a1.T @ dlogits                   (8,3)
    db2 = sum_i dlogits[i]                 (3,)
    da1 = dlogits @ W2.T                   (N,8)
    dz1 = da1 * (z1 > 0)                   (N,8)   -- похідна ReLU
    dW1 = X.T @ dz1                        (4,8)
    db1 = sum_i dz1[i]                     (8,)

  Розмірності проміжних значень, які зберігаються для backward:
    X      (N,4)  -- вхід
    z1     (N,8)  -- лінійний вихід першого шару (потрібен для маски ReLU)
    a1     (N,8)  -- вихід ReLU (потрібен для dW2)
    logits (N,3)  -- вхід у softmax
    p      (N,3)  -- softmax-імовірності (потрібні для dlogits)
    y      (N,)   -- цільові класи
"""
import numpy as np


def init_params(seed: int = 0):
    """He-ініціалізація W1, Xavier-ініціалізація W2, нульові зсуви.

    Порядок генерації: спочатку W1, потім W2 (один rng, один виклик на кожну матрицю).
    """
    rng = np.random.default_rng(seed)

    std_he = np.sqrt(2.0 / 4.0)          # fan_in = 4
    std_xavier = np.sqrt(2.0 / (8 + 3))  # fan_in + fan_out = 11

    W1 = rng.normal(loc=0.0, scale=std_he, size=(4, 8))
    W2 = rng.normal(loc=0.0, scale=std_xavier, size=(8, 3))

    b1 = np.zeros(8, dtype=np.float64)
    b2 = np.zeros(3, dtype=np.float64)

    return {"W1": W1.astype(np.float64), "b1": b1, "W2": W2.astype(np.float64), "b2": b2}


def log_softmax(logits: np.ndarray) -> np.ndarray:
    """Чисельно стабільний log-softmax зі зсувом на максимум."""
    shifted = logits - logits.max(axis=1, keepdims=True)
    log_sum_exp = np.log(np.sum(np.exp(shifted), axis=1, keepdims=True))
    return shifted - log_sum_exp


def forward(params: dict, X: np.ndarray, y: np.ndarray):
    """Прямий прохід. Повертає loss, logits та кеш проміжних значень."""
    W1, b1, W2, b2 = params["W1"], params["b1"], params["W2"], params["b2"]
    N = X.shape[0]

    z1 = X @ W1 + b1
    a1 = np.maximum(z1, 0.0)
    logits = a1 @ W2 + b2

    logp = log_softmax(logits)
    p = np.exp(logp)

    loss = -np.mean(logp[np.arange(N), y])

    cache = {"X": X, "y": y, "z1": z1, "a1": a1, "logits": logits, "p": p, "N": N}
    return loss, logits, cache


def backward(params: dict, cache: dict, bug_mode: bool = False):
    """Ручне обчислення градієнтів dW1, db1, dW2, db2.

    bug_mode=True відтворює навмисну помилку з п.4: ділення на N у dlogits прибрано.
    """
    W2 = params["W2"]
    X, y, z1, a1, p, N = cache["X"], cache["y"], cache["z1"], cache["a1"], cache["p"], cache["N"]

    onehot = np.zeros_like(p)
    onehot[np.arange(N), y] = 1.0

    dlogits = (p - onehot)
    if not bug_mode:
        dlogits = dlogits / N  # коректна реалізація

    dW2 = a1.T @ dlogits
    db2 = dlogits.sum(axis=0)

    da1 = dlogits @ W2.T
    dz1 = da1 * (z1 > 0).astype(np.float64)

    dW1 = X.T @ dz1
    db1 = dz1.sum(axis=0)

    return {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}


if __name__ == "__main__":
    from data import load_and_split

    Xtr, ytr, Xte, yte = load_and_split()
    params = init_params()

    loss, logits, cache = forward(params, Xtr, ytr)
    grads = backward(params, cache)

    print("loss:", loss)
    for k, v in grads.items():
        print(f"grad {k}: shape={v.shape}")
