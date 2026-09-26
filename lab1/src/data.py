"""
Завантаження Iris, стратифікований поділ 70/30 та стандартизація ознак.
"""
import numpy as np
from sklearn.datasets import load_iris


def load_and_split(seed: int = 0):
    """
    Повертає X_train, y_train, X_test, y_test (вже стандартизовані).

    Поділ: по 35 обʼєктів кожного класу у train, по 15 у test.
    rng = np.random.default_rng(0), класи в порядку 0, 1, 2,
    індекси кожного класу перемішуються, перші 35 -> train.
    Стандартизація: mean/std лише на train, ddof=0.
    """
    data = load_iris()
    X, y = data.data, data.target  # X: (150, 4), y: (150,)

    rng = np.random.default_rng(seed)

    train_idx = []
    test_idx = []
    for c in (0, 1, 2):
        class_idx = np.where(y == c)[0]
        shuffled = rng.permutation(class_idx)
        train_idx.append(shuffled[:35])
        test_idx.append(shuffled[35:])

    train_idx = np.concatenate(train_idx)
    test_idx = np.concatenate(test_idx)

    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0, ddof=0)

    X_train_std = (X_train - mean) / std
    X_test_std = (X_test - mean) / std

    return (
        X_train_std.astype(np.float64),
        y_train.astype(np.int64),
        X_test_std.astype(np.float64),
        y_test.astype(np.int64),
    )


if __name__ == "__main__":
    Xtr, ytr, Xte, yte = load_and_split()
    print("X_train:", Xtr.shape, "y_train:", ytr.shape)
    print("X_test:", Xte.shape, "y_test:", yte.shape)
    for c in (0, 1, 2):
        print(f"class {c}: train={np.sum(ytr == c)}, test={np.sum(yte == c)}")
    print("train mean (should be ~0):", Xtr.mean(axis=0))
    print("train std (should be ~1):", Xtr.std(axis=0, ddof=0))
