"""
Чисельна перевірка чотирьох елементів градієнта:
W1[0,0], b1[0], W2[0,0], b2[0].

g_num = (L(theta+eps) - L(theta-eps)) / (2*eps), eps = 1e-6
Критерій проходження: |g_num - g_manual| <= 1e-7
"""
import numpy as np
from model_numpy import forward, backward

EPS = 1e-6
TOL = 1e-7

# (назва_параметра, індекс у масиві параметра)
TARGETS = [
    ("W1", (0, 0)),
    ("b1", (0,)),
    ("W2", (0, 0)),
    ("b2", (0,)),
]


def loss_at(params: dict, X: np.ndarray, y: np.ndarray) -> float:
    loss, _, _ = forward(params, X, y)
    return loss


def numerical_gradient_check(params: dict, X: np.ndarray, y: np.ndarray, bug_mode: bool = False):
    """Повертає список рядків з результатами перевірки для кожного з 4 параметрів."""
    # ручний градієнт на початкових вагах
    loss0, _, cache = forward(params, X, y)
    grads = backward(params, cache, bug_mode=bug_mode)

    results = []
    for name, idx in TARGETS:
        original_value = params[name][idx]

        params[name][idx] = original_value + EPS
        loss_plus = loss_at(params, X, y)

        params[name][idx] = original_value - EPS
        loss_minus = loss_at(params, X, y)

        params[name][idx] = original_value  # відновлення

        g_num = (loss_plus - loss_minus) / (2 * EPS)
        g_manual = grads[name][idx]
        diff = abs(g_num - g_manual)
        passed = diff <= TOL

        results.append(
            {
                "param": f"{name}{list(idx)}",
                "grad_backward": g_manual,
                "grad_numeric": g_num,
                "abs_diff": diff,
                "passed": passed,
            }
        )

    return results, loss0


def print_report(results):
    header = f"{'Параметр':<12}{'backward()':>15}{'Чисельна':>15}{'|diff|':>15}{'Пройдено':>10}"
    print(header)
    print("-" * len(header))
    for r in results:
        print(
            f"{r['param']:<12}{r['grad_backward']:>15.9f}{r['grad_numeric']:>15.9f}"
            f"{r['abs_diff']:>15.2e}{str(r['passed']):>10}"
        )


if __name__ == "__main__":
    from data import load_and_split
    from model_numpy import init_params

    Xtr, ytr, _, _ = load_and_split()
    params = init_params()

    print("=== Коректна реалізація ===")
    results, loss0 = numerical_gradient_check(params, Xtr, ytr, bug_mode=False)
    print("loss на початкових вагах:", loss0)
    print_report(results)

    print("\n=== Режим з навмисною помилкою (без ділення на N) ===")
    results_bug, loss0_bug = numerical_gradient_check(params, Xtr, ytr, bug_mode=True)
    print("loss на початкових вагах:", loss0_bug)
    print_report(results_bug)
