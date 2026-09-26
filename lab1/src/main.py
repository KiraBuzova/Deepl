"""
Головний скрипт: запускає всі перевірки лабораторної роботи й друкує результати,
які потрібно перенести у README (можна також перенаправити вивід у файл:
`uv run python src/main.py > results.txt`).

Порядок:
  1. Прямий прохід + втрата (NumPy).
  2. Звірка NumPy vs PyTorch (втрата і 4 градієнти).
  3. Чисельна перевірка 4 елементів градієнта.
  4. Дослід із навмисною помилкою (п.4) + відновлення коректної реалізації.
"""
import numpy as np

from data import load_and_split
from model_numpy import init_params, forward, backward
from model_torch import build_torch_model, torch_forward_backward
from numeric_check import numerical_gradient_check, print_report, TOL
from bug_experiment import run_bug_experiment

SCALAR_TOL = 1e-12


def section(title: str):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def main():
    Xtr, ytr, Xte, yte = load_and_split()
    params = init_params()

    section("1. Прямий прохід (NumPy), повна навчальна вибірка, початкові ваги")
    loss_np, logits_np, cache = forward(params, Xtr, ytr)
    grads_np = backward(params, cache, bug_mode=False)
    print("Форма X_train:", Xtr.shape, " Форма y_train:", ytr.shape)
    print("Loss (NumPy):", loss_np)

    section("2. Звірка з PyTorch")
    model = build_torch_model(params)
    loss_torch, grads_torch = torch_forward_backward(model, Xtr, ytr)

    print(f"Loss NumPy  = {loss_np:.15f}")
    print(f"Loss PyTorch= {loss_torch:.15f}")

    header = f"{'Величина':<10}{'Макс.|diff|':>15}{'Перевірку пройдено':>22}"
    print(header)
    print("-" * len(header))
    loss_diff = abs(loss_np - loss_torch)
    print(f"{'Loss':<10}{loss_diff:>15.3e}{str(loss_diff <= SCALAR_TOL):>22}")
    for k in ("W1", "b1", "W2", "b2"):
        diff = np.max(np.abs(grads_np[k] - grads_torch[k]))
        passed = diff <= SCALAR_TOL and np.isfinite(diff)
        print(f"{'Grad ' + k:<10}{diff:>15.3e}{str(passed):>22}")

    section("3. Чисельна перевірка 4 елементів градієнта")
    results, loss0 = numerical_gradient_check(params, Xtr, ytr, bug_mode=False)
    print("Loss на початкових вагах:", loss0)
    print_report(results)

    section("4. Дослід із навмисною помилкою")
    run_bug_experiment()

    section("Готово")
    print("Скопіюйте потрібні числа й таблиці вище у README.md.")


if __name__ == "__main__":
    main()
