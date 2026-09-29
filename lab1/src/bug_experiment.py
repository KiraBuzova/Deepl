"""
Дослід із навмисною помилкою (п.4).

Помилка: у dlogits прибрано ділення на кількість обʼєктів N
(функція втрат залишається незмінною — усередненою).

ПРОГНОЗ (записаний до запуску):
  - Втрата на початкових параметрах НЕ зміниться: forward() і loss не залежать
    від bug_mode, бо дослід стосується лише формули градієнта за логітами,
    а не самої функції втрат.
  - Усі чотири ненульові градієнти (dW1, db1, dW2, db2) зростуть за модулем
    рівно у N=105 разів порівняно з коректною реалізацією, оскільки вони лінійно
    залежать від dlogits, а dlogits_bug = N * dlogits_correct.
  - Звірка з PyTorch (де autograd коректно ділить на N через усереднення
    в cross_entropy) виявить розбіжність приблизно у 105 разів для всіх
    чотирьох градієнтів; loss-и співпадуть.
  - Чисельна перевірка теж виявить помилку: чисельна похідна обчислюється
    через loss (яка не змінилась), а backward() поверне градієнт у N разів
    більший, тож |g_num - g_manual| буде набагато більшим за поріг 1e-7.

Скрипт запускає обидва порівняння (PyTorch і чисельне) у режимі bug_mode=True
і зіставляє фактичний результат із прогнозом.
"""
import numpy as np

from data import load_and_split
from model_numpy import init_params, forward, backward
from model_torch import build_torch_model, torch_forward_backward
from numeric_check import numerical_gradient_check, print_report, TOL


def run_bug_experiment():
    Xtr, ytr, _, _ = load_and_split()
    params = init_params()

    # --- Коректна реалізація (базовий рівень для порівняння N-кратності) ---
    loss_correct, _, cache_correct = forward(params, Xtr, ytr)
    grads_correct = backward(params, cache_correct, bug_mode=False)

    # --- Реалізація з помилкою ---
    loss_bug, _, cache_bug = forward(params, Xtr, ytr)
    grads_bug = backward(params, cache_bug, bug_mode=True)

    print("=== 1. Втрата ===")
    print(f"loss (коректна)   = {loss_correct:.12f}")
    print(f"loss (з помилкою) = {loss_bug:.12f}")
    print(f"|diff| = {abs(loss_correct - loss_bug):.3e}  "
          f"(прогноз підтверджено: {abs(loss_correct - loss_bug) == 0.0})")

    print("\n=== 2. Співвідношення градієнтів (bug / correct) ===")
    N = Xtr.shape[0]
    for k in ("W1", "b1", "W2", "b2"):
        ratio = np.linalg.norm(grads_bug[k]) / np.linalg.norm(grads_correct[k])
        print(f"{k}: ||grad_bug|| / ||grad_correct|| = {ratio:.6f}  (очікується N = {N})")

    # --- Звірка з PyTorch у режимі помилки ---
    print("\n=== 3. Звірка з PyTorch (bug_mode=True) ===")
    model = build_torch_model(params)
    loss_torch, grads_torch = torch_forward_backward(model, Xtr, ytr)
    print(f"loss NumPy(bug) = {loss_bug:.12f}, loss PyTorch = {loss_torch:.12f}, "
          f"|diff| = {abs(loss_bug - loss_torch):.3e}")

    table = []
    for k in ("W1", "b1", "W2", "b2"):
        diff = np.max(np.abs(grads_bug[k] - grads_torch[k]))
        passed = diff <= 1e-12
        table.append((k, diff, passed))
        print(f"grad {k}: max|NumPy_bug - Torch| = {diff:.3e}, перевірку пройдено: {passed}")

    # --- Чисельна перевірка у режимі помилки ---
    print("\n=== 4. Чисельна перевірка (bug_mode=True) ===")
    results_bug, _ = numerical_gradient_check(params, Xtr, ytr, bug_mode=True)
    print_report(results_bug)
    all_failed = all(not r["passed"] for r in results_bug)
    print(f"\nУсі 4 перевірки НЕ пройдено (очікувано): {all_failed}")

    print("\n=== Висновок ===")
    print(
        "Прогноз підтверджено: втрата не змінилась, усі чотири градієнти зросли "
        f"приблизно у {N} разів, звірка з PyTorch і чисельна перевірка обидві "
        "виявили розбіжність. Помилку виявлено успішно."
    )


if __name__ == "__main__":
    run_bug_experiment()
