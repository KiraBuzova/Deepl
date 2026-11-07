from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

import config as C
import data as D
import model as M
import utils as U


def save_batch_figure(all_rows, data_root, path):
    torch.manual_seed(1)
    loader = D.make_train_loader(all_rows, data_root)
    x, y = next(iter(loader))
    mean = torch.tensor(C.MEAN).view(3, 1, 1)
    std = torch.tensor(C.STD).view(3, 1, 1)
    fig, axes = plt.subplots(4, 4, figsize=(10, 10.6))
    for i, ax in enumerate(axes.flat):
        img = (x[i] * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()
        ax.imshow(img)
        ax.set_title(C.CLASSES[int(y[i])], fontsize=11)
        ax.axis("off")
    fig.suptitle("Перші 16 зображень навчального пакета після аугментацій (зерно 1)", fontsize=12)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def check_initial_loss(all_rows, data_root, device, md_path):
    crit = torch.nn.CrossEntropyLoss()
    rows = []
    for regime in ("scratch", "frozen"):
        model, _ = M.build_model(regime, 1, device)
        loader = D.make_train_loader(all_rows, data_root)
        x, y = next(iter(loader))
        M.set_train_mode(model, regime)
        with torch.no_grad():
            loss = crit(model(x.to(device)), y.to(device)).item()
        rows.append((regime, loss))
        del model
    table = U.md_table(["Режим", "L0 (перший пакет, 0 кроків)", "ln 6", "L0 − ln 6"],
                       [(r, f"{l:.4f}", f"{C.LN6:.4f}", f"{l - C.LN6:+.4f}") for r, l in rows])
    text = f"""# Початкова втрата (зерно 1, перший навчальний пакет, до жодного кроку оптимізатора)

{table}

Еталон: `ln 6 = {C.LN6:.4f}` - втрата моделі, яка дає всім шести класам однакову ймовірність 1/6.

**Пояснення відхилення.** Новий шар `fc = Linear(512, 6)` ініціалізований випадково, тому логіти не нульові й
розподіл імовірностей не рівномірний: на пакеті з 32 зображень втрата зазвичай трохи перевищує `ln 6`
(випадкові логіти з ненульовим розкидом дають втрату вищу за рівномірну; крім того, пакет малий і мітки на ньому
розподілені нерівномірно). У `frozen` ознаки на вході `fc` - це реальні ознаки ImageNet з великою нормою, тому
випадкова голова може давати більший розкид логітів, ніж у `scratch`, де ознаки після випадкової ініціалізації й
BN в режимі train мають інший масштаб. Відхилення такого порядку (десяті частки) - норма; значення, що на порядок
більше за `ln 6` або дорівнює NaN, указувало б на помилку в нормуванні чи в побудові моделі.
(Прямий прохід у режимі навчання оновив ковзні статистики BN цього екземпляра, тому він далі не використовується.)
"""
    U.write_text(md_path, text)
    return rows


def check_params(device, md_path):
    ok_all = True
    parts = []
    for regime in C.REGIMES:
        model, opt = M.build_model(regime, 1, device)
        total = M.count_params(model.parameters())
        trainable = M.count_params(p for p in model.parameters() if p.requires_grad)
        groups = M.group_info(model, opt)
        exp = C.EXPECTED_PARAMS[regime]
        ids_in_opt = {id(p) for g in opt.param_groups for p in g["params"]}
        frozen_in_opt = [n for n, p in model.named_parameters() if (not p.requires_grad) and id(p) in ids_in_opt]
        opt_trainable_ok = all(p.requires_grad for g in opt.param_groups for p in g["params"])
        groups_ok = [(g["name"], g["numel"], g["lr"]) for g in groups] == exp["groups"]
        ok = (total == C.TOTAL_PARAMS and trainable == exp["trainable"] and groups_ok
              and not frozen_in_opt and opt_trainable_ok)
        ok_all &= ok
        grp = "; ".join(f"{g['name']}: {g['numel']:,} парам., lr = {g['lr']:g}".replace(",", " ") for g in groups)
        parts.append((regime, f"{total:,}".replace(",", " "), f"{trainable:,}".replace(",", " "), grp,
                      "жоден" if not frozen_in_opt else f"ПОМИЛКА: {frozen_in_opt[:3]}",
                      "так" if ok else "НІ"))
    table = U.md_table(["Режим", "Всього параметрів", "requires_grad=True", "Групи оптимізатора",
                        "requires_grad=False в оптимізаторі", "Збігається з таблицею режимів"], parts)
    text = ("# Перевірка параметрів і груп оптимізатора\n\n" + table +
            "\n\nКількість параметрів - сума `numel()`, а не кількість тензорів. Очікування: "
            "scratch 11 179 590 / 11 179 590; frozen 11 179 590 / 3 078; unfreeze 11 179 590 / 8 396 806 "
            "(fc 3 078 + layer4 8 393 728).\n")
    U.write_text(md_path, text)
    return ok_all


def _classify_keys(model):
    pkeys = {n for n, _ in model.named_parameters()}
    bkeys = {n for n, _ in model.named_buffers()}
    return pkeys, bkeys


def _clone_state(model):
    return {k: v.detach().clone().cpu() for k, v in model.state_dict().items()}


def run_5_steps(regime, all_rows, data_root, device, plain_train=False):
    model, opt = M.build_model(regime, 1, device)
    loader = D.make_train_loader(all_rows, data_root)
    crit = torch.nn.CrossEntropyLoss()
    before = _clone_state(model)
    if plain_train:
        model.train()
    else:
        M.set_train_mode(model, regime)
    losses = []
    it = iter(loader)
    for _ in range(5):
        x, y = next(it)
        x, y = x.to(device), y.to(device)
        opt.zero_grad()
        loss = crit(model(x), y)
        loss.backward()
        opt.step()
        losses.append(loss.item())
    after = _clone_state(model)
    return before, after, losses, model


def module_changes(model, before, after):
    pkeys, bkeys = _classify_keys(model)
    rows = []
    for m in C.TOP_MODULES:
        prefix = m + "."
        pk = [k for k in before if k.startswith(prefix) and k in pkeys]
        bk = [k for k in before if k.startswith(prefix) and k in bkeys]
        stat_k = [k for k in bk if k.endswith(("running_mean", "running_var", "num_batches_tracked"))]
        p_changed = any(not torch.equal(before[k], after[k]) for k in pk)
        has_bn = len(stat_k) > 0
        s_changed = any(not torch.equal(before[k], after[k]) for k in stat_k) if has_bn else None
        rows.append({"module": m, "params_changed": p_changed, "has_bn": has_bn, "stats_changed": s_changed})
    return rows


def expected_change(regime, module, has_bn):
    frozen = module in C.EVAL_MODULES[regime]
    return {"params": not frozen, "stats": (None if not has_bn else (not frozen))}


def _yn(v):
    return "так" if v else "ні"


def check_module_changes(all_rows, data_root, device, md_path):
    ok_all = True
    out = ["# Що змінюється під час 5 кроків навчання (set_train_mode)\n",
           "Порівняння точне (`torch.equal`), для кожного модуля верхнього рівня; нова модель зі зерном 1, 5 перших пакетів.\n"]
    for regime in ("frozen", "unfreeze"):
        before, after, losses, model = run_5_steps(regime, all_rows, data_root, device, plain_train=False)
        res = module_changes(model, before, after)
        table_rows = []
        for r in res:
            exp = expected_change(regime, r["module"], r["has_bn"])
            ok = (r["params_changed"] == exp["params"]) and (r["stats_changed"] == exp["stats"])
            ok_all &= ok
            table_rows.append((regime, r["module"], _yn(r["params_changed"]),
                               "немає BN" if not r["has_bn"] else _yn(r["stats_changed"]), _yn(ok)))
        out.append(f"## Режим {regime}\n")
        out.append(U.md_table(["Режим", "Модуль", "Параметри змінилися", "Ковзні статистики BN змінилися",
                               "Збігається з очікуваним"], table_rows) + "\n")
        del model
    out.append("Очікування: модулі зі стовпця «У eval()» заморожені (не змінилося ні те, ні інше); "
               "у решти змінилися параметри, а в тих, що містять BN, - і ковзні статистики.\n")
    U.write_text(md_path, "\n".join(out))
    return ok_all


PREDICTION_TEXT = """# Прогноз досліду з навмисною помилкою (записано ДО запуску)

Режим `frozen`, але замість `set_train_mode` викликається звичайний `model.train()`.

**Прогноз.**
- **Параметри заморожених модулів (conv1, bn1, layer1-layer4) не зміняться**: `requires_grad=False` і вони не
  передані в оптимізатор, тому градієнтів і кроків для них немає.
- **Ковзні статистики BN (`running_mean`, `running_var`, `num_batches_tracked`) у всьому кістяку зміняться**: у режимі
  `train` кожен BN-шар під час прямого проходу нормує за статистиками пакета й оновлює буфери
  `running = (1 - momentum) * running + momentum * batch_stat`, незалежно від `requires_grad`.
  Параметри `fc` змінюються як і раніше.
- **З кривої втрати на train помилку діагностувати не вдасться**: втрата й далі правдоподібно спадатиме (BN у
  train-режимі нормує за статистиками пакета, а на `fc` це не заважає вчитися), хоча значення втрат на перших
  кроках можуть помірно відрізнятися від правильного варіанта, бо нормування інше (статистики пакета замість
  статистик ImageNet). Справжня шкода проявиться в режимі `eval()` (validation/test): там використовуються вже
  змінені ковзні статистики, оцінені на малих пакетах нового домену, а не ті, під які навчалися заморожені ваги.
"""


def check_bn_experiment(all_rows, data_root, device, pred_path, md_path):
    U.write_text(pred_path, PREDICTION_TEXT)
    b0, a0, loss_ok, m0 = run_5_steps("frozen", all_rows, data_root, device, plain_train=False)
    b1, a1, loss_bad, m1 = run_5_steps("frozen", all_rows, data_root, device, plain_train=True)

    res_ok = module_changes(m0, b0, a0)
    res_bad = module_changes(m1, b1, a1)
    rows = []
    prediction_holds = True
    for ok_r, bad_r in zip(res_ok, res_bad):
        m = bad_r["module"]
        frozen = m in C.EVAL_MODULES["frozen"]
        exp_p = not frozen
        exp_s = None if not bad_r["has_bn"] else True
        match = (bad_r["params_changed"] == exp_p) and (bad_r["stats_changed"] == exp_s)
        prediction_holds &= match
        rows.append((m, _yn(ok_r["params_changed"]), "немає BN" if not ok_r["has_bn"] else _yn(ok_r["stats_changed"]),
                     _yn(bad_r["params_changed"]), "немає BN" if not bad_r["has_bn"] else _yn(bad_r["stats_changed"]),
                     _yn(match)))
    table = U.md_table(["Модуль", "set_train_mode: параметри", "set_train_mode: статистики BN",
                        "model.train(): параметри", "model.train(): статистики BN", "Збіг із прогнозом"], rows)

    key = "layer1.0.bn1"
    d_mean = (a1[f"{key}.running_mean"] - b1[f"{key}.running_mean"]).abs().max().item()
    d_var = (a1[f"{key}.running_var"] - b1[f"{key}.running_var"]).abs().max().item()
    d_mean_ok = (a0[f"{key}.running_mean"] - b0[f"{key}.running_mean"]).abs().max().item()
    d_var_ok = (a0[f"{key}.running_var"] - b0[f"{key}.running_var"]).abs().max().item()
    nbt = int(a1[f"{key}.num_batches_tracked"].item())
    loss_gap = float(np.mean(np.abs(np.array(loss_ok) - np.array(loss_bad))))

    text = f"""# Результат досліду з навмисною помилкою (frozen + model.train())

{table}

**Шар `{key}` після 5 кроків** (максимальна абсолютна зміна):

| Варіант | max abs Δ running_mean | max abs Δ running_var | num_batches_tracked |
|---|---|---|---|
| `set_train_mode` (правильно) | {d_mean_ok:.3e} | {d_var_ok:.3e} | {int(a0[f'{key}.num_batches_tracked'].item())} |
| `model.train()` (помилка) | {d_mean:.3e} | {d_var:.3e} | {nbt} |

**Втрати на 5 кроках (ті самі пакети).**
- `set_train_mode`: {", ".join(f"{v:.4f}" for v in loss_ok)}
- `model.train()`: {", ".join(f"{v:.4f}" for v in loss_bad)}

Прогноз щодо параметрів і буферів підтвердився (параметри кістяка не змінилися, BN-статистики змінилися): **{_yn(prediction_holds)}**.
Середня абсолютна різниця між двома послідовностями втрат: **{loss_gap:.4f}** (середня втрата правильного варіанта
{sum(loss_ok)/5:.4f}, помилкового {sum(loss_bad)/5:.4f}). Обидві послідовності є «нормальними» числами без збоїв, тобто
сама по собі крива втрати на train не показує, що BN у кістяку оновлюється; це видно лише порівнянням буферів
або якістю в режимі `eval()`. (Оцініть різницю в цих числах і за потреби відкоригуйте формулювання в README.)

**Чому `requires_grad=False` не захищає ці значення.** Ковзні статистики - це *буфери*, а не параметри: вони не
беруть участі в градієнтному спуску й не мають `requires_grad`. Їх оновлює сам прямий прохід BN-шару, коли шар у
режимі `train` (`running = (1-momentum)*running + momentum*batch_stat`), без жодного градієнта й без оптимізатора.
Захищає їх тільки `eval()`: у цьому режимі BN бере збережені статистики й нічого не оновлює.

**Чому для `layer4` у `unfreeze` оновлення правильне.** `layer4` навчається: його ваги змінюються, а отже
змінюється й розподіл активацій на виході його згорток. BN-шари `layer4` мають стежити за цим новим розподілом -
інакше в `eval()` вони нормували б активації застарілими статистиками ImageNet, які вже не відповідають зміненим
вагам і новому домену. Тому `layer4` лишається в `train()`, а `conv1`, `bn1`, `layer1`-`layer3` - у `eval()`.
"""
    U.write_text(md_path, text)
    return prediction_holds
