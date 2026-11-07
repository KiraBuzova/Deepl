import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn.metrics import confusion_matrix, f1_score

import config as C
import selection as SEL
import split as S
import utils as U

ROOT = Path(__file__).resolve().parent


def stop(msg):
    print(f"\n[ПОМИЛКА] {msg}", file=sys.stderr)
    sys.exit(1)


def softmax(z):
    z = np.asarray(z, dtype=np.float64)
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def top_error_cases(labels, preds, files, probs, n=5):
    k = len(C.CLASSES)
    cm = confusion_matrix(labels, preds, labels=list(range(k)))
    pairs = [(int(cm[i, j]), i, j) for i in range(k) for j in range(k) if i != j and cm[i, j] > 0]
    pairs.sort(key=lambda t: (-t[0], t[1], t[2]))

    by_pair = {}
    for idx, (t, p) in enumerate(zip(labels, preds)):
        if t != p:
            by_pair.setdefault((int(t), int(p)), []).append(idx)
    for key in by_pair:
        by_pair[key].sort(key=lambda i: files[i])

    cases = []
    for cnt, t, p in pairs[:n]:
        i = by_pair[(t, p)][0]
        cases.append({"file": files[i], "true": t, "pred": p, "prob": float(probs[i, p]), "count": cnt, "index": i})
    depth = 1
    while len(cases) < n and depth < 10_000:
        added = False
        for cnt, t, p in pairs:
            if len(cases) >= n:
                break
            lst = by_pair[(t, p)]
            if len(lst) > depth:
                i = lst[depth]
                cases.append({"file": files[i], "true": t, "pred": p, "prob": float(probs[i, p]),
                              "count": cnt, "index": i})
                added = True
        if not added:
            break
        depth += 1
    return cm, cases


def pct(x):
    return f"{x:.4f}"


COLORS = {"scratch": "#444444", "frozen": "#1f77b4", "unfreeze": "#d95f02"}
STYLES = {1: "-", 2: "--", 3: ":"}


def plot_curves(logs, path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharex=True)
    ep = np.arange(1, C.EPOCHS + 1)
    for (regime, seed), lg in logs.items():
        kw = dict(color=COLORS[regime], linestyle=STYLES[seed], marker="o", markersize=3, linewidth=1.8,
                  label=f"{regime}, зерно {seed}")
        axes[0].plot(ep, lg["train_loss_per_epoch"], **kw)
        axes[1].plot(ep, lg["val_macro_f1_per_epoch"], **kw)
    axes[0].set_title("Втрата на train")
    axes[0].set_ylabel("CrossEntropy (середнє за епоху)")
    axes[1].set_title("macro-F1 на validation")
    axes[1].set_ylabel("macro-F1")
    for ax in axes:
        ax.set_xlabel("Епоха")
        ax.set_xticks(ep)
        ax.grid(alpha=0.3)
    axes[1].legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_confusion(cm, path, title):
    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(6)); ax.set_xticklabels(C.CLASSES, rotation=40, ha="right")
    ax.set_yticks(range(6)); ax.set_yticklabels(C.CLASSES)
    ax.set_xlabel("Передбачений клас"); ax.set_ylabel("Справжній клас")
    ax.set_title(title)
    thr = cm.max() / 2
    for i in range(6):
        for j in range(6):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center", color="white" if cm[i, j] > thr else "black")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_errors(cases, ds_paths, path):
    n = len(cases)
    fig, axes = plt.subplots(1, n, figsize=(3.4 * n, 4.2))
    axes = np.atleast_1d(axes)
    for ax, c in zip(axes, cases):
        ax.imshow(Image.open(ds_paths[c["index"]]).convert("RGB"))
        ax.set_title(f"{c['file']}\nсправжній: {C.CLASSES[c['true']]}\nпередбачено: {C.CLASSES[c['pred']]}\n"
                     f"p = {c['prob']:.3f}", fontsize=9)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="шлях до теки archive (потрібен лише для зображень test)")
    ap.add_argument("--runs", default=str(ROOT / "runs"))
    args = ap.parse_args()

    split_path = ROOT / C.SPLIT_CSV
    if not split_path.exists():
        stop("немає split.csv")
    cur_hash = U.sha256_file(split_path)

    logs = {}
    for regime, seed in C.RUNS:
        rj = Path(args.runs) / f"{regime}_seed{seed}" / "run.json"
        if not rj.exists():
            stop(f"немає журналу {rj}: потрібні рівно сім запусків протоколу")
        logs[(regime, seed)] = U.read_json(rj)
    hashes = {k: v["split_sha256"] for k, v in logs.items()}
    if len(set(hashes.values())) != 1:
        stop(f"у журналах різні хеші split.csv: {hashes}")
    if next(iter(hashes.values())) != cur_hash:
        stop("хеш split.csv у журналах не збігається з поточним split.csv - запуски на різних поділах порівнювати "
             "не можна; перенавчіть усі сім")
    print(f"[ok] SHA-256 split.csv однаковий у всіх 7 журналах і збігається з поточним: {cur_hash[:16]}…")

    all_rows = S.read_split(split_path)
    ok_split, _, _ = S.check_split(all_rows)
    if not ok_split:
        stop("поточний split.csv не збігається з контрольною таблицею")

    y_val = np.array([C.CLASSES.index(r["label"]) for r in all_rows if r["part"] == "validation"])
    warn = []
    for (regime, seed), lg in logs.items():
        lp = Path(args.runs) / f"{regime}_seed{seed}" / "val_logits.npy"
        if not lp.exists():
            warn.append(f"{regime}_seed{seed}: немає val_logits.npy")
            continue
        z = np.load(lp)
        f = f1_score(y_val, z.argmax(1), average="macro", labels=list(range(6)), zero_division=0)
        if z.shape != (len(y_val), 6) or abs(f - lg["final_val_macro_f1"]) > 1e-6:
            warn.append(f"{regime}_seed{seed}: F1 із val_logits.npy ({f:.6f}) ≠ у run.json ({lg['final_val_macro_f1']:.6f})")
    for w in warn:
        print("[увага]", w)
    if not warn:
        print("[ok] macro-F1, перераховані з val_logits.npy, збігаються з run.json")

    f1 = {k: v["final_val_macro_f1"] for k, v in logs.items()}
    table_rows = []
    for n, (regime, seed) in enumerate(C.RUNS, 1):
        lg = logs[(regime, seed)]
        table_rows.append((n, regime, seed, f"{lg['final_train_loss']:.4f}", f"{lg['final_val_macro_f1']:.4f}",
                           f"{lg['train_time_seconds']:.1f}"))
    runs_table = U.md_table(["#", "Режим", "Зерно", "Втрата train, епоха 8", "macro-F1 validation, епоха 8",
                             "Час навчання, с"], table_rows)
    plot_curves(logs, ROOT / "figures" / "curves.png")

    eff = SEL.effects(f1)
    R = eff["R"]
    effects_table = U.md_table(["Величина", "Значення"], [
        ("F_scratch,1", pct(eff["F_scratch"])),
        ("F̄_frozen", pct(eff["Fbar_frozen"])), ("R_frozen", pct(eff["R_frozen"])),
        ("F̄_unfreeze", pct(eff["Fbar_unfreeze"])), ("R_unfreeze", pct(eff["R_unfreeze"])),
        ("R = max(R_frozen, R_unfreeze)", pct(R)),
        ("Δpre = F̄_frozen − F_scratch,1", f"{eff['delta_pre']:+.4f}"),
        ("Δunf = F̄_unfreeze − F̄_frozen", f"{eff['delta_unf']:+.4f}"),
        ("Δpre + Δunf = F̄_unfreeze − F_scratch,1", f"{eff['delta_total']:+.4f}"),
        ("|Δpre| ≤ R ?", "так" if abs(eff["delta_pre"]) <= R else "ні"),
        ("|Δunf| ≤ R ?", "так" if abs(eff["delta_unf"]) <= R else "ні"),
    ])

    def last3_range(lg):
        v = lg["val_macro_f1_per_epoch"][-3:]
        return max(v) - min(v)
    scratch_curve = logs[("scratch", 1)]["val_macro_f1_per_epoch"]
    r3 = {k: last3_range(v) for k, v in logs.items()}
    mean_time = {r: float(np.mean([logs[(rr, s)]["train_time_seconds"] for (rr, s) in logs if rr == r]))
                 for r in C.REGIMES}
    time_table = U.md_table(["Режим", "Середній час навчання, с", "Відношення до scratch", "Кроків оптимізатора"],
                            [(r, f"{mean_time[r]:.1f}", f"{mean_time[r] / mean_time['scratch']:.2f}",
                              logs[(r, 1)]["settings"]["budget"]["total_steps"]) for r in C.REGIMES])
    hints = (f"- Крива scratch (зерно 1), macro-F1 за епохами: {', '.join(f'{v:.3f}' for v in scratch_curve)}.\n"
             f"- Розмах macro-F1 за останні 3 епохи: scratch {r3[('scratch', 1)]:.4f}; "
             f"frozen {np.mean([r3[('frozen', s)] for s in (1, 2, 3)]):.4f} (середнє за зернами); "
             f"unfreeze {np.mean([r3[('unfreeze', s)] for s in (1, 2, 3)]):.4f} (середнє за зернами).\n"
             f"- |Δpre| / R = {abs(eff['delta_pre']) / R:.2f}." if R > 0 else "- R = 0.")

    best_regime, best_seed, means = SEL.choose(f1)
    model_path = ROOT / C.MODEL_FILE
    if not model_path.exists():
        stop(f"немає {C.MODEL_FILE}; спершу виконайте команду навчання")
    import torch
    import test_eval as T
    ckpt = T.load_checkpoint(model_path)
    problems = []
    if (ckpt["regime"], ckpt["seed"]) != (best_regime, best_seed):
        problems.append(f"у {C.MODEL_FILE}: ({ckpt['regime']}, {ckpt['seed']}), за журналами має бути ({best_regime}, {best_seed})")
    if list(ckpt["classes"]) != C.CLASSES or ckpt["img_size"] != C.IMG_SIZE:
        problems.append("classes/img_size у файлі не збігаються з очікуваними")
    local_w = Path(args.runs) / f"{best_regime}_seed{best_seed}" / "model.pt"
    weights_note = "локальні ваги запуску відсутні - порівняння ваг пропущено"
    if local_w.exists():
        sd = torch.load(local_w, map_location="cpu", weights_only=True)
        same = all(torch.equal(sd[k], ckpt["state_dict"][k]) for k in sd) and set(sd) == set(ckpt["state_dict"])
        weights_note = "ваги в lab3_model.pt збігаються з вагами вибраного запуску" if same else "ВАГИ НЕ ЗБІГАЮТЬСЯ"
        if not same:
            problems.append("state_dict у lab3_model.pt відрізняється від ваг вибраного запуску")
    if problems:
        stop("перевірка вибору не пройдена:\n  - " + "\n  - ".join(problems))
    print(f"[ok] lab3_model.pt відповідає правилу вибору: {best_regime}, зерно {best_seed}; {weights_note}")
    noise_decided = abs(eff["delta_unf"]) <= R

    print("Тестова оцінка на CPU…")
    logits, labels, files = T.evaluate_test(model_path, all_rows, args.data)
    preds = logits.argmax(1)
    probs = softmax(logits)
    f1_test = float(f1_score(labels, preds, average="macro", labels=list(range(6)), zero_division=0))
    acc = float((preds == labels).mean())
    n_test = len(labels)
    se = float(np.sqrt(acc * (1 - acc) / n_test))
    f1_val_sel = f1[(best_regime, best_seed)]
    cm, cases = top_error_cases(labels, preds, files, probs, 5)
    np.save(ROOT / "results" / "test_logits.npy", logits)
    plot_confusion(cm, ROOT / "figures" / "confusion.png",
                   f"Матриця плутанини на test ({best_regime}, зерно {best_seed}), macro-F1 = {f1_test:.4f}")

    train_dir, test_dir = S.resolve_dirs(args.data)
    test_rows = [r for r in all_rows if r["part"] == "test"]
    paths = [test_dir / r["label"] / r["file"] for r in test_rows]
    plot_errors(cases, paths, ROOT / "figures" / "errors.png")

    cm_table = U.md_table(["справжній \\ передбачений"] + C.CLASSES,
                          [[C.CLASSES[i]] + [int(v) for v in cm[i]] + [] for i in range(6)])
    err_table = U.md_table(["Файл", "Справжній клас", "Передбачений клас", "Імовірність передбаченого класу",
                            "Кількість таких помилок у test"],
                           [(c["file"], C.CLASSES[c["true"]], C.CLASSES[c["pred"]], f"{c['prob']:.4f}", c["count"])
                            for c in cases])
    bias_within = f1_val_sel - means[best_regime]
    selection_table = U.md_table(["Режим", "Середнє F̄ (validation)"], [(r, f"{v:.4f}") for r, v in means.items()])

    md = f"""# Результати аналізу (згенеровано analyze.py)

SHA-256 split.csv: `{cur_hash}` (однаковий у всіх семи журналах).

## Таблиця семи запусків (п. 3)

{runs_table}

Криві: `figures/curves.png`.

### Час навчання за режимами

{time_table}

## Розклад ефектів (п. 4)

{effects_table}

Правило: окремо для Δpre і Δunf - якщо |Δ| ≤ R, різниця не відрізняється від розкиду (результату немає); якщо |Δ| > R,
ефект вартий перевірки, але не доведений.

- **Δpre = {eff['delta_pre']:+.4f}**: {eff['verdict_pre']}.
- **Δunf = {eff['delta_unf']:+.4f}**: {eff['verdict_unf']}.

Допоміжні числа для пояснень:

{hints}

## Вибір конфігурації (п. 5, лише за validation)

{selection_table}

Вибрано: **{best_regime}, зерно {best_seed}**, macro-F1(validation) = {f1_val_sel:.4f}.
{"Увага: |Δunf| ≤ R, тобто вибір між frozen і unfreeze визначив шум (процедура від цього не змінюється)." if noise_decided else "|Δunf| > R: вибір режиму не зводиться до шуму."}
Перевірка `{C.MODEL_FILE}`: відповідає правилу вибору за журналами; {weights_note}.

## Тестова оцінка (один раз, CPU)

- macro-F1 на test: **{f1_test:.4f}**; accuracy: {acc:.4f} ({int(round(acc * n_test))}/{n_test}).
- macro-F1 на validation вибраного запуску: {f1_val_sel:.4f}; різниця test − validation: {f1_test - f1_val_sel:+.4f}.
- Стандартна похибка частки правильних відповідей √(p(1−p)/n) при n = {n_test}, p = {acc:.4f}: **{se:.4f}**.
- Оптимізм вибору «найкращого з трьох» усередині режиму (F вибраного − F̄ режиму): {bias_within:+.4f}.

### Матриця плутанини (рядки - справжній клас, стовпці - передбачений)

{cm_table}

Рисунок: `figures/confusion.png`.

### П'ять характерних помилок

{err_table}

Рисунок: `figures/errors.png`. Категорію кожної помилки (сцена містить обидва класи / ймовірна помилка розмітки /
помилка моделі на однозначному зображенні) визначте, подивившись на рисунок, - див. README.
"""
    U.write_text(ROOT / "results" / "analysis.md", md)

    def direction(d):
        return "вищий" if d > 0 else ("нижчий" if d < 0 else "однаковий")
    concl = f"""# Чернетка висновку (числа згенеровано; відредагуйте формулювання)

**Ефект попереднього навчання (Δpre).** Напрям: frozen {direction(eff['delta_pre'])} за scratch
(F̄_frozen = {eff['Fbar_frozen']:.4f} проти {eff['F_scratch']:.4f}). Величина: {eff['delta_pre'] * 100:+.2f} п. macro-F1.
Співвідношення з розкидом: |Δpre| = {abs(eff['delta_pre']):.4f} {'≤' if abs(eff['delta_pre']) <= R else '>'} R = {R:.4f}
(R_frozen = {eff['R_frozen']:.4f}, R_unfreeze = {eff['R_unfreeze']:.4f}) → {eff['verdict_pre']}.

**Ефект розморожування (Δunf).** Напрям: unfreeze {direction(eff['delta_unf'])} за frozen
(F̄_unfreeze = {eff['Fbar_unfreeze']:.4f} проти {eff['Fbar_frozen']:.4f}). Величина: {eff['delta_unf'] * 100:+.2f} п. macro-F1.
Співвідношення з розкидом: |Δunf| = {abs(eff['delta_unf']):.4f} {'≤' if abs(eff['delta_unf']) <= R else '>'} R = {R:.4f} → {eff['verdict_unf']}.

Повний виграш трансферу відносно навчання з нуля: Δpre + Δunf = {eff['delta_total'] * 100:+.2f} п. macro-F1.
Обидва ефекти виміряно за конкретного бюджету 304 кроки оптимізатора: з довшим навчанням scratch міг би скоротити відставання.
"""
    U.write_text(ROOT / "results" / "conclusion_draft.md", concl)

    U.write_json({
        "split_sha256": cur_hash,
        "runs": {f"{r}_seed{s}": {"final_train_loss": logs[(r, s)]["final_train_loss"],
                                  "final_val_macro_f1": logs[(r, s)]["final_val_macro_f1"],
                                  "train_time_seconds": logs[(r, s)]["train_time_seconds"]} for r, s in C.RUNS},
        "effects": eff, "regime_means": means,
        "selected": {"regime": best_regime, "seed": best_seed, "val_macro_f1": f1_val_sel,
                     "noise_decided_selection": bool(noise_decided)},
        "test": {"macro_f1": f1_test, "accuracy": acc, "n": n_test, "se_accuracy": se,
                 "confusion_matrix": cm.tolist(),
                 "top_errors": [{**c, "true_name": C.CLASSES[c["true"]], "pred_name": C.CLASSES[c["pred"]]} for c in cases]},
    }, ROOT / "results" / "analysis.json")

    print("\n" + md)
    print("Файли: results/analysis.md, results/analysis.json, results/conclusion_draft.md, figures/curves.png, "
          "figures/confusion.png, figures/errors.png")


if __name__ == "__main__":
    main()
