import argparse
import shutil
import sys
from pathlib import Path

import torch

import checks as K
import config as C
import data as D
import engine as E
import selection as SEL
import split as S
import utils as U

ROOT = Path(__file__).resolve().parent


def stop(msg):
    print(f"\n[ЗУПИНКА] {msg}", file=sys.stderr)
    sys.exit(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="шлях до теки archive (де лежать seg_train, seg_test)")
    ap.add_argument("--resume", action="store_true",
                    help="пропустити запуски, у яких уже є run.json з тим самим хешем split.csv")
    ap.add_argument("--skip-checks", action="store_true", help="пропустити перевірки пп. 1-2 (лише для налагодження)")
    args = ap.parse_args()

    device = U.pick_device()
    print(f"Пристрій: {U.device_name(device)}  ({device})")

    print("\n== 1. Поділ ==")
    ok_raw, n_tr, n_te = S.check_raw_counts(args.data)
    print(f"seg_train: {n_tr} файлів (очікується 14034), seg_test: {n_te} (очікується 3000)")
    if not ok_raw:
        stop("кількість зображень у наборі не збігається з умовою; перевірте шлях --data")
    rows_t = S.build_split(args.data)
    S.write_split(rows_t, ROOT / C.SPLIT_CSV)
    ok_split, md, _ = S.check_split(rows_t)
    U.write_text(ROOT / "results" / "split_check.md", "# Контроль поділу\n\n" + md + "\n")
    print(md)
    if not ok_split:
        stop("поділ не збігається з контрольною таблицею - це помилка реалізації; далі не йдемо")
    split_hash = U.sha256_file(ROOT / C.SPLIT_CSV)
    print(f"SHA-256 split.csv: {split_hash}")
    all_rows = S.read_split(ROOT / C.SPLIT_CSV)

    if not args.skip_checks:
        print("\n== 2. Перевірки до навчання ==")
        K.save_batch_figure(all_rows, args.data, ROOT / "figures" / "batch.png")
        print("figures/batch.png збережено")
        K.check_initial_loss(all_rows, args.data, device, ROOT / "results" / "initial_loss.md")
        print("results/initial_loss.md збережено")
        ok_params = K.check_params(device, ROOT / "results" / "param_table.md")
        print(f"параметри й групи оптимізатора збігаються з таблицею: {ok_params}")
        ok_mod = K.check_module_changes(all_rows, args.data, device, ROOT / "results" / "module_changes.md")
        print(f"зміни модулів збігаються з очікуваними: {ok_mod}")
        K.check_bn_experiment(all_rows, args.data, device,
                              ROOT / "results" / "bn_experiment_prediction.md",
                              ROOT / "results" / "bn_experiment.md")
        print("дослід із BN виконано (results/bn_experiment*.md)")
        if not (ok_params and ok_mod):
            stop("перевірки пп. 1-2 не збіглися з очікуваним; знайдіть помилку перед основним експериментом")

    print("\n== 3. Сім запусків ==")
    f1 = {}
    for regime, seed in C.RUNS:
        run_dir = ROOT / "runs" / f"{regime}_seed{seed}"
        rj = run_dir / "run.json"
        if args.resume and rj.exists() and (run_dir / "model.pt").exists():
            log = U.read_json(rj)
            if log.get("split_sha256") == split_hash:
                print(f"- {regime} seed{seed}: уже є, пропускаю (F1={log['final_val_macro_f1']:.4f})")
                f1[(regime, seed)] = log["final_val_macro_f1"]
                continue
        print(f"- {regime} seed{seed}: навчання")
        log = E.run_training(regime, seed, all_rows, args.data, device, run_dir, split_hash)
        f1[(regime, seed)] = log["final_val_macro_f1"]
        print(f"  підсумок: train_loss={log['final_train_loss']:.4f}  val_macro_F1={log['final_val_macro_f1']:.4f}  "
              f"час навчання={log['train_time_seconds']:.1f} с")

    print("\n== 5. Вибір конфігурації (за validation) ==")
    regime, seed, means = SEL.choose(f1)
    print("Середні F за режимами:", {k: round(v, 4) for k, v in means.items()})
    print(f"Вибрано: {regime}, зерно {seed}, F1(val) = {f1[(regime, seed)]:.4f}")
    state = torch.load(ROOT / "runs" / f"{regime}_seed{seed}" / "model.pt", map_location="cpu", weights_only=True)
    torch.save({"state_dict": state, "regime": regime, "seed": seed,
                "classes": list(C.CLASSES), "img_size": C.IMG_SIZE}, ROOT / C.MODEL_FILE)
    print(f"Збережено {C.MODEL_FILE}")
    print("\nГотово. Наступний крок:  uv run python analyze.py --data", args.data)


if __name__ == "__main__":
    main()
