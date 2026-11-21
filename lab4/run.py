"""ЛР4. FP32 проти INT8: єдина команда запуску.

uv run python run.py --model ../lab3/lab3_model.pt --split ../lab3/split.csv --data path/to/archive

Порядок: експорт і перевірка -> калібрування -> якість -> діапазони каналів -> (потрібен PREDICTION.md)
-> потензорний дослід -> затримка. Якщо PREDICTION.md відсутній або містить TODO, запуск зупиняється
після діапазонів (частковий lab4_results.json зберігається): напишіть прогноз і запустіть команду ще раз.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

import data
import export
import latency
import quality
import quantize
import ranges


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(obj, path: Path) -> None:
    def default(o):
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(type(o))
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=default), encoding="utf-8")


def prediction_ready(p: Path) -> bool:
    return p.exists() and "TODO" not in p.read_text(encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="шлях до lab3_model.pt")
    ap.add_argument("--split", required=True, help="шлях до split.csv з ЛР3")
    ap.add_argument("--data", required=True, help="тека archive (seg_train, seg_test, seg_pred)")
    ap.add_argument("--work", default="artifacts", help="тека для .onnx (не додається до git)")
    ap.add_argument("--out", default="lab4_results.json")
    ap.add_argument("--prediction", default="PREDICTION.md")
    a = ap.parse_args()

    work = Path(a.work)
    work.mkdir(exist_ok=True)
    fp32, prep = work / "lab4_fp32.onnx", work / "lab4_fp32_prep.onnx"
    int8, int8_pt = work / "lab4_int8.onnx", work / "lab4_int8_per_tensor.onnx"
    out_path, pred_path = Path(a.out), Path(a.prediction)
    R: dict = {"model_file": str(a.model), "split_file": str(a.split), "split_sha256": sha256(Path(a.split))}

    # --- дані й контролі ---
    rows = data.read_split(a.split)
    R["split_check"] = data.check_split(rows)
    test_rows = data.part_rows(rows, "test")          # порядок як у split.csv
    cal_rows = data.calibration_rows(rows)
    R["calibration_check"] = data.check_calibration(cal_rows)

    model, meta = export.load_model(a.model)
    R["model_meta"] = meta
    S = meta["img_size"]
    print(f"Завантаження {len(test_rows)} test і {len(cal_rows)} калібрувальних зображень ...")
    x_test = data.load_array(a.data, test_rows, S)
    y_test = data.labels_of(test_rows)
    x_cal = data.load_array(a.data, cal_rows, S)

    # --- п. 1: експорт і перевірка ---
    t_export = export.export_onnx(model, fp32, S)
    R["export_time_s"] = t_export
    R["fp32_graph_ops"] = export.count_ops(fp32)
    sess_fp32 = export.make_session(fp32)
    lg_torch = export.torch_logits(model, x_test)
    lg_fp32 = export.onnx_logits(sess_fp32, x_test)
    R["export_check"] = quality.export_check(lg_torch, lg_fp32, y_test)
    print("Експорт:", R["export_check"], R["fp32_graph_ops"])

    # --- п. 2: квантизація (per-channel) ---
    R["quant_preprocess_time_s"] = quantize.preprocess(fp32, prep)
    R["quantize_time_s"] = quantize.quantize(prep, int8, x_cal, per_channel=True)
    sess_int8 = export.make_session(int8)

    # --- п. 3: якість ---
    lg_int8 = export.onnx_logits(sess_int8, x_test)
    R["quality"] = {
        "fp32": {**quality.basic_metrics(lg_fp32, y_test), "agreement_with_fp32": 1.0},
        "int8": quality.compare_to_fp32(lg_fp32, lg_int8, y_test),
    }
    print("INT8:", R["quality"]["int8"])

    # --- п. 4: діапазони каналів ---
    R["ranges"] = ranges.conv_ranges(fp32)
    print("Топ-3 r:", [(t["node"], round(t["r"], 2)) for t in R["ranges"]["top3"]],
          "медіана r:", round(R["ranges"]["median_r"], 3),
          "біт:", round(R["ranges"]["median_log2_r_bits"], 3))

    if not prediction_ready(pred_path):
        R["status"] = "partial: потрібен PREDICTION.md без TODO перед потензорним запуском"
        dump(R, out_path)
        print(f"\nЗУПИНКА: заповніть {pred_path} (приберіть TODO) за числами діапазонів вище й запустіть "
              "ту саму команду ще раз. Прогноз не можна змінювати після потензорного запуску.")
        return 2
    R["prediction_sha256"] = sha256(pred_path)

    R["per_tensor_quantize_time_s"] = quantize.quantize(prep, int8_pt, x_cal, per_channel=False)
    sess_pt = export.make_session(int8_pt)
    lg_pt = export.onnx_logits(sess_pt, x_test)
    R["per_tensor"] = quality.compare_to_fp32(lg_fp32, lg_pt, y_test)
    print("INT8 per-tensor:", R["per_tensor"])

    # --- п. 5: затримка ---
    print("Вимірювання затримки ...")
    R["latency"] = latency.measure(sess_fp32, sess_int8, x_test[:1], fp32, int8)
    R["environment"] = latency.environment()
    print({k: R["latency"][k] for k in ("speedup_p50", "size_ratio", "p95_ratio", "latency_change_measured")})

    R["status"] = "complete"
    dump(R, out_path)
    print(f"Готово: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
