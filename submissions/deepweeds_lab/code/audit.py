"""Validate the finished submission against fold 0 and the official evaluator."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from eval import check_against_csv, load_group, read_pred

SUB = Path(__file__).resolve().parent.parent
ROOT = SUB.parents[1]
RUNS = SUB / "runs"
PREDS = SUB / "predictions"
CURVES = SUB / "curves"
LABELS = ROOT / "data" / "labels"


def audit():
    import pandas as pd

    eda = json.loads((SUB / "eda.json").read_text())
    split = eda["split"]
    assert split["union"] == 17509 and split["n"] == {"train": 10501, "val": 3501, "test": 3507}
    assert all(n == 0 for n in split["overlap"].values())
    assert (SUB / "sanity.json").is_file()
    assert (CURVES / "class_distribution.png").is_file()
    assert (CURVES / "class_samples.png").is_file()
    assert (CURVES / "sanity_augmented.png").is_file()

    training_ids = [f"B{i:02d}" for i in range(1, 6)] + [f"T{i:02d}" for i in range(1, 13)]
    for exp in training_ids:
        rd = RUNS / exp / "seed0"
        for filename in ("config.json", "history.csv", "summary.json", "best.pt", "val_logits.npz"):
            assert (rd / filename).is_file(), rd / filename
        assert (CURVES / f"{exp}_seed0.png").is_file(), exp
        assert len(pd.read_csv(rd / "history.csv")) == json.loads((rd / "config.json").read_text())["epochs"]
    baseline = json.loads((RUNS / "B01" / "seed0" / "config.json").read_text())
    fair_keys = ("fold", "seed", "init", "img_size", "aug", "sampler", "mix", "loss", "epochs",
                 "batch_size", "lr_backbone", "lr_head", "weight_decay", "warmup_epochs", "amp")
    for exp in training_ids[:5]:
        cfg = json.loads((RUNS / exp / "seed0" / "config.json").read_text())
        assert all(cfg[k] == baseline[k] for k in fair_keys), exp
    assert len({json.loads((RUNS / exp / "seed0" / "config.json").read_text())["backbone"] for exp in training_ids[:5]}) == 5

    selection = json.loads((SUB / "final_selection.json").read_text())
    selection_time=(SUB / "final_selection.json").stat().st_mtime_ns
    infer = json.loads((SUB / "inference_results.json").read_text())
    methods = {r["method"] for r in infer}
    assert "I00" in methods and len(methods - {"I00"}) >= 4, methods
    assert any(r["source"] == selection["source"] and r["method"] == selection["method"] for r in infer)
    for r in infer:
        for field in ("latency", "latency_batch16"):
            lat = r[field]
            assert lat["n"] >= 50 and 0 < lat["p50"] <= lat["p95"] <= lat["p99"], (r["method"], field)

    final_sheet = pd.read_excel(SUB / "results.xlsx", sheet_name="Final")
    with pd.ExcelFile(SUB / "results.xlsx") as xls:
        assert set(xls.sheet_names) == {"Backbones", "Training", "Inference", "Final", "PerClass", "Latency", "Summary"}
        assert len(pd.read_excel(xls, "Backbones")) == 5
        assert len(pd.read_excel(xls, "Training")) == 12
        assert len(pd.read_excel(xls, "Inference")) >= 5
        assert set(pd.read_excel(xls, "Latency")["batch"]) >= {1, 16}
    result = {}
    for exp in ("T00", "F01"):
        group = load_group(str(PREDS / f"{exp}_seed*_test.csv"), str(LABELS / "test_subset0.csv"))
        assert set(group.seeds) == {0, 1, 2}, (exp, group.seeds)
        for pred, metric in zip(group.preds, group.metrics):
            assert metric["n"] == 3507
            rd = RUNS / exp / f"seed{pred.seed}"
            assert (rd / "test_started.flag").is_file()
            assert selection_time <= (rd / "test_started.flag").stat().st_mtime_ns <= Path(pred.path).stat().st_mtime_ns
            assert (rd / "test_logits.npz").is_file()
            assert (CURVES / f"{exp}_seed{pred.seed}.png").is_file()
            val = read_pred(str(PREDS / f"{exp}_seed{pred.seed}_val.csv"))
            check_against_csv(val, str(LABELS / "val_subset0.csv"), "val")
            assert len(val.y_true) == 3501
            row = final_sheet[(final_sheet.exp_id == exp) & (final_sheet.seed == pred.seed)]
            assert len(row) == 1, (exp, pred.seed)
            for column, key in (("macro_F1_test", "macro_f1"), ("top1_test", "top1"), ("ECE_test", "ece")):
                assert abs(float(row.iloc[0][column]) - metric[key]) < 1e-8, (exp, pred.seed, column)
        result[exp] = {key: group.summary[key] for key in ("macro_f1", "top1", "ece")}

    for filename in ("report.md", "results.xlsx"):
        assert (SUB / filename).is_file(), filename
    for filename in ("F01_confusion_seed0.png", "F01_errors_seed0.png", "inference_tradeoff.png"):
        assert (CURVES / filename).is_file(), filename
    report = (SUB / "report.md").read_text()
    for term in ("Chinee Apple", "Snake Weed", "## 7. Kết luận", "test", "val"):
        assert term in report, term
    print(json.dumps({"status": "PASS", "backbones": 5, "training": 12,
                      "inference_methods": sorted(methods), "selection": selection,
                      "final": result}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    audit()
