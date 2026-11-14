"""Check 1: end-to-end detection and classification per fault, D3/P0, L = 6/18/36.

The end-to-end pipeline table in RESULT_SUMMARY2 said 0 of 50 F4 injections were
detected by the autoencoder while T7 gave D3 F4 PR-AUC 0.83 to 0.90. The two
cannot both hold at the p99.5 threshold: the pipeline table was written before
the F4 re-injection and so read the pre-fix (NaN) injections.

This script rebuilds the pipeline from the current artifacts:

- for each fault, the injection segments and whether the AE flagged them at
  p99.5, taken from the run's event_recall.csv
- of the detected injections, the Section 7 classification applied to the
  injected series, so the answer is what the operator would actually get
- the injected-classifier confusion matrix broken down per fault type

Detection comes from the saved event_recall.csv. Classification is recomputed
here because the classifier must see the injected values, not the clean ones.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from audit import classify_event  # noqa: E402
from data import split_bounds  # noqa: E402
from inject import inject_copy  # noqa: E402
from utils import ensure_dir, load_config  # noqa: E402

TABLES = ROOT / "outputs" / "tables"
FAULTS = ("F1", "F2", "F3", "F4", "F5", "F6")


def classify_injected_segments(series: pd.Series, testset: str, fault: str, copy_id: int, cfg: dict, jump_threshold: float) -> pd.DataFrame:
    """Classify every injected segment of one copy with the Section 7 rules."""
    injected, positions = inject_copy(series, testset, fault, copy_id, cfg)
    rows = []
    for _, seg in positions.iterrows():
        start = pd.Timestamp(seg["start"])
        end = pd.Timestamp(seg["end"])
        window = injected.loc[start:end]
        raw_rule = classify_event(window, jump_threshold)
        rows.append({
            "testset": testset,
            "fault": fault,
            "copy": copy_id,
            "start": start,
            "end": end,
            "raw_rule": raw_rule,
            "predicted_class": "sensor fault" if raw_rule == "sensor-suspect" else "demand anomaly",
        })
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", default="E2")
    parser.add_argument("--prep", default="P0")
    parser.add_argument("--depth", default="D3")
    args = parser.parse_args()
    cfg = load_config()
    ensure_dir(TABLES)

    processed = pd.read_parquet(cfg["data"]["processed"]).set_index("timestamp")
    series = processed["available"].astype(float)
    bounds = split_bounds(series, cfg["splits"]["history_end"], cfg["splits"]["test_b_start"], cfg["splits"]["test_b_end"])
    train = series.loc[bounds.train[0]:bounds.train[1]]
    jump_threshold = float(np.percentile(train.diff().abs().dropna(), 99.99))

    runs = sorted(
        p for p in (ROOT / "outputs" / "runs" / args.exp).glob(f"{args.depth}_{args.prep}_L*")
        if p.name.split("_")[1] == args.prep
    )
    if not runs:
        print("no runs found")
        return 1

    detection_rows = []
    classification_rows = []
    for run_dir in runs:
        name = run_dir.name
        length = int([p for p in name.split("_") if p.startswith("L")][0][1:])
        seed = int([p for p in name.split("_") if p.startswith("s")][0][1:])
        recall_path = run_dir / "event_recall.csv"
        if not recall_path.exists():
            print(f"skip {name}: no event_recall.csv")
            continue
        recall = pd.read_csv(recall_path, parse_dates=["start", "end"])
        recall["run"] = name
        recall["length"] = length
        recall["seed"] = seed
        detection_rows.append(recall)

        # the classifier must see the injected series, so rebuild each copy
        for testset in ("testA", "testB"):
            span = bounds.test_a if testset == "testA" else bounds.test_b
            raw_eval = series.loc[span[0]:span[1]]
            for fault in FAULTS:
                for copy_id in range(cfg["injection"]["copies"]):
                    classified = classify_injected_segments(raw_eval, testset, fault, copy_id, cfg, jump_threshold)
                    classified["run"] = name
                    classified["length"] = length
                    classified["seed"] = seed
                    classification_rows.append(classified)

    if not detection_rows:
        print("no detection data")
        return 1
    detection = pd.concat(detection_rows, ignore_index=True)
    classification = pd.concat(classification_rows, ignore_index=True)

    # join the detection flag onto the classified segments
    merged = classification.merge(
        detection[["run", "testset", "fault", "copy", "start", "end", "detected"]],
        on=["run", "testset", "fault", "copy", "start", "end"],
        how="left",
    )
    merged["detected"] = merged["detected"].fillna(False).astype(bool)
    merged["true_class"] = "sensor fault"
    merged.to_csv(TABLES / "T_check1_injections_classified.csv", index=False)

    # per fault: injections, detected, rate, and the class of the detected ones
    per_run = merged.groupby(["run", "length", "seed", "testset", "fault"]).agg(
        injections=("detected", "size"),
        detected_by_ae=("detected", "sum"),
        detected_sensor=("predicted_class", lambda s: int(((s == "sensor fault") & merged.loc[s.index, "detected"]).sum())),
        detected_demand=("predicted_class", lambda s: int(((s == "demand anomaly") & merged.loc[s.index, "detected"]).sum())),
        injected_sensor=("predicted_class", lambda s: int((s == "sensor fault").sum())),
        injected_demand=("predicted_class", lambda s: int((s == "demand anomaly").sum())),
    ).reset_index()
    per_run["detection_rate"] = (per_run["detected_by_ae"] / per_run["injections"]).round(4)
    per_run.to_csv(TABLES / "T_check1_pipeline_per_run.csv", index=False)

    agg = per_run.groupby(["length", "testset", "fault"]).agg(
        injections=("injections", "sum"),
        detected_by_ae=("detected_by_ae", "sum"),
        detected_sensor=("detected_sensor", "sum"),
        detected_demand=("detected_demand", "sum"),
        injected_sensor=("injected_sensor", "sum"),
        injected_demand=("injected_demand", "sum"),
        seeds=("seed", "nunique"),
    ).reset_index()
    agg["detection_rate"] = (agg["detected_by_ae"] / agg["injections"]).round(4)
    # the class of the detected ones, as a share of the detected ones
    agg["detected_sensor_share"] = (agg["detected_sensor"] / agg["detected_by_ae"].replace(0, np.nan)).round(4)
    agg.to_csv(TABLES / "T_check1_pipeline.csv", index=False)
    print(agg.to_string(index=False))

    # the injected-classifier confusion matrix, broken down per fault type
    confusion = merged.groupby(["length", "testset", "fault", "true_class", "predicted_class"]).size().reset_index(name="count")
    confusion.to_csv(TABLES / "T_check1_confusion_by_fault.csv", index=False)

    total = merged.groupby(["length", "testset", "fault", "true_class", "predicted_class"]).size().unstack(
        fill_value=0
    ).reset_index()
    total.to_csv(TABLES / "T_check1_confusion_wide.csv", index=False)
    print()
    print(total.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
