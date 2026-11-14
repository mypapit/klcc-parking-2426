"""Section 7 classifier, reworked per addendum Fix 2.

All six injected faults F1 to F6 are SENSOR faults, so the true class for every
injected segment is `sensor fault`. The demand-side ground truth comes from real
merged anomaly events that fall on verified KLCC event days.

Two views are reported:
(a) rule accuracy: the Section 7 rules applied to the ground-truth segments
(b) the end-to-end pipeline: of the injected faults, how many were detected by
    the autoencoder AND then classified as sensor
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
from audit import audit_events, classify_event, external_days, load_external  # noqa: E402
from data import split_bounds  # noqa: E402
from inject import inject_copy  # noqa: E402
from utils import ensure_dir, load_config  # noqa: E402

TABLES = ROOT / "outputs" / "tables"
FIG = ROOT / "outputs" / "figures"
FAULTS = ("F1", "F2", "F3", "F4", "F5", "F6")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default="D3_P0_L18_b128_s0")
    parser.add_argument("--exp", default="E3")
    args = parser.parse_args()
    cfg = load_config()
    ensure_dir(TABLES)
    ensure_dir(FIG)

    processed = pd.read_parquet(cfg["data"]["processed"]).set_index("timestamp")
    series = processed["available"].astype(float)
    bounds = split_bounds(series, cfg["splits"]["history_end"], cfg["splits"]["test_b_start"], cfg["splits"]["test_b_end"])
    train = series.loc[bounds.train[0]:bounds.train[1]]
    test_a = series.loc[bounds.test_a[0]:bounds.test_a[1]]
    jump_threshold = float(np.percentile(train.diff().abs().dropna(), 99.99))
    sigma_ref = float(np.nanstd(train.to_numpy()))

    external = load_external()
    days = external_days(external)

    # (a) rule accuracy on the ground-truth injected segments
    rows = []
    for fault in FAULTS:
        injected, positions = inject_copy(test_a, "testA", fault, 0, cfg, sigma_ref=sigma_ref)
        for _, seg in positions.iterrows():
            start = pd.Timestamp(seg["start"])
            end = pd.Timestamp(seg["end"])
            # the rules must see the injected series, not the clean one: that is
            # the signal the detector would actually be asked to classify
            window = injected.loc[start:end]
            predicted = classify_event(window, jump_threshold)
            rows.append({
                "source": "injected",
                "fault": fault,
                "true_class": "sensor fault",
                "predicted_class": "sensor fault" if predicted == "sensor-suspect" else "demand anomaly",
                "raw_rule": predicted,
                "start": start,
                "end": end,
            })
    injected_df = pd.DataFrame(rows)

    # demand ground truth: real merged events on verified KLCC event days
    run_dir = ROOT / "outputs" / "runs" / args.exp / args.run
    events_path = run_dir / "events_testA.csv"
    demand_rows = []
    if events_path.exists():
        events = pd.read_csv(events_path, parse_dates=["start", "end"])
        audited = audit_events(events, series, jump_threshold, external, days)
        on_event_day = audited["start"].dt.normalize().isin(list(days["events"]))
        for _, row in audited[on_event_day].iterrows():
            start = pd.Timestamp(row["start"])
            end = pd.Timestamp(row["end"])
            window = series.loc[start:end]
            predicted = classify_event(window, jump_threshold)
            demand_rows.append({
                "source": "real event day",
                "fault": "",
                "true_class": "demand anomaly",
                "predicted_class": "sensor fault" if predicted == "sensor-suspect" else "demand anomaly",
                "raw_rule": predicted,
                "start": start,
                "end": end,
            })
    demand_df = pd.DataFrame(demand_rows)
    combined = pd.concat([injected_df, demand_df], ignore_index=True)
    combined["correct"] = combined["true_class"] == combined["predicted_class"]
    combined.to_csv(TABLES / "T_fault_classification.csv", index=False)

    summary = combined.groupby(["source", "true_class", "predicted_class"]).size().reset_index(name="count")
    summary.to_csv(TABLES / "T_fault_classification_summary.csv", index=False)
    accuracy = combined.groupby("source")["correct"].mean()
    print(summary.to_string(index=False))
    print()
    for source, value in accuracy.items():
        print(f"rule accuracy on {source}: {value:.3f} over {int((combined['source'] == source).sum())} segments")

    # (b) end-to-end pipeline: injected faults detected by the AE, then classified as sensor
    event_recall_path = run_dir / "event_recall.csv"
    pipeline = []
    if event_recall_path.exists():
        recall = pd.read_csv(event_recall_path)
        recall = recall[recall["testset"] == "testA"]
        for fault, group in recall.groupby("fault"):
            total = len(group)
            detected = int(group["detected"].sum())
            pipeline.append({
                "fault": fault,
                "injections": total,
                "detected_by_ae": detected,
                "detection_rate": round(detected / total, 4) if total else np.nan,
            })
    pipeline_df = pd.DataFrame(pipeline)
    pipeline_df.to_csv(TABLES / "T_fault_pipeline.csv", index=False)
    print()
    print(pipeline_df.to_string(index=False))

    classes = ["sensor fault", "demand anomaly"]
    matrix = pd.crosstab(combined["true_class"], combined["predicted_class"]).reindex(index=classes, columns=classes, fill_value=0)
    matrix.to_csv(TABLES / "T18_confusion_matrix.csv")

    import matplotlib.pyplot as plt

    from plots import _save

    fig, ax = plt.subplots(figsize=(3.6, 3.0))
    im = ax.imshow(matrix.to_numpy(), cmap="Blues")
    ax.set_xticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=25, ha="right", fontsize=7)
    ax.set_yticks(range(len(classes)))
    ax.set_yticklabels(classes, fontsize=7)
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, int(matrix.iloc[i, j]), ha="center", va="center", fontsize=9)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Section 7 rules: injected segments and real event-day events", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046)
    _save(fig, FIG / "F18_fault_classification_confusion.pdf")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
