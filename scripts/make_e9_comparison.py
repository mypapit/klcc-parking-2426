"""E9 versus E2: D3/P0 with and without the full-capacity mask.

E9 masks long full-capacity runs from train and validation, on the hypothesis
that the model otherwise learns long flat stretches at capacity as normal,
which is why the stuck-sensor fault F1 sits near chance.

This compares PR-AUC per fault and the merged-event count on Test A, per
window length, averaged over the five seeds of each experiment.
"""
from __future__ import annotations

import sys
from pathlib import Path

import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from utils import ensure_dir  # noqa: E402

TABLES = ROOT / "outputs" / "tables"
FAULTS = ("F1", "F2", "F3", "F4", "F5", "F6")


def collect(exp: str) -> pd.DataFrame:
    rows = []
    for run_dir in sorted((ROOT / "outputs" / "runs" / exp).glob("D3_P0_L*")):
        fault_path = run_dir / "fault_metrics_allcopies.csv"
        if not fault_path.exists():
            fault_path = run_dir / "fault_metrics.csv"
            if not fault_path.exists():
                continue
        df = pd.read_csv(fault_path)
        df["run"] = run_dir.name
        df["length"] = int(run_dir.name.split("_L")[1].split("_")[0])
        df["seed"] = int(run_dir.name.split("_s")[1])
        rows.append(df)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def merged_events(exp: str) -> pd.DataFrame:
    rows = []
    for run_dir in sorted((ROOT / "outputs" / "runs" / exp).glob("D3_P0_L*")):
        path = run_dir / "events_testA.csv"
        if not path.exists():
            continue
        events = pd.read_csv(path)
        rows.append({
            "run": run_dir.name,
            "experiment": exp,
            "length": int(run_dir.name.split("_L")[1].split("_")[0]),
            "seed": int(run_dir.name.split("_s")[1]),
            "events_testA": len(events),
        })
    return pd.DataFrame(rows)


def event_recall(exp: str) -> pd.DataFrame:
    """Per-fault event recall: was each injected segment flagged at p99.5?"""
    rows = []
    for run_dir in sorted((ROOT / "outputs" / "runs" / exp).glob("D3_P0_L*")):
        path = run_dir / "event_recall.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        for (testset, fault), group in df.groupby(["testset", "fault"]):
            rows.append({
                "experiment": exp,
                "run": run_dir.name,
                "length": int(run_dir.name.split("_L")[1].split("_")[0]),
                "seed": int(run_dir.name.split("_s")[1]),
                "testset": testset,
                "fault": fault,
                "detected_rate": float(group["detected"].mean()),
            })
    return pd.DataFrame(rows)


def main() -> int:
    ensure_dir(TABLES)
    e2 = collect("E2")
    e9 = collect("E9")
    if e2.empty or e9.empty:
        print(f"missing data: E2 rows {len(e2)}, E9 rows {len(e9)}")
        return 1

    def prepare(df: pd.DataFrame, label: str) -> pd.DataFrame:
        out = df.copy()
        out["experiment"] = label
        return out

    combined = pd.concat([prepare(e2, "E2"), prepare(e9, "E9")], ignore_index=True)
    # the single-copy fault_metrics.csv has no testset column; default to Test A
    if "testset" not in combined.columns:
        combined["testset"] = "testA"
    else:
        combined["testset"] = combined["testset"].fillna("testA")

    per_fault = combined.groupby(["experiment", "length", "testset", "fault"]).agg(
        pr_auc_mean=("pr_auc", "mean"),
        pr_auc_std=("pr_auc", "std"),
        roc_auc_mean=("roc_auc", "mean"),
        roc_auc_std=("roc_auc", "std"),
        positive_windows=("positive_windows", "mean"),
        seeds=("seed", "nunique"),
    ).reset_index()
    per_fault.to_csv(TABLES / "T_e9_fault_metrics.csv", index=False)

    # the comparison the addendum asks for: E2 D3 against E9 D3
    pivot = per_fault.pivot_table(
        index=["length", "testset", "fault"], columns="experiment", values="pr_auc_mean"
    ).reset_index()
    if "E2" in pivot.columns and "E9" in pivot.columns:
        pivot["delta"] = (pivot["E9"] - pivot["E2"]).round(6)
        pivot["E2"] = pivot["E2"].round(6)
        pivot["E9"] = pivot["E9"].round(6)
    pivot.to_csv(TABLES / "T_e9_vs_e2_pr_auc.csv", index=False)

    events = pd.concat([merged_events("E2"), merged_events("E9")], ignore_index=True)
    if not events.empty:
        # the experiment label is attached at collection time, because the run
        # names are identical across experiments
        ev = events.groupby(["experiment", "length"]).agg(
            events_testA_mean=("events_testA", "mean"),
            events_testA_std=("events_testA", "std"),
            seeds=("seed", "nunique"),
        ).reset_index()
        ev.to_csv(TABLES / "T_e9_vs_e2_events.csv", index=False)
        ev_pivot = ev.pivot_table(index="length", columns="experiment", values="events_testA_mean").reset_index()
        if "E2" in ev_pivot.columns and "E9" in ev_pivot.columns:
            ev_pivot["delta"] = (ev_pivot["E9"] - ev_pivot["E2"]).round(4)
            ev_pivot["E2"] = ev_pivot["E2"].round(4)
            ev_pivot["E9"] = ev_pivot["E9"].round(4)
        ev_pivot.to_csv(TABLES / "T_e9_vs_e2.csv", index=False)
        print("merged events on Test A")
        print(ev_pivot.to_string(index=False))
        print()

    recall = pd.concat([event_recall("E2"), event_recall("E9")], ignore_index=True)
    if not recall.empty:
        recall_agg = recall.groupby(["experiment", "length", "testset", "fault"]).agg(
            detected_rate_mean=("detected_rate", "mean"),
            detected_rate_std=("detected_rate", "std"),
            seeds=("seed", "nunique"),
        ).reset_index()
        recall_agg.to_csv(TABLES / "T_e9_vs_e2_event_recall.csv", index=False)
        recall_pivot = recall_agg.pivot_table(
            index=["length", "testset", "fault"], columns="experiment", values="detected_rate_mean"
        ).reset_index()
        if "E2" in recall_pivot.columns and "E9" in recall_pivot.columns:
            recall_pivot["delta"] = (recall_pivot["E9"] - recall_pivot["E2"]).round(6)
            recall_pivot["E2"] = recall_pivot["E2"].round(6)
            recall_pivot["E9"] = recall_pivot["E9"].round(6)
        recall_pivot.to_csv(TABLES / "T_e9_vs_e2_event_recall_wide.csv", index=False)
        print("event recall per fault, E2 against E9 (Test A)")
        sub = recall_pivot[recall_pivot["testset"] == "testA"]
        print(sub.to_string(index=False))
        print()

    print("PR-AUC per fault, E2 against E9")
    print(pivot.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
