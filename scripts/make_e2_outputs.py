"""Depth-ablation table (T5) and figure (F12) from E2 runs.

T5 needs PR-AUC, ROC-AUC, F1 and event recall per depth and window length.
Event recall is the fraction of injected events that produced at least one
flagged window, which requires the injection positions, not just the metrics.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from plots import _save  # noqa: E402
from utils import load_config  # noqa: E402

RUNS = ROOT / "outputs" / "runs" / "E2"
TABLES = ROOT / "outputs" / "tables"
FIG = ROOT / "outputs" / "figures"


def event_recall(run_dir: Path, cfg: dict) -> tuple[float, int]:
    """Fraction of injected events detected at least once, per fault and copy.

    The per-run file is written by scripts/run_injection_copies.py, which scores
    the injected series itself. Using clean-series scores at injected timestamps
    would measure the wrong thing.
    """
    path = run_dir / "event_recall.csv"
    if not path.exists():
        return np.nan, 0
    events = pd.read_csv(path)
    if events.empty:
        return np.nan, 0
    return float(events["detected"].mean()), len(events)


def main() -> int:
    cfg = load_config()
    rows = []
    for metrics_path in RUNS.glob("*/metrics.json"):
        payload = json.loads(metrics_path.read_text())
        payload["run"] = metrics_path.parent.name
        fault_path = metrics_path.parent / "fault_metrics.csv"
        if fault_path.exists():
            faults = pd.read_csv(fault_path)
            valid = faults[faults["positive_windows"] > 0]
            payload["mean_pr_auc"] = valid["pr_auc"].mean()
            payload["mean_roc_auc"] = valid["roc_auc"].mean()
            # F1 at the p99.5 operating point, the threshold used throughout
            if "p99_5_f1" in valid.columns:
                payload["mean_f1"] = valid["p99_5_f1"].mean()
            else:
                payload["mean_f1"] = np.nan
        else:
            payload["mean_pr_auc"] = np.nan
            payload["mean_roc_auc"] = np.nan
            payload["mean_f1"] = np.nan
        recall, n_events = event_recall(metrics_path.parent, cfg)
        payload["event_recall"] = recall
        payload["injected_events"] = n_events
        rows.append(payload)
    if not rows:
        print("no E2 runs yet")
        return 1
    df = pd.DataFrame(rows)
    df["depth"] = df["run"].str.split("_").str[0]
    df["length"] = df["run"].str.extract(r"_L(\d+)_").astype(int)
    df["seed"] = df["run"].str.extract(r"_s(\d+)$").astype(int)
    # event recall per fault, pooled over the runs that have it
    recall_rows = []
    for metrics_path in sorted(RUNS.glob("*/metrics.json")):
        run_dir = metrics_path.parent
        events_path = run_dir / "event_recall.csv"
        if not events_path.exists():
            continue
        events = pd.read_csv(events_path)
        if events.empty:
            continue
        for (testset, fault), group in events.groupby(["testset", "fault"]):
            recall_rows.append({
                "run": run_dir.name,
                "depth": run_dir.name.split("_")[0],
                "length": int(run_dir.name.split("_L")[1].split("_")[0]),
                "testset": testset,
                "fault": fault,
                "event_recall": float(group["detected"].mean()),
                "events": len(group),
            })
    if recall_rows:
        recall_df = pd.DataFrame(recall_rows)
        recall_df.to_csv(TABLES / "T_e2_event_recall.csv", index=False)
        pooled = recall_df.groupby(["depth", "length", "testset", "fault"])["event_recall"].mean().reset_index()
        pooled.to_csv(TABLES / "T5b_event_recall_by_fault.csv", index=False)
    df.to_csv(TABLES / "T_e2_runs.csv", index=False)
    cols = ["final_val_loss", "events_testA", "mean_pr_auc", "mean_roc_auc", "mean_f1", "event_recall"]
    cols = [c for c in cols if c in df.columns]
    summary = df.groupby(["depth", "length"])[cols].agg(["mean", "std"])
    summary.columns = [f"{a}_{b}" for a, b in summary.columns]
    summary = summary.reset_index()
    summary.to_csv(TABLES / "T5_depth_ablation.csv", index=False)
    print(summary.to_string(index=False))

    if len(summary) >= 3:
        fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.9))
        lengths = sorted(df["length"].unique())
        for ax, column, label in (
            (axes[0], "mean_pr_auc_mean", "Mean PR-AUC over injected faults"),
            (axes[1], "mean_f1_mean", "Mean F1 on injected faults (p99.5)"),
        ):
            for depth in ("D1", "D2", "D3"):
                sub = summary[summary["depth"] == depth].set_index("length").reindex(lengths)
                ax.errorbar(lengths, sub[column], yerr=sub[column.replace("_mean", "_std")], marker="o", capsize=3, label=depth)
            ax.set_xlabel("Window length (10-min steps)")
            ax.set_ylabel(label, fontsize=8)
            ax.set_xticks(lengths)
            ax.legend(fontsize=8, frameon=False)
        fig.tight_layout()
        _save(fig, FIG / "F12_depth_ablation.pdf")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
