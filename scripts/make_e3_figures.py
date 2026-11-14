"""Figures from completed E3 runs: threshold sensitivity and preprocessing comparison."""
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

FIG = ROOT / "outputs" / "figures"
TABLES = ROOT / "outputs" / "tables"


def threshold_sensitivity():
    run = ROOT / "outputs" / "runs" / "E3" / "D3_P0_L18_b128_s0"
    scores = pd.read_parquet(run / "scores_testA.parquet")
    faults = pd.read_csv(run / "fault_metrics.csv")
    percentiles = [95, 97, 99, 99.5, 99.9]
    events = []
    for p in percentiles:
        tau = np.percentile(scores["score"], p)
        events.append(int((scores["score"] > tau).sum()))
    fig, ax = plt.subplots(figsize=(7.16, 2.8))
    ax.plot(percentiles, events, marker="o")
    ax.set_xlabel("Threshold percentile (train scores)")
    ax.set_ylabel("Flagged windows (Test A)")
    ax.set_yscale("log")
    _save(fig, FIG / "F11_threshold_sensitivity.pdf")


def prep_bars():
    df = pd.read_csv(TABLES / "T_e3_summary.csv")
    lengths = sorted(df["length"].unique())
    preps = ["P0", "P1", "P2"]
    x = np.arange(len(lengths))
    width = 0.25
    fig, ax = plt.subplots(figsize=(7.16, 3.0))
    for i, prep in enumerate(preps):
        sub = df[df["prep"] == prep].set_index("length").reindex(lengths)
        ax.bar(x + (i - 1) * width, sub["mean_pr_auc_mean"], width, yerr=sub["mean_pr_auc_std"], label=prep, capsize=3)
    ax.set_xticks(x)
    ax.set_xticklabels([f"L={L}" for L in lengths])
    ax.set_xlabel("Window length (10-min steps)")
    ax.set_ylabel("Mean PR-AUC over injected faults")
    ax.legend()
    _save(fig, FIG / "F13_prep_pr_auc.pdf")


def main() -> int:
    threshold_sensitivity()
    prep_bars()
    print("wrote F11, F13")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
