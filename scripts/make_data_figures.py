"""F02 data quality over time, F04 MSTL decomposition, F19 cost against performance."""
from __future__ import annotations

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

FIG = ROOT / "outputs" / "figures"
TABLES = ROOT / "outputs" / "tables"


def data_quality_figure(cfg):
    raw = pd.read_csv(cfg["data"]["raw"], parse_dates=["lastupdate"])
    raw = raw[raw["siteid"] == cfg["data"]["siteid"]].sort_values("lastupdate")
    interval = raw["lastupdate"].diff().dt.total_seconds() / 60
    median_interval = interval.groupby(raw["lastupdate"].dt.floor("D")).median()
    dupes = raw[raw.duplicated("lastupdate", keep=False)].groupby(raw["lastupdate"].dt.floor("D")).size()
    out_of_range = raw[(raw["available"] < 0) | (raw["available"] > raw["totalbay"])]
    oor_monthly = out_of_range.groupby(out_of_range["lastupdate"].dt.to_period("M")).size()
    # Fix 5: full-capacity (5300) readings per month. These are valid values, so
    # the out-of-range rule never flagged them, but they behave like sensor
    # states and they sit inside the training split.
    full = raw[raw["available"] == raw["totalbay"]]
    full_monthly = full.groupby(full["lastupdate"].dt.to_period("M")).size()
    fig, axes = plt.subplots(3, 1, figsize=(7.16, 5.6), sharex=False)
    axes[0].plot(median_interval.index, median_interval.to_numpy(), lw=0.7, color="#4C72B0")
    axes[0].set_ylabel("Median sampling interval (min)")
    axes[0].axhline(10, ls="--", color="grey", lw=0.7)
    axes[0].axhline(30, ls=":", color="#C44E52", lw=0.7)
    counts = dupes.groupby(dupes.index.to_period("M")).size()
    axes[1].bar([str(p) for p in counts.index], counts.to_numpy(), color="#DD8452")
    axes[1].set_ylabel("Duplicate-timestamp rows per month")
    axes[1].set_xlabel("Month")
    axes[2].bar([str(p) for p in full_monthly.index], full_monthly.to_numpy(), color="#55A868")
    axes[2].set_ylabel("Full-capacity (5300) readings per month")
    axes[2].set_xlabel("Month")
    fig.tight_layout()
    _save(fig, FIG / "F02_data_quality.pdf")
    quality = pd.DataFrame({
        "metric": [
            "median_interval_min_mean", "median_interval_min_median", "median_interval_min_max",
            "duplicate_rows_per_month_mean", "duplicate_rows_per_month_max", "out_of_range_total",
            "full_capacity_total", "full_capacity_months_max",
        ],
        "value": [
            float(median_interval.mean()), float(median_interval.median()), float(median_interval.max()),
            float(counts.mean()) if len(counts) else 0.0, float(counts.max()) if len(counts) else 0.0,
            float(len(out_of_range)),
            float(len(full)), float(full_monthly.max()) if len(full_monthly) else 0.0,
        ],
    })
    quality.to_csv(TABLES / "T2_data_quality.csv", index=False)
    return True


def mstl_figure(cfg):
    from statsmodels.tsa.seasonal import MSTL

    processed = pd.read_parquet(cfg["data"]["processed"]).set_index("timestamp")["available"].astype(float)
    train = processed.loc[:"2023-03-14 09:20"].dropna()
    if len(train) < 2016:
        return False
    month = train.loc["2023-02-01":"2023-03-01"]
    if len(month) < 2016:
        month = train.iloc[-2016:]
    result = MSTL(month, periods=(144, 1008), stl_kwargs={"robust": True}).fit()
    seasonal_cols = list(result.seasonal.columns)
    daily = result.seasonal[seasonal_cols[0]] if seasonal_cols else None
    weekly = result.seasonal[seasonal_cols[1]] if len(seasonal_cols) > 1 else None
    fig, axes = plt.subplots(5, 1, figsize=(7.16, 6.2), sharex=True)
    for ax, (label, series) in zip(axes, [
        ("Observed", result.observed),
        ("Trend", result.trend),
        ("Daily", daily),
        ("Weekly", weekly),
        ("Residual", result.resid),
    ]):
        if series is None:
            ax.set_visible(False)
            continue
        ax.plot(series.index, series.to_numpy(), lw=0.6)
        ax.set_ylabel(label, fontsize=7)
    axes[-1].set_xlabel("Time (Asia/Kuala_Lumpur)")
    fig.tight_layout()
    fig.savefig(FIG / "F04_mstl_decomposition.pdf")
    fig.savefig(FIG / "F04_mstl_decomposition.png", dpi=300)
    plt.close(fig)
    return True


def cost_performance_figure():
    cost_path = TABLES / "T12_computational_cost.csv"
    perf_path = TABLES / "T7_methods_by_fault.csv"
    if not cost_path.exists() or not perf_path.exists():
        return False
    cost = pd.read_csv(cost_path)
    perf = pd.read_csv(perf_path)
    neural = perf[perf["method"].str.startswith("D")]
    merged = cost.merge(neural.groupby(["method", "length"])["pr_auc_mean"].mean().reset_index(), left_on=["depth", "length"], right_on=["method", "length"], how="inner")
    if merged.empty:
        return False
    fig, ax = plt.subplots(figsize=(7.16, 3.0))
    for depth, group in merged.groupby("depth"):
        ax.scatter(group["params"], group["pr_auc_mean"], s=40, label=depth)
    ax.set_xscale("log")
    ax.set_xlabel("Trainable parameters (log scale)")
    ax.set_ylabel("Mean PR-AUC over injected faults")
    ax.legend()
    _save(fig, FIG / "F19_cost_vs_performance.pdf")
    return True


def main() -> int:
    cfg = load_config()
    print("F02:", data_quality_figure(cfg))
    print("F04:", mstl_figure(cfg))
    print("F19:", cost_performance_figure())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
