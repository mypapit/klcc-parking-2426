"""Seasonality and preprocessing comparison figures from real data."""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from data import split_bounds  # noqa: E402
from features import Preprocessor  # noqa: E402
from plots import _save  # noqa: E402

FIG = ROOT / "outputs" / "figures"


def main() -> int:
    df = pd.read_parquet(ROOT / "data" / "processed" / "klcc_10min.parquet").set_index("timestamp")
    series = df["available"].astype(float)
    bounds = split_bounds(series, "2023-12-31 23:59:59", "2024-01-01 00:00:00", "2024-03-31 23:59:59")
    train = series.loc[bounds.train[0]:bounds.train[1]]

    # F03: seasonality heatmap (train only)
    keys = pd.DataFrame({"dow": train.index.day_of_week, "slot": train.index.hour * 6 + train.index.minute // 10, "value": train.to_numpy()})
    pivot = keys.pivot_table(index="dow", columns="slot", values="value", aggfunc="mean")
    fig, ax = plt.subplots(figsize=(7.16, 2.6))
    im = ax.imshow(pivot.to_numpy(), aspect="auto", origin="lower", cmap="viridis")
    ax.set_yticks(range(7))
    ax.set_yticklabels(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
    ticks = [i for i in range(0, 144, 12)]
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{i // 6:02d}:00" for i in ticks])
    ax.set_xlabel("Time of day")
    ax.set_ylabel("Day of week")
    fig.colorbar(im, ax=ax, label="Available bays")
    _save(fig, FIG / "F03_seasonality_heatmap.pdf")

    # F05: preprocessing comparison, one week
    week = train.iloc[-1008:]
    fig, axes = plt.subplots(3, 1, figsize=(7.16, 5.4), sharex=True)
    axes[0].plot(week.index, week.to_numpy(), lw=0.7)
    axes[0].set_ylabel("Raw (bays)")
    for ax, (variant, length) in zip(axes[1:], (("P1", 18), ("P2", 18))):
        pre = Preprocessor(variant)
        pre.window_length = length
        pre.fit(train)
        values = pre.transform(week)
        ax.plot(week.index, values.to_numpy(), lw=0.7)
        ax.set_ylabel(f"{variant} residual")
    axes[-1].set_xlabel("Time (Asia/Kuala_Lumpur)")
    _save(fig, FIG / "F05_preprocessing_comparison.pdf")
    print("wrote F03, F05")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
