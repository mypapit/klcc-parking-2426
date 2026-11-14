"""F01: the 10-minute available series with the excluded bands and the splits.

The previous version plotted the processed parquet, which starts on
2022-04-25, so it showed neither the excluded 2021 period nor the excluded
early-2022 period that the manuscript text describes. This version goes back
to the raw CSV and plots from 1 Jan 2021.

Double column, 7.16 x 2.2 in, no title inside the figure.
"""
from __future__ import annotations

import sys
from pathlib import Path

import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from figure_style import apply_style, save_figure  # noqa: E402

FIG = ROOT / "outputs" / "figures"
SPLITS = ROOT / "outputs" / "tables" / "T2b_splits.csv"


def exclusion_bands() -> list[dict]:
    """The two excluded periods, with the labels the manuscript uses."""
    return [
        {"start": "2021-01-01", "end": "2021-12-31", "label": "Excluded: COVID-19 restrictions"},
        {"start": "2022-01-01", "end": "2022-04-24", "label": "Excluded: inconsistent sensor readings"},
    ]


def load_raw_series() -> pd.Series:
    """The raw 10-minute series from 1 Jan 2021, after duplicate removal."""
    path = ROOT / "data" / "raw" / "klcc-parking.csv"
    df = pd.read_csv(path)
    df = df[df["siteid"] == 7].copy()
    df["lastupdate"] = pd.to_datetime(df["lastupdate"])
    # the cleaning step keeps the last row for a repeated timestamp
    df = df.sort_values(["lastupdate", "rowid"]).drop_duplicates(subset=["lastupdate"], keep="last")
    series = df.set_index("lastupdate")["available"].astype(float)
    series = series.resample("10min").mean()
    return series.loc["2021-01-01":"2024-03-31"]


def split_bands() -> list[dict]:
    """Train, validation, Test A and Test B, from the T2b split table.

    T2b stores the split names lowercase; the manuscript labels are title case.
    """
    table = pd.read_csv(SPLITS)
    labels = {"train": "Train", "validation": "Validation", "testA": "Test A", "testB": "Test B"}
    return [
        {"label": labels.get(row["split"], row["split"]), "start": row["start"], "end": row["end"]}
        for _, row in table.iterrows()
    ]


def main() -> int:
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    series = load_raw_series()
    apply_style()
    fig, ax = plt.subplots(figsize=(7.16, 2.2))
    ax.plot(series.index, series.to_numpy(), lw=0.4, color="#0072B2", label="Available bays (10-min)")

    # the excluded periods: grey for COVID-19, hatched for the sensor readings
    first = exclusion_bands()[0]
    ax.axvspan(
        pd.Timestamp(first["start"]), pd.Timestamp(first["end"]),
        color="grey", alpha=0.18, lw=0, label=first["label"],
    )
    second = exclusion_bands()[1]
    ax.axvspan(
        pd.Timestamp(second["start"]), pd.Timestamp(second["end"]),
        facecolor="none", edgecolor="grey", hatch="///", lw=0.3, alpha=0.6,
        label=second["label"],
    )

    # the four splits as light background bands, each with a legend entry
    palette = {"train": "#009E73", "validation": "#E69F00", "testA": "#56B4E9", "testB": "#CC79A7"}
    for band in split_bands():
        key = band["label"].lower().replace(" ", "")
        colour = palette.get(key, "#999999")
        ax.axvspan(
            pd.Timestamp(band["start"]), pd.Timestamp(band["end"]),
            color=colour, alpha=0.07, lw=0, label=f"Split: {band['label']}",
        )

    ax.set_xlabel("Time (Asia/Kuala_Lumpur)")
    ax.set_ylabel("Available bays")
    ax.set_xlim(series.index.min(), series.index.max())
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1, 7)))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    # every plotted element has a legend entry. The series fills the axes, so
    # the legend goes in a reserved band UNDER the axis labels, which is why
    # the layout rect reserves space for both of them.
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, fontsize=6,
               frameon=False, handlelength=1.2, columnspacing=0.8,
               borderaxespad=0.0)
    # reserve room below the axes for the legend and the x-label, and above
    # for nothing (there is no title inside the figure)
    fig.tight_layout(rect=(0, 0.16, 1, 1))
    save_figure(fig, FIG / "F01_raw_series.pdf")
    print("wrote F01_raw_series.pdf and .png")
    print(f"  series: {len(series)} bins from {series.index.min()} to {series.index.max()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
