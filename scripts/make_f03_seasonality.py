"""F03: the weekly seasonality heatmap, resized for a single column.

The addendum says "resize only", with two corrections:
- Monday at the top and Sunday at the bottom. imshow with origin="lower"
  puts index 0 at the BOTTOM, so the day axis needs origin="upper".
- x ticks every 4 hours.

The addendum's caption basis is unchanged: the mean available bays over the
train split, per day of week and 10-minute slot.
"""
from __future__ import annotations

import sys
from pathlib import Path

import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from figure_style import save_figure  # noqa: E402

FIG = ROOT / "outputs" / "figures"

ORIGIN = "upper"


def day_labels() -> list[str]:
    return ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def hour_ticks() -> list[str]:
    # every 4 hours: 00:00, 04:00 ... 20:00
    return [f"{h:02d}:00" for h in range(0, 24, 4)]


def pivot_table() -> pd.DataFrame:
    """Mean available bays over the train split, per day and 10-min slot."""
    df = pd.read_parquet(ROOT / "data" / "processed" / "klcc_10min.parquet").set_index("timestamp")
    series = df["available"].astype(float)
    # the T2b train split, the same basis the seasonality figure has always used
    end_train = pd.Timestamp("2023-03-14 09:20:00")
    train = series.loc[:end_train]
    keys = pd.DataFrame({
        "dow": train.index.day_of_week,
        "slot": train.index.hour * 6 + train.index.minute // 10,
        "value": train.to_numpy(),
    })
    return keys.pivot_table(index="dow", columns="slot", values="value", aggfunc="mean")


def build():
    """Return the completed figure and axis, so the test can inspect it."""
    import matplotlib.pyplot as plt

    from figure_style import apply_style

    pivot = pivot_table()
    apply_style()
    fig, ax = plt.subplots(figsize=(3.5, 2.3))
    # oriGIN="upper" draws row 0 (Monday) at the top, which the addendum wants.
    im = ax.imshow(pivot.to_numpy(), aspect="auto", origin=ORIGIN, cmap="viridis")
    ax.set_yticks(range(7))
    ax.set_yticklabels(day_labels())
    slot_per_4h = 24
    ticks = list(range(0, 144, slot_per_4h))
    ax.set_xticks(ticks)
    ax.set_xticklabels(hour_ticks())
    ax.set_xlabel("Time of day")
    ax.set_ylabel("Day of week")
    fig.colorbar(im, ax=ax, label="Available bays")
    return fig, ax


def main() -> int:
    fig, _ = build()
    # the y-label "Day of week" needs a little more left margin than the
    # default at this size, otherwise it clips
    fig.tight_layout(rect=(0.02, 0, 1, 1))
    save_figure(fig, FIG / "F03_seasonality_heatmap.pdf")
    print("wrote F03_seasonality_heatmap.pdf and .png")
    # geometry check: 3.5 x 2.3 in at 300 dpi must be 1050 x 690 pixels
    from PIL import Image

    png = FIG / "F03_seasonality_heatmap.png"
    size = Image.open(png).size
    ok = size == (1050, 690)
    print(f"  PNG size {size[0]}x{size[1]} (expected 1050x690) -> {'OK' if ok else 'MISMATCH'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
