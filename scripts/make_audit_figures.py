"""F16 audit breakdown and F17 base-rate figures, from the Section 8 audit tables."""
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

FIG = ROOT / "outputs" / "figures"
TABLES = ROOT / "outputs" / "tables"

# Section 8 priority order, sensor-suspect first
CATEGORIES = [
    ("cat_sensor-suspect", "sensor-suspect", "#C44E52"),
    ("cat_KLCC event", "KLCC event", "#4C72B0"),
    ("cat_public holiday", "public holiday", "#55A868"),
    ("cat_festive period", "festive period", "#64B5CD"),
    ("cat_school holiday", "school holiday", "#8172B2"),
    ("cat_heavy rain", "heavy rain", "#DA8BC3"),
    ("cat_weekend", "weekend", "#8C8C8C"),
    ("cat_unexplained", "unexplained", "#DD8452"),
]


def audit_breakdown():
    path = TABLES / "T11_base_rate_tests.csv"
    if not path.exists():
        return False
    df = pd.read_csv(path)
    lengths = sorted(df["length"].unique())
    fig, ax = plt.subplots(figsize=(7.16, 3.4))
    bottom = np.zeros(len(lengths))
    for column, label, colour in CATEGORIES:
        if column not in df.columns:
            continue
        values = df.groupby("length")[column].mean().reindex(lengths).to_numpy()
        if values.sum() == 0:
            continue
        ax.bar([f"L={length}" for length in lengths], values, bottom=bottom, label=label, color=colour)
        bottom += values
    ax.set_xlabel("Window length (10-min steps)")
    ax.set_ylabel("Merged anomaly events (mean over seeds)")
    ax.legend(ncol=4, fontsize=7, frameon=False)
    _save(fig, FIG / "F16_audit_breakdown.pdf")
    return True


def base_rate_figure():
    path = TABLES / "T11_base_rate_tests.csv"
    if not path.exists():
        return False
    df = pd.read_csv(path)
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.9))
    for ax, cause, title in (
        (axes[0], "holiday", "Public holidays"),
        (axes[1], "event", "KLCC event days"),
    ):
        for length, group in df.groupby("length"):
            fraction = group[f"matched_{cause}"] / group["events"]
            pvalues = group[f"permutation_p_{cause}"]
            ax.scatter(fraction, np.maximum(pvalues, 1e-4), label=f"L={length}", s=30)
        ax.axhline(0.05, ls="--", color="grey", lw=0.8)
        ax.set_yscale("log")
        ax.set_xlabel("Fraction of merged events on those days")
        ax.set_ylabel("Permutation p-value")
        ax.set_title(title, fontsize=9)
        ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    _save(fig, FIG / "F17_permutation_test.pdf")
    return True


def main() -> int:
    print("audit breakdown:", audit_breakdown())
    print("base rate:", base_rate_figure())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
