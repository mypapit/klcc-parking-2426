"""F13: the method x fault heatmap, re-laid out for the manuscript.

The previous version was 7.8 in tall, nearly a full page. The addendum wants
the same information at 7.16 x 2.6 in: three panels side by side (L = 6, 18,
36), a shared colour scale, one colour bar on the right.

The label strings are the addendum's short forms, which differ slightly from
`src/labels.fault_short` ("F2 zero" rather than "F2 zero block", "F4 exc."
rather than "F4 excursion"). They are defined here so the addendum's wording
is what appears in the figures.
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
TABLE = ROOT / "outputs" / "tables" / "T7_methods_by_fault.csv"

METHODS = ["D1", "D2", "D3", "B1", "B2", "B3", "B4"]
SHORT_FAULTS = ["F1 stuck", "F2 zero", "F3 full", "F4 exc.", "F5 drift", "F6 noise"]
FAULT_CODES = ["F1", "F2", "F3", "F4", "F5", "F6"]

FIGSIZE = (7.16, 2.6)


def build(testset: str):
    """Return the completed figure and its axes for one test set."""
    import matplotlib.pyplot as plt

    from figure_style import apply_style

    t7 = pd.read_csv(TABLE)
    sub = t7[t7["testset"] == testset]
    lengths = sorted(sub["length"].unique())
    # one shared colour scale across all three panels, so the gap between the
    # neural methods and the baselines stays visible
    vmax = float(np.nanmax(sub["pr_auc_mean"].to_numpy(dtype=float)))
    apply_style()
    fig, axes = plt.subplots(1, len(lengths), figsize=FIGSIZE, squeeze=False, layout="constrained")
    for ax, length in zip(axes[0], lengths):
        pivot = (
            sub[sub["length"] == length]
            .pivot_table(index="method", columns="fault", values="pr_auc_mean")
            .reindex(index=METHODS, columns=FAULT_CODES)
        )
        values = pivot.to_numpy(dtype=float)
        im = ax.imshow(values, aspect="auto", cmap="viridis", vmin=0.0, vmax=vmax)
        ax.set_xticks(range(len(FAULT_CODES)))
        ax.set_xticklabels(SHORT_FAULTS, rotation=45, ha="right", fontsize=6.5)
        ax.set_yticks(range(len(METHODS)))
        ax.set_yticklabels(METHODS, fontsize=6.5)
        # a short L label at the top of each panel; no full title inside the
        # figure, the caption carries that
        ax.set_xlabel(f"L={length}", fontsize=6.5, labelpad=1.0)
        for i in range(len(METHODS)):
            for j in range(len(FAULT_CODES)):
                value = values[i, j]
                if not np.isfinite(value):
                    ax.text(j, i, "n/a", ha="center", va="center", fontsize=6, color="white")
                    continue
                shade = "white" if value < vmax * 0.55 else "black"
                ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=6, color=shade)
    fig.colorbar(im, ax=list(axes.ravel()), fraction=0.03, pad=0.015, label="PR-AUC")
    return fig, axes


def main() -> int:
    for testset in ("testA", "testB"):
        fig, _ = build(testset)
        save_figure(fig, FIG / f"F13_method_fault_heatmap_{testset}.pdf")
        print(f"wrote F13_method_fault_heatmap_{testset}.pdf and .png")
    # geometry check: 7.16 x 2.6 in at 300 dpi must be 2148 x 780
    from PIL import Image

    png = FIG / "F13_method_fault_heatmap_testA.png"
    size = Image.open(png).size
    ok = size == (2148, 780)
    print(f"  PNG size {size[0]}x{size[1]} (expected 2148x780) -> {'OK' if ok else 'MISMATCH'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
