"""F12: the depth ablation, resized for a single column.

The addendum says "resize only": keep both panels side by side (a) mean PR-AUC
and (b) mean F1 at p99.5, error bars = std over 5 seeds, one shared legend
(D1, D2, D3) above the panels, and panel labels "(a)" and "(b)".

Caption basis, to be stated in FIGURE_NOTES.md: the means cover F1, F2, F3,
F5 and F6 on both test sets, matching T5. F4 is absent from T5 because the
single-copy fault_metrics.csv it is built from recorded zero positive windows
for F4 at the time T5 was generated; the corrected fault_metrics_allcopies.csv
gives F4 PR-AUC about 0.92 on Test A, so the exclusion is an artifact of the
stale file, not a property of the fault. State the basis rather than implying
that F4 is undetectable.
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
TABLE = ROOT / "outputs" / "tables" / "T5_depth_ablation.csv"

CAPTION_FAULTS = ("F1", "F2", "F3", "F5", "F6")

# the faults whose single-copy positive_windows were zero when T5 was generated
STALE_FAULTS = ("F4",)

COLOURS = {"D1": "#0072B2", "D2": "#D55E00", "D3": "#009E73"}


def t5_basis_faults() -> tuple[str, ...]:
    """The faults present with positive windows in T5's stored raw basis."""
    rows = []
    for seed in range(5):
        path = ROOT / "outputs" / "runs" / "E2" / f"D3_P0_L18_b128_s{seed}" / "fault_metrics.csv"
        d = pd.read_csv(path)
        rows.extend(d[d["positive_windows"] > 0]["fault"].tolist())
    return tuple(sorted(set(rows)))


def main() -> int:
    import matplotlib.pyplot as plt

    from figure_style import apply_style

    t5 = pd.read_csv(TABLE)
    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(3.5, 2.0))
    lengths = sorted(t5["length"].unique())

    for ax, column, panel_label, ylabel in zip(
        axes,
        ("mean_pr_auc_mean", "mean_f1_mean"),
        ("(a)", "(b)"),
        ("Mean PR-AUC", "Mean F1 @ p99.5"),
    ):
        for depth in ("D1", "D2", "D3"):
            sub = t5[t5["depth"] == depth].set_index("length").reindex(lengths)
            yerr = sub[column.replace("_mean", "_std")]
            ax.errorbar(
                lengths, sub[column], yerr=yerr, marker="o", capsize=2,
                label=depth, color=COLOURS[depth], lw=1.0, markersize=3.5, elinewidth=0.7,
            )
        # the panel label sits above the axes, not as a title, so the figure
        # carries no title inside the plotted area while still identifying
        # the panels as (a) and (b)
        ax.text(0.0, 1.02, panel_label, transform=ax.transAxes,
                fontsize=8, va="bottom", ha="left")
        ax.set_ylabel(ylabel, fontsize=8, labelpad=1.0)
        # one figure-level x-label instead of one per panel: at 3.5 in two
        # copies of the same long label collide with each other and clip at
        # the right edge
        ax.set_xticks(lengths)
        ax.set_xticklabels([str(v) for v in lengths])
    # one shared legend above the panels
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, fontsize=7,
               frameon=False, handlelength=1.4, columnspacing=1.0,
               borderaxespad=0.0)
    fig.supxlabel("Window length (10-min steps)", fontsize=8)
    # the rect reserves the top band for the shared legend
    fig.tight_layout(rect=(0, 0.04, 1, 0.88), w_pad=1.2, h_pad=1.0)
    save_figure(fig, FIG / "F12_depth_ablation.pdf")
    print("wrote F12_depth_ablation.pdf and .png")
    # geometry check: 3.5 x 2.0 in at 300 dpi must be 1050 x 600 pixels
    from PIL import Image

    png = FIG / "F12_depth_ablation.png"
    size = Image.open(png).size
    ok = size == (1050, 600)
    print(f"  PNG size {size[0]}x{size[1]} (expected 1050x600) -> {'OK' if ok else 'MISMATCH'}")
    # record the caption basis where the notes will quote it
    print(f"  caption basis: {CAPTION_FAULTS} on Test A and Test B")
    print(f"  T5 raw basis confirms: {t5_basis_faults()}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
