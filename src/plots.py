"""Plot helpers."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def _save(fig, path, metadata: dict | None = None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # tight_layout conflicts with the constrained layout engine and with figures
    # that already carry a colorbar, so only apply it when the figure has neither
    engine = getattr(fig, "get_layout_engine", None)
    has_constrained = engine is not None and type(engine()).__name__ == "ConstrainedLayoutEngine"
    if not has_constrained and not getattr(fig, "_legend_handles", None):
        try:
            fig.tight_layout()
        except (RuntimeError, ValueError):
            pass
    fig.savefig(path, metadata=metadata) if metadata else fig.savefig(path)
    png_path = path.with_suffix(".png")
    if metadata:
        fig.savefig(png_path, dpi=300, metadata=metadata)
    else:
        fig.savefig(png_path, dpi=300)
    plt.close(fig)


def line_plot(x, y, path, xlabel="Time", ylabel="Value"):
    fig, ax = plt.subplots(figsize=(7.16, 2.6))
    ax.plot(x, y, lw=0.8)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    _save(fig, path)


def raw_series_figure(series: pd.Series, path):
    line_plot(series.index, series.to_numpy(), path, xlabel="Time (Asia/Kuala_Lumpur)", ylabel="Available bays")


def error_hist_figure(train_scores, test_scores, path):
    fig, ax = plt.subplots(figsize=(7.16, 2.8))
    ax.hist(train_scores, bins=80, alpha=0.6, label="Train", density=True)
    ax.hist(test_scores, bins=80, alpha=0.6, label="Test A", density=True)
    ax.set_yscale("log")
    ax.set_xlabel("Reconstruction error (MSE)")
    ax.set_ylabel("Density")
    ax.legend()
    _save(fig, path)
