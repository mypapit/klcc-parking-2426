"""Figures from a completed run: F09 error over time, F10 reconstructions, F15 delay."""
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


def error_over_time(run_dir: Path, path):
    scores = pd.read_parquet(run_dir / "scores_testA.parquet")
    events_path = run_dir / "events_testA.csv"
    threshold = pd.read_json(run_dir / "metrics.json")["thresholds"]["p99_5"]
    fig, ax = plt.subplots(figsize=(7.16, 2.6))
    ax.plot(scores["timestamp"], scores["score"], lw=0.4, color="#4C72B0")
    ax.axhline(threshold, ls="--", color="#C44E52", lw=0.8, label=f"p99.5 threshold ({threshold:.4g})")
    if events_path.exists():
        events = pd.read_csv(events_path, parse_dates=["start", "end"])
        for _, event in events.iterrows():
            ax.axvspan(event["start"], event["end"], color="#C44E52", alpha=0.25, lw=0)
    ax.set_xlabel("Time (Asia/Kuala_Lumpur)")
    ax.set_ylabel("Reconstruction error (MSE)")
    ax.legend(loc="upper right")
    _save(fig, path)
    return True


def reconstruction_examples(model_path: Path, series: pd.Series, path):
    import tensorflow as tf

    from windows import make_windows

    model = tf.keras.models.load_model(model_path)
    length = model.input_shape[1]
    fig, axes = plt.subplots(2, 2, figsize=(7.16, 4.0))
    for ax, (title, start) in zip(axes.ravel(), (("Normal window", 5000), ("Normal window 2", 12000), ("Late window", 20000), ("Final window", 25000))):
        if start + length >= len(series):
            continue
        window = series.iloc[start:start + length].to_numpy(dtype="float32")
        x = window.reshape(1, length, 1)
        pred = model.predict(x, verbose=0).ravel()
        ax.plot(window, lw=0.9, label="observed")
        ax.plot(pred, lw=0.9, ls="--", label="reconstruction")
        ax.set_title(title, fontsize=8)
        ax.set_xlabel("10-min step")
    axes[0, 0].set_ylabel("Available bays")
    axes[1, 0].set_ylabel("Available bays")
    axes[0, 0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path)
    fig.savefig(path.with_suffix(".png"), dpi=300)
    plt.close(fig)
    return True


def main() -> int:
    run_dir = ROOT / "outputs" / "runs" / "E3" / "D3_P0_L18_b128_s0"
    if not (run_dir / "scores_testA.parquet").exists():
        print("run artifacts missing")
        return 1
    error_over_time(run_dir, FIG / "F09_error_over_time.pdf")
    model_path = run_dir / "model.keras"
    series = pd.read_parquet(ROOT / "data" / "processed" / "klcc_10min.parquet").set_index("timestamp")["available"].astype(float)
    if model_path.exists():
        reconstruction_examples(model_path, series, FIG / "F10_reconstructions.pdf")
    print("wrote F09, F10")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
