"""F14 PR curves and F15 detection delay, computed from saved scores."""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve

ROOT = Path(__file__).resolve()[0].parent if False else Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from inject import inject_copy  # noqa: E402
from metrics import event_labels  # noqa: E402
from plots import _save  # noqa: E402
from utils import load_config  # noqa: E402

FIG = ROOT / "outputs" / "figures"


def pr_curves(cfg, length: int, seed: int):
    import tensorflow as tf

    from data import split_bounds
    from features import Preprocessor
    from windows import make_windows

    run_dir = ROOT / "outputs" / "runs" / "E3" / f"D3_P0_L{length}_b128_s{seed}"
    model_path = run_dir / "model.keras"
    if not model_path.exists():
        return None
    df = pd.read_parquet(cfg["data"]["processed"]).set_index("timestamp")
    series = df["available"].astype(float)
    bounds = split_bounds(series, cfg["splits"]["history_end"], cfg["splits"]["test_b_start"], cfg["splits"]["test_b_end"])
    pre = Preprocessor("P0")
    pre.window_length = length
    pre.fit(series.loc[bounds.train[0]:bounds.train[1]])
    test_a = series.loc[bounds.test_a[0]:bounds.test_a[1]]
    model = tf.keras.models.load_model(model_path)
    faults = ["F1", "F2", "F3", "F5", "F6"]
    curves = {}
    for fault in faults:
        injected, positions = inject_copy(test_a, "testA", fault, seed, cfg)
        values = pre.transform(injected)
        x, ts = make_windows(values.to_numpy(), values.index, length)
        pred = model.predict(x, batch_size=1024, verbose=0)
        scores = np.mean((x - pred) ** 2, axis=(1, 2))
        y = event_labels(ts, positions, length)
        if y.sum() == 0:
            continue
        precision, recall, _ = precision_recall_curve(y, scores)
        curves[fault] = (precision, recall, y, scores, ts, positions)
    return curves


def main() -> int:
    cfg = load_config()
    length = 18
    seed = 0
    curves = pr_curves(cfg, length, seed)
    if not curves:
        print("no curves")
        return 1
    faults = list(curves)
    fig, axes = plt.subplots(1, len(faults), figsize=(7.16, 2.2 * len(faults)), squeeze=False)
    for ax, fault in zip(axes[0], faults):
        precision, recall, _, _, _, _ = curves[fault]
        ax.plot(recall, precision, lw=1.0, color="#4C72B0")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title(fault, fontsize=8)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)
    fig.suptitle(f"Precision-recall curves, D3/P0/L{length} (Test A)", fontsize=9)
    _save(fig, FIG / "F14_pr_curves.pdf")

    # F15 detection delay: first flag inside each injected segment minus segment start
    delays = []
    for fault, (_, _, y, scores, ts, positions) in curves.items():
        for _, segment in positions.iterrows():
            start = pd.Timestamp(segment["start"])
            end = pd.Timestamp(segment["end"]) + pd.Timedelta(minutes=10 * (length - 1))
            mask = (ts >= start) & (ts <= end)
            mask_values = np.asarray(mask)
            if not mask_values.any():
                continue
            window_scores = scores[mask_values]
            flagged = window_scores > np.percentile(scores, 99.5)
            if flagged.any():
                first = ts[mask_values][np.argmax(flagged)]
                delays.append({"fault": fault, "delay_minutes": (first - start).total_seconds() / 60})
    if delays:
        df = pd.DataFrame(delays)
        df.to_csv(ROOT / "outputs" / "tables" / "T_fault_detection_delay.csv", index=False)
        fig, ax = plt.subplots(figsize=(7.16, 2.8))
        order = [f for f in ["F1", "F2", "F3", "F5", "F6"] if f in df["fault"].unique()]
        data = [df[df["fault"] == f]["delay_minutes"].to_numpy() for f in order]
        ax.boxplot(data, tick_labels=order)
        ax.set_ylabel("Detection delay (minutes)")
        ax.set_xlabel("Injected fault type")
        _save(fig, FIG / "F15_detection_delay.pdf")
    print("wrote F14, F15")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
