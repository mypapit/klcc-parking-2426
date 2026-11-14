"""F14: precision-recall curves per fault type, D3 against the best two baselines.

The stored metrics carry operating points, not the score vectors, so the curves
are recomputed here for one neural run and for the two best baselines at the same
window length, using the same injection positions and the same scoring paths as
the runners.
"""
from __future__ import annotations

import sys
from pathlib import Path

import _bootstrap  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from inject import inject_copy  # noqa: E402
from labels import fault_label  # noqa: E402
from metrics import event_labels  # noqa: E402
from plots import _save  # noqa: E402
from utils import load_config  # noqa: E402

FIG = ROOT / "outputs" / "figures"
FAULTS = ("F1", "F2", "F3", "F4", "F5", "F6")


def neural_scores(cfg, run_dir: Path, length: int, test_a: pd.Series, train_sigma: float):
    import tensorflow as tf

    from data import split_bounds
    from features import Preprocessor
    from models import build_autoencoder
    from windows import make_windows

    model = tf.keras.models.load_model(run_dir / "model.keras")
    processed = pd.read_parquet(cfg["data"]["processed"]).set_index("timestamp")
    series = processed["available"].astype(float)
    bounds = split_bounds(series, cfg["splits"]["history_end"], cfg["splits"]["test_b_start"], cfg["splits"]["test_b_end"])
    pre = Preprocessor("P0")
    pre.window_length = length
    pre.fit(series.loc[bounds.train[0]:bounds.train[1]])
    out = {}
    for fault in FAULTS:
        pooled_scores = []
        pooled_labels = []
        for copy_id in range(cfg["injection"]["copies"]):
            injected_raw, positions = inject_copy(test_a, "testA", fault, copy_id, cfg, sigma_ref=train_sigma)
            injected = pre.transform(injected_raw)
            x, ts = make_windows(injected.to_numpy(), injected.index, length)
            pred = model.predict(x, batch_size=1024, verbose=0)
            scores = np.mean((x - pred) ** 2, axis=(1, 2))
            labels = event_labels(ts, positions, length)
            pooled_scores.append(scores)
            pooled_labels.append(labels)
        out[fault] = (np.concatenate(pooled_scores), np.concatenate(pooled_labels))
    return out


def baseline_scores(cfg, method: str, length: int, seed: int, test_a: pd.Series, train_sigma: float):
    import run_baselines_v2 as rb
    from data import split_bounds

    processed = pd.read_parquet(cfg["data"]["processed"]).set_index("timestamp")
    series = processed["available"].astype(float)
    bounds = split_bounds(series, cfg["splits"]["history_end"], cfg["splits"]["test_b_start"], cfg["splits"]["test_b_end"])
    train = series.loc[bounds.train[0]:bounds.train[1]]
    ctx = rb.build_ctx(cfg, train, length)
    run_dir = ROOT / "outputs" / "runs" / "E4" / f"{method}_L{length}_s{seed}"
    clean_score, clean_ts = rb.score_clean(method, test_a, length, seed, ctx)
    out = {}
    for fault in FAULTS:
        pooled_scores = []
        pooled_labels = []
        for copy_id in range(cfg["injection"]["copies"]):
            injected_raw, positions = inject_copy(test_a, "testA", fault, copy_id, cfg, sigma_ref=train_sigma)
            score, ts = rb.score_injected(method, clean_score, injected_raw, positions, length, seed, ctx)
            labels = event_labels(ts, positions, length)
            pooled_scores.append(np.asarray(score, dtype=float))
            pooled_labels.append(labels)
        out[fault] = (np.concatenate(pooled_scores), np.concatenate(pooled_labels))
    return out


def main() -> int:
    cfg = load_config()
    length = 18
    seed = 0
    t7 = pd.read_csv(ROOT / "outputs" / "tables" / "T7_methods_by_fault.csv")
    sub = t7[(t7["length"] == length) & (t7["testset"] == "testA")]
    methods = sub.groupby("method")["pr_auc_mean"].mean().sort_values(ascending=False)
    best_baselines = [m for m in methods.index if m.startswith("B")][:2]
    print(f"best baselines at L={length}: {best_baselines}")

    processed = pd.read_parquet(cfg["data"]["processed"]).set_index("timestamp")
    series = processed["available"].astype(float)
    from data import split_bounds

    bounds = split_bounds(series, cfg["splits"]["history_end"], cfg["splits"]["test_b_start"], cfg["splits"]["test_b_end"])
    train = series.loc[bounds.train[0]:bounds.train[1]]
    test_a = series.loc[bounds.test_a[0]:bounds.test_a[1]]
    train_sigma = float(np.nanstd(train.to_numpy()))

    curves = {}
    run_dir = ROOT / "outputs" / "runs" / "E3" / f"D3_P0_L{length}_b128_s{seed}"
    if run_dir.exists():
        curves["D3"] = neural_scores(cfg, run_dir, length, test_a, train_sigma)
    for method in best_baselines:
        if (ROOT / "outputs" / "runs" / "E4" / f"{method}_L{length}_s{seed}").exists():
            curves[method] = baseline_scores(cfg, method, length, seed, test_a, train_sigma)
    if "D3" not in curves:
        print("no neural run available")
        return 1

    colours = {"D3": "#4C72B0", "B1": "#DD8452", "B2": "#55A868", "B3": "#C44E52", "B4": "#8172B2"}
    faults = [f for f in FAULTS if f in curves["D3"] and curves["D3"][f][1].sum() > 0]
    ncols = 3
    nrows = (len(faults) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(7.16, 2.2 * nrows), squeeze=False)
    for ax, fault in zip(axes.ravel(), faults):
        for method, data in curves.items():
            scores, labels = data.get(fault, (None, None))
            if scores is None or labels.sum() == 0:
                continue
            precision, recall, _ = precision_recall_curve(labels, scores)
            ax.plot(recall, precision, lw=1.1, color=colours.get(method, "grey"), label=method)
        ax.set_title(fault_label(fault), fontsize=8)
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)
    for ax in axes.ravel()[len(faults):]:
        ax.set_visible(False)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=8, frameon=False,
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(f"Precision-recall curves per fault type, D3 against the best baselines (L={length})", fontsize=9)
    fig.tight_layout(rect=(0, 0.10, 1, 0.96))
    _save(fig, FIG / "F14_pr_curves.pdf")
    print("wrote F14")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
