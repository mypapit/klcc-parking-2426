"""F11: threshold sensitivity (merged events, F1 on injected faults, false alarms/week).

Thresholds are fitted on train scores, as the protocol requires. F1 on the injected
faults is computed by re-scoring the injected series once and evaluating every
candidate threshold against the labels, rather than reusing stored metrics.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from inject import inject_copy  # noqa: E402
from metrics import event_labels  # noqa: E402
from plots import _save  # noqa: E402
from threshold import merge_events, percentile_threshold, pot_threshold  # noqa: E402
from utils import load_config  # noqa: E402
from windows import make_windows  # noqa: E402

FIG = ROOT / "outputs" / "figures"
TABLES = ROOT / "outputs" / "tables"
# all six faults: F4 was excluded from this tuple before, which contradicted
# the addendum's "all six faults" for panel (b); see make_f11_thresholds.py
FAULTS = ("F1", "F2", "F3", "F4", "F5", "F6")


def main() -> int:
    import tensorflow as tf

    from data import split_bounds
    from features import Preprocessor
    from models import build_autoencoder

    cfg = load_config()
    length, seed, prep = 18, 0, "P0"
    run = ROOT / "outputs" / "runs" / "E3" / f"D3_{prep}_L{length}_b128_s{seed}"
    if not (run / "model.keras").exists():
        print("model missing")
        return 1

    processed = pd.read_parquet(cfg["data"]["processed"]).set_index("timestamp")
    series = processed["available"].astype(float)
    bounds = split_bounds(series, cfg["splits"]["history_end"], cfg["splits"]["test_b_start"], cfg["splits"]["test_b_end"])
    pre = Preprocessor(prep)
    pre.window_length = length
    pre.fit(series.loc[bounds.train[0]:bounds.train[1]])
    test_a = series.loc[bounds.test_a[0]:bounds.test_a[1]]

    train_scores = pd.read_parquet(run / "scores_train.parquet")["score"].to_numpy()
    test = pd.read_parquet(run / "scores_testA.parquet")
    test_scores = test["score"].to_numpy()
    ts = pd.DatetimeIndex(test["timestamp"])

    model = tf.keras.models.load_model(run / "model.keras")

    # score every injected fault once, then evaluate all thresholds against the labels
    injected_scores = {}
    for fault in FAULTS:
        for copy_id in range(cfg["injection"]["copies"]):
            injected_raw, positions = inject_copy(test_a, "testA", fault, copy_id, cfg)
            injected = pre.transform(injected_raw)
            x, ts_inj = make_windows(injected.to_numpy(), injected.index, length)
            pred = model.predict(x, batch_size=1024, verbose=0)
            scores = np.mean((x - pred) ** 2, axis=(1, 2))
            labels = event_labels(ts_inj, positions, length)
            if labels.sum() == 0:
                continue
            injected_scores[(fault, copy_id)] = (scores, labels)

    percentiles = [90, 95, 97, 99, 99.5, 99.9, 99.95]
    candidates = [("pct", p, percentile_threshold(train_scores, p)) for p in percentiles]
    candidates += [("pot", r, pot_threshold(train_scores, r)) for r in (1e-3, 1e-4)]

    rows = []
    weeks = (ts[-1] - ts[0]).total_seconds() / (7 * 24 * 3600)
    for family, parameter, tau in candidates:
        events = merge_events(ts, test_scores, tau, length)
        flagged = test_scores > tau
        # per-fault precision/recall/F1, then the unweighted mean (macro).
        # The previous version pooled TP/FP/FN across every fault before
        # computing precision, a micro-average dominated by F5's 530 positive
        # windows against F4's 180, which made F11 incomparable with F12.
        per_fault = []
        for fault in FAULTS:
            p_vals, r_vals, f1_vals = [], [], []
            for (f, _copy), (scores, labels) in injected_scores.items():
                if f != fault:
                    continue
                predicted = scores > tau
                tp = int((predicted & (labels == 1)).sum())
                fp = int((predicted & (labels == 0)).sum())
                fn = int((~predicted & (labels == 1)).sum())
                precision = tp / (tp + fp) if tp + fp else np.nan
                recall = tp / (tp + fn) if tp + fn else np.nan
                f1 = 2 * precision * recall / (precision + recall) if precision and recall and (precision + recall) else np.nan
                p_vals.append(precision)
                r_vals.append(recall)
                f1_vals.append(f1)
            per_fault.append((np.nanmean(p_vals), np.nanmean(r_vals), np.nanmean(f1_vals)))
        # macro over the faults, skipping faults whose precision is undefined
        # (no flagged window at this threshold) but recording how many faults
        # contributed, so the denominator is explicit. At p99.9/p99.95 at
        # least one fault produces no flagged windows, so the macro covers
        # fewer than six faults there.
        precision = float(np.nanmean([p for p, _, _ in per_fault]))
        recall = float(np.nanmean([r for _, r, _ in per_fault]))
        f1 = float(np.nanmean([v for _, _, v in per_fault]))
        n_precision = int(np.isfinite([p for p, _, _ in per_fault]).sum())
        n_recall = int(np.isfinite([r for _, r, _ in per_fault]).sum())
        n_f1 = int(np.isfinite([v for _, _, v in per_fault]).sum())
        rows.append({
            "family": family,
            "parameter": parameter,
            "threshold": tau,
            "merged_events": len(events),
            "flagged_windows": int(flagged.sum()),
            "n_faults_precision": n_precision,
            "n_faults_recall": n_recall,
            "n_faults_f1": n_f1,
            "false_alarms_per_week": float(flagged.sum()) / weeks if weeks > 0 else np.nan,
            "injected_precision": precision,
            "injected_recall": recall,
            "injected_f1": f1,
        })
    table = pd.DataFrame(rows)
    table.to_csv(TABLES / "T8b_threshold_sensitivity_detail.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.9))
    ax = axes[0]
    ax.plot(table["parameter"], table["merged_events"], marker="o", color="#4C72B0")
    ax.set_xscale("log")
    ax.set_xlabel("Threshold (percentile on the left, POT risk on the right)")
    ax.set_ylabel("Merged event count (Test A)")
    ax.set_yscale("log")
    ax2 = axes[1]
    ax2.plot(table["parameter"], table["injected_f1"], marker="s", color="#DD8452")
    ax2.set_xscale("log")
    ax2.set_xlabel("Threshold (percentile on the left, POT risk on the right)")
    ax2.set_ylabel("F1 on injected faults")
    ax2.set_ylim(0, 1.02)
    fig.tight_layout()
    _save(fig, FIG / "F11_threshold_sensitivity.pdf")
    print(table.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
