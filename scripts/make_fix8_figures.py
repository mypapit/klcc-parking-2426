"""Fix 8 small corrections: F07 mean +/- std band, F11 dual axis, F17 null histogram."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import _bootstrap  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from plots import _save  # noqa: E402

FIG = ROOT / "outputs" / "figures"
TABLES = ROOT / "outputs" / "tables"


def f07():
    """Learning curves as the mean +/- std band over seeds per window length."""
    frames = []
    for metrics_path in sorted((ROOT / "outputs" / "runs" / "E2").glob("D[123]_P0_*/metrics.json")):
        history = metrics_path.parent / "history.csv"
        if not history.exists():
            continue
        df = pd.read_csv(history)
        name = metrics_path.parent.name
        parts = name.split("_")
        df["depth"] = parts[0]
        df["length"] = int([p for p in parts if p.startswith("L")][0][1:])
        df["seed"] = int([p for p in parts if p.startswith("s")][0][1:])
        df["epoch"] = range(1, len(df) + 1)
        frames.append(df)
    if not frames:
        print("no histories")
        return False
    hist = pd.concat(frames, ignore_index=True)
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.6), squeeze=False)
    for ax, length in zip(axes[0], sorted(hist["length"].unique())):
        for depth in ("D1", "D2", "D3"):
            sub = hist[(hist["depth"] == depth) & (hist["length"] == length)]
            if sub.empty:
                continue
            pivot = sub.pivot_table(index="epoch", columns="seed", values="val_loss", aggfunc="first")
            mean = pivot.mean(axis=1)
            std = pivot.std(axis=1).fillna(0.0)
            ax.plot(mean.index, mean.to_numpy(), lw=0.9, label=depth)
            ax.fill_between(mean.index, (mean - std).to_numpy(), (mean + std).to_numpy(), alpha=0.18)
        ax.set_title(f"L={length}", fontsize=8)
        ax.set_xlabel("Epoch")
        ax.set_yscale("log")
    axes[0][0].set_ylabel("Validation loss")
    axes[0][0].legend(fontsize=7, frameon=False)
    fig.suptitle("Learning curves: mean +/- std over five seeds", fontsize=9)
    _save(fig, FIG / "F07_learning_curves.pdf")
    return True


def f11():
    path = TABLES / "T8b_threshold_sensitivity_detail.csv"
    if not path.exists():
        return False
    df = pd.read_csv(path)
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.9))
    ax = axes[0]
    ax.plot(df["parameter"], df["merged_events"], marker="o", color="#4C72B0", label="merged events (Test A)")
    ax.set_xscale("log")
    ax.set_xlabel("Threshold percentile (left) or POT risk (right)")
    ax.set_ylabel("Merged event count", color="#4C72B0")
    ax.set_yscale("log")
    ax.tick_params(axis="y", labelcolor="#4C72B0")
    ax2 = ax.twinx()
    ax2.plot(df["parameter"], df["false_alarms_per_week"], marker="s", color="#DD8452", label="false alarms/week")
    ax2.set_ylabel("False alarms per week", color="#DD8452")
    ax2.set_yscale("log")
    ax2.tick_params(axis="y", labelcolor="#DD8452")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=7, frameon=False)
    axes[1].plot(df["parameter"], df["injected_f1"], marker="s", color="#55A868")
    axes[1].set_xscale("log")
    axes[1].set_xlabel("Threshold percentile (left) or POT risk (right)")
    axes[1].set_ylabel("F1 on injected faults")
    axes[1].set_ylim(0, 1.02)
    fig.tight_layout()
    _save(fig, FIG / "F11_threshold_sensitivity.pdf")
    return True


def f17():
    """Null distribution for the day-level event test, one panel per length."""
    from base_rate_day import N_PERM, SEED

    from audit import load_external, external_days
    from utils import load_config

    cfg = load_config()
    external = load_external()
    days = external_days(external)
    event_days = set(days["events"])

    from data import split_bounds

    series = pd.read_parquet(cfg["data"]["processed"]).set_index("timestamp")["available"].astype(float)
    bounds = split_bounds(series, cfg["splits"]["history_end"], cfg["splits"]["test_b_start"], cfg["splits"]["test_b_end"])
    split_days = {}
    for split, span in (("testA", bounds.test_a), ("testB", bounds.test_b)):
        block = series.loc[span[0]:span[1]]
        split_days[split] = pd.DatetimeIndex(sorted(set(block.index.normalize())))

    runs = sorted((ROOT / "outputs" / "runs" / "E3").glob("D3_P0_*"))
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.5), squeeze=False)
    for ax, length in zip(axes[0], (6, 18, 36)):
        run = None
        for candidate in runs:
            if f"_L{length}_" in candidate.name:
                run = candidate
                break
        if run is None:
            continue
        events_path = run / "events_testA.csv"
        if not events_path.exists():
            continue
        events = pd.read_csv(events_path, parse_dates=["start", "end"])
        test_days = split_days["testA"]
        n_days = len(test_days)
        anom = pd.DatetimeIndex(sorted(set(events["start"].dt.normalize())))
        n_anom = len(anom)
        if n_anom == 0:
            continue
        cause = np.asarray(test_days.isin(list(event_days)), dtype=bool)
        special_idx = np.flatnonzero(cause)
        rng = np.random.default_rng(SEED)
        hits = np.zeros(N_PERM, dtype=int)
        for i in range(N_PERM):
            draw = rng.choice(n_days, size=n_anom, replace=False)
            hits[i] = np.isin(draw, special_idx).sum()
        k = int(anom.isin(list(event_days)).sum())
        ax.hist(hits, bins=40, color="#4C72B0", alpha=0.8)
        ax.axvline(k, color="#C44E52", lw=1.4)
        p_value = float((hits >= k).mean())
        # the permutation p is bounded below by 1/(n_perm+1); report the bound
        p_text = "< 0.0001" if p_value <= 0 else f"{p_value:.4f}"
        ax.set_title(f"L={length} (observed {k}, p {p_text})", fontsize=8)
        ax.set_ylabel("Permutations")
        # leave room at the right edge so the observed-value line is not clipped
        ax.set_xlim(0, max(float(hits.max()), float(k)) + 2)
    fig.suptitle("Day-level event-day test: null distribution and observed value", fontsize=9)
    fig.supxlabel("Anomaly days that are event days")
    fig.tight_layout()
    _save(fig, FIG / "F17_permutation_test.pdf")
    return True


def main() -> int:
    print("F07:", f07())
    print("F11:", f11())
    print("F17:", f17())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
