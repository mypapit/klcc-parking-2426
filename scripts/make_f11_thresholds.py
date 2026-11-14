"""F11/T8c: the threshold sensitivity table and figure at 5 seeds, all six faults.

What changed and why:

- The addendum's panel (b) asks for injected precision, recall and F1 over
  "all six faults", and `make_f11.py:28` carried a FAULTS tuple that omitted
  F4. This script includes all six.
- The summary and manuscript previously quoted single-run numbers. This table
  is the five-seed mean, so the text can be updated to the mean.
- The aggregation matches T5: pool per seed per fault over the 5 injection
  copies, then take the unweighted mean over the six faults (macro), then the
  mean and std over the 5 seeds. `make_f11.py:83-91` pooled TP/FP/FN across
  every fault before computing precision, a micro-average dominated by F5's
  530 positive windows against F4's 180; that would make F11 incomparable
  with F12.
- USE_POT is False for the figure: the POT family saturates the x axis
  (its two risk values land inside the percentile range on a log scale), so
  the addendum's panel (a)/(b) shows the percentile family only. The POT rows
  stay in the detail table T8b (see make_f11.py) for completeness.

Outputs: T8c_threshold_5seeds.csv (threshold level, macro over faults),
T8c_threshold_5seeds_byfault.csv (fault level), and the F11 figure PDF/PNG
drawn from T8c with the 5-seed mean and std as an error band.

Definitions, to be quoted in FIGURE_NOTES.md:

- `false_alarms_per_week` is FLAGGED WINDOWS per week on clean Test A, the
  same definition `make_f11.py:98` and the T8b table use. At p99.5 it is
  687 flagged windows over 26.38 weeks = 26.04. The addendum's "26 false
  alarms/week" quotes this definition. `merged_false_alarm_events_per_week`
  is also reported, for the reading where the manuscript means events.
- The thresholds are fitted on the train scores of each run, as the protocol
  requires.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from threshold import merge_events, percentile_threshold  # noqa: E402
from utils import ensure_dir, load_config  # noqa: E402

TABLES = ROOT / "outputs" / "tables"
FIG = ROOT / "outputs" / "figures"
FAULTS = ("F1", "F2", "F3", "F4", "F5", "F6")
PERCENTILES = [95, 97, 99, 99.5, 99.9]
# the POT family saturates the figure's log x axis; its rows stay in T8b
USE_POT = False


def score_one_run(run_dir: Path, cfg: dict) -> pd.DataFrame:
    """One run's per-threshold, per-fault metrics for Test A, in long form.

    No retraining: the model and the saved scores are reused, and the injected
    series are re-scored (inference only). One row per threshold x fault, so
    the macro-average and the by-fault views stay consistent by construction.
    """
    import tensorflow as tf

    from data import split_bounds
    from features import Preprocessor
    from inject import inject_copy
    from metrics import event_labels
    from windows import make_windows

    name = run_dir.name
    length = int([p for p in name.split("_") if p.startswith("L")][0][1:])
    seed = int([p for p in name.split("_") if p.startswith("s")][0][1:])
    prep = name.split("_")[1]

    processed = pd.read_parquet(cfg["data"]["processed"]).set_index("timestamp")
    series = processed["available"].astype(float)
    bounds = split_bounds(series, cfg["splits"]["history_end"], cfg["splits"]["test_b_start"], cfg["splits"]["test_b_end"])
    pre = Preprocessor(prep)
    pre.window_length = length
    pre.fit(series.loc[bounds.train[0]:bounds.train[1]])
    test_a = series.loc[bounds.test_a[0]:bounds.test_a[1]]

    train_scores = pd.read_parquet(run_dir / "scores_train.parquet")["score"].to_numpy()
    clean = pd.read_parquet(run_dir / "scores_testA.parquet")
    clean_scores = clean["score"].to_numpy()
    clean_ts = pd.DatetimeIndex(clean["timestamp"])

    model = tf.keras.models.load_model(run_dir / "model.keras")
    thresholds = {f"pct_{p}": percentile_threshold(train_scores, p) for p in PERCENTILES}

    # per fault: the injected scores and labels, pooled over the 5 copies
    injected = {}
    for fault in FAULTS:
        fault_scores, fault_labels = [], []
        for copy_id in range(cfg["injection"]["copies"]):
            injected_raw, positions = inject_copy(test_a, "testA", fault, copy_id, cfg)
            values = pre.transform(injected_raw)
            x, ts_inj = make_windows(values.to_numpy(), values.index, length)
            pred = model.predict(x, batch_size=1024, verbose=0)
            scores = np.mean((x - pred) ** 2, axis=(1, 2))
            labels = event_labels(ts_inj, positions, length)
            fault_scores.append(scores)
            fault_labels.append(labels)
        injected[fault] = (
            np.concatenate(fault_scores),
            np.concatenate(fault_labels),
        )

    weeks = (clean_ts[-1] - clean_ts[0]).total_seconds() / (7 * 24 * 3600)
    rows = []
    for family, tau in thresholds.items():
        # false alarms on the clean Test A: flagged windows AND merged events
        flagged = int((clean_scores > tau).sum())
        events = merge_events(clean_ts, clean_scores, tau, length)
        common = {
            "run": name,
            "length": length,
            "seed": seed,
            "threshold_family": family.split("_")[0],
            "parameter": float(family.split("_")[1]),
            "threshold": tau,
            "merged_false_alarm_events_per_week": len(events) / weeks if weeks else np.nan,
            "false_alarms_per_week": flagged / weeks if weeks else np.nan,
            "false_alarm_events": len(events),
            "flagged_windows": flagged,
            "weeks": weeks,
        }
        # one row per fault; the macro-average is taken at aggregation time
        for fault in FAULTS:
            scores, labels = injected[fault]
            predicted = scores > tau
            tp = int((predicted & (labels == 1)).sum())
            fp = int((predicted & (labels == 0)).sum())
            fn = int((~predicted & (labels == 1)).sum())
            precision = tp / (tp + fp) if tp + fp else np.nan
            recall = tp / (tp + fn) if tp + fn else np.nan
            f1 = 2 * precision * recall / (precision + recall) if precision and recall and (precision + recall) else np.nan
            rows.append({**common, "fault": fault, "injected_precision": precision, "injected_recall": recall, "injected_f1": f1})
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", default="E3")
    parser.add_argument("--prep", default="P0")
    parser.add_argument("--length", type=int, default=18)
    args = parser.parse_args()
    cfg = load_config()
    ensure_dir(TABLES)

    runs = sorted((ROOT / "outputs" / "runs" / args.exp).glob(f"D3_{args.prep}_L{args.length}_b128_s*"))
    if len(runs) < 5:
        print(f"expected 5 runs, found {len(runs)}")
        return 1
    frames = []
    for run_dir in runs:
        df = score_one_run(run_dir, cfg)
        frames.append(df)
        print(f"{run_dir.name}: scored {len(df)} thresholds")
    raw = pd.concat(frames, ignore_index=True)

    # per seed per fault over the copies (pooled in score_one_run), then the
    # fault level over seeds, then the macro over the six faults
    per_seed_fault = raw.groupby(["length", "threshold_family", "parameter", "fault", "seed"]).agg(
        injected_precision=("injected_precision", "mean"),
        injected_recall=("injected_recall", "mean"),
        injected_f1=("injected_f1", "mean"),
    ).reset_index()

    by_fault = per_seed_fault.groupby(["length", "threshold_family", "parameter", "fault"]).agg(
        injected_precision_mean=("injected_precision", "mean"),
        injected_precision_std=("injected_precision", "std"),
        injected_recall_mean=("injected_recall", "mean"),
        injected_recall_std=("injected_recall", "std"),
        injected_f1_mean=("injected_f1", "mean"),
        injected_f1_std=("injected_f1", "std"),
        seeds=("seed", "nunique"),
    ).reset_index()
    by_fault.to_csv(TABLES / "T8c_threshold_5seeds_byfault.csv", index=False)

    # macro over the faults, then mean/std over seeds, for the false alarms too.
    # The mean skips NaN (undefined precision when a fault produces no flagged
    # window at this threshold); record how many of the six faults contributed
    # per cell so the denominator is explicit.
    macro_seed = per_seed_fault.groupby(["length", "threshold_family", "parameter", "seed"]).agg(
        injected_precision=("injected_precision", "mean"),
        injected_recall=("injected_recall", "mean"),
        injected_f1=("injected_f1", "mean"),
        n_faults_precision=("injected_precision", lambda v: int(v.notna().sum())),
        n_faults_recall=("injected_recall", lambda v: int(v.notna().sum())),
        n_faults_f1=("injected_f1", lambda v: int(v.notna().sum())),
    ).reset_index()
    clean_cols = ["length", "threshold_family", "parameter", "seed", "threshold",
                  "merged_false_alarm_events_per_week", "false_alarms_per_week",
                  "false_alarm_events", "flagged_windows", "weeks"]
    macro_seed = macro_seed.merge(raw[clean_cols].drop_duplicates(["threshold_family", "parameter", "seed"]),
                                  on=["length", "threshold_family", "parameter", "seed"])
    agg = macro_seed.groupby(["length", "threshold_family", "parameter"]).agg(
        threshold=("threshold", "first"),
        merged_false_alarm_events_per_week_mean=("merged_false_alarm_events_per_week", "mean"),
        merged_false_alarm_events_per_week_std=("merged_false_alarm_events_per_week", "std"),
        false_alarms_per_week_mean=("false_alarms_per_week", "mean"),
        false_alarms_per_week_std=("false_alarms_per_week", "std"),
        false_alarm_events_mean=("false_alarm_events", "mean"),
        flagged_windows_mean=("flagged_windows", "mean"),
        weeks=("weeks", "first"),
        injected_precision_mean=("injected_precision", "mean"),
        injected_precision_std=("injected_precision", "std"),
        injected_recall_mean=("injected_recall", "mean"),
        injected_recall_std=("injected_recall", "std"),
        injected_f1_mean=("injected_f1", "mean"),
        injected_f1_std=("injected_f1", "std"),
        n_faults_precision_min=("n_faults_precision", "min"),
        n_faults_recall_min=("n_faults_recall", "min"),
        n_faults_f1_min=("n_faults_f1", "min"),
        seeds=("seed", "nunique"),
    ).reset_index()
    out = TABLES / "T8c_threshold_5seeds.csv"
    agg.to_csv(out, index=False)
    raw.to_csv(TABLES / "T8c_threshold_5seeds_per_run.csv", index=False)
    print(out)
    print(agg.to_string(index=False))

    # F11 from the 5-seed table: percentile family only (USE_POT False), the
    # addendum's two panels, shared legend, panel labels, no title
    figure(agg)
    return 0


def figure(agg: pd.DataFrame) -> None:
    """The F11 figure, from the aggregated table.

    Addendum spec (Fig. 6): single column 3.5 x 2.4 in. Panel (a) shows
    merged events on Test A (left y, log scale) AND false alarms per week on
    clean Test A (right y), both the mean over 5 seeds. Panel (b) shows
    injected-fault precision, recall and F1 (mean over 5 seeds, all six
    faults, Test A, L=18). X axis uses the fixed percentiles as CATEGORICAL
    positions; POT is left out.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    from figure_style import apply_style, save_figure

    sub = agg[agg["threshold_family"] == "pct"].sort_values("parameter")
    labels = [f"p{p:g}" for p in sub["parameter"]]
    # categorical x positions, evenly spaced, so p99.5 and p99.9 do not
    # collapse onto each other the way a log axis would place them
    x = np.arange(len(sub))

    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(3.5, 2.4))

    # panel (a): merged events (left y, log) + false alarms/week (right y)
    ax = axes[0]
    axr = ax.twinx()
    ln1 = ax.errorbar(x, sub["false_alarm_events_mean"], marker="o", color="#0072B2",
                      capsize=2, lw=1.0, markersize=3.0, elinewidth=0.7,
                      label="Merged events")
    ln2 = axr.errorbar(x, sub["false_alarms_per_week_mean"], yerr=sub["false_alarms_per_week_std"],
                       marker="s", color="#D55E00", capsize=2, lw=1.0, markersize=3.0,
                       elinewidth=0.7, label="False alarms/week")
    ax.set_yscale("log")
    ax.set_ylabel("Merged events (log)", fontsize=7, color="#0072B2")
    axr.set_ylabel("False alarms / week", fontsize=7, color="#D55E00")
    ax.tick_params(axis="y", labelsize=7, colors="#0072B2")
    axr.tick_params(axis="y", labelsize=7, colors="#D55E00")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=6.5, rotation=45, ha="right")
    ax.set_xlabel("Percentile threshold", fontsize=7)
    ax.text(0.0, 1.02, "(a)", transform=ax.transAxes, fontsize=8, va="bottom", ha="left")
    # no in-plot legend: at 3.5 in the descending curves fill the panel and any
    # box overlaps them. Both entries join the shared top legend instead.
    panel_a_handles = [ln1, ln2]

    # panel (b): injected precision / recall / F1
    ax2 = axes[1]
    for column, label, colour, marker in (
        ("injected_precision", "Precision", "#D55E00", "o"),
        ("injected_recall", "Recall", "#009E73", "s"),
        ("injected_f1", "F1", "#CC79A7", "^"),
    ):
        ax2.errorbar(x, sub[f"{column}_mean"], yerr=sub[f"{column}_std"],
                     marker=marker, color=colour, capsize=2, lw=1.0, markersize=3.0,
                     elinewidth=0.7, label=label)
    ax2.set_xlabel("Percentile threshold", fontsize=7)
    ax2.set_ylabel("Injected faults (macro)", fontsize=7)
    ax2.set_ylim(0, 1.02)
    ax2.tick_params(labelsize=7)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=6.5, rotation=45, ha="right")
    ax2.text(0.0, 1.02, "(b)", transform=ax2.transAxes, fontsize=8, va="bottom", ha="left")

    # a single shared figure-level legend covering both panels' five series,
    # in two rows, inside the figure bounds at the top
    handles_b, lbls_b = ax2.get_legend_handles_labels()
    all_handles = panel_a_handles + handles_b
    all_labels = ["Merged events", "False alarms/week"] + lbls_b
    fig.legend(all_handles, all_labels, loc="upper center", bbox_to_anchor=(0.5, 1.0),
               fontsize=5.5, frameon=False, ncol=3, handlelength=1.0,
               columnspacing=0.6, borderaxespad=0.0, labelspacing=0.3)

    # reserve room at the top for the two-row figure legend
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    save_figure(fig, FIG / "F11_threshold_sensitivity_5seeds.pdf")
    print("wrote F11_threshold_sensitivity_5seeds.pdf and .png")


if __name__ == "__main__":
    raise SystemExit(main())
