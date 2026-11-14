"""T13: Test A against Test B, without the PR-AUC prevalence artefact.

PR-AUC depends on the positive prevalence, and Test B has roughly half the
windows of Test A but the same number of injections, so its prevalence is about
twice as high. ROC-AUC does not depend on prevalence, so it is the main
cross-set comparison. PR-AUC is reported next to the prevalence and normalised
by it, plus the flag rate per week for each split.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from data import split_bounds  # noqa: E402
from utils import ensure_dir, load_config  # noqa: E402

TABLES = ROOT / "outputs" / "tables"


def main() -> int:
    cfg = load_config()
    ensure_dir(TABLES)
    processed = pd.read_parquet(cfg["data"]["processed"]).set_index("timestamp")
    series = processed["available"].astype(float)
    bounds = split_bounds(series, cfg["splits"]["history_end"], cfg["splits"]["test_b_start"], cfg["splits"]["test_b_end"])
    weeks = {
        "testA": (bounds.test_a[1] - bounds.test_a[0]).total_seconds() / (7 * 24 * 3600),
        "testB": (bounds.test_b[1] - bounds.test_b[0]).total_seconds() / (7 * 24 * 3600),
    }

    rows = []
    flag_rows = []
    for metrics_path in sorted((ROOT / "outputs" / "runs" / "E3").glob("D3_P0_*/metrics.json")):
        run_dir = metrics_path.parent
        name = run_dir.name
        parts = name.split("_")
        length = int([p for p in parts if p.startswith("L")][0][1:])
        seed = int([p for p in parts if p.startswith("s")][0][1:])
        payload = json.loads(metrics_path.read_text())
        thresholds = payload.get("thresholds", {})
        tau = thresholds.get("p99_5")
        fault_path = run_dir / "fault_metrics_allcopies.csv"
        if not fault_path.exists():
            continue
        faults = pd.read_csv(fault_path)
        for testset in ("testA", "testB"):
            sub = faults[faults["testset"] == testset]
            if sub.empty:
                continue
            valid = sub[sub["positive_windows"] > 0]
            scores_path = run_dir / f"scores_{testset}.parquet"
            n_scored = len(pd.read_parquet(scores_path)) if scores_path.exists() else np.nan
            copies = max(sub["copy"].nunique(), 1)
            # prevalence per copy: positive windows over scored windows
            prevalence = float(sub["positive_windows"].sum() / (n_scored * copies)) if n_scored else np.nan
            mean_pr = float(valid["pr_auc"].mean()) if len(valid) else np.nan
            rows.append({
                "run": name,
                "length": length,
                "seed": seed,
                "testset": testset,
                "scored_windows": n_scored,
                "split_weeks": round(weeks[testset], 2),
                "mean_prevalence": prevalence,
                "mean_pr_auc": mean_pr,
                "pr_auc_over_prevalence": mean_pr / prevalence if prevalence else np.nan,
                "pr_auc_minus_prevalence": mean_pr - prevalence if prevalence else np.nan,
                "mean_roc_auc": float(valid["roc_auc"].mean()) if len(valid) else np.nan,
            })
            if tau is not None and scores_path.exists():
                frame = pd.read_parquet(scores_path)
                flagged = int((frame["score"].to_numpy() > tau).sum())
                flag_rows.append({
                    "run": name,
                    "length": length,
                    "seed": seed,
                    "testset": testset,
                    "flagged_windows": flagged,
                    "split_weeks": round(weeks[testset], 2),
                    "flagged_per_week": round(flagged / weeks[testset], 2) if weeks[testset] else np.nan,
                    "flagged_per_1000_windows": round(1000.0 * flagged / len(frame), 3) if len(frame) else np.nan,
                })
    if not rows:
        print("no runs with 5-copy fault metrics")
        return 1
    df = pd.DataFrame(rows)
    df.to_csv(TABLES / "T13_testA_vs_testB.csv", index=False)
    summary = df.groupby(["length", "testset"])[[
        "scored_windows", "split_weeks", "mean_prevalence", "mean_pr_auc",
        "pr_auc_over_prevalence", "pr_auc_minus_prevalence", "mean_roc_auc",
    ]].agg(["mean", "std"])
    summary.columns = [f"{a}_{b}" for a, b in summary.columns]
    summary = summary.reset_index()
    summary.to_csv(TABLES / "T13_testA_vs_testB_summary.csv", index=False)
    print(summary.to_string(index=False))

    flags = pd.DataFrame(flag_rows)
    flags.to_csv(TABLES / "T13_flag_rates.csv", index=False)
    if len(flags):
        print()
        print(flags.groupby(["length", "testset"])[["flagged_per_week", "flagged_per_1000_windows"]].agg(["mean", "std"]).round(3).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
