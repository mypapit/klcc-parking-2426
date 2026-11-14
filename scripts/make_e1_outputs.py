"""T4: E1 replication table, including flagged-window counts per threshold.

The runner stores merged event counts, not raw flagged-window counts, so those are
derived here from the saved scores rather than by retraining.
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
from threshold import merge_events  # noqa: E402
from utils import ensure_dir  # noqa: E402

TABLES = ROOT / "outputs" / "tables"


def main() -> int:
    ensure_dir(TABLES)
    rows = []
    for metrics_path in sorted((ROOT / "outputs" / "runs" / "E1").glob("*/metrics.json")):
        payload = json.loads(metrics_path.read_text())
        run_dir = metrics_path.parent
        name = run_dir.name
        parts = name.split("_")
        row = {
            "run": name,
            "prep": parts[1],
            "length": int([p for p in parts if p.startswith("L")][0][1:]),
            "batch": int([p for p in parts if p.startswith("b")][0][1:]),
            "seed": int([p for p in parts if p.startswith("s")][0][1:]),
            "epochs": payload["epochs"],
            "final_train_loss": payload["final_train_loss"],
            "final_val_loss": payload["final_val_loss"],
            "merged_events_testA": payload["events_testA"],
            "params": payload.get("params"),
            "deterministic_ops": payload.get("deterministic_ops"),
        }
        thresholds = payload.get("thresholds", {})
        scores = {}
        for split in ("train", "val", "testA", "testB"):
            path = run_dir / f"scores_{split}.parquet"
            if path.exists():
                scores[split] = pd.read_parquet(path)
        for tname, tvalue in thresholds.items():
            row[f"threshold_{tname}"] = tvalue
            for split, frame in scores.items():
                values = frame["score"].to_numpy()
                row[f"flagged_{split}_{tname}"] = int((values > tvalue).sum())
        # recompute the merged event count from the saved scores as a cross-check.
        # merge_events expects the full series (it derives the flags itself), so
        # pass the unfiltered scores and threshold rather than a filtered subset.
        if "testA" in scores and "p99_5" in thresholds:
            frame = scores["testA"]
            ts = pd.DatetimeIndex(frame["timestamp"])
            values = frame["score"].to_numpy()
            events = merge_events(ts, values, thresholds["p99_5"], row["length"])
            row["merged_events_testA_recomputed"] = len(events)
        rows.append(row)
    if not rows:
        print("no E1 runs")
        return 1
    df = pd.DataFrame(rows)
    df.to_csv(TABLES / "T_e1_runs.csv", index=False)
    agg = ["epochs", "final_train_loss", "final_val_loss", "merged_events_testA",
           "flagged_train_p99_5", "flagged_val_p99_5", "flagged_testA_p99_5", "flagged_testB_p99_5"]
    agg = [c for c in agg if c in df.columns]
    summary = df.groupby(["length", "batch"])[agg].agg(["mean", "std"])
    summary.columns = [f"{a}_{b}" for a, b in summary.columns]
    summary = summary.reset_index()
    summary.to_csv(TABLES / "T_e1_summary.csv", index=False)
    print(summary.to_string(index=False))
    mismatches = df[df["merged_events_testA"] != df["merged_events_testA_recomputed"]]
    print(f"\nmerged event count cross-check: {len(mismatches)} of {len(df)} runs disagree")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
