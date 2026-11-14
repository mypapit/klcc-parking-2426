"""E5: threshold sensitivity from saved scores, no retraining."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from threshold import merge_events, percentile_threshold, pot_threshold, rolling_threshold  # noqa: E402
from utils import ensure_dir, load_config, write_json  # noqa: E402

TABLES = ROOT / "outputs" / "tables"


def main() -> int:
    cfg = load_config()
    ensure_dir(TABLES)
    rows = []
    for run_dir in sorted((ROOT / "outputs" / "runs" / "E3").glob("D3_P0_*")):
        scores = {}
        for split in ("train", "val", "testA", "testB"):
            path = run_dir / f"scores_{split}.parquet"
            if path.exists():
                scores[split] = pd.read_parquet(path).set_index("timestamp")["score"]
        if "train" not in scores or "testA" not in scores:
            continue
        name = run_dir.name
        length = int([p for p in name.split("_") if p.startswith("L")][0][1:])
        seed = int([p for p in name.split("_") if p.startswith("s")][0][1:])
        train_scores = scores["train"].to_numpy()
        val_scores = scores["val"].to_numpy() if "val" in scores else None
        test_scores = scores["testA"]
        thresholds = {f"p{p}": percentile_threshold(train_scores, p) for p in (95, 97, 99, 99.5, 99.9)}
        if val_scores is not None:
            thresholds["val_p99_5"] = percentile_threshold(val_scores, 99.5)
        thresholds["pot_1e3"] = pot_threshold(train_scores, 1e-3)
        thresholds["pot_1e4"] = pot_threshold(train_scores, 1e-4)
        roll = rolling_threshold(test_scores, cfg["thresholds"]["rolling_days"], 99.5).fillna(float("inf"))
        for method, tau in thresholds.items():
            events = merge_events(test_scores.index, test_scores.to_numpy(), tau, length)
            rows.append({
                "run": name,
                "length": length,
                "seed": seed,
                "method": method,
                "threshold": tau,
                "flagged_windows": int((test_scores.to_numpy() > tau).sum()),
                "merged_events": len(events),
            })
        roll = rolling_threshold(test_scores, cfg["thresholds"]["rolling_days"], 99.5)
        roll_values = roll.to_numpy(dtype=float)
        rolling_flags = (test_scores.to_numpy() > roll_values) & np.isfinite(roll_values)
        # merge consecutive flagged timestamps; the gap rule uses the window length
        flag_ts = test_scores.index[rolling_flags]
        flag_scores = test_scores.to_numpy()[rolling_flags]
        if len(flag_ts):
            roll_events = merge_events(flag_ts, flag_scores, -np.inf, length)
        else:
            roll_events = pd.DataFrame()
        rows.append({
            "run": name,
            "length": length,
            "seed": seed,
            "method": "rolling_7d_p99_5",
            "threshold": np.nan,
            "flagged_windows": int(rolling_flags.sum()),
            "merged_events": len(roll_events),
        })
    if not rows:
        print("no scores found")
        return 1
    df = pd.DataFrame(rows)
    df.to_csv(TABLES / "T8_threshold_sensitivity.csv", index=False)
    summary = df.groupby(["length", "method"])[["flagged_windows", "merged_events"]].agg(["mean", "std"])
    summary.columns = [f"{a}_{b}" for a, b in summary.columns]
    summary = summary.reset_index()
    summary.to_csv(TABLES / "T8_threshold_summary.csv", index=False)
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
