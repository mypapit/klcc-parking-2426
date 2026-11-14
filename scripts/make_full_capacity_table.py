"""Fix 5: full-capacity readings (5300) per split, their run lengths, and the
alignment with the audit's sensor-suspect events. Report only, no retraining."""
from __future__ import annotations

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


def run_lengths(mask: np.ndarray) -> np.ndarray:
    idx = np.flatnonzero(mask)
    if len(idx) == 0:
        return np.array([], dtype=int)
    breaks = np.flatnonzero(np.diff(idx) != 1)
    starts = np.concatenate([[idx[0]], idx[breaks + 1]])
    return np.diff(np.concatenate([starts, [idx[-1] + 1]]))


def main() -> int:
    cfg = load_config()
    ensure_dir(TABLES)
    raw = pd.read_csv(cfg["data"]["raw"], parse_dates=["lastupdate"])
    raw = raw[(raw["siteid"] == cfg["data"]["siteid"])].sort_values("lastupdate").set_index("lastupdate")
    processed = pd.read_parquet(cfg["data"]["processed"]).set_index("timestamp")
    series = processed["available"].astype(float)
    bounds = split_bounds(series, cfg["splits"]["history_end"], cfg["splits"]["test_b_start"], cfg["splits"]["test_b_end"])
    splits = {
        "train": bounds.train,
        "validation": bounds.val,
        "testA": bounds.test_a,
        "testB": bounds.test_b,
    }
    rows = []
    monthly = []
    for name, (start, end) in splits.items():
        block = raw.loc[start:end, "available"].astype(float)
        total = len(block)
        mask = (block == 5300).to_numpy()
        n_full = int(mask.sum())
        lens = run_lengths(mask)
        rows.append({
            "split": name,
            "points": total,
            "full_readings": n_full,
            "full_pct": round(100.0 * n_full / total, 3) if total else np.nan,
            "full_runs": len(lens),
            "run_median_bins": float(np.median(lens)) if len(lens) else 0.0,
            "run_max_bins": int(lens.max()) if len(lens) else 0,
            "run_mean_bins": float(lens.mean()) if len(lens) else 0.0,
            "run_max_days": round(float(lens.max()) * 10 / (24 * 60), 2) if len(lens) else 0.0,
        })
        full_mask = block == 5300
        per_month = full_mask.groupby(full_mask.index.to_period("M")).sum()
        for month, value in per_month.items():
            if value > 0:
                monthly.append({"split": name, "month": str(month), "full_readings": int(value)})
    table = pd.DataFrame(rows)
    table.to_csv(TABLES / "T_full_capacity_readings.csv", index=False)
    pd.DataFrame(monthly).to_csv(TABLES / "T_full_capacity_monthly.csv", index=False)
    print(table.to_string(index=False))

    # T9 cross-check: do the audit's sensor-suspect events coincide with these runs?
    audit = pd.read_csv(TABLES / "T9_anomaly_audit.csv", parse_dates=["start", "end"])
    sensor = audit[audit["category"] == "sensor-suspect"]
    matched = 0
    for _, row in sensor.iterrows():
        start = pd.Timestamp(row["start"])
        end = pd.Timestamp(row["end"])
        block = raw.loc[start:end, "available"].astype(float)
        if len(block) and (block == 5300).any():
            matched += 1
    print(f"\nsensor-suspect events overlapping a 5300 run: {matched} of {len(sensor)}")
    pd.DataFrame([{"sensor_suspect_events": len(sensor), "overlap_5300": matched}]).to_csv(TABLES / "T_sensor_suspect_vs_full.csv", index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
