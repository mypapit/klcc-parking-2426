"""T10: the verified KLCC event archive, with the detections it explains."""
from __future__ import annotations

import sys
from pathlib import Path

import _bootstrap  # noqa: F401
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from audit import load_external  # noqa: E402
from utils import ensure_dir  # noqa: E402

TABLES = ROOT / "outputs" / "tables"


def main() -> int:
    ensure_dir(TABLES)
    external = load_external()
    events = external["events"]
    if events.empty:
        print("klcc_events.csv is empty")
        return 1
    events = events.copy()
    events["date_start"] = pd.to_datetime(events["date_start"])
    events["date_end"] = pd.to_datetime(events["date_end"])
    events["duration_days"] = (events["date_end"] - events["date_start"]).dt.days + 1
    events = events.sort_values("date_start").reset_index(drop=True)

    # Fix 8: only show events inside a scored period (Test A or Test B).
    # Events in train/val were never scored, so their detection count is zero by
    # construction and listing them would mislead.
    from data import split_bounds  # noqa: E402
    from utils import load_config  # noqa: E402

    cfg = load_config()
    series = pd.read_parquet(cfg["data"]["processed"]).set_index("timestamp")["available"].astype(float)
    bounds = split_bounds(series, cfg["splits"]["history_end"], cfg["splits"]["test_b_start"], cfg["splits"]["test_b_end"])
    in_scored = (
        ((events["date_start"] <= bounds.test_b[1]) & (events["date_end"] >= bounds.test_a[0]))
        & ~((events["date_end"] < bounds.test_a[0]) | (events["date_start"] > bounds.test_b[1]))
    )
    events = events[in_scored].reset_index(drop=True)
    if events.empty:
        print("no events inside the scored test periods")
        return 1

    # how many audited detections fall inside each event window
    audit_path = TABLES / "T9_anomaly_audit.csv"
    if audit_path.exists():
        audit = pd.read_csv(audit_path, parse_dates=["start", "end"])
        counts = []
        for _, row in events.iterrows():
            mask = (audit["start"] <= row["date_end"]) & (audit["end"] >= row["date_start"])
            counts.append(int(mask.sum()))
        events["detections_in_window"] = counts
    events["source_url"] = events["source_url"].astype(str).str.strip()
    events["has_source"] = events["source_url"] != ""
    keep = ["date_start", "date_end", "duration_days", "event_name", "venue", "source_url", "has_source"]
    if "detections_in_window" in events.columns:
        keep.append("detections_in_window")
    events[keep].to_csv(TABLES / "T10_klcc_events.csv", index=False)
    print(f"events: {len(events)}, with source: {int(events['has_source'].sum())}, "
          f"total detections inside event windows: {int(events['detections_in_window'].sum()) if 'detections_in_window' in events.columns else 0}")
    print(events[["date_start", "event_name", "duration_days"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
