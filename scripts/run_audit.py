"""Section 8: real-anomaly audit and base-rate significance tests."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from audit import audit_events  # noqa: E402
from events import load_public_holidays  # noqa: E402
from utils import ensure_dir, load_config, write_json  # noqa: E402

TABLES = ROOT / "outputs" / "tables"


def match_mask(timestamps: pd.DatetimeIndex, special_days: set, tolerance_days: int = 0) -> np.ndarray:
    days = timestamps.normalize()
    allowed = set()
    for delta in range(-tolerance_days, tolerance_days + 1):
        allowed |= {d + pd.Timedelta(days=delta) for d in special_days}
    return days.isin(list(allowed))


def permutation_test(events: pd.DataFrame, timestamps: pd.DatetimeIndex, special_days: set, tolerance_days: int = 0, seed: int = 0, n_perm: int = 2000) -> dict:
    """Place the same number of events at random positions and count how often they land on special days.

    Vectorised over permutations: one boolean array of special-day positions is rolled per draw.
    """
    n = len(events)
    if n == 0 or not special_days:
        return {"n": n, "p_perm": float("nan"), "null_mean": float("nan")}
    days = timestamps.normalize()
    flag = np.zeros(len(timestamps), dtype=bool)
    allowed = set()
    for delta in range(-tolerance_days, tolerance_days + 1):
        allowed |= {d + pd.Timedelta(days=delta) for d in special_days}
    flag[:] = days.isin(list(allowed))

    durations = ((pd.to_datetime(events["end"]) - pd.to_datetime(events["start"])).dt.total_seconds() / 600).to_numpy()
    durations = np.maximum(durations.astype(int) + 1, 1)
    observed = int(match_mask(pd.DatetimeIndex(events["start"]), special_days, tolerance_days).sum())

    rng = np.random.default_rng(seed)
    horizon = len(timestamps)
    hits = np.zeros(n_perm, dtype=int)
    for i in range(n_perm):
        starts = rng.integers(0, max(1, horizon - int(durations.max())), size=n)
        cnt = 0
        for start, dur in zip(starts, durations):
            if flag[start: start + dur].any():
                cnt += 1
        hits[i] = cnt
    return {
        "n": n,
        "observed_matched": observed,
        "null_mean": float(hits.mean()),
        "p_perm": float((hits >= observed).mean()),
        "n_perm": n_perm,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", default="E3")
    parser.add_argument("--run", default="D3_P0_L18_b128_s0")
    args = parser.parse_args()
    cfg = load_config()
    ensure_dir(TABLES)
    run_dir = ROOT / "outputs" / "runs" / args.exp / args.run
    events_path = run_dir / "events_testA.csv"
    if not events_path.exists():
        print(f"missing {events_path}")
        return 1
    events = pd.read_csv(events_path, parse_dates=["start", "end"])
    scores = pd.read_parquet(run_dir / "scores_testA.parquet")
    raw = pd.read_parquet(cfg["data"]["processed"]).set_index("timestamp")["available"].astype(float)
    train_raw = raw.loc[: cfg["splits"]["history_end"]]
    jump_threshold = float(np.percentile(train_raw.diff().abs().dropna(), 99.99))

    audited = audit_events(events, raw, jump_threshold)
    holidays_path = ROOT / "data" / "external" / "public_holidays.csv"
    if holidays_path.exists():
        hol = load_public_holidays(holidays_path)
        special = set(pd.to_datetime(hol["date"]).dt.normalize())
    else:
        special = set()
    audited["is_holiday"] = match_mask(pd.DatetimeIndex(audited["start"]), special, 0)
    audited.to_csv(TABLES / "T9_anomaly_audit.csv", index=False)

    n = len(audited)
    k = int(audited["is_holiday"].sum())
    ts = pd.DatetimeIndex(scores["timestamp"])
    p0 = float(match_mask(ts, special, 0).mean()) if special else 0.0
    binom = binomtest(k, n, p0, alternative="greater") if n and p0 > 0 else None
    perm = permutation_test(audited, ts, special, 0) if n else {"p_perm": np.nan}
    result = {
        "run": args.run,
        "events": n,
        "matched_to_holiday": k,
        "p0": p0,
        "binomial_p": float(binom.pvalue) if binom else None,
        "permutation_p": perm.get("p_perm"),
        "audit_categories": audited["category"].value_counts().to_dict() if "category" in audited else {},
    }
    write_json(run_dir / "audit_summary.json", result)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
