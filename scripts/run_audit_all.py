"""Run the Section 8 audit and the day-level base-rate tests across a whole experiment.

Addendum Fix 4 makes the day-level test the main one: merged events cluster within
a day, so the event-level binomial and permutation tests over-count independence.
Addendum Fix 7 merges the weekend category into unexplained and adds a weekend arm.

Tests run on both Test A and Test B. 10,000 permutations, seeded.
"""
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
sys.path.insert(0, str(ROOT / "scripts"))
from audit import audit_events, external_days, load_external  # noqa: E402
from base_rate_day import N_PERM, SEED, anomaly_days, run_arms  # noqa: E402
from data import split_bounds  # noqa: E402
from run_audit import match_mask, permutation_test  # noqa: E402
from utils import ensure_dir, load_config, write_json  # noqa: E402

TABLES = ROOT / "outputs" / "tables"

CATEGORIES = (
    "sensor-suspect", "KLCC event", "public holiday", "festive period",
    "school holiday", "heavy rain", "unexplained",
)


def _split_days(bounds, cfg) -> dict:
    series = pd.read_parquet(cfg["data"]["processed"]).set_index("timestamp")["available"].astype(float)
    out = {}
    for name, span in (("testA", bounds.test_a), ("testB", bounds.test_b)):
        block = series.loc[span[0]:span[1]]
        out[name] = pd.DatetimeIndex(sorted(set(block.index.normalize())))
    return out


def _event_level(events, ts, special, tolerance, n_perm=2000):
    """Event-level test, kept as the secondary result (addendum Fix 4)."""
    n = len(events)
    p0 = float(match_mask(ts, special, tolerance).mean()) if special else 0.0
    k = int(match_mask(pd.DatetimeIndex(events["start"]), special, tolerance).sum())
    binom = binomtest(k, n, p0, alternative="greater") if n and p0 > 0 else None
    perm = permutation_test(events, ts, special, tolerance, n_perm=n_perm) if n else {"p_perm": np.nan}
    return {"p0": p0, "k": k, "n": n, "binomial_p": float(binom.pvalue) if binom else np.nan, "permutation_p": perm.get("p_perm", np.nan)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", default="E3")
    parser.add_argument("--prep", default="P0")
    args = parser.parse_args()
    cfg = load_config()
    ensure_dir(TABLES)

    external = load_external()
    days = external_days(external)
    special = set(days["holidays"]) | set(days["festive"]) | set(days["events"])
    print(f"special days: {len(special)} (holiday {len(days['holidays'])}, festive {len(days['festive'])}, event {len(days['events'])})")

    raw = pd.read_parquet(cfg["data"]["processed"]).set_index("timestamp")["available"].astype(float)
    train_raw = raw.loc[: cfg["splits"]["history_end"]]
    jump_threshold = float(np.percentile(train_raw.diff().abs().dropna(), 99.99))
    bounds = split_bounds(raw, cfg["splits"]["history_end"], cfg["splits"]["test_b_start"], cfg["splits"]["test_b_end"])
    split_days = _split_days(bounds, cfg)

    rows = []
    audit_frames = []
    null_rows = []
    for run_dir in sorted((ROOT / "outputs" / "runs" / args.exp).glob(f"D3_{args.prep}_*")):
        name = run_dir.name
        length = int(name.split("_L")[1].split("_")[0])
        for split in ("testA", "testB"):
            events_path = run_dir / f"events_{split}.csv"
            scores_path = run_dir / f"scores_{split}.parquet"
            if not scores_path.exists():
                continue
            scores = pd.read_parquet(scores_path)
            ts = pd.DatetimeIndex(scores["timestamp"])
            if events_path.exists():
                events = pd.read_csv(events_path, parse_dates=["start", "end"])
            else:
                # the neural runs only stored Test A events; rebuild the split's
                # merged events from the saved scores and the run's own threshold
                tau = json.loads((run_dir / "metrics.json").read_text()).get("thresholds", {}).get("p99_5")
                if tau is None:
                    continue
                from threshold import merge_events
                events = merge_events(ts, scores["score"].to_numpy(), tau, length)
                events.to_csv(run_dir / f"events_{split}.csv", index=False)
            audited = audit_events(events, raw, jump_threshold, external, days)
            n = len(audited)

            audited = audited.copy()
            if "event_id" in audited.columns:
                audited = audited.drop(columns=["event_id"])
            audited.insert(0, "event_id", range(len(audited)))
            if "L" in audited.columns:
                audited = audited.drop(columns=["L"])
            audited.insert(1, "L", length)
            audited.insert(2, "split", split)
            if "peak_score" not in audited.columns and "score" in audited.columns:
                audited["peak_score"] = audited["score"]
            audited["run"] = name
            audit_frames.append(audited)

            row = {"run": name, "length": length, "split": split, "events": n, "anomaly_days": len(anomaly_days(events))}
            for cause, dayset in (("special", special), ("holiday", set(days["holidays"])), ("event", set(days["events"]))):
                result = _event_level(audited, ts, dayset, 0, n_perm=2000)
                for key, value in result.items():
                    row[f"eventlevel_{cause}_{key}"] = value
            weekend_days = {d for d in split_days[split] if d.day_of_week >= 5}
            arms = {
                "special": special,
                "holiday": set(days["holidays"]),
                "event": set(days["events"]),
                "weekend": weekend_days,
            }
            results = run_arms(split_days[split], arms, events, n_perm=N_PERM, seed=SEED)
            for arm, result in results.items():
                for key in ("n_days", "n_anomaly_days", "k", "p0", "binomial_p", "permutation_p", "null_mean", "n_perm"):
                    row[f"day_{arm}_{key}"] = result[key]
            anom = anomaly_days(events)
            observed_fraction = float(anom.isin(list(days["events"])).sum() / len(anom)) if len(anom) else 0.0
            null_rows.append({"run": name, "length": length, "split": split, "observed_day_event_fraction": observed_fraction})
            for category in CATEGORIES:
                row[f"cat_{category}"] = int((audited["category"] == category).sum()) if n else 0
            row["unexplained_on_weekend"] = int(((audited["category"] == "unexplained") & audited["on_weekend"]).sum()) if n and "on_weekend" in audited.columns else 0
            rows.append(row)
            write_json(run_dir / f"audit_summary_{split}.json", {
                "run": name, "split": split, "events": n,
                "anomaly_days": len(anomaly_days(events)),
                "day_level_event_permutation_p": results["event_exact"]["permutation_p"],
                "day_level_event_binomial_p": results["event_exact"]["binomial_p"],
                "day_level_weekend_permutation_p": results["weekend_exact"]["permutation_p"],
                "audit_categories": audited["category"].value_counts().to_dict() if n else {},
                "unexplained_on_weekend": row["unexplained_on_weekend"],
            })

    if not rows:
        print("no runs audited")
        return 1
    df = pd.DataFrame(rows)
    # the suffix keeps a second experiment from overwriting the first
    suffix = "" if args.exp == "E3" else f"_{args.exp}"
    df.to_csv(TABLES / f"T11_base_rate_tests{suffix}.csv", index=False)
    cat_cols = [f"cat_{c}" for c in CATEGORIES]
    suffixes = ("p0", "k", "binomial_p", "permutation_p", "n_anomaly_days", "n_days", "null_mean")
    day_cols = [c for c in df.columns if c.startswith("day_") and c.endswith(suffixes)]
    agg_cols = ["events", "anomaly_days"] + day_cols + cat_cols + ["unexplained_on_weekend"]
    grouped = df.groupby(["length", "split"])[agg_cols].agg(["mean", "std"])
    grouped.columns = [f"{a}_{b}" for a, b in grouped.columns]
    grouped.reset_index().to_csv(TABLES / f"T11_base_rate_summary{suffix}.csv", index=False)

    if audit_frames:
        audit_df = pd.concat(audit_frames, ignore_index=True)
        keep = [c for c in ["event_id", "L", "split", "start", "end", "peak_score", "category", "matched_name", "source_url", "rule_detail", "on_weekend", "run"] if c in audit_df.columns]
        audit_df[keep].to_csv(TABLES / f"T9_anomaly_audit{suffix}.csv", index=False)

    pd.DataFrame(null_rows).to_csv(TABLES / f"T11_null_summary{suffix}.csv", index=False)
    print(df[["run", "length", "split", "events", "anomaly_days", "day_event_exact_p0", "day_event_exact_k", "day_event_exact_permutation_p"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
