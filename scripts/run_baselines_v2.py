"""Run classical baselines on the same windows, labels and injection protocol.

Addendum Fix 6: bring B1-B4 onto the same footing as the neural models.
- per-timestamp scores saved as parquet (schema: timestamp, score) for every split
- the 5-copy x 10-injection protocol on Test A and Test B, not one copy of Test A
- B3 (Isolation Forest) with five seeds; B1, B2, B4 with seed 0
- B4 stays CPU-only; its exact injected profile comes from scripts/b4_partial
- event_recall.csv and fault_metrics_allcopies.csv match the neural run schema
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from b4_partial import inject_local_profile  # noqa: E402
from baselines import isolation_forest_scores, iqr_residual_score, matrix_profile_score  # noqa: E402
from data import split_bounds  # noqa: E402
from features import Preprocessor  # noqa: E402
from inject import inject_copy  # noqa: E402
from metrics import event_labels, point_metrics, ranking_metrics  # noqa: E402
from threshold import merge_events, percentile_threshold  # noqa: E402
from utils import ensure_dir, load_config, write_json  # noqa: E402
from windows import make_windows  # noqa: E402

FAULTS = ("F1", "F2", "F3", "F4", "F5", "F6")


def build_ctx(cfg: dict, train: pd.Series, length: int) -> dict:
    pre_p2 = Preprocessor("P2").fit(train)
    residual = pre_p2.transform(train).to_numpy()
    med = float(np.nanmedian(residual))
    mad = float(np.nanmedian(np.abs(residual - med))) or 1.0

    def b2_residual(values: pd.Series) -> np.ndarray:
        base = train.tail(144).copy()
        history = pd.concat([base, values])
        line = history.rolling(144, min_periods=1).median().shift(1).iloc[len(base):]
        return np.nan_to_num((values - line).to_numpy(), nan=0.0)

    train_res = np.nan_to_num((train - train.rolling(144, min_periods=1).median().shift(1)).to_numpy(), nan=0.0)
    x_p2, _ = make_windows(residual, train.index, length)
    return {
        "cfg": cfg,
        "pre": pre_p2,
        "b1": (pre_p2, med, mad),
        "b2_residual": b2_residual,
        "train_res": train_res,
        "train_arr": train.to_numpy(),
        "x_train_p2": x_p2,
        "sigma_ref": float(np.nanstd(train.to_numpy())),
    }


def seasonal_scores(pre, med: float, mad: float, values: pd.Series, length: int) -> tuple[np.ndarray, pd.DatetimeIndex]:
    """B1: robust z-score of the seasonal residual, maximum over the window."""
    residual = pre.transform(values).to_numpy()
    score = np.abs(residual - med) / mad
    xs, ts = make_windows(score, values.index, length)
    return np.nanmax(xs, axis=(1, 2)), ts


def score_clean(baseline: str, values: pd.Series, length: int, seed: int, ctx: dict) -> tuple[np.ndarray, pd.DatetimeIndex]:
    if baseline == "B1":
        pre, med, mad = ctx["b1"]
        return seasonal_scores(pre, med, mad, values, length)
    if baseline == "B2":
        xs, ts = make_windows(iqr_residual_score(ctx["train_res"], ctx["b2_residual"](values)), values.index, length)
        return xs.max(axis=(1, 2)), ts
    if baseline == "B4":
        profile = matrix_profile_score(ctx["train_arr"], values.to_numpy(), length)
        ts = values.index[length - 1:][: len(profile)]
        return profile[: len(ts)], ts
    res = ctx["pre"].transform(values).to_numpy()
    x_eval, ts = make_windows(res, values.index, length)
    return isolation_forest_scores(ctx["x_train_p2"], x_eval, seed), ts


def score_injected(baseline: str, clean_score: np.ndarray, injected: pd.Series, positions: pd.DataFrame, length: int, seed: int, ctx: dict) -> tuple[np.ndarray, pd.DatetimeIndex]:
    if baseline == "B1":
        pre, med, mad = ctx["b1"]
        return seasonal_scores(pre, med, mad, injected, length)
    if baseline == "B2":
        xs, ts = make_windows(iqr_residual_score(ctx["train_res"], ctx["b2_residual"](injected)), injected.index, length)
        return xs.max(axis=(1, 2)), ts
    if baseline == "B4":
        return inject_local_profile(clean_score, ctx["train_arr"], injected.to_numpy(), positions, length), pd.DatetimeIndex(injected.index[length - 1:][: len(clean_score)])
    res = ctx["pre"].transform(injected).to_numpy()
    x_inj, ts = make_windows(res, injected.index, length)
    return isolation_forest_scores(ctx["x_train_p2"], x_inj, seed), ts


def run_one(baseline: str, length: int, seed: int) -> int:
    cfg = load_config()
    name = f"{baseline}_L{length}_s{seed}"
    out = ensure_dir(Path(cfg["outputs"]["runs"]) / "E4" / name)
    metrics_path = out / "metrics.json"
    if metrics_path.exists():
        print(json.dumps({"skipped": True, "path": str(metrics_path)}), flush=True)
        return 0
    df = pd.read_parquet(cfg["data"]["processed"]).set_index("timestamp")
    series = df["available"].astype(float)
    bounds = split_bounds(series, cfg["splits"]["history_end"], cfg["splits"]["test_b_start"], cfg["splits"]["test_b_end"])
    train = series.loc[bounds.train[0]:bounds.train[1]]
    ctx = build_ctx(cfg, train, length)

    clean: dict[str, tuple[np.ndarray, pd.DatetimeIndex]] = {}
    for split in ("testA", "testB"):
        values = series.loc[bounds.test_a[0]:bounds.test_a[1]] if split == "testA" else series.loc[bounds.test_b[0]:bounds.test_b[1]]
        score, ts = score_clean(baseline, values, length, seed, ctx)
        pd.DataFrame({"timestamp": ts, "score": score}).to_parquet(out / f"scores_{split}.parquet", index=False)
        clean[split] = (score, ts)

    train_score, _ = score_clean(baseline, train, length, seed, ctx)
    train_windows = train_score[np.isfinite(train_score)]
    if baseline == "B4":
        # the self-join profile is dominated by trivial self-matches (63 percent
        # of windows have distance 0 to themselves), so its p99.5 is not a usable
        # anomaly threshold. Score each train window against the rest of the
        # train series instead, i.e. an AB-join of the first half against the
        # second half and vice versa, which keeps the reference causal in form.
        half = len(train) // 2
        a = train.iloc[:half]
        b = train.iloc[half:]
        profile_a = matrix_profile_score(b.to_numpy(), a.to_numpy(), length)
        profile_b = matrix_profile_score(a.to_numpy(), b.to_numpy(), length)
        combined = np.concatenate([profile_a, profile_b])
        # stumpy returns exactly sqrt(length) as the z-normalised distance
        # between a constant window and any other window, which is a degenerate
        # value rather than an anomaly score. The feed produces such windows in
        # quantity (long stretches at full capacity), and at L=36 they alone
        # occupy the p99.5 percentile, so excluding them from the reference is
        # what makes the threshold usable.
        degenerate = np.sqrt(length)
        combined = combined[np.isfinite(combined) & ~np.isclose(combined, degenerate)]
        threshold = float(percentile_threshold(combined, 99.5))
    else:
        threshold = float(percentile_threshold(train_score, 99.5))
    del train_windows

    faults, events = [], []
    for split in ("testA", "testB"):
        values = series.loc[bounds.test_a[0]:bounds.test_a[1]] if split == "testA" else series.loc[bounds.test_b[0]:bounds.test_b[1]]
        clean_score, clean_ts = clean[split]
        inj_dir = ensure_dir(out / "injections")
        for fault in FAULTS:
            for copy_id in range(cfg["injection"]["copies"]):
                injected, positions = inject_copy(values, split, fault, copy_id, cfg, sigma_ref=ctx["sigma_ref"])
                positions.to_csv(inj_dir / f"{split}_{fault}_copy{copy_id}.csv", index=False)
                score, ts = score_injected(baseline, clean_score, injected, positions, length, seed, ctx)
                y = event_labels(ts, positions, length)
                finite = np.isfinite(score)
                if y.sum() == 0 or not finite.any():
                    row = {"testset": split, "fault": fault, "copy": copy_id, "positive_windows": int(y.sum())}
                    row["roc_auc"] = np.nan
                    row["pr_auc"] = np.nan
                else:
                    # stumpy returns inf/NaN on degenerate windows; the ranking
                    # metric must not see them
                    row = {"testset": split, "fault": fault, "copy": copy_id, "positive_windows": int(y[finite].sum())}
                    row.update(ranking_metrics(y[finite], score[finite]))
                row.update({f"p99_5_{k}": v for k, v in point_metrics(y, score > threshold).items()})
                faults.append(row)
                flagged = score > threshold
                for _, seg in positions.iterrows():
                    start = pd.Timestamp(seg["start"])
                    end = pd.Timestamp(seg["end"])
                    mask = np.asarray((ts >= start) & (ts <= end + pd.Timedelta(minutes=10 * (length - 1))))
                    detected = bool(flagged[mask].any())
                    events.append({
                        "testset": split, "fault": fault, "copy": copy_id,
                        "start": start, "end": end, "detected": detected,
                    })
    pd.DataFrame(faults).to_csv(out / "fault_metrics_allcopies.csv", index=False)
    events_df = pd.DataFrame(events)
    events_df.to_csv(out / "event_recall.csv", index=False)

    counts = {}
    for split in ("testA", "testB"):
        _, ts = clean[split]
        merged = merge_events(ts, clean[split][0], threshold, length)
        merged.to_csv(out / f"events_{split}.csv", index=False)
        counts[split] = int(len(merged))
    result = {
        "baseline": baseline,
        "length": length,
        "seed": seed,
        "threshold_p99_5": threshold,
        "events_testA": counts["testA"],
        "events_testB": counts["testB"],
        "event_recall": round(float(events_df["detected"].mean()), 4) if len(events_df) else None,
        "saves_scores": True,
        "protocol": "5copies",
    }
    write_json(metrics_path, result)
    print(json.dumps(result, indent=2), flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", choices=["B1", "B2", "B3", "B4"], required=True)
    parser.add_argument("--length", type=int, required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    raise SystemExit(run_one(args.baseline, args.length, args.seed))


if __name__ == "__main__":
    main()
