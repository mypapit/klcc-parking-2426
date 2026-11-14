"""Run classical baselines on the same windows and fault labels."""
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
from baselines import isolation_forest_scores, iqr_residual_score, matrix_profile_score  # noqa: E402
from data import split_bounds  # noqa: E402
from features import Preprocessor  # noqa: E402
from inject import inject_copy  # noqa: E402
from metrics import event_labels, ranking_metrics  # noqa: E402
from threshold import merge_events, percentile_threshold  # noqa: E402
from utils import ensure_dir, load_config, write_json  # noqa: E402
from windows import make_windows  # noqa: E402


def fit_seasonal(train_values: pd.Series, length: int):
    pre = Preprocessor("P2").fit(train_values)
    residual = pre.transform(train_values).to_numpy()
    med = float(np.nanmedian(residual))
    mad = float(np.nanmedian(np.abs(residual - med))) or 1.0
    return pre, med, mad


def seasonal_scores(pre, med: float, mad: float, values: pd.Series, length: int):
    residual = pre.transform(values).to_numpy()
    score = np.abs(residual - med) / mad
    xs, ts = make_windows(score, values.index, length)
    return np.nanmax(xs, axis=(1, 2)), ts


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", choices=["B1", "B2", "B3", "B4"], required=True)
    parser.add_argument("--length", type=int, required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    cfg = load_config()
    name = f"{args.baseline}_L{args.length}_s{args.seed}"
    out = ensure_dir(Path(cfg["outputs"]["runs"]) / "E4" / name)
    metrics_path = out / "metrics.json"
    if metrics_path.exists():
        print(json.dumps({"skipped": True, "path": str(metrics_path)}))
        return 0
    df = pd.read_parquet(cfg["data"]["processed"]).set_index("timestamp")
    series = df["available"].astype(float)
    bounds = split_bounds(series, cfg["splits"]["history_end"], cfg["splits"]["test_b_start"], cfg["splits"]["test_b_end"])
    train = series.loc[bounds.train[0]:bounds.train[1]]
    test_a = series.loc[bounds.test_a[0]:bounds.test_a[1]]
    if args.baseline == "B1":
        pre, med, mad = fit_seasonal(train, args.length)
        train_score, train_ts = seasonal_scores(pre, med, mad, train, args.length)
        test_score, test_ts = seasonal_scores(pre, med, mad, test_a, args.length)
    elif args.baseline == "B2":
        # causal rolling-median residual computed on a train-anchored baseline.
        # The baseline is extended from the train tail into evaluation time, so an
        # injected fault cannot redefine its own reference level.
        def residual(values: pd.Series) -> np.ndarray:
            base = train.tail(144).copy()
            history = pd.concat([base, values])
            baseline = history.rolling(144, min_periods=1).median().shift(1).iloc[len(base):]
            return np.nan_to_num((values - baseline).to_numpy(), nan=0.0)
        train_res = (train - train.rolling(144, min_periods=1).median().shift(1)).to_numpy()
        train_res = np.nan_to_num(train_res, nan=0.0)
        test_res = residual(test_a)
        train_score, train_ts = make_windows(iqr_residual_score(train_res, train_res), train.index, args.length)
        test_score, test_ts = make_windows(iqr_residual_score(train_res, test_res), test_a.index, args.length)
        train_score = train_score.max(axis=(1, 2))
        test_score = test_score.max(axis=(1, 2))
    elif args.baseline == "B4":
        # AB-join of the eval series against the full train series (causal reference).
        # The train self-profile gives the threshold distribution.
        train_arr = train.to_numpy()
        test_arr = test_a.to_numpy()
        train_score = matrix_profile_score(train_arr, train_arr, args.length)
        train_ts = train.index[args.length - 1:]
        test_score = matrix_profile_score(train_arr, test_arr, args.length)
        test_ts = test_a.index[args.length - 1:]
        valid = np.isfinite(train_score)
        train_score, train_ts = train_score[valid], train_ts[valid]
        valid = np.isfinite(test_score)
        test_score, test_ts = test_score[valid], test_ts[valid]
    else:
        pre = Preprocessor("P2").fit(train)
        train_res = pre.transform(train).to_numpy()
        test_res = pre.transform(test_a).to_numpy()
        x_train, train_ts = make_windows(train_res, train.index, args.length)
        x_test, test_ts = make_windows(test_res, test_a.index, args.length)
        train_score = isolation_forest_scores(x_train, x_train, args.seed)
        test_score = isolation_forest_scores(x_train, x_test, args.seed)
    threshold = percentile_threshold(train_score, 99.5)
    events = merge_events(test_ts, test_score, threshold, args.length)
    fault_rows = []
    for fault in ("F1", "F2", "F3", "F4", "F5", "F6"):
        injected, positions = inject_copy(test_a, "testA", fault, args.seed, cfg)
        if args.baseline == "B1":
            score, ts = seasonal_scores(pre, med, mad, injected, args.length)
        elif args.baseline == "B2":
            baseline = injected.rolling(144, min_periods=1).median().shift(1)
            baseline.iloc[:144] = train.tail(144).median()
            inj_res = np.nan_to_num((injected - baseline).to_numpy(), nan=0.0)
            xs, ts = make_windows(iqr_residual_score(train_res, inj_res), injected.index, args.length)
            score = xs.max(axis=(1, 2))
        elif args.baseline == "B4":
            score = matrix_profile_score(train_arr, injected.to_numpy(), args.length)
            ts = injected.index[args.length - 1:]
            finite = np.isfinite(score)
            score, ts = score[finite], ts[finite]
        else:
            res = pre.transform(injected).to_numpy()
            x_inj, ts = make_windows(res, injected.index, args.length)
            score = isolation_forest_scores(x_train, x_inj, args.seed)
        y = event_labels(ts, positions, args.length)
        fault_rows.append({"fault": fault, "positive_windows": int(y.sum()), **({"roc_auc": np.nan, "pr_auc": np.nan} if y.sum() == 0 else ranking_metrics(y, score))})
    pd.DataFrame(fault_rows).to_csv(out / "fault_metrics.csv", index=False)
    result = {"baseline": args.baseline, "length": args.length, "seed": args.seed, "events_testA": len(events), "threshold": threshold}
    write_json(metrics_path, result)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
