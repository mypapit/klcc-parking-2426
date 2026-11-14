"""Re-score the neural fault metrics after the Fix 1 injection repair.

The old F4 spikes were NaN (sigma computed as np.std over a series that contains
NaN), so every F4 copy was invisible. The repair in src/inject.py changes the
injection positions wherever the old path would have hit a NaN gap, so this
script re-injects every fault and re-scores every run: the archived manifests and
allcopy metrics are replaced, not patched. Inference only; no retraining.

Covers E2 and E3 runs with a saved model. Writes per run:
- fault_metrics_allcopies.csv (2 splits x 6 faults x 5 copies)
- event_recall.csv
- injections/{split}_{fault}_copy{n}.csv
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
from data import split_bounds  # noqa: E402
from features import Preprocessor  # noqa: E402
from inject import inject_copy  # noqa: E402
from metrics import event_labels, point_metrics, ranking_metrics  # noqa: E402
from threshold import percentile_threshold, pot_threshold  # noqa: E402
from utils import ensure_dir, load_config, write_json  # noqa: E402
from windows import make_windows  # noqa: E402

FAULTS = ("F1", "F2", "F3", "F4", "F5", "F6")


def rescore_run(run_dir: Path, cfg: dict, train_sigma: float) -> dict | None:
    import tensorflow as tf

    model_path = run_dir / "model.keras"
    if not model_path.exists():
        return None
    name = run_dir.name
    parts = name.split("_")
    depth, prep = parts[0], parts[1]
    length = int([p for p in parts if p.startswith("L")][0][1:])

    processed = pd.read_parquet(cfg["data"]["processed"]).set_index("timestamp")
    series = processed["available"].astype(float)
    bounds = split_bounds(series, cfg["splits"]["history_end"], cfg["splits"]["test_b_start"], cfg["splits"]["test_b_end"])
    pre = Preprocessor(prep)
    pre.window_length = length
    pre.fit(series.loc[bounds.train[0]:bounds.train[1]])

    scores_path = run_dir / "scores_train.parquet"
    if scores_path.exists():
        train_scores = pd.read_parquet(scores_path)["score"].to_numpy()
        thresholds = {
            "p95": percentile_threshold(train_scores, 95),
            "p99_5": percentile_threshold(train_scores, 99.5),
            "pot_1e3": pot_threshold(train_scores, 1e-3),
        }
    else:
        thresholds = {}

    model = tf.keras.models.load_model(model_path)
    rows = []
    event_rows = []
    inj_dir = ensure_dir(run_dir / "injections")
    for split, span in (("testA", bounds.test_a), ("testB", bounds.test_b)):
        raw_eval = series.loc[span[0]:span[1]]
        for fault in FAULTS:
            for copy_id in range(cfg["injection"]["copies"]):
                injected_raw, positions = inject_copy(raw_eval, split, fault, copy_id, cfg, sigma_ref=train_sigma)
                positions.to_csv(inj_dir / f"{split}_{fault}_copy{copy_id}.csv", index=False)
                injected = pre.transform(injected_raw)
                x, ts = make_windows(injected.to_numpy(), injected.index, length)
                pred = model.predict(x, batch_size=1024, verbose=0)
                scores = np.mean((x - pred) ** 2, axis=(1, 2))
                y = event_labels(ts, positions, length)
                row = {"testset": split, "fault": fault, "copy": copy_id, "positive_windows": int(y.sum())}
                if y.sum() == 0:
                    row["roc_auc"] = np.nan
                    row["pr_auc"] = np.nan
                else:
                    row.update(ranking_metrics(y, scores))
                for tname, tval in thresholds.items():
                    row.update({f"{tname}_{k}": v for k, v in point_metrics(y, scores > tval).items()})
                rows.append(row)
                if thresholds:
                    flagged = scores > thresholds["p99_5"]
                else:
                    flagged = np.zeros(len(scores), dtype=bool)
                for _, seg in positions.iterrows():
                    start = pd.Timestamp(seg["start"])
                    end = pd.Timestamp(seg["end"])
                    mask = np.asarray((ts >= start) & (ts <= end + pd.Timedelta(minutes=10 * (length - 1))))
                    event_rows.append({
                        "testset": split, "fault": fault, "copy": copy_id,
                        "start": start, "end": end,
                        "detected": bool(flagged[mask].any()) if mask.any() else False,
                    })
    faults_df = pd.DataFrame(rows)
    faults_df.to_csv(run_dir / "fault_metrics_allcopies.csv", index=False)
    events_df = pd.DataFrame(event_rows)
    events_df.to_csv(run_dir / "event_recall.csv", index=False)
    return {
        "run": name,
        "rows": len(faults_df),
        "f4_positives_testA": float(faults_df[(faults_df.testset == "testA") & (faults_df.fault == "F4")]["positive_windows"].mean()),
        "event_recall": round(float(events_df["detected"].mean()), 4) if len(events_df) else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", action="append", required=True)
    parser.add_argument("--limit", type=int, default=None, help="rescore only the first N runs (for smoke tests)")
    args = parser.parse_args()
    cfg = load_config()
    processed = pd.read_parquet(cfg["data"]["processed"]).set_index("timestamp")
    series = processed["available"].astype(float)
    bounds = split_bounds(series, cfg["splits"]["history_end"], cfg["splits"]["test_b_start"], cfg["splits"]["test_b_end"])
    train = series.loc[bounds.train[0]:bounds.train[1]]
    train_sigma = float(np.nanstd(train.to_numpy()))
    print(f"train sigma_ref = {train_sigma:.1f}")
    total = 0
    for exp in args.exp:
        runs = sorted((ROOT / "outputs" / "runs" / exp).glob("D[123]_*"))
        if args.limit:
            runs = runs[: args.limit]
        for run_dir in runs:
            result = rescore_run(run_dir, cfg, train_sigma)
            if result:
                print(json.dumps(result), flush=True)
                total += 1
    print(json.dumps({"rescored": total}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
