"""Score all injection copies (5 per fault per test set) for trained runs.

Rescoring only: no retraining. The trainer stores one copy per run (the seed);
this expands to the full protocol of `injection.copies` x `injection.injections_per_copy`.
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
from threshold import merge_events, percentile_threshold, pot_threshold  # noqa: E402
from utils import ensure_dir, load_config, write_json  # noqa: E402
from windows import make_windows  # noqa: E402

FAULTS = ("F1", "F2", "F3", "F4", "F5", "F6")


def score_run(run_dir: Path, cfg: dict) -> dict | None:
    import tensorflow as tf

    from models import build_autoencoder

    model_path = run_dir / "model.keras"
    scores_path = run_dir / "scores_train.parquet"
    if not model_path.exists() or not scores_path.exists():
        return None
    name = run_dir.name
    parts = name.split("_")
    prep = parts[1]
    length = int([p for p in parts if p.startswith("L")][0][1:])
    depth = parts[0]
    out_path = run_dir / "fault_metrics_allcopies.csv"
    events_path = run_dir / "event_recall.csv"
    if out_path.exists() and events_path.exists():
        return {"run": name, "status": "cached", "rows": len(pd.read_csv(out_path))}

    processed = pd.read_parquet(cfg["data"]["processed"]).set_index("timestamp")
    series = processed["available"].astype(float)
    bounds = split_bounds(series, cfg["splits"]["history_end"], cfg["splits"]["test_b_start"], cfg["splits"]["test_b_end"])
    pre = Preprocessor(prep)
    pre.window_length = length
    pre.fit(series.loc[bounds.train[0]:bounds.train[1]])
    train_scores = pd.read_parquet(scores_path)["score"].to_numpy()
    thresholds = {
        "p95": percentile_threshold(train_scores, 95),
        "p99_5": percentile_threshold(train_scores, 99.5),
        "pot_1e3": pot_threshold(train_scores, 1e-3),
    }
    model = tf.keras.models.load_model(model_path)
    rows = []
    event_rows = []
    for test_name in ("testA", "testB"):
        span = bounds.test_a if test_name == "testA" else bounds.test_b
        raw_eval = series.loc[span[0]:span[1]]
        for fault in FAULTS:
            for copy_id in range(cfg["injection"]["copies"]):
                injected_raw, positions = inject_copy(raw_eval, test_name, fault, copy_id, cfg)
                inj_dir = run_dir / "injections"
                ensure_dir(inj_dir)
                positions.to_csv(inj_dir / f"{test_name}_{fault}_copy{copy_id}.csv", index=False)
                injected = pre.transform(injected_raw)
                x_inj, ts_inj = make_windows(injected.to_numpy(), injected.index, length)
                pred = model.predict(x_inj, batch_size=1024, verbose=0)
                scores_inj = np.mean((x_inj - pred) ** 2, axis=(1, 2))
                y = event_labels(ts_inj, positions, length)
                row = {"testset": test_name, "fault": fault, "copy": copy_id, "positive_windows": int(y.sum())}
                row.update(ranking_metrics(y, scores_inj))
                for tname, tval in thresholds.items():
                    row.update({f"{tname}_{k}": v for k, v in point_metrics(y, scores_inj > tval).items()})
                rows.append(row)
                # event recall: was the injected segment flagged at least once?
                # this must use the injected scores, not the clean-series scores
                tau = thresholds["p99_5"]
                flagged = scores_inj > tau
                for _, segment in positions.iterrows():
                    start = pd.Timestamp(segment["start"])
                    end = pd.Timestamp(segment["end"])
                    mask = np.asarray((ts_inj >= start) & (ts_inj <= end))
                    detected = bool(flagged[mask].any()) if mask.any() else False
                    event_rows.append({
                        "testset": test_name,
                        "fault": fault,
                        "copy": copy_id,
                        "start": start,
                        "end": end,
                        "detected": detected,
                    })
    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    events = pd.DataFrame(event_rows)
    events.to_csv(run_dir / "event_recall.csv", index=False)
    return {
        "run": name,
        "status": "done",
        "rows": len(df),
        "event_recall": round(float(events["detected"].mean()), 4) if len(events) else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", default="E3")
    parser.add_argument("--prep", default=None)
    args = parser.parse_args()
    cfg = load_config()
    runs = sorted((ROOT / "outputs" / "runs" / args.exp).glob("D[123]_*"))
    if args.prep:
        runs = [r for r in runs if r.name.split("_")[1] == args.prep]
    for run_dir in runs:
        result = score_run(run_dir, cfg)
        if result:
            print(json.dumps(result), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
