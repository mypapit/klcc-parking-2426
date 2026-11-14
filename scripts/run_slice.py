"""Run one complete slice through scoring, thresholds, and injected faults."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd
import tensorflow as tf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from data import split_bounds  # noqa: E402
from features import Preprocessor  # noqa: E402
from inject import inject_copy  # noqa: E402
from metrics import event_labels, point_metrics, ranking_metrics  # noqa: E402
from models import build_autoencoder  # noqa: E402
from threshold import merge_events, percentile_threshold, pot_threshold  # noqa: E402
from utils import ensure_dir, load_config, set_all_seeds, write_json  # noqa: E402
from windows import make_windows  # noqa: E402


def score_windows(model, x):
    pred = model.predict(x, batch_size=1024, verbose=0)
    return np.mean((x - pred) ** 2, axis=(1, 2))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--depth", default="D3")
    parser.add_argument("--prep", default="P2")
    parser.add_argument("--length", type=int, default=18)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--faults", default="F1,F2,F3,F4,F5,F6")
    args = parser.parse_args()
    cfg = load_config()
    set_all_seeds(args.seed)
    tf.keras.utils.set_random_seed(args.seed)
    run_root = ensure_dir(Path(cfg["outputs"]["runs"]) / "SLICE_FULL" / f"{args.depth}_{args.prep}_L{args.length}_b{args.batch}_s{args.seed}")
    write_json(run_root / "config.json", {"depth": args.depth, "prep": args.prep, "length": args.length, "seed": args.seed, "batch": args.batch})
    df = pd.read_parquet(cfg["data"]["processed"]).set_index("timestamp")
    series = df["available"].astype(float)
    bounds = split_bounds(series, cfg["splits"]["history_end"], cfg["splits"]["test_b_start"], cfg["splits"]["test_b_end"])
    pre = Preprocessor(args.prep)
    pre.window_length = args.length
    pre.fit(series.loc[bounds.train[0]:bounds.train[1]])
    train = pre.transform(series.loc[bounds.train[0]:bounds.train[1]])
    val = pre.transform(series.loc[bounds.val[0]:bounds.val[1]])
    test_a = pre.transform(series.loc[bounds.test_a[0]:bounds.test_a[1]])
    test_b = pre.transform(series.loc[bounds.test_b[0]:bounds.test_b[1]])
    x_train, _ = make_windows(train.to_numpy(), train.index, args.length)
    x_val, ts_val = make_windows(val.to_numpy(), val.index, args.length)
    x_test_a, ts_test_a = make_windows(test_a.to_numpy(), test_a.index, args.length)
    x_test_b, ts_test_b = make_windows(test_b.to_numpy(), test_b.index, args.length)
    model = build_autoencoder(args.depth, args.length)
    if args.depth == "D3":
        assert model.count_params() == 584257
    opt = tf.keras.optimizers.Adam(learning_rate=cfg["models"]["learning_rate"], beta_1=cfg["models"]["adam_beta1"], beta_2=cfg["models"]["adam_beta2"], epsilon=cfg["models"]["adam_epsilon"])
    model.compile(optimizer=opt, loss="mse")
    cb = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=cfg["models"]["patience"], restore_best_weights=True)
    hist = model.fit(x_train, x_train, validation_data=(x_val, x_val), epochs=cfg["models"]["max_epochs"], batch_size=args.batch, callbacks=[cb], shuffle=True, verbose=0)
    scores_train = score_windows(model, x_train)
    scores_val = score_windows(model, x_val)
    scores_test_a = score_windows(model, x_test_a)
    scores_test_b = score_windows(model, x_test_b)
    pd.DataFrame({"timestamp": ts_test_a, "score": scores_test_a}).to_parquet(run_root / "scores_testA.parquet", index=False)
    pd.DataFrame({"timestamp": ts_test_b, "score": scores_test_b}).to_parquet(run_root / "scores_testB.parquet", index=False)
    thresholds = {
        "p95": percentile_threshold(scores_train, 95),
        "p99_5": percentile_threshold(scores_train, 99.5),
        "pot_1e3": pot_threshold(scores_train, 1e-3),
    }
    events = merge_events(ts_test_a, scores_test_a, thresholds["p99_5"], args.length)
    events.to_csv(run_root / "events_testA.csv", index=False)
    fault_rows = []
    inj_dir = ensure_dir(Path(cfg["outputs"]["injections"]))
    for test_name, clean_series, ts_eval in (("testA", test_a, ts_test_a), ("testB", test_b, ts_test_b)):
        raw_eval = series.loc[bounds.test_a[0]:bounds.test_a[1]] if test_name == "testA" else series.loc[bounds.test_b[0]:bounds.test_b[1]]
        for fault in [f.strip() for f in args.faults.split(",") if f.strip()]:
            injected_raw, positions = inject_copy(raw_eval, test_name, fault, args.seed, cfg)
            positions.to_csv(inj_dir / f"{test_name}_{fault}_copy{args.seed}.csv", index=False)
            injected = pre.transform(injected_raw)
            x_inj, ts_inj = make_windows(injected.to_numpy(), injected.index, args.length)
            scores_inj = score_windows(model, x_inj)
            y = event_labels(ts_inj, positions, args.length)
            row = {"testset": test_name, "fault": fault, "positive_windows": int(y.sum())}
            if y.sum() == 0:
                row.update({"roc_auc": np.nan, "pr_auc": np.nan})
            else:
                row.update(ranking_metrics(y, scores_inj))
            for tname, tval in thresholds.items():
                flags = scores_inj > tval
                pm = point_metrics(y, flags)
                row.update({f"{tname}_{k}": v for k, v in pm.items()})
            fault_rows.append(row)
    faults_df = pd.DataFrame(fault_rows)
    faults_df.to_csv(run_root / "fault_metrics.csv", index=False)
    result = {
        "run": run_root.name,
        "params": int(model.count_params()),
        "epochs": len(hist.history["loss"]),
        "final_train_loss": float(hist.history["loss"][-1]),
        "final_val_loss": float(hist.history["val_loss"][-1]),
        "thresholds": thresholds,
        "events_testA": int(len(events)),
        "fault_rows": int(len(faults_df)),
    }
    write_json(run_root / "metrics.json", result)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
