"""Train and score one full experiment run."""
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
from mask import mask_full_capacity, mask_statistics  # noqa: E402
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
    parser.add_argument("--exp", required=True)
    parser.add_argument("--depth", required=True)
    parser.add_argument("--prep", required=True)
    parser.add_argument("--length", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument(
        "--mask-full-capacity",
        type=int,
        default=0,
        help="E9: mask runs of 5300 readings of at least this many bins in train and validation. 0 disables it.",
    )
    args = parser.parse_args()
    cfg = load_config()
    set_all_seeds(args.seed)
    tf.keras.utils.set_random_seed(args.seed)
    try:
        tf.config.experimental.enable_op_determinism()
        determinism = True
    except Exception as exc:  # cuDNN LSTM can refuse
        determinism = False
        print(f"WARNING: op determinism unavailable: {exc}", file=sys.stderr)
    name = f"{args.depth}_{args.prep}_L{args.length}_b{args.batch}_s{args.seed}"
    run_root = ensure_dir(Path(cfg["outputs"]["runs"]) / args.exp / name)
    metrics_path = run_root / "metrics.json"
    if metrics_path.exists():
        print(json.dumps({"skipped": True, "path": str(metrics_path)}))
        return 0
    write_json(run_root / "config.json", vars(args))
    df = pd.read_parquet(cfg["data"]["processed"]).set_index("timestamp")
    series = df["available"].astype(float)
    bounds = split_bounds(series, cfg["splits"]["history_end"], cfg["splits"]["test_b_start"], cfg["splits"]["test_b_end"])
    # E9: mask long full-capacity runs in train and validation only. Test A and
    # Test B stay untouched so the comparison against E2 is like for like.
    # The preprocessor is fitted on the masked train series, otherwise the
    # z-score would still carry the masked points.
    split_values = {
        "train": series.loc[bounds.train[0]:bounds.train[1]],
        "val": series.loc[bounds.val[0]:bounds.val[1]],
        "testA": series.loc[bounds.test_a[0]:bounds.test_a[1]],
        "testB": series.loc[bounds.test_b[0]:bounds.test_b[1]],
    }
    mask_stats: dict[str, dict] = {}
    if args.mask_full_capacity > 0:
        for split_name in ("train", "val"):
            original = split_values[split_name]
            masked = mask_full_capacity(original, min_run=args.mask_full_capacity)
            mask_stats[split_name] = mask_statistics(original, masked)
            split_values[split_name] = masked
    pre = Preprocessor(args.prep)
    pre.window_length = args.length
    pre.fit(split_values["train"])
    transformed = {name: pre.transform(values) for name, values in split_values.items()}
    windows = {name: make_windows(values.to_numpy(), values.index, args.length) for name, values in transformed.items()}
    if args.mask_full_capacity > 0:
        for split_name in ("train", "val"):
            mask_stats[split_name]["windows"] = int(len(windows[split_name][0]))
        mask_stats["mask_min_run"] = args.mask_full_capacity
    x_train, _ = windows["train"]
    x_val, ts_val = windows["val"]
    x_test_a, ts_test_a = windows["testA"]
    x_test_b, ts_test_b = windows["testB"]
    model = build_autoencoder(args.depth, args.length)
    if args.depth == "D3":
        assert model.count_params() == 584257
    opt = tf.keras.optimizers.Adam(learning_rate=cfg["models"]["learning_rate"], beta_1=cfg["models"]["adam_beta1"], beta_2=cfg["models"]["adam_beta2"], epsilon=cfg["models"]["adam_epsilon"])
    model.compile(optimizer=opt, loss="mse")
    cb = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=cfg["models"]["patience"], restore_best_weights=True)
    hist = model.fit(x_train, x_train, validation_data=(x_val, x_val), epochs=cfg["models"]["max_epochs"], batch_size=args.batch, callbacks=[cb], shuffle=True, verbose=0)
    model.save(run_root / "model.keras")
    pd.DataFrame(hist.history).to_csv(run_root / "history.csv", index=False)
    scores = {name: score_windows(model, x) for name, (x, _) in windows.items()}
    for split, (_, ts) in windows.items():
        pd.DataFrame({"timestamp": ts, "score": scores[split]}).to_parquet(run_root / f"scores_{split}.parquet", index=False)
    thresholds = {"p95": percentile_threshold(scores["train"], 95), "p99_5": percentile_threshold(scores["train"], 99.5), "pot_1e3": pot_threshold(scores["train"], 1e-3)}
    events = merge_events(ts_test_a, scores["testA"], thresholds["p99_5"], args.length)
    events.to_csv(run_root / "events_testA.csv", index=False)
    fault_rows = []
    inj_dir = ensure_dir(Path(cfg["outputs"]["injections"]))
    for test_name in ("testA", "testB"):
        raw_eval = series.loc[getattr(bounds, "test_a" if test_name == "testA" else "test_b")[0]:getattr(bounds, "test_a" if test_name == "testA" else "test_b")[1]]
        for fault in ("F1", "F2", "F3", "F4", "F5", "F6"):
            injected_raw, positions = inject_copy(raw_eval, test_name, fault, args.seed, cfg)
            positions.to_csv(inj_dir / f"{test_name}_{fault}_copy{args.seed}.csv", index=False)
            injected = pre.transform(injected_raw)
            x_inj, ts_inj = make_windows(injected.to_numpy(), injected.index, args.length)
            scores_inj = score_windows(model, x_inj)
            y = event_labels(ts_inj, positions, args.length)
            row = {"testset": test_name, "fault": fault, "positive_windows": int(y.sum())}
            row.update({"roc_auc": np.nan, "pr_auc": np.nan} if y.sum() == 0 else ranking_metrics(y, scores_inj))
            for tname, tval in thresholds.items():
                row.update({f"{tname}_{k}": v for k, v in point_metrics(y, scores_inj > tval).items()})
            fault_rows.append(row)
    pd.DataFrame(fault_rows).to_csv(run_root / "fault_metrics.csv", index=False)
    result = {"run": name, "params": int(model.count_params()), "epochs": len(hist.history["loss"]), "final_train_loss": float(hist.history["loss"][-1]), "final_val_loss": float(hist.history["val_loss"][-1]), "thresholds": thresholds, "events_testA": int(len(events)), "fault_rows": len(fault_rows), "deterministic_ops": determinism}
    if mask_stats:
        result["mask"] = mask_stats
    write_json(metrics_path, result)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
