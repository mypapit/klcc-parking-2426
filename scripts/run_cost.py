"""E8: computational cost per model. Uses saved models, no retraining.

Covers the neural models and, with --include-baselines, the four classical
baselines, so the cost table compares every method that appears in T7 rather
than the neural runs alone.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd
import tensorflow as tf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from baselines import isolation_forest_scores, iqr_residual_score, matrix_profile_score  # noqa: E402
from data import split_bounds  # noqa: E402
from models import build_autoencoder  # noqa: E402
from utils import ensure_dir, load_config, write_json  # noqa: E402

TABLES = ROOT / "outputs" / "tables"


def median_latency(model, x, batch: int, runs: int = 100, warmup: int = 10) -> float:
    for _ in range(warmup):
        model.predict(x, batch_size=batch, verbose=0)
    times = []
    for _ in range(runs):
        start = time.perf_counter()
        model.predict(x, batch_size=batch, verbose=0)
        times.append(time.perf_counter() - start)
    return float(np.median(times))


def throughput(model, length: int, windows: int = 4096, batch: int = 1024) -> float:
    """Windows per second: the operationally meaningful cost measure."""
    x = np.zeros((windows, length, 1), dtype="float32")
    model.predict(x[:batch], batch_size=batch, verbose=0)
    start = time.perf_counter()
    model.predict(x, batch_size=batch, verbose=0)
    elapsed = time.perf_counter() - start
    return windows / elapsed


def baseline_cost(baseline: str, train: pd.Series, length: int, seed: int) -> dict:
    """Time one baseline scoring pass over the train-sized series."""
    n = len(train) - length + 1
    if n <= 0:
        return {}
    values = train.to_numpy()
    if baseline == "B4":
        start = time.perf_counter()
        matrix_profile_score(values, values, length)
        elapsed = time.perf_counter() - start
        return {"windows_per_s": round(n / elapsed, 1), "single_window_ms": round(elapsed / n * 1000, 6)}

    from features import Preprocessor  # noqa: E402

    pre = Preprocessor("P2").fit(train)
    if baseline == "B1":
        start = time.perf_counter()
        pre.transform(train)
        elapsed = time.perf_counter() - start
    elif baseline == "B2":
        start = time.perf_counter()
        iqr_residual_score(values, values)
        elapsed = time.perf_counter() - start
    else:  # B3
        from windows import make_windows  # noqa: E402

        residual = pre.transform(train).to_numpy()
        x_train, _ = make_windows(residual, train.index, length)
        start = time.perf_counter()
        isolation_forest_scores(x_train, x_train, seed)
        elapsed = time.perf_counter() - start
    return {"windows_per_s": round(n / elapsed, 1), "single_window_ms": round(elapsed / n * 1000, 6)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", default="E3")
    parser.add_argument("--include-baselines", action="store_true")
    args = parser.parse_args()
    cfg = load_config()
    ensure_dir(TABLES)
    rows = []
    for metrics_path in sorted((ROOT / "outputs" / "runs" / args.exp).glob("*/metrics.json")):
        payload = json.loads(metrics_path.read_text())
        name = metrics_path.parent.name
        parts = name.split("_")
        depth, prep = parts[0], parts[1]
        length = int([p for p in parts if p.startswith("L")][0][1:])
        seed = int([p for p in parts if p.startswith("s")][0][1:])
        model = build_autoencoder(depth, length)
        model.compile("adam", "mse")
        x_one = np.zeros((1, length, 1), dtype="float32")
        x_batch = np.zeros((1024, length, 1), dtype="float32")
        gpu_ms = median_latency(model, x_one, batch=1)
        gpu_batch_ms = median_latency(model, x_batch, batch=1024, runs=20)
        gpu_windows_per_s = throughput(model, length)
        with tf.device("/CPU:0"):
            cpu_model = build_autoencoder(depth, length)
            cpu_model.compile("adam", "mse")
            cpu_ms = median_latency(cpu_model, x_one, batch=1, runs=50)
            cpu_windows_per_s = throughput(cpu_model, length, windows=2048)
        rows.append({
            "run": name,
            "method": depth,
            "depth": depth,
            "prep": prep,
            "length": length,
            "seed": seed,
            "params": int(model.count_params()),
            "epochs": payload.get("epochs"),
            "final_train_loss": payload.get("final_train_loss"),
            "final_val_loss": payload.get("final_val_loss"),
            "inference_gpu_batch1_ms": round(gpu_ms * 1000, 4),
            "inference_gpu_batch1024_ms": round(gpu_batch_ms * 1000, 4),
            "inference_cpu_batch1_ms": round(cpu_ms * 1000, 4),
            "gpu_windows_per_s": round(gpu_windows_per_s, 1),
            "cpu_windows_per_s": round(cpu_windows_per_s, 1),
        })
        print(f"{name} params={model.count_params()} gpu_b1={gpu_ms*1000:.3f}ms gpu_b1024={gpu_batch_ms*1000:.2f}ms cpu_b1={cpu_ms*1000:.3f}ms gpu_win/s={gpu_windows_per_s:.0f} cpu_win/s={cpu_windows_per_s:.0f}")

    if args.include_baselines:
        series = pd.read_parquet(cfg["data"]["processed"]).set_index("timestamp")["available"].astype(float)
        bounds = split_bounds(series, cfg["splits"]["history_end"], cfg["splits"]["test_b_start"], cfg["splits"]["test_b_end"])
        train = series.loc[bounds.train[0]:bounds.train[1]]
        for baseline in ("B1", "B2", "B3", "B4"):
            for length in (6, 18, 36):
                cost = baseline_cost(baseline, train, length, 0)
                if not cost:
                    continue
                rows.append({
                    "run": f"{baseline}_L{length}_s0",
                    "method": baseline,
                    "depth": baseline,
                    "prep": "n/a",
                    "length": length,
                    "seed": 0,
                    "params": 0,
                    "epochs": None,
                    "final_train_loss": None,
                    "final_val_loss": None,
                    "inference_gpu_batch1_ms": None,
                    "inference_gpu_batch1024_ms": None,
                    "inference_cpu_batch1_ms": cost["single_window_ms"],
                    "gpu_windows_per_s": None,
                    "cpu_windows_per_s": cost["windows_per_s"],
                })
                print(f"{baseline} L={length} cpu_win/s={cost['windows_per_s']}")

    if not rows:
        print("no runs")
        return 1
    df = pd.DataFrame(rows)
    df.to_csv(TABLES / "T12_computational_cost.csv", index=False)
    summary = df.groupby(["method", "length"]).agg({
        "params": "first",
        "epochs": "mean",
        "inference_gpu_batch1_ms": "mean",
        "inference_gpu_batch1024_ms": "mean",
        "inference_cpu_batch1_ms": "mean",
        "gpu_windows_per_s": "mean",
        "cpu_windows_per_s": "mean",
    }).reset_index()
    summary.to_csv(TABLES / "T12_cost_summary.csv", index=False)
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
