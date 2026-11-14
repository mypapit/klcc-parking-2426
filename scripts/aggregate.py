"""Aggregate experiment runs into seed-averaged tables and figures."""
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
from plots import _save  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

RUNS = ROOT / "outputs" / "runs"
TABLES = ROOT / "outputs" / "tables"
FIGURES = ROOT / "outputs" / "figures"


def parse_name(name: str) -> dict:
    parts = name.split("_")
    out = {"run": name}
    if len(parts) >= 2:
        out["depth"] = parts[0]
        out["prep"] = parts[1]
    for part in parts[2:]:
        if part.startswith("L") and part[1:].isdigit():
            out["length"] = int(part[1:])
        elif part.startswith("b") and part[1:].isdigit():
            out["batch"] = int(part[1:])
        elif part.startswith("s") and part[1:].isdigit():
            out["seed"] = int(part[1:])
    return out


def load_runs(exp: str) -> pd.DataFrame:
    rows = []
    for metrics_path in (RUNS / exp).glob("*/metrics.json"):
        payload = json.loads(metrics_path.read_text())
        payload.update(parse_name(metrics_path.parent.name))
        payload["experiment"] = exp
        payload["run_dir"] = str(metrics_path.parent)
        fault_path = metrics_path.parent / "fault_metrics.csv"
        if fault_path.exists():
            faults = pd.read_csv(fault_path)
            valid = faults[faults["positive_windows"] > 0]
            if not valid.empty:
                payload["mean_pr_auc"] = float(valid["pr_auc"].mean())
                payload["mean_roc_auc"] = float(valid["roc_auc"].mean())
            else:
                payload["mean_pr_auc"] = np.nan
                payload["mean_roc_auc"] = np.nan
            f2 = faults[faults["fault"] == "F2"]
            payload["f2_pr_auc"] = float(f2["pr_auc"].iloc[0]) if not f2.empty else np.nan
        rows.append(payload)
    return pd.DataFrame(rows)


def summarise(df: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    if df.empty:
        return df
    metrics = ["final_val_loss", "final_train_loss", "epochs", "events_testA", "mean_pr_auc", "mean_roc_auc", "f2_pr_auc"]
    metrics = [m for m in metrics if m in df.columns]
    grouped = df.groupby(by)[metrics].agg(["mean", "std"])
    grouped.columns = [f"{metric}_{stat}" for metric, stat in grouped.columns]
    return grouped.reset_index()


def figure_learning_curves(df: pd.DataFrame, path):
    frames = []
    for run_dir in df["run_dir"]:
        hist = Path(run_dir) / "history.csv"
        if hist.exists():
            h = pd.read_csv(hist)
            h["epoch"] = np.arange(1, len(h) + 1)
            h["run"] = Path(run_dir).name
            frames.append(h)
    if not frames:
        return False
    all_hist = pd.concat(frames)
    fig, ax = plt.subplots(figsize=(7.16, 3.0))
    for name, group in all_hist.groupby("run"):
        ax.plot(group["epoch"], group["val_loss"], lw=0.7, alpha=0.6)
    ax.set_yscale("log")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation loss (MSE)")
    _save(fig, path)
    return True


def figure_prep_comparison(series: pd.Series, preps: dict, path, n_points=1008):
    window = series.iloc[:n_points]
    fig, axes = plt.subplots(3, 1, figsize=(7.16, 5.4), sharex=True)
    for ax, (label, values) in zip(axes, preps.items()):
        ax.plot(values.index[:n_points], values.to_numpy()[:n_points], lw=0.7)
        ax.set_ylabel(label)
    axes[-1].set_xlabel("Time (Asia/Kuala_Lumpur)")
    _save(fig, path)
    return True


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", default="E3")
    args = parser.parse_args()
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    df = load_runs(args.exp)
    if df.empty:
        print("no runs found")
        return 1
    df.to_csv(TABLES / f"T_{args.exp.lower()}_runs.csv", index=False)
    by = [c for c in ("prep", "depth", "length", "batch") if c in df.columns]
    summary = summarise(df, by)
    summary.to_csv(TABLES / f"T_{args.exp.lower()}_summary.csv", index=False)
    if args.exp == "E3":
        figure_learning_curves(df, FIGURES / "F07_learning_curves.pdf")
    print(summary.to_string(index=False))
    print(f"\nwrote {TABLES / f'T_{args.exp.lower()}_summary.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
