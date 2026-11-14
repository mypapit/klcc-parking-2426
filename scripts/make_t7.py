"""T7: methods x fault types on Test A and Test B.

Covers every method: the neural depths D1/D2/D3 and the classical baselines
B1 to B4, as the handover's figure spec requires (rows = D1 to D3, B1 to B4).

The baselines' fault_metrics.csv predates the testset column, so the test set is
recovered from the file layout where it is missing.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from labels import fault_short  # noqa: E402
from utils import ensure_dir  # noqa: E402

TABLES = ROOT / "outputs" / "tables"
FIG = ROOT / "outputs" / "figures"


def collect_fault_metrics(exp: str, depths: tuple[str, ...] | None = None) -> pd.DataFrame:
    """Collect the fault metrics of one experiment, optionally for chosen depths.

    `depths` exists because E2 and E3 both carry D3/P0 run directories and the
    artifacts are byte-identical between them (same seeds, same deterministic
    training), so pooling both experiments double-counts D3. The dedupe policy
    is: D1 and D2 come from E2, D3 comes from E3, and each experiment only
    contributes the baselines it actually ran (E4).
    """
    rows = []
    for run_dir in sorted((ROOT / "outputs" / "runs" / exp).glob("*")):
        # prefer the full injection protocol (5 copies x 10 injections) when present
        full = run_dir / "fault_metrics_allcopies.csv"
        fault_path = full if full.exists() else run_dir / "fault_metrics.csv"
        if not fault_path.exists():
            continue
        run = run_dir.name
        parts = run.split("_")
        depth = parts[0]
        if depths is not None and depth not in depths:
            continue
        df = pd.read_csv(fault_path)
        df["run"] = run
        df["method"] = depth
        df["length"] = int([p for p in parts if p.startswith("L")][0][1:])
        df["seed"] = int([p for p in parts if p.startswith("s")][0][1:])
        df["prep"] = None if depth.startswith("B") else parts[1]
        df["experiment"] = exp
        if "testset" not in df.columns:
            # the baseline runner writes one row per fault for Test A only
            df["testset"] = "testA"
        df["protocol"] = "5copies" if full.exists() else "1copy"
        rows.append(df)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def main() -> int:
    ensure_dir(TABLES)
    ensure_dir(FIG)
    frames = []
    # dedupe policy: D1 and D2 from E2, D3 from E3, baselines from E4. E2 and
    # E3 both carry D3/P0 run directories with byte-identical artifacts, so
    # pooling both double-counts D3.
    frames.append(collect_fault_metrics("E2", depths=("D1", "D2")))
    frames.append(collect_fault_metrics("E3", depths=("D3",)))
    frames.append(collect_fault_metrics("E4", depths=None))
    if not frames or all(f.empty for f in frames):
        print("no fault metrics found")
        return 1
    allf = pd.concat(frames, ignore_index=True)
    allf.to_csv(TABLES / "T_fault_metrics_raw.csv", index=False)

    # neural: keep only the selected preprocessing (P0) and average over seeds
    neural = allf[allf["method"].str.startswith("D")].copy()
    neural = neural[neural["prep"] == "P0"]
    baselines = allf[allf["method"].str.startswith("B")].copy()
    selected = pd.concat([neural, baselines], ignore_index=True)
    # average per seed first, then over seeds. Pooling the copy-level rows
    # before aggregating inflates the std about 5x, because the 5 injection
    # copies of one seed are correlated with each other.
    per_seed = selected.groupby(
        ["method", "length", "testset", "fault", "seed"]
    ).agg(pr_auc=("pr_auc", "mean"), roc_auc=("roc_auc", "mean"), positive_windows=("positive_windows", "mean")).reset_index()
    agg = per_seed.groupby(["method", "length", "testset", "fault"]).agg(
        pr_auc_mean=("pr_auc", "mean"),
        pr_auc_std=("pr_auc", "std"),
        roc_auc_mean=("roc_auc", "mean"),
        roc_auc_std=("roc_auc", "std"),
        positive_windows=("positive_windows", "mean"),
        seeds=("seed", "nunique"),
    ).reset_index()
    agg.to_csv(TABLES / "T7_methods_by_fault.csv", index=False)
    print(agg.to_string(index=False))

    # heatmap of PR-AUC by method and fault at each length
    import matplotlib.pyplot as plt

    from plots import _save

    methods = ["D1", "D2", "D3", "B1", "B2", "B3", "B4"]
    for testset in ("testA", "testB"):
        sub = agg[agg["testset"] == testset]
        if sub.empty:
            continue
        faults = ["F1", "F2", "F3", "F4", "F5", "F6"]
        lengths = sorted(sub["length"].unique())
        fig, axes = plt.subplots(1, len(lengths), figsize=(7.16, 2.6 * len(lengths)), squeeze=False, layout="constrained")
        # one shared scale across panels, otherwise each panel rescales and the
        # gap between the neural methods and the baselines becomes unreadable
        vmax = float(np.nanmax(sub["pr_auc_mean"].to_numpy(dtype=float)))
        for ax, length in zip(axes[0], lengths):
            pivot = sub[sub["length"] == length].pivot_table(index="method", columns="fault", values="pr_auc_mean").reindex(index=methods, columns=faults)
            values = pivot.to_numpy(dtype=float)
            im = ax.imshow(values, aspect="auto", cmap="viridis", vmin=0.0, vmax=vmax)
            ax.set_xticks(range(len(faults)))
            ax.set_xticklabels([fault_short(f) for f in faults], rotation=30, ha="right", fontsize=6)
            ax.set_yticks(range(len(methods)))
            ax.set_yticklabels(methods)
            ax.set_title(f"L={length}")
            for i in range(len(methods)):
                for j in range(len(faults)):
                    value = values[i, j]
                    if not np.isfinite(value):
                        # F4 has no positive windows under the window-end label
                        # rule, so its metric is undefined rather than zero
                        ax.text(j, i, "n/a", ha="center", va="center", fontsize=5, color="white")
                        continue
                    shade = "white" if value < vmax * 0.55 else "black"
                    ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=5, color=shade)
        axes[0][0].set_ylabel("Method")
        fig.suptitle(f"PR-AUC by fault type ({testset}), shared colour scale", fontsize=9)
        # constrained layout keeps the colourbar in its own column, clear of the
        # rightmost panel's cells
        fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.025, pad=0.02)
        _save(fig, FIG / f"F13_method_fault_heatmap_{testset}.pdf")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
