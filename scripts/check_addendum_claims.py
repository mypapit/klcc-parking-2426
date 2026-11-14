import _bootstrap  # noqa: F401
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import json  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

print("=== FIX 3: prevalence check (D3/P0/L18) ===")
run = ROOT / "outputs" / "runs" / "E3" / "D3_P0_L18_b128_s0"
fm = pd.read_csv(run / "fault_metrics_allcopies.csv")
for testset in ("testA", "testB"):
    sub = fm[fm.testset == testset]
    f1 = sub[sub.fault == "F1"]
    print(f"  {testset}: F1 positive_windows = {f1.positive_windows.mean():.0f}")
# split lengths
from data import split_bounds  # noqa: E402
from utils import load_config  # noqa: E402

cfg = load_config()
series = pd.read_parquet(cfg["data"]["processed"]).set_index("timestamp")["available"].astype(float)
bounds = split_bounds(series, cfg["splits"]["history_end"], cfg["splits"]["test_b_start"], cfg["splits"]["test_b_end"])
for name, span in (("testA", bounds.test_a), ("testB", bounds.test_b)):
    block = series.loc[span[0]:span[1]]
    print(f"  {name}: {len(block)} points, {len(block)/144:.1f} days")

print()
print("=== FIX 5: 0 and 5300 readings by split ===")
raw = pd.read_csv(ROOT.parent / "klcc_parking.csv")
raw = raw[(raw.sitename == "NWD") & (raw.siterename == "KLCC")].copy()
raw["lastupdate"] = pd.to_datetime(raw["lastupdate"])
raw = raw.sort_values("lastupdate").set_index("lastupdate")
post = raw.loc["2022-04-25":]
for name, span in (("train", bounds.train), ("val", bounds.val), ("testA", bounds.test_a), ("testB", bounds.test_b)):
    block = post.loc[span[0]:span[1], "available"].astype(float)
    n_zero = int((block == 0).sum())
    n_full = int((block == 5300).sum())
    runs = []
    for value, label in ((0, "zero"), (5300, "full")):
        mask = (block == value).to_numpy()
        idx = np.where(mask)[0]
        if len(idx) == 0:
            continue
        breaks = np.where(np.diff(idx) != 1)[0]
        run_starts = np.concatenate([[idx[0]], idx[breaks + 1]]) if len(breaks) else np.array([idx[0]])
        run_lens = np.diff(np.concatenate([run_starts, [idx[-1] + 1]]))
        runs.append(f"{label}: {len(run_starts)} runs, median len {np.median(run_lens):.0f}, max {run_lens.max()}")
    total = len(block)
    print(f"  {name}: 0={n_zero} ({100*n_zero/total:.2f}%), 5300={n_full} ({100*n_full/total:.2f}%), points={total}")
    for r in runs:
        print(f"      {r}")
