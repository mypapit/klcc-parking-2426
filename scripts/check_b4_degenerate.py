import _bootstrap  # noqa: F401
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from data import split_bounds  # noqa: E402
from utils import load_config  # noqa: E402

cfg = load_config()
series = pd.read_parquet(cfg["data"]["processed"]).set_index("timestamp")["available"].astype(float)
bounds = split_bounds(series, cfg["splits"]["history_end"], cfg["splits"]["test_b_start"], cfg["splits"]["test_b_end"])
train = series.loc[bounds.train[0]:bounds.train[1]]
a = train.iloc[: len(train) // 2].to_numpy()
b = train.iloc[len(train) // 2 :].to_numpy()

for name, arr in (("a", a), ("b", b)):
    length = 36
    n = len(arr) - length + 1
    const = np.array([np.nanstd(arr[i : i + length]) == 0 for i in range(n)])
    print(f"{name}: {n} windows, {const.sum()} constant ({const.mean():.4f})")
    vals = arr[~np.isnan(arr)]
    print(f"   series: {len(vals)} finite, {int((vals == 5300).sum())} at 5300")

# what does a constant window score against a normal one?
import stumpy  # noqa: E402

normal = b[:200]
const_window = np.full(36, 5300.0)
prof = stumpy.stump(T_A=normal, m=36, T_B=const_window[None, :], ignore_trivial=False)
print("constant-vs-normal profile values (unique):", np.unique(prof[:, 0])[:5])
prof2 = stumpy.stump(T_A=const_window[None, :], m=36, T_B=normal, ignore_trivial=False)
print("normal-vs-constant profile values (unique):", np.unique(prof2[:, 0])[:5])
