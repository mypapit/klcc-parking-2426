import _bootstrap  # noqa: F401
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from data import split_bounds  # noqa: E402
from inject import inject_copy  # noqa: E402
from utils import load_config  # noqa: E402

cfg = load_config()
series = pd.read_parquet(cfg["data"]["processed"]).set_index("timestamp")["available"].astype(float)
bounds = split_bounds(series, cfg["splits"]["history_end"], cfg["splits"]["test_b_start"], cfg["splits"]["test_b_end"])
test_a = series.loc[bounds.test_a[0]:bounds.test_a[1]]

sigma_train = float(np.std(test_a.to_numpy()))
print(f"sigma_train (std of the whole test series) = {sigma_train:.1f}")
print(f"5 * sigma_train = {5 * sigma_train:.1f}")
print(f"series value range: {test_a.min():.0f} to {test_a.max():.0f}")
print()
print("F4 injection writes: value + (-1 or +1) * 5 * sigma_train")
print("A value near 3911 becomes 3911 +/- 2850, which can exceed totalbay 5300 or go negative.")
print()
injected, positions = inject_copy(test_a, "testA", "F4", 0, cfg)
for i, seg in positions.head(5).iterrows():
    lo = pd.Timestamp(seg["start"])
    print(f"seg {i}: before={test_a.loc[lo]:8.1f}  after={injected.loc[lo]}")
print()
print("nan count in injected:", int(injected.isna().sum()), "of", len(injected))
print("nan count in original test_a:", int(test_a.isna().sum()))
