import _bootstrap  # noqa: F401
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from baselines import matrix_profile_score  # noqa: E402
from data import split_bounds  # noqa: E402
from utils import load_config  # noqa: E402

cfg = load_config()
series = pd.read_parquet(cfg["data"]["processed"]).set_index("timestamp")["available"].astype(float)
bounds = split_bounds(series, cfg["splits"]["history_end"], cfg["splits"]["test_b_start"], cfg["splits"]["test_b_end"])
train = series.loc[bounds.train[0]:bounds.train[1]]
length = 36
half = len(train) // 2
a = train.iloc[:half].to_numpy()
b = train.iloc[half:].to_numpy()
pa = matrix_profile_score(b, a, length)
pb = matrix_profile_score(a, b, length)
# indices where pa is exactly 6.0
idx = np.flatnonzero(np.isclose(pa, 6.0))
print("profile pa has", len(idx), "values at 6.0; first 5 windows:")
for i in idx[:5]:
    w = a[i:i+length]
    print("  window", i, "min", np.nanmin(w), "max", np.nanmax(w), "std", np.nanstd(w), "unique", len(np.unique(w[~np.isnan(w)])))
print()
# check how many distinct values the series has
u, c = np.unique(a[~np.isnan(a)], return_counts=True)
print("most common values in first half:", u[np.argsort(-c)[:5]], c[np.argsort(-c)[:5]])
