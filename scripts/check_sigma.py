import _bootstrap  # noqa: F401
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from data import split_bounds  # noqa: E402
from utils import load_config  # noqa: E402

cfg = load_config()
series = pd.read_parquet(cfg["data"]["processed"]).set_index("timestamp")["available"].astype(float)
bounds = split_bounds(series, cfg["splits"]["history_end"], cfg["splits"]["test_b_start"], cfg["splits"]["test_b_end"])
train = series.loc[bounds.train[0]:bounds.train[1]]
ta = series.loc[bounds.test_a[0]:bounds.test_a[1]]
print("train sigma (nanstd):", float(np.nanstd(train.to_numpy())))
print("testA sigma (nanstd):", float(np.nanstd(ta.to_numpy())))
print("train NaNs:", int(train.isna().sum()), "of", len(train))
print("testA NaNs:", int(ta.isna().sum()), "of", len(ta))
print("5 sigma = ", float(np.nanstd(train.to_numpy())) * 5)
