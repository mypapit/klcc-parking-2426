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
for length in (6, 18, 36):
    half = len(train) // 2
    a = train.iloc[:half].to_numpy()
    b = train.iloc[half:].to_numpy()
    pa = matrix_profile_score(b, a, length)
    pb = matrix_profile_score(a, b, length)
    c = np.concatenate([pa, pb])
    c = c[np.isfinite(c)]
    print(f"L={length}: n={len(c)} p50={np.median(c):.4f} p95={np.percentile(c,95):.4f} p99.5={np.percentile(c,99.5):.4f} max={c.max():.4f}")
