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
test_a = series.loc[bounds.test_a[0]:bounds.test_a[1]]

for length in (36,):
    half = len(train) // 2
    a = train.iloc[:half].to_numpy()
    b = train.iloc[half:].to_numpy()
    combined = np.concatenate([matrix_profile_score(b, a, length), matrix_profile_score(a, b, length)])
    combined = combined[np.isfinite(combined)]
    print(f"train cross-half L={length}: p99.5={np.percentile(combined,99.5):.6f} max={combined.max():.6f}")
    print("  values at exactly 6.0:", int(np.isclose(combined, 6.0).sum()), "of", len(combined))
    test_profile = matrix_profile_score(train.to_numpy(), test_a.to_numpy(), length)
    tf = test_profile[np.isfinite(test_profile)]
    print(f"testA profile L={length}: p50={np.median(tf):.4f} p95={np.percentile(tf,95):.4f} max={tf.max():.6f}")
    print("  values at exactly 6.0:", int(np.isclose(tf, 6.0).sum()), "of", len(tf))
    print("  values above 6.0:", int((tf > 6.0 + 1e-9).sum()))
