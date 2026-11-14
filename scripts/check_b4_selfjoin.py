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
prof = matrix_profile_score(train.to_numpy(), train.to_numpy(), 6)
f = prof[np.isfinite(prof)]
print("train self-join profile (L=6):")
print("  min %.6f median %.6f p95 %.6f p99.5 %.8f" % (f.min(), np.median(f), np.percentile(f, 95), np.percentile(f, 99.5)))
print("  fraction exactly zero: %.4f" % (f == 0).mean())
print("  fraction > 1e-6: %.4f" % (f > 1e-6).mean())
