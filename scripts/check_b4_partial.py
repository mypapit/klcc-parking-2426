"""Check the B4 partial profile against a full stumpy AB-join on a small case."""
import _bootstrap  # noqa: F401
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from b4_partial import inject_local_profile  # noqa: E402
from baselines import matrix_profile_score  # noqa: E402
from inject import inject_copy  # noqa: E402
from utils import load_config  # noqa: E402

cfg = load_config()
length = 6
rng = np.random.default_rng(0)
n_train, n_eval = 400, 300
train = pd.Series(rng.normal(1000, 200, n_train).cumsum() * 0.01 + 1000)
base = pd.Series(rng.normal(1000, 200, n_eval).cumsum() * 0.01 + 1000)
index = pd.date_range("2023-01-01", periods=n_eval, freq="10min")
base.index = index

cfg_small = {"injection": {"injections_per_copy": 3, "min_gap_days": 0}, "data": {"totalbay": 5300}}
injected, positions = inject_copy(base, "testA", "F4", 0, cfg_small, sigma_ref=50.0)

clean_full = matrix_profile_score(train.to_numpy(), base.to_numpy(), length)
partial = inject_local_profile(clean_full, train.to_numpy(), injected.to_numpy(), positions, length)
full = matrix_profile_score(train.to_numpy(), injected.to_numpy(), length)

n = min(len(partial), len(full))
diff = np.abs(partial[:n] - full[:n])
finite = np.isfinite(diff)
print(f"windows compared: {n}")
print(f"max abs diff (finite): {diff[finite].max():.6f}")
print(f"mean abs diff (finite): {diff[finite].mean():.8f}")
print(f"nan pattern equal: {bool((np.isfinite(partial[:n]) == np.isfinite(full[:n])).all())}")
affected = []
for _, seg in positions.iterrows():
    lo = max(0, int(seg["start_i"]) - length + 1)
    hi = min(len(full) - length + 1, int(seg["end_i"]) + 1)
    affected.extend(range(lo, hi))
if affected:
    a = np.array(sorted(set(affected)))
    print(f"affected windows: {len(a)}, max diff among them: {diff[a].max():.6f}")
