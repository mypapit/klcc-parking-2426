"""Check the B4 partial profile on F1: NaN pattern mismatch."""
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

rng = np.random.default_rng(3)
n_train, n_eval, length = 400, 300, 6
train = pd.Series(rng.normal(1000, 200, n_train).cumsum() * 0.01 + 1000)
base = pd.Series(rng.normal(1000, 200, n_eval).cumsum() * 0.01 + 1000, index=pd.date_range("2023-01-01", periods=n_eval, freq="10min"))

cfg = {"injection": {"injections_per_copy": 3, "min_gap_days": 0}, "data": {"totalbay": 5300}}
injected, positions = inject_copy(base, "testA", "F1", 0, cfg, sigma_ref=50.0)
print("positions:", positions[["start_i", "end_i"]].to_string(index=False))
print("injected has NaN:", injected.isna().any())

clean = matrix_profile_score(train.to_numpy(), base.to_numpy(), length)
partial = inject_local_profile(clean, train.to_numpy(), injected.to_numpy(), positions, length)
full = matrix_profile_score(train.to_numpy(), injected.to_numpy(), length)
n = min(len(partial), len(full))
print("lengths:", len(clean), len(partial), len(full))
p_finite = np.isfinite(partial[:n])
f_finite = np.isfinite(full[:n])
print("mismatch count:", int((p_finite != f_finite).sum()))
mis = np.flatnonzero(p_finite != f_finite)
print("mismatch idx:", mis[:10])
print("partial there:", partial[:n][mis[:10]])
print("full there:", full[:n][mis[:10]])
