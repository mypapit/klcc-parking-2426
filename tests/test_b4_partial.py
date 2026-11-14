"""The B4 partial profile must equal a full stumpy AB-join (addendum Fix 6).

The partial computation is only a legitimate shortcut if it is exact.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import _bootstrap  # noqa: E402,F401
from b4_partial import inject_local_profile  # noqa: E402
from baselines import matrix_profile_score  # noqa: E402
from inject import inject_copy  # noqa: E402


def _series(n: int, seed: int) -> pd.Series:
    rng = np.random.default_rng(seed)
    values = rng.normal(1000, 200, n).cumsum() * 0.01 + 1000
    index = pd.date_range("2023-01-01", periods=n, freq="10min")
    return pd.Series(values, index=index)


def test_partial_profile_matches_full_abjoin():
    length = 6
    train = _series(400, 0)
    base = _series(300, 1)
    cfg = {"injection": {"injections_per_copy": 3, "min_gap_days": 0}, "data": {"totalbay": 5300}}
    injected, positions = inject_copy(base, "testA", "F4", 0, cfg, sigma_ref=50.0)
    clean = matrix_profile_score(train.to_numpy(), base.to_numpy(), length)
    partial = inject_local_profile(clean, train.to_numpy(), injected.to_numpy(), positions, length)
    full = matrix_profile_score(train.to_numpy(), injected.to_numpy(), length)
    n = min(len(partial), len(full))
    assert n > 0
    diff = np.abs(partial[:n] - full[:n])
    finite = np.isfinite(diff)
    assert (np.isfinite(partial[:n]) == np.isfinite(full[:n])).all()
    assert diff[finite].max() < 1e-6


def test_partial_profile_matches_for_every_fault():
    length = 6
    train = _series(400, 2)
    base = _series(300, 3)
    cfg = {"injection": {"injections_per_copy": 3, "min_gap_days": 0}, "data": {"totalbay": 5300}}
    for fault in ("F1", "F2", "F3", "F4", "F5", "F6"):
        injected, positions = inject_copy(base, "testA", fault, 0, cfg, sigma_ref=50.0)
        clean = matrix_profile_score(train.to_numpy(), base.to_numpy(), length)
        partial = inject_local_profile(clean, train.to_numpy(), injected.to_numpy(), positions, length)
        full = matrix_profile_score(train.to_numpy(), injected.to_numpy(), length)
        n = min(len(partial), len(full))
        diff = np.abs(partial[:n] - full[:n])
        finite = np.isfinite(diff)
        assert (np.isfinite(partial[:n]) == np.isfinite(full[:n])).all(), fault
        assert diff[finite].max() < 1e-6, fault
