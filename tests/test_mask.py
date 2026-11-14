"""Regression tests for the E9 full-capacity mask."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mask import full_capacity_runs, mask_full_capacity, mask_statistics  # noqa: E402


def _series(values: list[float]) -> pd.Series:
    return pd.Series(values, index=pd.date_range("2023-01-01", periods=len(values), freq="10min"))


def test_short_run_is_not_masked():
    # five bins is below the six-bin threshold
    s = _series([100.0, 5300, 5300, 5300, 5300, 5300, 100.0])
    masked = mask_full_capacity(s, min_run=6)
    assert masked.notna().all()


def test_long_run_is_masked():
    s = _series([100.0] + [5300.0] * 6 + [100.0])
    masked = mask_full_capacity(s, min_run=6)
    assert masked.iloc[1:7].isna().all()
    assert masked.iloc[0] == 100.0
    assert masked.iloc[7] == 100.0


def test_mask_does_not_change_non_full_capacity_values():
    s = _series([100.0, 250.0, 300.0, 12.0, 5.0])
    masked = mask_full_capacity(s, min_run=6)
    assert masked.equals(s)


def test_existing_nan_is_preserved():
    s = _series([np.nan, 100.0, 200.0])
    masked = mask_full_capacity(s, min_run=6)
    assert np.isnan(masked.iloc[0])
    assert masked.iloc[1] == 100.0


def test_run_detection():
    s = _series([5300.0, 5300.0, 100.0, 5300.0])
    runs = full_capacity_runs(s)
    assert runs == [(0, 1), (3, 3)]


def test_mask_statistics_counts_only_newly_masked():
    s = _series([np.nan, 100.0] + [5300.0] * 6)
    masked = mask_full_capacity(s, min_run=6)
    stats = mask_statistics(s, masked)
    assert stats["already_missing"] == 1
    assert stats["newly_masked"] == 6
    assert stats["points"] == 8
