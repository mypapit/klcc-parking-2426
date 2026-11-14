"""Regression test: thresholds must ignore non-finite scores.

np.percentile returns NaN when the input contains NaN, which silently produced a
NaN threshold for B4 and would do the same for any method whose scores contain a
non-finite value.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from threshold import percentile_threshold, pot_threshold  # noqa: E402


def test_percentile_threshold_ignores_nan():
    scores = np.array([1.0, 2.0, 3.0, 4.0, np.nan, np.inf])
    value = percentile_threshold(scores, 99.5)
    assert np.isfinite(value)
    assert value == np.percentile(np.array([1.0, 2.0, 3.0, 4.0]), 99.5)


def test_percentile_threshold_matches_clean_input():
    clean = np.linspace(0.0, 1.0, 500)
    dirty = clean.copy()
    dirty[10] = np.nan
    dirty[20] = np.inf
    # the dirty array has fewer elements, so the percentile differs slightly
    assert abs(percentile_threshold(clean, 99.5) - percentile_threshold(dirty, 99.5)) < 1e-3


def test_percentile_threshold_all_non_finite_is_nan():
    value = percentile_threshold(np.array([np.nan, np.inf, -np.inf]), 99.5)
    assert np.isnan(value)


def test_pot_threshold_ignores_nan():
    scores = np.concatenate([np.linspace(0, 1, 400), np.array([np.nan, np.inf])])
    value = pot_threshold(scores, 1e-3)
    assert np.isfinite(value)
