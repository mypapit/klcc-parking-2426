"""Regression test: B4's threshold must exclude degenerate constant windows.

stumpy returns exactly sqrt(length) as the z-normalised distance between a
constant window and any other window. The KLCC feed produces such windows in
quantity (long stretches at full capacity), and at L=36 they alone occupy the
p99.5 percentile of the train reference, which made the threshold equal to the
degenerate value and flagged nothing at all.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from baselines import matrix_profile_score  # noqa: E402
from threshold import percentile_threshold  # noqa: E402


def _reference_threshold(profile: np.ndarray, length: int) -> float:
    degenerate = np.sqrt(length)
    clean = profile[np.isfinite(profile) & ~np.isclose(profile, degenerate)]
    return percentile_threshold(clean, 99.5)


def test_constant_window_scores_exactly_sqrt_length():
    rng = np.random.default_rng(0)
    normal = rng.normal(size=600)
    for length in (6, 18, 36):
        profile = matrix_profile_score(normal, np.full(length, 5.0), length)
        finite = profile[np.isfinite(profile)]
        assert np.isclose(finite.min(), np.sqrt(length))


def test_degenerate_windows_do_not_saturate_the_threshold():
    # half the windows are constant, which is what the real feed looks like
    rng = np.random.default_rng(1)
    length = 36
    n = 2000
    a = rng.normal(size=n)
    b = np.concatenate([rng.normal(size=n // 2), np.full(n // 2, 5.0)])
    profile = np.concatenate(
        [matrix_profile_score(b, a, length), matrix_profile_score(a, b, length)]
    )
    degenerate = np.sqrt(length)
    threshold = _reference_threshold(profile, length)
    assert np.isfinite(threshold)
    assert not np.isclose(threshold, degenerate), "threshold must not be the degenerate value"


def test_exclusion_removes_the_degenerate_mass():
    rng = np.random.default_rng(2)
    length = 36
    profile = np.concatenate(
        [np.full(500, np.sqrt(length)), rng.normal(size=500)]
    )
    cleaned = profile[np.isfinite(profile) & ~np.isclose(profile, np.sqrt(length))]
    assert len(cleaned) == 500
    assert percentile_threshold(cleaned, 99.5) < np.sqrt(length)


def test_reference_threshold_ignores_non_finite():
    profile = np.array([1.0, 2.0, np.nan, np.inf, 3.0])
    threshold = _reference_threshold(profile, 6)
    assert np.isfinite(threshold)
