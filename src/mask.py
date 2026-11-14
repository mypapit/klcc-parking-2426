"""Masking of full-capacity runs for experiment E9.

The feed contains long stretches of 5300 (full-capacity) readings, up to 26.6
days. These are operational closures rather than sensor faults, but a model
trained on them learns that a long flat stretch at capacity is normal, which
is one plausible reason the stuck-sensor fault F1 sits near chance.

E9 masks every run of 5300 readings of at least a minimum length in the train
and validation splits only. The masked bins become NaN, and `make_windows`
already drops any window containing NaN, so no window touching a masked bin
reaches the model.

Test A and Test B are never masked: the evaluation data must stay exactly as
it is, or the comparison against E2 would not be like for like.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

FULL_CAPACITY = 5300


def full_capacity_runs(values: pd.Series, full_capacity: int = FULL_CAPACITY) -> list[tuple[int, int]]:
    """Return the (start, end) positions of contiguous full-capacity runs."""
    numeric = values.to_numpy(dtype=float)
    runs = []
    start = None
    for i, value in enumerate(numeric):
        if value == full_capacity:
            if start is None:
                start = i
        elif start is not None:
            runs.append((start, i - 1))
            start = None
    if start is not None:
        runs.append((start, len(numeric) - 1))
    return runs


def mask_full_capacity(values: pd.Series, min_run: int = 6, full_capacity: int = FULL_CAPACITY) -> pd.Series:
    """Replace runs of full-capacity readings of at least min_run bins with NaN.

    Only runs that were already a single constant value are masked; a single
    off reading inside a run does not start a new run, because the run
    detection works on the value changing.
    """
    out = values.astype(float).copy()
    numeric = values.to_numpy(dtype=float)
    start = None
    for i in range(len(numeric) + 1):
        at_end = i == len(numeric)
        if not at_end and numeric[i] == full_capacity:
            if start is None:
                start = i
            continue
        if start is not None:
            if i - start >= min_run:
                out.iloc[start:i] = np.nan
            start = None
    return out


def mask_statistics(original: pd.Series, masked: pd.Series) -> dict:
    """Points removed by the mask, and how many of them were already missing."""
    was_missing = original.isna()
    now_missing = masked.isna()
    newly_masked = int((now_missing & ~was_missing).sum())
    return {
        "points": int(len(original)),
        "already_missing": int(was_missing.sum()),
        "newly_masked": newly_masked,
        "masked_share": round(newly_masked / len(original), 6) if len(original) else 0.0,
    }
