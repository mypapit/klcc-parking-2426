"""Thresholding and event merging."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import genpareto


def percentile_threshold(scores: np.ndarray, percentile: float) -> float:
    """Percentile of the score distribution, ignoring non-finite values.

    np.percentile returns NaN when the input contains NaN, which silently
    poisons every downstream comparison (the B4 matrix profile has NaN on
    degenerate windows). Only the finite scores define the distribution.
    """
    values = np.asarray(scores, dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return float("nan")
    return float(np.percentile(finite, percentile))


def pot_threshold(scores: np.ndarray, risk: float, init_percentile: float = 98.0) -> float:
    values = np.asarray(scores, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan")
    init = np.percentile(values, init_percentile)
    exceedances = values[values > init] - init
    if len(exceedances) < 10:
        return float(init)
    shape, loc, scale = genpareto.fit(exceedances, floc=0)
    return float(init + genpareto.ppf(1.0 - risk, shape, loc=loc, scale=scale))


def rolling_threshold(scores: pd.Series, days: int, percentile: float = 99.5) -> pd.Series:
    return scores.rolling(f"{days}D", min_periods=1).quantile(percentile / 100.0).shift(1)


def merge_events(timestamps: pd.DatetimeIndex, scores: np.ndarray, threshold: float, max_gap_steps: int) -> pd.DataFrame:
    flags = scores > threshold
    events = []
    start_i = None
    for i, flag in enumerate(flags):
        if flag and start_i is None:
            start_i = i
        elif flag and (timestamps[i] - timestamps[i - 1]).total_seconds() / 600 > max_gap_steps:
            events.append((start_i, i - 1))
            start_i = i
        elif not flag and start_i is not None:
            events.append((start_i, i - 1))
            start_i = None
    if start_i is not None:
        events.append((start_i, len(flags) - 1))
    rows = []
    for event_id, (start_i, end_i) in enumerate(events, start=1):
        rows.append({
            "event_id": event_id,
            "start": timestamps[start_i],
            "end": timestamps[end_i],
            "peak_score": float(scores[start_i:end_i + 1].max()),
            "flagged_windows": int(end_i - start_i + 1),
        })
    return pd.DataFrame(rows)
