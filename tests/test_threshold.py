import numpy as np
import pandas as pd

from src.threshold import merge_events, percentile_threshold


def test_percentile_threshold():
    scores = np.arange(100, dtype=float)
    assert percentile_threshold(scores, 50) == 49.5


def test_merge_events_groups_consecutive_flags():
    ts = pd.date_range("2023-01-01", periods=6, freq="10min")
    scores = np.array([0.1, 2.0, 2.5, 0.2, 3.0, 0.1])
    events = merge_events(pd.DatetimeIndex(ts), scores, threshold=1.0, max_gap_steps=1)
    assert len(events) == 2
    assert events.iloc[0]["flagged_windows"] == 2
