import numpy as np
import pandas as pd

from src.data import split_bounds


def test_split_bounds_are_chronological_and_non_overlapping():
    idx = pd.date_range("2022-04-25", "2023-12-31 23:50", freq="10min")
    series = pd.Series(np.arange(len(idx), dtype=float), index=idx)
    bounds = split_bounds(series, "2023-12-31 23:59:59", "2024-01-01 00:00:00", "2024-03-31 23:59:59")
    assert bounds.train[0] == idx[0]
    assert bounds.train[1] < bounds.val[0]
    assert bounds.val[1] < bounds.test_a[0]
    assert bounds.test_a[1] == idx[-1]
    assert bounds.test_b[0] == pd.Timestamp("2024-01-01 00:00:00")
