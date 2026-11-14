"""Window extraction with window-end scoring."""
from __future__ import annotations

import numpy as np
import pandas as pd


def make_windows(values: np.ndarray, timestamps: pd.DatetimeIndex, length: int) -> tuple[np.ndarray, pd.DatetimeIndex]:
    xs = []
    ts = []
    for end in range(length - 1, len(values)):
        window = values[end - length + 1 : end + 1]
        if np.isnan(window).any():
            continue
        xs.append(window)
        ts.append(timestamps[end])
    return np.asarray(xs, dtype=np.float32).reshape(-1, length, 1), pd.DatetimeIndex(ts)
