import numpy as np
import pandas as pd

from src.windows import make_windows


def test_make_windows_uses_window_end_timestamps():
    idx = pd.date_range("2023-01-01", periods=10, freq="10min")
    values = np.arange(10, dtype=float)
    xs, ts = make_windows(values, idx, 4)
    assert xs.shape == (7, 4, 1)
    assert ts[0] == idx[3]
    assert ts[-1] == idx[-1]
