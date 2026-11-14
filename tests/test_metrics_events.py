import numpy as np
import pandas as pd

from src.metrics import event_labels


def test_event_labels_include_one_bin_fault():
    ts = pd.date_range("2023-01-01", periods=50, freq="10min")
    segments = pd.DataFrame({"start": [ts[10]], "end": [ts[10]]})
    labels = event_labels(pd.DatetimeIndex(ts), segments, window_len=18)
    assert labels.sum() == 18
    assert labels[10] == 1
    assert labels[27] == 1
    assert labels[28] == 0
