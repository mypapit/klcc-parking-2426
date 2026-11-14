import numpy as np
import pandas as pd

from src.inject import inject_fault


def test_inject_fault_f2_writes_zero_block():
    idx = pd.date_range("2023-01-01", periods=100, freq="10min")
    series = pd.Series(np.full(len(idx), 100.0), index=idx)
    out = inject_fault(series, 10, "F2", sigma_ref=5.0, seed=0)
    assert out.iloc[10:21].eq(0).all()
    assert out.iloc[9] == 100.0
    assert out.iloc[21] == 100.0
