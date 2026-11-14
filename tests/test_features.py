import numpy as np
import pandas as pd

from src.features import Preprocessor


def test_p2_weekly_profile_is_train_fit_and_aligned():
    idx = pd.date_range("2022-04-25", periods=14 * 144, freq="10min")
    values = pd.Series(np.arange(len(idx), dtype=float), index=idx)
    pre = Preprocessor("P2").fit(values)
    transformed = pre.transform(values.iloc[:288])
    assert np.isfinite(transformed.to_numpy()).all()
    assert pre.weekly_profile_ is not None
    assert len(pre.weekly_profile_) == 1008
