"""Preprocessing variants P0/P1/P2."""
from __future__ import annotations

import numpy as np
import pandas as pd


class Preprocessor:
    def __init__(self, variant: str):
        self.variant = variant
        self.mean_: float | None = None
        self.std_: float | None = None
        self.weekly_profile_: pd.Series | None = None

    def fit(self, train_values: pd.Series) -> "Preprocessor":
        values = train_values.astype(float)
        self.mean_ = float(values.mean())
        std = float(values.std(ddof=0))
        self.std_ = std if std else 1.0
        z = (values - self.mean_) / self.std_
        if self.variant == "P2":
            idx = values.index
            keys = pd.MultiIndex.from_arrays([idx.day_of_week, idx.hour * 6 + idx.minute // 10])
            self.weekly_profile_ = z.groupby(keys).mean()
        return self

    def transform(self, values: pd.Series) -> pd.Series:
        z = (values.astype(float) - self.mean_) / self.std_
        if self.variant == "P0":
            return z
        if self.variant == "P1":
            # Causal trailing mean; the window length is set by the runner.
            length = getattr(self, "window_length", None)
            if length is None:
                raise ValueError("P1 requires window_length")
            trailing = z.rolling(length, min_periods=length).mean().shift(1).fillna(0.0)
            return z - trailing
        if self.variant == "P2":
            idx = values.index
            keys = pd.MultiIndex.from_arrays([idx.day_of_week, idx.hour * 6 + idx.minute // 10])
            return z - pd.Series(keys.map(self.weekly_profile_), index=values.index)
        raise ValueError(self.variant)

    def inverse_transform(self, values: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
        """Map standardised values back to bay counts (exact for P0/P2; P1 loses the trailing-mean term)."""
        target = values
        column = None
        if isinstance(values, pd.DataFrame):
            column = "available" if "available" in values.columns else values.columns[0]
            target = values[column]
        out = target.astype(float) * self.std_ + self.mean_
        if column is not None:
            return out.to_frame(name=column)
        return out
