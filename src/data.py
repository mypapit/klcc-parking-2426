"""Load, clean, resample, and split the KLCC parking series."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class SplitBounds:
    train: tuple[pd.Timestamp, pd.Timestamp]
    val: tuple[pd.Timestamp, pd.Timestamp]
    test_a: tuple[pd.Timestamp, pd.Timestamp]
    test_b: tuple[pd.Timestamp, pd.Timestamp]


def load_raw(path: str | Path, siteid: int) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["lastupdate"] = pd.to_datetime(df["lastupdate"])
    df = df[df["siteid"] == siteid].sort_values(["lastupdate", "rowid"]).reset_index(drop=True)
    return df


def split_bounds(series: pd.Series, history_end: str, test_b_start: str, test_b_end: str) -> SplitBounds:
    hist = series.loc[:pd.Timestamp(history_end)]
    end_hist = hist.index.max()
    split_ts = hist.index[int(len(hist) * 0.70) - 1]
    train_core_end = hist.index[int(len(hist.loc[:split_ts]) * 0.75) - 1]
    return SplitBounds(
        train=(hist.index.min(), train_core_end),
        val=(hist.index[hist.index.get_loc(train_core_end) + 1], split_ts),
        test_a=(hist.index[hist.index.get_loc(split_ts) + 1], end_hist),
        test_b=(pd.Timestamp(test_b_start), pd.Timestamp(test_b_end)),
    )
