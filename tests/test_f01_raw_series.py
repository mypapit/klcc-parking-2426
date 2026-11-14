"""Test F01's exclusion bands and split boundaries."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import pandas as pd  # noqa: E402


def test_exclusion_bands_present():
    from make_f01_raw_series import exclusion_bands

    bands = exclusion_bands()
    assert bands[0]["start"] == "2021-01-01"
    assert bands[0]["end"] == "2021-12-31"
    assert bands[0]["label"] == "Excluded: COVID-19 restrictions"
    assert bands[1]["start"] == "2022-01-01"
    assert bands[1]["end"] == "2022-04-24"
    assert bands[1]["label"] == "Excluded: inconsistent sensor readings"


def test_series_covers_the_excluded_periods():
    from make_f01_raw_series import load_raw_series

    series = load_raw_series()
    # the current figure starts in April 2022 and shows neither excluded band
    assert series.index.min() < pd.Timestamp("2021-06-01")
    assert series.index.max() >= pd.Timestamp("2024-03-01")
    assert (series.index < pd.Timestamp("2022-04-25")).sum() > 10000


def test_split_boundaries_come_from_t2b():
    from make_f01_raw_series import split_bands

    bands = split_bands()
    assert [b["label"] for b in bands] == ["Train", "Validation", "Test A", "Test B"]
    assert bands[0]["start"] == "2022-04-25 00:00:00"
    assert bands[-1]["end"] == "2024-03-31 23:59:59"
    # the splits must be contiguous and cover the scored period
    for earlier, later in zip(bands, bands[1:]):
        assert earlier["end"] <= later["start"]
