"""Test the T8c 5-seed threshold table and the F11 figure."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


def test_percentiles_are_categorical_not_log():
    from make_f11_thresholds import PERCENTILES

    assert PERCENTILES == [95, 97, 99, 99.5, 99.9]


def test_pot_is_left_out_of_the_figure():
    from make_f11_thresholds import USE_POT

    assert USE_POT is False


def test_t8c_covers_all_six_faults():
    # make_f11.py:28 excluded F4, but the addendum asks for all six faults
    table = pd.read_csv("/edata/opt/klcc-experiment/klcc_ad/outputs/tables/T8c_threshold_5seeds_byfault.csv")
    assert set(table["fault"].unique()) == {"F1", "F2", "F3", "F4", "F5", "F6"}


def test_t8c_is_5_seed():
    table = pd.read_csv("/edata/opt/klcc-experiment/klcc_ad/outputs/tables/T8c_threshold_5seeds.csv")
    assert table["seeds"].unique().tolist() == [5]


def test_t8c_false_alarm_definition_is_flagged_windows_per_week():
    # 26.04 at p99.5 = 687 flagged windows over 26.38 weeks, as in T8b
    table = pd.read_csv("/edata/opt/klcc-experiment/klcc_ad/outputs/tables/T8c_threshold_5seeds.csv")
    row = table[(table.threshold_family == "pct") & (table.parameter == 99.5)]
    assert abs(row.false_alarms_per_week_mean.iloc[0] - 26.04) < 1.0
