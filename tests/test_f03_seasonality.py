"""Test F03's day order, tick spacing and rendered geometry."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


def test_day_order_and_tick_spacing():
    from make_f03_seasonality import ORIGIN, day_labels, hour_ticks

    assert day_labels()[0] == "Mon"
    assert day_labels()[-1] == "Sun"
    # Monday must be at the TOP: origin="upper" puts row 0 at the top,
    # origin="lower" would put it at the bottom (the current bug)
    assert ORIGIN == "upper"
    ticks = hour_ticks()
    assert ticks[0] == "00:00"
    assert ticks[-1] == "20:00"
    assert len(ticks) == 6


def test_rendered_geometry_matches_the_spec(tmp_path):
    from make_f03_seasonality import build

    fig, ax = build()
    from figure_style import SINGLE_WIDTH

    assert abs(fig.get_figwidth() - SINGLE_WIDTH) < 1e-9
    assert abs(fig.get_figheight() - 2.3) < 1e-9
