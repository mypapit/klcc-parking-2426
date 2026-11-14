"""Test F09's log axis, threshold label and seed metadata."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


def test_log_axis_and_threshold_label():
    from make_f09_error_over_time import SEED, THRESHOLD_LABEL, USE_LOG_Y

    assert USE_LOG_Y is True
    assert THRESHOLD_LABEL == "p99.5 threshold"
    assert SEED == 0


def test_png_carries_seed_metadata():
    from PIL import Image

    png = ROOT / "outputs" / "figures" / "F09_error_over_time.png"
    assert png.exists(), "run make_f09_error_over_time.py first"
    meta = Image.open(png).info
    assert str(meta.get("Seed")) == "0"
    assert meta.get("Run") == "D3_P0_L18_b128_s0"
    assert str(meta.get("Length")) == "18"
