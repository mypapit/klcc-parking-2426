"""Acceptance checks for the rendered JTEC figure artifacts (Task 9 Step 3)."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

FIG = ROOT / "outputs" / "figures"
from PIL import Image


def png_size(name):
    return Image.open(FIG / name).size


def test_f03_png_geometry_and_monday_at_top():
    # 3.5 x 2.3 in at 300 dpi = 1050 x 690
    assert png_size("F03_seasonality_heatmap.png") == (1050, 690)
    # Monday must be at the TOP: the build() origin constant pins this
    from make_f03_seasonality import ORIGIN
    assert ORIGIN == "upper"


def test_f10_png_geometry_and_seed():
    # 3.5 x 3.0 in at 300 dpi = 1050 x 900
    meta = Image.open(FIG / "F10_reconstructions.png").info
    assert png_size("F10_reconstructions.png") == (1050, 900)
    assert str(meta.get("Seed")) == "0"
    assert meta.get("Run") == "D3_P0_L18_b128_s0"
    assert str(meta.get("Length")) == "18"


def test_f12_png_geometry():
    # 3.5 x 2.0 in at 300 dpi = 1050 x 600
    assert png_size("F12_depth_ablation.png") == (1050, 600)


def test_f11_png_geometry_and_size():
    # the addendum's Fig. 6 is single column 3.5 x 2.4 in = 1050 x 720 px
    assert png_size("F11_threshold_sensitivity_5seeds.png") == (1050, 720)
    # and it uses categorical percentile positions, not a log axis
    from make_f11_thresholds import PERCENTILES, USE_POT
    assert PERCENTILES == [95, 97, 99, 99.5, 99.9]
    assert USE_POT is False


def test_f09_png_carries_seed():
    meta = Image.open(FIG / "F09_error_over_time.png").info
    assert str(meta.get("Seed")) == "0"
    assert meta.get("Run") == "D3_P0_L18_b128_s0"
    assert str(meta.get("Length")) == "18"


@pytest.mark.parametrize("name,width,height", [
    ("F01_raw_series.pdf", 7.16, 2.2),
    ("F03_seasonality_heatmap.pdf", 3.5, 2.3),
    ("F09_error_over_time.pdf", 7.16, 2.2),
    ("F10_reconstructions.pdf", 3.5, 3.0),
    ("F11_threshold_sensitivity_5seeds.pdf", 3.5, 2.4),
    ("F12_depth_ablation.pdf", 3.5, 2.0),
    ("F13_method_fault_heatmap_testA.pdf", 7.16, 2.6),
])
def test_pdf_mediabox_matches_declared_size(name, width, height):
    import subprocess

    out = subprocess.run(["pdfinfo", str(FIG / name)], capture_output=True, text=True)
    assert out.returncode == 0
    # "Page size:  257.76 x 158.4 pts" ; 1 in = 72 pt
    for line in out.stdout.splitlines():
        if line.startswith("Page size:"):
            parts = line.split()
            w_pts, h_pts = float(parts[2]), float(parts[4])
            assert abs(w_pts - width * 72) < 1.5, f"{name}: width {w_pts/72:.3f} in != {width}"
            assert abs(h_pts - height * 72) < 1.5, f"{name}: height {h_pts/72:.3f} in != {height}"
            return
    pytest.fail(f"{name}: no Page size line in pdfinfo")


def test_all_16_deliverables_exist():
    names = [
        "F01_raw_series.pdf", "F01_raw_series.png",
        "F03_seasonality_heatmap.pdf", "F03_seasonality_heatmap.png",
        "F09_error_over_time.pdf", "F09_error_over_time.png",
        "F10_reconstructions.pdf", "F10_reconstructions.png",
        "F11_threshold_sensitivity.pdf", "F11_threshold_sensitivity.png",
        "F12_depth_ablation.pdf", "F12_depth_ablation.png",
        "F13_method_fault_heatmap_testA.pdf", "F13_method_fault_heatmap_testA.png",
    ]
    for n in names:
        assert (FIG / n).exists(), n
    assert (ROOT / "outputs" / "tables" / "T8c_threshold_5seeds.csv").exists()
    assert (ROOT / "outputs" / "FIGURE_NOTES.md").exists()
