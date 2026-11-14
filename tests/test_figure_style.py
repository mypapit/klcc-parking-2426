"""Test the shared JTEC figure style and metadata plumbing."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def test_figure_dimensions_and_fonts():
    from figure_style import DOUBLE_WIDTH, SINGLE_WIDTH, new_figure

    fig, axes = new_figure(1, 1, 2.0, single=True)
    assert abs(fig.get_figwidth() - SINGLE_WIDTH) < 1e-9
    assert abs(fig.get_figheight() - 2.0) < 1e-9
    assert fig.get_figwidth() < DOUBLE_WIDTH
    plt.close(fig)


def test_double_column_width():
    from figure_style import DOUBLE_WIDTH, new_figure

    fig, axes = new_figure(1, 1, 2.2, single=False)
    assert abs(fig.get_figwidth() - DOUBLE_WIDTH) < 1e-9
    plt.close(fig)


def test_font_sizes_are_set_not_shrunk():
    from figure_style import apply_style, new_figure

    fig, ax = new_figure(1, 1, 2.0, single=True)
    assert matplotlib.rcParams["font.size"] == 8.0
    assert matplotlib.rcParams["axes.labelsize"] == 8.0
    assert matplotlib.rcParams["xtick.labelsize"] == 8.0
    assert matplotlib.rcParams["ytick.labelsize"] == 8.0
    assert matplotlib.rcParams["legend.fontsize"] == 8.0
    plt.close(fig)


def test_seed_lands_in_png_metadata(tmp_path):
    from figure_style import new_figure, save_figure
    from PIL import Image

    fig, ax = new_figure(1, 1, 2.0, single=True)
    ax.plot([0, 1], [0, 1])
    out = tmp_path / "test.png"
    save_figure(fig, out, seed=0, run="D3_P0_L18_b128_s0", length=18)
    info = Image.open(out).info
    assert info.get("Seed") == "0"
    assert info.get("Run") == "D3_P0_L18_b128_s0"
    assert info.get("Length") == "18"
    plt.close(fig)


def test_seed_lands_in_pdf_info_not_dropped(tmp_path):
    # PDF silently drops unknown keys, so the seed must go into the standard
    # PDF Info fields rather than into a custom key
    from figure_style import new_figure, save_figure

    fig, ax = new_figure(1, 1, 2.0, single=True)
    ax.plot([0, 1], [0, 1])
    out = tmp_path / "test.pdf"
    save_figure(fig, out, seed=0, run="D3_P0_L18_b128_s0", length=18)
    text = out.read_bytes().decode("latin-1")
    assert "seed 0" in text or "seed=0" in text
    plt.close(fig)
