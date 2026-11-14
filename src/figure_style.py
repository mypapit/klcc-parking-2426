"""Shared plotting style for the JTEC manuscript figures.

JTEC uses two columns, so most figures are single-column width (3.5 in) and
must stay readable there. This module pins the widths and the font sizes in
one place, and provides the metadata plumbing the addendum requires: a
single-seed figure records its seed in the file metadata.

Two things are deliberate here:

- The figure size is set BEFORE the fonts, because matplotlib shrinks text to
  fit a figure that is too small, which silently breaks a point-size
  requirement.
- PNG and PDF take metadata differently. PNG accepts arbitrary text keys.
  PDF silently drops unknown keys with a `UserWarning: Unknown infodict
  keyword`, so the seed goes into the standard Info fields Title, Subject and
  Keywords.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt

SINGLE_WIDTH = 3.5
DOUBLE_WIDTH = 7.16

# a colour-blind-safe cycle (Okabe-Ito), so the figures survive greyscale
# printing and the common forms of colour vision deficiency
COLOURBLIND_SAFE = [
    "#0072B2",  # blue
    "#D55E00",  # vermillion
    "#009E73",  # bluish green
    "#CC79A7",  # reddish purple
    "#E69F00",  # orange
    "#56B4E9",  # sky blue
    "#F0E442",  # yellow
    "#000000",  # black
]

FONT_SIZE = 8.0
ANNOTATION_SIZE = 7.0


def apply_style() -> None:
    """Set the font sizes and the palette.

    Called after the figure size is set, so matplotlib has nothing to
    shrink.
    """
    matplotlib.rcParams.update({
        "font.size": FONT_SIZE,
        "axes.labelsize": FONT_SIZE,
        "axes.titlesize": FONT_SIZE,
        "xtick.labelsize": FONT_SIZE,
        "ytick.labelsize": FONT_SIZE,
        "legend.fontsize": FONT_SIZE,
        "figure.titlesize": FONT_SIZE,
        "axes.prop_cycle": matplotlib.cycler(color=COLOURBLIND_SAFE),
        "savefig.dpi": 300,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def new_figure(nrows: int = 1, ncols: int = 1, height: float = 2.4, single: bool = True):
    """Create a figure at the JTEC column width, with the fonts applied after.

    `height` is in inches. `single` selects the 3.5 in or 7.16 in width.
    """
    width = SINGLE_WIDTH if single else DOUBLE_WIDTH
    fig, axes = plt.subplots(nrows, ncols, figsize=(width, height))
    apply_style()
    return fig, axes


def png_metadata(suffix: str, seed, run: str | None, length: int | None) -> dict:
    """The metadata dict the PNG accepts as arbitrary text keys."""
    metadata = {"Software": f"klcc_ad, {suffix}" if suffix else "klcc_ad"}
    if seed is not None:
        metadata["Seed"] = str(seed)
    if run:
        metadata["Run"] = str(run)
    if length is not None:
        metadata["Length"] = str(length)
    return metadata


def save_figure(fig, path, seed=None, run: str | None = None, length: int | None = None) -> None:
    """Save the figure as PDF and as a 300 dpi PNG, with the seed in the metadata.

    The PDF receives the seed in its standard Info fields because matplotlib
    drops unknown PDF keys; the PNG receives it as an ordinary text key.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    name = path.stem
    details = []
    if seed is not None:
        details.append(f"seed {seed}")
    if run:
        details.append(f"run {run}")
    if length is not None:
        details.append(f"length {length}")
    suffix = ", ".join(details)

    if path.suffix.lower() == ".pdf":
        metadata = {}
        if suffix:
            metadata["Title"] = f"{name}, {suffix}"
            metadata["Subject"] = suffix
            metadata["Keywords"] = suffix.replace(", ", " ").replace(" ", "=")
        fig.savefig(path, metadata=metadata)
        # the PNG is half the deliverable, so write it in the same call
        fig.savefig(path.with_suffix(".png"), dpi=300, metadata=png_metadata(suffix, seed, run, length))
    else:
        metadata = png_metadata(suffix, seed, run, length)
        fig.savefig(path, dpi=300, metadata=metadata)
    plt.close(fig)
