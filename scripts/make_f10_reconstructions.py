"""F10: four captioned reconstruction panels with consistent units.

The plan's Task 8, implementing the reviewer's two-sided fix. The previous
`make_run_figures.py:48-52` fed RAW bays to a z-score-trained model with no
`pre.transform`, so the reconstruction saturated: the model expects
standardised input and produces standardised output, and inverse-transforming
the output alone cannot undo a missing transform of the input. Here both
sides are handled: `prepare_window` transforms the input with `pre.transform`,
and the model's output is mapped back to bays with `inverse_transform`.

Panels (D3/P0/L=18, seed 0, the same run as F09):
(a) a normal weekday window (2023-08-02, all 144 of that day's scores below
    the threshold) - not flagged;
(b) a real anomaly on a verified event day (Beautyexpo & Cosmobeaute Malaysia
    2023, 27-30 Sep), window ending 2023-09-28 17:50, peak score 0.0526 -
    flagged;
(c) an injected F3 full-capacity block (copy 0, first segment) - flagged;
(d) an injected F6 noise burst (copy 0, first segment) - NOT flagged, which
    is the honest result and the panel label says so.

The injected panels are re-scored with `inject_copy` + `prepare_window`
(inference only, no retraining); no injected per-timestamp score file exists
on disk, the saved runs carry injection positions only.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

FIG = ROOT / "outputs" / "figures"
RUN = ROOT / "outputs" / "runs" / "E3" / "D3_P0_L18_b128_s0"
LENGTH = 18
SEED = 0


def inverse_transform(z: float | np.ndarray, mean: float, std: float) -> float | np.ndarray:
    """Map standardised values back to bay counts: z * std + mean."""
    return z * std + mean


def prepare_window(values: pd.Series, pre, model, start: int, length: int, tau: float):
    """Score one window of `values` and return (observed, reconstruction, score, flagged).

    `start` is the window START index: `values.iloc[start:start+length]`. The
    window is transformed with pre.transform BEFORE the model sees it, the
    reconstruction is mapped back with inverse_transform, and the score is
    mean((z_obs - z_recon)^2), the same formula as the scorer.
    """
    observed = values.iloc[start:start + length].astype(float).to_numpy()
    z = pre.transform(pd.Series(observed, index=values.index[start:start + length]))
    # the model expects the same 3D float32 shape make_windows produces:
    # (batch, length, channels). Passing a 2D window is accepted silently but
    # returns a different reconstruction, which is how the panel scores first
    # diverged from the stored pipeline scores.
    x = z.to_numpy().reshape(1, length, 1).astype(np.float32)
    pred = model.predict(x, verbose=0)[0]
    # the score in z units, exactly like the scorer; the reconstruction for
    # the display goes back to bays through the same mean/std. Squeeze the
    # channel axis so the reconstruction is 1-D like the observed values.
    score = float(np.mean((x[0] - pred) ** 2))
    reconstruction = inverse_transform(pred.reshape(-1), pre.mean_, pre.std_)
    return observed, reconstruction, score, bool(score > tau)


def main() -> int:
    import matplotlib.pyplot as plt

    from data import split_bounds
    from features import Preprocessor
    from figure_style import apply_style, save_figure
    from inject import inject_copy
    from utils import load_config

    cfg = load_config()
    metrics = json.loads((RUN / "metrics.json").read_text())
    tau = float(metrics["thresholds"]["p99_5"])

    processed = pd.read_parquet(cfg["data"]["processed"]).set_index("timestamp")
    series = processed["available"].astype(float)
    bounds = split_bounds(series, cfg["splits"]["history_end"], cfg["splits"]["test_b_start"], cfg["splits"]["test_b_end"])
    pre = Preprocessor("P0")
    pre.window_length = LENGTH
    pre.fit(series.loc[bounds.train[0]:bounds.train[1]])
    test_a = series.loc[bounds.test_a[0]:bounds.test_a[1]]
    pos = {ts: i for i, ts in enumerate(test_a.index)}

    import tensorflow as tf

    model = tf.keras.models.load_model(RUN / "model.keras")

    # (a) the day's max-score window on 2023-08-02, still below the threshold
    scores = pd.read_parquet(RUN / "scores_testA.parquet")
    day = scores[(scores["timestamp"] >= "2023-08-02") & (scores["timestamp"] < "2023-08-03")]
    end_a = day.loc[day["score"].idxmax(), "timestamp"]
    start_a = pos[pd.Timestamp(end_a)] - LENGTH + 1

    # (b) the real anomaly window ending 2023-09-28 17:50 (peak 0.0526)
    start_b = pos[pd.Timestamp("2023-09-28 17:50:00")] - LENGTH + 1

    # (c)/(d) rebuild the injected series for copy 0 (inference only)
    panels = {}
    injected = {}
    for fault, seg in (("F3", 0), ("F6", 0)):
        inj, positions = inject_copy(test_a, "testA", fault, 0, cfg)
        injected[fault] = (inj, positions)
    inj_f3, pos_f3 = injected["F3"]
    start_c = int(pos_f3.iloc[seg]["start_i"])
    inj_f6, pos_f6 = injected["F6"]
    start_d = int(pos_f6.iloc[seg]["start_i"])

    specs = [
        ("(a)", test_a, start_a),
        ("(b)", test_a, start_b),
        ("(c)", inj_f3, start_c),
        ("(d)", inj_f6, start_d),
    ]
    results = []
    for label, values, start in specs:
        observed, reconstruction, score, flagged = prepare_window(values, pre, model, start, LENGTH, tau)
        results.append((label, observed, reconstruction, score, flagged))
        print(f"{label} start={start} score={score:.5f} flagged={flagged}")

    apply_style()
    fig, axes = plt.subplots(2, 2, figsize=(3.5, 3.0))
    for ax, (label, observed, reconstruction, score, flagged) in zip(axes.ravel(), results):
        x = np.arange(len(observed))
        ax.plot(x, observed, lw=1.0, color="#0072B2", label="Observed")
        ax.plot(x, reconstruction, lw=1.0, ls="--", color="#D55E00", label="Reconstruction")
        ax.set_ylabel("Available bays", fontsize=7)
        ax.tick_params(labelsize=7)
        # headroom so the corner text never sits on the data: 18% at the top
        # for the score/flag line, and the panel label above the axes
        lo = float(min(observed.min(), reconstruction.min()))
        hi = float(max(observed.max(), reconstruction.max()))
        span = max(hi - lo, 1.0)
        ax.set_ylim(lo - 0.06 * span, hi + 0.22 * span)
        ax.text(0.0, 1.02, label, transform=ax.transAxes, fontsize=8, va="bottom", ha="left")
        ax.text(0.02, 0.985, f"score {score:.5f}\n{'flagged' if flagged else 'not flagged'}",
                transform=ax.transAxes, fontsize=7, va="top", ha="left")
    axes[0, 0].legend(fontsize=6, frameon=False, loc="lower right")
    fig.tight_layout()
    save_figure(fig, FIG / "F10_reconstructions.pdf", seed=SEED, run="D3_P0_L18_b128_s0", length=LENGTH)
    print("wrote F10_reconstructions.pdf and .png")

    from PIL import Image

    size = Image.open(FIG / "F10_reconstructions.png").size
    ok = size == (1050, 900)
    print(f"  PNG {size[0]}x{size[1]} (expected 1050x900) -> {'OK' if ok else 'MISMATCH'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
