"""F09: anomaly score over clean Test A, log y axis, KLCC event bands.

The addendum's corrections over the previous version:
- log-scale y axis, so the p99.5 threshold line (~0.0038) and the small
  errors stay visible next to the ~0.12 peaks;
- verified KLCC event days from data/external/klcc_events.csv inside Test A
  as light grey bands, labelled in the legend;
- merged anomaly events as small red markers along the bottom, labelled;
- the threshold legend label is "p99.5 threshold" with no decimal.

Seed 0 (run E3/D3_P0_L18_b128_s0) goes into the file metadata: PNG carries
Seed/Run/Length text keys, PDF carries them in Title/Subject/Keywords because
PDF silently drops unknown keys.
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
from figure_style import save_figure  # noqa: E402

FIG = ROOT / "outputs" / "figures"
RUN = ROOT / "outputs" / "runs" / "E3" / "D3_P0_L18_b128_s0"
EVENTS_CSV = ROOT / "data" / "external" / "klcc_events.csv"

USE_LOG_Y = True
THRESHOLD_LABEL = "p99.5 threshold"
SEED = 0


def main() -> int:
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    from figure_style import apply_style

    scores = pd.read_parquet(RUN / "scores_testA.parquet")
    ts = pd.DatetimeIndex(scores["timestamp"])
    values = scores["score"].to_numpy()
    metrics = json.loads((RUN / "metrics.json").read_text())
    tau = float(metrics["thresholds"]["p99_5"])

    events = pd.read_csv(RUN / "events_testA.csv", parse_dates=["start", "end"])

    klcc = pd.read_csv(EVENTS_CSV, parse_dates=["date_start", "date_end"])
    lo, hi = ts.min(), ts.max()
    # keep only the verified KLCC events that overlap Test A
    klcc = klcc[(klcc["date_end"] >= lo) & (klcc["date_start"] <= hi)]

    apply_style()
    fig, ax = plt.subplots(figsize=(7.16, 2.2))
    ax.plot(ts, values, lw=0.4, color="#0072B2", label="Anomaly score")

    # verified KLCC event days as light grey bands
    first = True
    for _, row in klcc.iterrows():
        ax.axvspan(row["date_start"], row["date_end"] + pd.Timedelta(days=1),
                   color="grey", alpha=0.15, lw=0,
                   label="Verified KLCC event" if first else None)
        first = False

    # merged anomaly events as small red markers along the bottom
    ax.plot(events["start"], np.full(len(events), values.min() * 0.7), "|",
            color="#D62728", markersize=4, label="Merged anomaly event")

    ax.axhline(tau, color="#444444", lw=0.8, ls="--", label=THRESHOLD_LABEL)
    if USE_LOG_Y:
        ax.set_yscale("log")
    ax.set_ylabel("Anomaly score (log)")
    ax.set_xlabel("Time (Asia/Kuala_Lumpur)")
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.legend(loc="upper left", ncol=2, fontsize=6.5, frameon=False)
    fig.tight_layout()
    save_figure(fig, FIG / "F09_error_over_time.pdf", seed=SEED, run="D3_P0_L18_b128_s0", length=18)
    print(f"wrote F09_error_over_time.pdf and .png (tau={tau:.6f}, {len(events)} events, {len(klcc)} KLCC events)")

    # confirm the metadata landed
    from PIL import Image

    meta = Image.open(FIG / "F09_error_over_time.png").info
    print(f"  PNG metadata: Seed={meta.get('Seed')} Run={meta.get('Run')} Length={meta.get('Length')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
