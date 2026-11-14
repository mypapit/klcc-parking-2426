"""Figure generation entrypoint for the series-level figures."""
from __future__ import annotations

import sys
from pathlib import Path

import _bootstrap  # noqa: F401
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from plots import error_hist_figure, raw_series_figure  # noqa: E402

FIGURES = ROOT / "outputs" / "figures"
RUNS = ROOT / "outputs" / "runs"


def main() -> int:
    FIGURES.mkdir(parents=True, exist_ok=True)
    processed = ROOT / "data" / "processed" / "klcc_10min.parquet"
    if processed.exists():
        df = pd.read_parquet(processed).set_index("timestamp")
        raw_series_figure(df["available"], FIGURES / "F01_raw_series.pdf")

    # the error distribution must come from a deterministic current run, not the
    # superseded slice, otherwise the figure describes an archived model
    candidate = None
    for relative in (
        "E3/D3_P0_L18_b128_s0",
        "E2/D3_P0_L18_b128_s0",
    ):
        path = RUNS / relative / "scores_testA.parquet"
        if path.exists():
            candidate = path
            break
    if candidate is None:
        found = sorted(RUNS.glob("E*/D3_*/scores_testA.parquet"))
        candidate = found[0] if found else None
    if candidate is not None:
        scores = pd.read_parquet(candidate)["score"].to_numpy()
        error_hist_figure(scores[: max(1, len(scores) // 2)], scores, FIGURES / "F08_error_distribution.pdf")
        print(f"F08 uses {candidate.parent.name}")
    else:
        print("no scores found for F08")
    print(f"wrote figures to {FIGURES}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
