"""Synthetic fault injection."""
from __future__ import annotations

import numpy as np
import pandas as pd


def inject_fault(series: pd.Series, start: int, fault: str, sigma_ref: float, seed: int, totalbay: int = 5300) -> pd.Series:
    rng = np.random.default_rng(seed)
    out = series.copy()
    if fault == "F1":
        out.iloc[start:start + 12] = series.iloc[start]
    elif fault == "F2":
        out.iloc[start:start + 11] = 0
    elif fault == "F3":
        out.iloc[start:start + 11] = 5300
    elif fault == "F4":
        # one-bin spike. sigma_ref must be finite (a NaN here would poison the
        # injected value, as in the pre-2026-11-10 code), and the result must be
        # a valid parking reading, because the cleaning step would otherwise
        # replace an out-of-range spike with NaN and the windows would be dropped.
        value = series.iloc[start]
        delta = 5.0 * sigma_ref
        target = value + rng.choice([-1, 1]) * delta
        if target > totalbay or target < 0:
            target = value - delta
        if target > totalbay or target < 0 or not np.isfinite(target):
            target = float(np.clip(value + delta, 0, totalbay))
        if target == value:
            target = 0.0 if value > totalbay / 2 else float(totalbay)
        out.iloc[start:start + 1] = target
    elif fault == "F5":
        ramp = np.linspace(0, 530, 36)
        out.iloc[start:start + 36] = series.iloc[start:start + 36].to_numpy() + ramp
    elif fault == "F6":
        out.iloc[start:start + 12] = series.iloc[start:start + 12].to_numpy() + rng.normal(0, 265, 12)
    else:
        raise ValueError(fault)
    return out


def fault_length(fault: str) -> int:
    return {"F1": 12, "F2": 11, "F3": 11, "F4": 1, "F5": 36, "F6": 12}[fault]


def inject_copy(series: pd.Series, testset: str, fault: str, copy_id: int, cfg: dict, sigma_ref: float | None = None) -> tuple[pd.Series, pd.DataFrame]:
    rng = np.random.default_rng(abs(sum((i + 1) * ord(ch) for i, ch in enumerate(f"{testset}:{fault}:{copy_id}"))) % (2**32))
    count = cfg["injection"]["injections_per_copy"]
    min_gap = pd.Timedelta(days=cfg["injection"]["min_gap_days"])
    start_max = len(series) - fault_length(fault) - 1
    positions = []
    values = series.to_numpy()
    while len(positions) < count:
        start = int(rng.integers(0, start_max + 1))
        end = start + fault_length(fault) - 1
        # refuse positions where the source data is missing: injecting onto a NaN
        # produces NaN again, the affected window is dropped, and the fault
        # becomes invisible (the original F4 failure mode)
        if not np.isfinite(values[start: end + 1]).all():
            continue
        start_ts = series.index[start]
        if all(abs(start_ts - row["start"]) >= min_gap for row in positions):
            positions.append({"copy": copy_id, "fault": fault, "start": start_ts, "end": series.index[end], "start_i": start, "end_i": end})
    out = series.copy()
    # sigma must come from finite values only and default to the train split when
    # the caller supplies it; a std over a series containing NaN silently poisons
    # everything computed from it
    if sigma_ref is None:
        sigma_ref = float(np.nanstd(values))
    totalbay = int(cfg["data"].get("totalbay", 5300))
    for row in positions:
        out = inject_fault(out, row["start_i"], fault, sigma_ref, seed=copy_id, totalbay=totalbay)
    return out, pd.DataFrame(positions)
