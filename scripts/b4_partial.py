"""B4 matrix-profile scoring on an injected series, computed exactly.

A full AB-join for every injected copy is wasteful: an injection only changes the
windows that overlap it. The clean profile gives every unaffected window its
distance for free, and only the affected windows need a direct nearest-neighbour
computation. This is mathematically exact, one stumpy stump call per split
instead of one per injection.

Indexing: `matrix_profile_score` returns one value per window start index i
(the window covering [i, i+length-1]), matching stumpy's profile rows. A fault
over raw positions [start_i, end_i] is contained in the windows with
i in [start_i - length + 1, end_i], so only those rows are recomputed.

Distance: stumpy.stump defaults to normalize=True (z-normalised Euclidean
distance), so the recomputed distances use the same normalisation. Constant
windows get infinite distance, as in stumpy.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _stats(windows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return windows.mean(axis=1), windows.std(axis=1, ddof=0)


def zn_distances(train_windows: np.ndarray, needles: np.ndarray) -> np.ndarray:
    """Z-normalised Euclidean distance (stumpy's normalize=True default)."""
    m = needles.shape[1]
    t_mean, t_std = _stats(train_windows)
    n_mean, n_std = _stats(needles)
    t_std_safe = np.where(t_std <= 0, 1.0, t_std)
    n_std_safe = np.where(n_std <= 0, 1.0, n_std)
    dot = needles @ train_windows.T
    corr = (dot - m * n_mean[:, None] * t_mean[None, :]) / (m * n_std_safe[:, None] * t_std_safe[None, :])
    d2 = 2.0 * m * (1.0 - corr)
    dist = np.sqrt(np.clip(d2, 0.0, None))
    const_n = (n_std <= 0) | ~np.isfinite(needles).all(axis=1)
    const_t = (t_std <= 0) | ~np.isfinite(train_windows).all(axis=1)
    dist[:, const_t] = np.inf
    dist[const_n, :] = np.inf
    return dist


def inject_local_profile(
    clean_scores: np.ndarray,
    train: np.ndarray,
    injected: np.ndarray,
    positions: pd.DataFrame,
    length: int,
) -> np.ndarray:
    """Exact profile for an injected series, reusing the clean profile elsewhere.

    Windows whose needle is constant or contains NaN have no well-defined
    z-normalised distance, and stumpy's own handling of them (a finite value for
    constant needles, NaN for NaN needles) does not follow from the correlation
    formula. Those windows are handed back to stumpy directly, so the result
    matches a full AB-join exactly.
    """
    import stumpy

    scores = clean_scores.copy()
    n_eval = len(injected)
    end_cap = n_eval - length + 1
    if end_cap <= 0:
        return scores[:0]
    from numpy.lib.stride_tricks import sliding_window_view

    train_windows = sliding_window_view(train, length).astype(np.float64)
    for _, seg in positions.iterrows():
        start_i = int(seg["start_i"])
        end_i = int(seg["end_i"])
        lo = max(0, start_i - length + 1)
        hi = min(end_cap, end_i + 1)
        if hi <= lo:
            continue
        # the block must extend length-1 beyond hi so sliding_window_view yields
        # exactly the windows starting at [lo, hi)
        block = injected[lo: min(n_eval, hi + length - 1)]
        needles = sliding_window_view(block, length).astype(np.float64)[: hi - lo]
        if len(needles) < hi - lo:
            continue
        degenerate = ~np.isfinite(needles).all(axis=1) | (needles.std(axis=1, ddof=0) <= 0)
        d = zn_distances(train_windows, needles)
        best = np.full(len(needles), np.inf)
        for j in range(len(needles)):
            if np.isfinite(d[j]).any():
                best[j] = d[j].min()
        values = np.where(np.isfinite(best), best, np.nan)
        if degenerate.any():
            # hand the degenerate windows back to stumpy, in contiguous runs
            for run_lo, run_hi in _contiguous_runs(np.flatnonzero(degenerate)):
                sub_block = injected[lo + run_lo: min(n_eval, lo + run_hi + length - 1)]
                profile = stumpy.stump(T_A=sub_block, m=length, T_B=train, ignore_trivial=False)
                values[run_lo:run_hi] = profile[: run_hi - run_lo, 0]
        scores[lo:hi] = values
    return scores[: min(len(scores), end_cap)]


def _contiguous_runs(indices: np.ndarray) -> list[tuple[int, int]]:
    """Split sorted indices into (start, stop) half-open contiguous runs."""
    runs: list[tuple[int, int]] = []
    if len(indices) == 0:
        return runs
    start = int(indices[0])
    previous = start
    for value in indices[1:]:
        value = int(value)
        if value != previous + 1:
            runs.append((start, previous + 1))
            start = value
        previous = value
    runs.append((start, previous + 1))
    return runs
