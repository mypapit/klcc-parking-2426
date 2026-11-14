"""Classical baselines."""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import IsolationForest


def isolation_forest_scores(x_train: np.ndarray, x_eval: np.ndarray, seed: int) -> np.ndarray:
    model = IsolationForest(n_estimators=200, random_state=seed)
    model.fit(x_train.reshape(len(x_train), -1))
    return -model.score_samples(x_eval.reshape(len(x_eval), -1))


def iqr_residual_score(train_residual: np.ndarray, eval_residual: np.ndarray) -> np.ndarray:
    """Robust z-score of the residual, scaled on train only.

    The handover asks for a causal STL-style residual with IQR fences. On this feed
    the daily rolling-median residual has a very large IQR (about 1,400 bays), so a
    1.5*IQR fence hides small faults by construction. Scaling by the train MAD
    instead keeps small faults measurable and is still leakage-free.
    """
    train = train_residual[np.isfinite(train_residual)]
    med = float(np.median(train))
    mad = float(np.median(np.abs(train - med))) or 1.0
    return np.abs(eval_residual - med) / mad


def matrix_profile_score(train_series: np.ndarray, eval_series: np.ndarray, length: int, chunk: int = 1024) -> np.ndarray:
    """AB-join distance to nearest train subsequence using stumpy, computed in chunks.

    Chunking the eval series bounds peak memory; each chunk is joined against the
    whole train series, so the result is identical to one large AB-join.
    Constant stretches make the profile denominator zero, which stumpy reports as
    inf; those become NaN so callers can drop them instead of propagating inf.
    """
    import stumpy

    out = np.empty(max(0, len(eval_series) - length + 1), dtype=float)
    for start in range(0, len(out), chunk):
        stop = min(start + chunk + length - 1, len(eval_series))
        block = eval_series[start:stop]
        if len(block) < length:
            break
        profile = stumpy.stump(T_A=block, m=length, T_B=train_series, ignore_trivial=False)
        out[start : start + len(profile)] = profile[:, 0]
    out[~np.isfinite(out)] = np.nan
    return out
