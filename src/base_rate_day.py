"""Day-level base-rate tests (addendum Fix 4).

Merged events cluster within a day, so the event-level binomial and permutation
tests over-count independence. The day-level test makes the calendar day the
unit of analysis:

- anomaly day = a day in the split with at least one merged event
- p0 = the fraction of split days that belong to the tested cause
- k = anomaly days that belong to the cause, n = the number of anomaly days
- a binomial test and a permutation test that redraws which n days are anomaly
  days, uniformly at random, seeded, N_PERM times

Used by scripts/run_audit_all.py; the event-level tests stay in run_audit.py.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import binomtest

N_PERM = 10_000
SEED = 20261110


def anomaly_days(events: pd.DataFrame) -> pd.DatetimeIndex:
    """Calendar days (normalised) that carry at least one merged event."""
    if events is None or len(events) == 0:
        return pd.DatetimeIndex([])
    days = pd.to_datetime(events["start"]).dt.normalize()
    return pd.DatetimeIndex(sorted(set(days)))


def day_level_test(
    test_days: pd.DatetimeIndex,
    cause_days: set,
    events: pd.DataFrame,
    n_perm: int = N_PERM,
    seed: int = SEED,
) -> dict:
    """One cause, one split, at the day level."""
    n_days = len(test_days)
    anom = anomaly_days(events)
    n_anom = len(anom)
    cause = test_days.isin(list(cause_days))
    p0 = float(cause.mean()) if n_days else 0.0
    anom_is_cause = anom.isin(list(cause_days))
    k = int(anom_is_cause.sum())

    if n_anom == 0 or p0 <= 0:
        return {
            "unit": "day",
            "n_days": n_days,
            "n_anomaly_days": n_anom,
            "k": k,
            "p0": p0,
            "binomial_p": float("nan"),
            "permutation_p": float("nan"),
            "null_mean": float("nan"),
            "n_perm": 0,
        }

    binom = binomtest(k, n_anom, p0, alternative="greater")
    special_idx = np.flatnonzero(np.asarray(cause, dtype=bool))
    n_test = n_days
    rng = np.random.default_rng(seed)
    hits = np.zeros(n_perm, dtype=int)
    for i in range(n_perm):
        draw = rng.choice(n_test, size=n_anom, replace=False)
        hits[i] = np.isin(draw, special_idx).sum()
    return {
        "unit": "day",
        "n_days": n_days,
        "n_anomaly_days": n_anom,
        "k": k,
        "p0": p0,
        "binomial_p": float(binom.pvalue),
        "permutation_p": float((hits >= k).mean()),
        "null_mean": float(hits.mean()),
        "n_perm": n_perm,
    }


def run_arms(
    test_days: pd.DatetimeIndex,
    arms: dict,
    events: pd.DataFrame,
    n_perm: int = N_PERM,
    seed: int = SEED,
) -> dict:
    """arms maps an arm name to a set of cause days; +/-1 day variants included."""
    out: dict = {}
    for name, dayset in arms.items():
        out[f"{name}_exact"] = day_level_test(test_days, dayset, events, n_perm=n_perm, seed=seed)
        grown = set()
        for d in dayset:
            grown.update({d - pd.Timedelta(days=1), d, d + pd.Timedelta(days=1)})
        out[f"{name}_plusminus1"] = day_level_test(test_days, grown, events, n_perm=n_perm, seed=seed)
    return out
