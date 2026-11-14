"""Metric helpers."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score


def ranking_metrics(y_true: np.ndarray, scores: np.ndarray) -> dict:
    return {
        "roc_auc": float(roc_auc_score(y_true, scores)),
        "pr_auc": float(average_precision_score(y_true, scores)),
    }


def point_metrics(y_true: np.ndarray, flags: np.ndarray) -> dict:
    return {
        "precision": float(precision_score(y_true, flags, zero_division=0)),
        "recall": float(recall_score(y_true, flags, zero_division=0)),
        "f1": float(f1_score(y_true, flags, zero_division=0)),
    }


def event_labels(timestamps: pd.DatetimeIndex, segments: pd.DataFrame, window_len: int) -> np.ndarray:
    labels = np.zeros(len(timestamps), dtype=int)
    for _, seg in segments.iterrows():
        start = pd.Timestamp(seg["start"])
        end = pd.Timestamp(seg["end"])
        lo = start
        hi = end + pd.Timedelta(minutes=10 * (window_len - 1))
        labels[(timestamps >= lo) & (timestamps <= hi)] = 1
    return labels
