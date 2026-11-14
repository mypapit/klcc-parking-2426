import numpy as np

from src.metrics import ranking_metrics


def test_ranking_metrics_returns_expected_keys():
    y = np.array([0, 0, 1, 1])
    s = np.array([0.1, 0.2, 0.8, 0.9])
    out = ranking_metrics(y, s)
    assert set(out) == {"roc_auc", "pr_auc"}
    assert out["roc_auc"] == 1.0
