"""Test F12's caption basis and layout."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


def test_caption_faults_match_t5():
    from make_f12_depth import CAPTION_FAULTS

    assert CAPTION_FAULTS == ("F1", "F2", "F3", "F5", "F6")


def test_t5_table_fits_the_caption_basis():
    import pandas as pd

    # T5 is built from the single-copy fault_metrics.csv over F1,F2,F3,F5,F6
    # on both test sets. F4 is present there but with zero positive windows,
    # so T5's mean excludes it.
    from make_f12_depth import t5_basis_faults

    basis = t5_basis_faults()
    assert "F4" not in basis
    assert set(basis) == {"F1", "F2", "F3", "F5", "F6"}


def test_t5_covers_both_test_sets():
    import pandas as pd

    # recompute T5's mean from the raw run files for D3/P0/L18 and check the
    # stored value matches the both-testsets average
    rows = []
    for seed in range(5):
        d = pd.read_csv(
            f"/edata/opt/klcc-experiment/klcc_ad/outputs/runs/E2/D3_P0_L18_b128_s{seed}/fault_metrics.csv"
        )
        v = d[(d.positive_windows > 0) & (d.fault.isin(["F1", "F2", "F3", "F5", "F6"]))]
        rows.extend(v.pr_auc.tolist())
    recomputed = sum(rows) / len(rows)
    t5 = pd.read_csv("/edata/opt/klcc-experiment/klcc_ad/outputs/tables/T5_depth_ablation.csv")
    stored = t5[(t5.depth == "D3") & (t5.length == 18)].mean_pr_auc_mean.iloc[0]
    assert abs(recomputed - stored) < 1e-6
