"""Test F13's labels, method order and the T7 deduplication."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


def test_fault_labels_short_and_methods_ordered():
    from make_f13_heatmap import METHODS, SHORT_FAULTS

    assert SHORT_FAULTS[0] == "F1 stuck"
    assert SHORT_FAULTS[3] == "F4 exc."
    assert SHORT_FAULTS[5] == "F6 noise"
    assert METHODS == ["D1", "D2", "D3", "B1", "B2", "B3", "B4"]


def test_t7_does_not_double_count_d3():
    # make_t7.py pools E3 and E2, whose D3_P0_L18_b128_s* artifacts are
    # byte-identical, so D3/P0 rows were counted twice. The dedupe selects
    # E2 for D1/D2 and E3 for D3.
    import pandas as pd

    t7 = pd.read_csv("/edata/opt/klcc-experiment/klcc_ad/outputs/tables/T7_methods_by_fault.csv")
    sub = t7[(t7.method == "D3") & (t7.length == 18) & (t7.testset == "testA") & (t7.fault == "F4")]
    assert sub.seeds.iloc[0] == 5


def test_t7_per_seed_std_is_finite_and_small():
    # the inflated std came from pooling copy-level rows; after the fix, the
    # std over the 5 seeds of D3/P0/L18/testA/F4 should be well under 0.05
    import pandas as pd

    t7 = pd.read_csv("/edata/opt/klcc-experiment/klcc_ad/outputs/tables/T7_methods_by_fault.csv")
    sub = t7[(t7.method == "D3") & (t7.length == 18) & (t7.testset == "testA") & (t7.fault == "F4")]
    assert sub.pr_auc_std.iloc[0] < 0.05


def test_figure_size_matches_the_addendum(tmp_path):
    # the current heatmap is 7.16 x 7.8 in, nearly a full page; the addendum
    # wants 7.16 x 2.6
    import matplotlib

    matplotlib.use("Agg")
    from make_f13_heatmap import build
    from figure_style import DOUBLE_WIDTH

    fig, axes = build("testA")
    assert abs(fig.get_figwidth() - DOUBLE_WIDTH) < 1e-9
    assert abs(fig.get_figheight() - 2.6) < 1e-9
