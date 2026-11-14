"""Unit tests for the two-sided reconstruction fix in make_f10_reconstructions.

`make_run_figures.py:48-52` fed raw bays to a z-score-trained model with no
`pre.transform`, so the reconstruction saturated: the model expects
standardised inputs and produces standardised outputs, which inverse-
transforming the OUTPUT alone cannot undo. The fix transforms the input first
and reconstructs from the transformed window, then inverse-transforms both.
"""
from __future__ import annotations

import inspect
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


def test_inverse_transform_maps_z_back_to_bays():
    from make_f10_reconstructions import inverse_transform

    mean, std = 2000.0, 1000.0
    assert inverse_transform(1.5, mean, std) == 3500.0


def test_window_is_transformed_before_the_model_sees_it():
    from make_f10_reconstructions import prepare_window

    src = inspect.getsource(prepare_window)
    # the model must receive preprocessed values, not raw bays
    assert "pre.transform" in src


def test_prepare_window_round_trips_through_a_stub_model():
    # a stub model that returns its input: the score must be 0 and the
    # reconstruction must come back in bays, which only works when the input
    # went through pre.transform and the output through inverse_transform
    import numpy as np
    import pandas as pd

    from features import Preprocessor
    from make_f10_reconstructions import prepare_window

    class StubModel:
        def predict(self, x, verbose=0):
            return np.array(x)

    pre = Preprocessor("P0")
    pre.window_length = 6
    values = pd.Series(np.linspace(100, 500, 100, dtype=float))
    pre.fit(values)
    observed, reconstruction, score, flagged = prepare_window(values, pre, StubModel(), 10, 6, tau=1e-9)
    assert score == 0.0
    assert not flagged
    # float32 model tensors round-trip with about 1e-4 bay error, not 1e-9
    assert np.allclose(observed, reconstruction, atol=1e-3)
