import tensorflow as tf

from src.models import build_autoencoder


def test_d3_param_count_is_fixed():
    model = build_autoencoder("D3", 18)
    assert model.count_params() == 584257
    assert isinstance(model, tf.keras.Model)
