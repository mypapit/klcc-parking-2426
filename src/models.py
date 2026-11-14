"""LSTM autoencoder builders."""
from __future__ import annotations

import tensorflow as tf

# Trainable parameter counts the handover specifies for the univariate case.
# D3 must be exactly 584,257; a different value means the architecture is wrong
# and must be reported in ISSUES.md rather than papered over.
EXPECTED_PARAMS = {"D1": 444865, "D2": 556225, "D3": 584257}


def build_autoencoder(depth: str, seq_len: int) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=(seq_len, 1))
    if depth == "D1":
        x = tf.keras.layers.LSTM(192)(inputs)
        x = tf.keras.layers.RepeatVector(seq_len)(x)
        x = tf.keras.layers.LSTM(192, return_sequences=True)(x)
    elif depth == "D2":
        x = tf.keras.layers.LSTM(192, return_sequences=True)(inputs)
        x = tf.keras.layers.LSTM(96)(x)
        x = tf.keras.layers.RepeatVector(seq_len)(x)
        x = tf.keras.layers.LSTM(96, return_sequences=True)(x)
        x = tf.keras.layers.LSTM(192, return_sequences=True)(x)
    elif depth == "D3":
        x = tf.keras.layers.LSTM(192, return_sequences=True)(inputs)
        x = tf.keras.layers.LSTM(96, return_sequences=True)(x)
        x = tf.keras.layers.LSTM(48)(x)
        x = tf.keras.layers.RepeatVector(seq_len)(x)
        x = tf.keras.layers.LSTM(48, return_sequences=True)(x)
        x = tf.keras.layers.LSTM(96, return_sequences=True)(x)
        x = tf.keras.layers.LSTM(192, return_sequences=True)(x)
    else:
        raise ValueError(depth)
    outputs = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(1))(x)
    model = tf.keras.Model(inputs, outputs)
    expected = EXPECTED_PARAMS.get(depth)
    if expected is not None and model.count_params() != expected:
        raise AssertionError(
            f"{depth} has {model.count_params()} trainable parameters, "
            f"expected {expected}; report this in ISSUES.md rather than changing the model"
        )
    return model
