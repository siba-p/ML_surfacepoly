"""Combiners for backward/forward pipelines (tandem + extension)."""
from __future__ import annotations

import tensorflow as tf
from tensorflow import keras
from keras.layers import Concatenate, Dense, Input, Lambda, LeakyReLU
from keras import regularizers


def identity_fn(x):
    return x


def compute_pmf_fn(x):
    mean_after_90 = tf.reduce_mean(x[:, 90:], axis=1, keepdims=True)
    min_all = tf.reduce_min(x, axis=1, keepdims=True)
    return mean_after_90 - min_all


def transfer_input_fn(x):
    return x[:, 1:]


def build_tandem_model(
    backward_model: keras.Model,
    forward_model: keras.Model,
    input_dim: int = 100,
    polymer_dim: int = 40,
    learning_rate: float = 1e-4,
) -> keras.Model:
    for layer in forward_model.layers:
        layer.trainable = False

    backward_input = keras.Input(shape=(input_dim,), name="tandem_input")
    polymer_input = keras.Input(shape=(polymer_dim,), name="polymer_input")

    pred_surface = backward_model(backward_input)
    forward_input = Concatenate(axis=1)([pred_surface[:, :400], polymer_input])
    forward_output = forward_model(forward_input)

    tandem = keras.Model(inputs=[backward_input, polymer_input], outputs=forward_output, name="TandemModel")
    tandem.compile(optimizer=tf.optimizers.Adam(learning_rate=learning_rate), loss="mae", metrics=["mse"])
    return tandem


def build_backext_model(
    input_dim: int = 41,
    learning_rate: float = 1e-4,
    activation_alpha: float = 0.1,
) -> keras.Model:
    inputs = keras.Input(shape=(input_dim,), name="backext_input")
    x = Dense(80, name="E1")(inputs)
    x = LeakyReLU(alpha=activation_alpha, name="E1_act")(x)

    x = Dense(100, name="E2", kernel_regularizer=regularizers.l2(1e-4), bias_regularizer=regularizers.l2(1e-5))(x)
    x = LeakyReLU(alpha=activation_alpha, name="E2_act")(x)

    pmf_output = Dense(100, name="E3A")(x)
    deltaF_output = Lambda(compute_pmf_fn, name="E3B")(pmf_output)

    model = keras.Model(inputs=inputs, outputs=[pmf_output, deltaF_output], name="BackExtModel")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="mae",
        metrics={"E3A": ["mse", "mae"], "E3B": ["mse", "mae"]},
    )
    return model


def build_extended_tandem_model(
    backext_model: keras.Model,
    backward_model: keras.Model,
    forward_model: keras.Model,
    learning_rate: float = 1e-5,
) -> keras.Model:
    for layer in forward_model.layers:
        layer.trainable = False

    extend_input = keras.Input(shape=(41,), name="extend_input")
    transfer_input = Lambda(transfer_input_fn, name="lambda_transfer_input")(extend_input)

    backext_output, _ = backext_model(extend_input)
    temp_output = backward_model(backext_output)
    rounded_output = Lambda(identity_fn, name="lambda_round_extend")(temp_output)
    concat_input = Concatenate(name="bypass_concat")([rounded_output[:, :400], transfer_input])
    final_output = forward_model(concat_input)

    extend_model = keras.Model(inputs=extend_input, outputs=final_output, name="ExtendedTandemModel")
    extend_model.compile(optimizer=tf.optimizers.Adam(learning_rate=learning_rate), loss="mae", metrics=["mse"])
    return extend_model
