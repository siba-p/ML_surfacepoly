"""Inverse/backward architectures."""
from __future__ import annotations

import tensorflow as tf
from tensorflow import keras
from keras import regularizers
from keras.layers import (
    Dense,
    Dropout,
    Flatten,
    Input,
    Lambda,
    LeakyReLU,
    Model,
    Reshape,
    UpSampling2D,
    Conv2D,
)

from ..callbacks import AnnealedSmoothBinary


def build_DNN_backward_model(
    input_dim: int = 100,
    output_dim: int = 440,
    learning_rate: float = 1e-3,
    alpha_focal: float = 0.25,
    gamma_focal: float = 1.0,
    leakyrelu_alpha: float = 0.1,
) -> keras.Model:
    """Dense inverse network mapping PMF -> sequence."""

    kernel_initializer = keras.initializers.he_normal(seed=40)
    bias_initializer = keras.initializers.zeros()

    loss_fn = tf.keras.losses.BinaryFocalCrossentropy(alpha=alpha_focal, gamma=gamma_focal)

    inputs = keras.Input(shape=(input_dim,), name="backward_input")
    x = Dense(141, name="backward_dense_1", kernel_regularizer=regularizers.l2(1e-7),
              kernel_initializer=kernel_initializer, bias_initializer=bias_initializer)(inputs)
    x = LeakyReLU(alpha=leakyrelu_alpha)(x)

    x = Dense(248, name="backward_dense_2", kernel_regularizer=regularizers.l2(1e-6),
              kernel_initializer=kernel_initializer, bias_initializer=bias_initializer)(x)
    x = LeakyReLU(alpha=leakyrelu_alpha)(x)

    x = Dense(368, name="backward_dense_3", kernel_regularizer=regularizers.l2(2.88015e-6),
              kernel_initializer=kernel_initializer, bias_initializer=bias_initializer)(x)
    x = LeakyReLU(alpha=leakyrelu_alpha)(x)

    outputs = Dense(output_dim, activation="sigmoid", name="sequence_output")(x)

    model = keras.Model(inputs, outputs, name="BackwardDNN")
    model.compile(
        optimizer=tf.optimizers.Nadam(learning_rate=learning_rate),
        loss=loss_fn,
        metrics=["binary_accuracy", keras.metrics.Precision(), keras.metrics.Recall(), "mse", "mae"],
    )
    return model


def build_HybridCNN_backward_model(
    input_dim: int = 100,
    surface_shape: tuple[int, int, int] = (20, 20, 1),
    learning_rate: float = 1e-4,
    leakyrelu_alpha: float = 0.1,
) -> keras.Model:
    """CNN decoder for surfaces driven by PMF embeddings."""

    pmf_input = Input(shape=(input_dim,), name="pmf_input")
    x = Dense(200)(pmf_input)
    x = LeakyReLU(alpha=leakyrelu_alpha)(x)
    x = Dense(150)(x)
    x = LeakyReLU(alpha=leakyrelu_alpha)(x)
    x = Dropout(0.1)(x)

    x_surf = Dense(5 * 5 * 32, name="surf_fc")(x)
    x_surf = LeakyReLU(alpha=leakyrelu_alpha)(x_surf)
    x_surf = Reshape((5, 5, 32))(x_surf)

    x_surf = UpSampling2D((2, 2))(x_surf)
    x_surf = Conv2D(16, kernel_size=3, padding="same", kernel_regularizer=regularizers.l2(1e-3))(x_surf)
    x_surf = LeakyReLU(alpha=leakyrelu_alpha)(x_surf)

    x_surf = UpSampling2D((2, 2))(x_surf)
    x_surf = Conv2D(16, kernel_size=3, padding="same", kernel_regularizer=regularizers.l2(1e-2))(x_surf)
    x_surf = LeakyReLU(alpha=leakyrelu_alpha)(x_surf)
    x_surf = Conv2D(1, kernel_size=3, padding="same", activation=None, name="surf_logits")(x_surf)

    x_surf = AnnealedSmoothBinary(name="annealed_surface_activation")(x_surf)
    surface_output = Flatten(name="surface_output")(x_surf)

    model = Model(inputs=pmf_input, outputs=surface_output, name="BackwardHybridCNN")
    model.compile(
        optimizer=keras.optimizers.Nadam(learning_rate=learning_rate),
        loss="mae",
        metrics=["binary_accuracy", "mae", "mse"],
    )
    return model
