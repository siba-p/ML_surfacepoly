"""Forward-model architectures (DNN + Hybrid CNN/Transformer)."""
from __future__ import annotations

import tensorflow as tf
from tensorflow import keras
from keras import regularizers
from keras.layers import (
    Add,
    Conv2D,
    Dense,
    Dropout,
    Flatten,
    GRU,
    Input,
    Lambda,
    LayerNormalization,
    LeakyReLU,
    MaxPooling2D,
    Model,
    MultiHeadAttention,
    Reshape,
)
from keras import optimizers


def build_DNN_forward_model(
    input_shape: tuple[int, ...] = (440,),
    act_alpha: float = 0.1,
    lambtha: float = 10 ** -2.449,
    learning_rate: float = 1e-4,
    loss_function: str = "mae",
) -> keras.Model:

    l2_reg = [1e-6, 1e-6, 2.88015e-5]
    num_neuron = [368, 248, 141]
    dropout_frac = [0.0, 0.1, 0.2]

    kernel_initializer = keras.initializers.he_normal(seed=40)
    bias_initializer = keras.initializers.zeros()
    optimizer = optimizers.Nadam(learning_rate=learning_rate)

    inputs = Input(shape=input_shape, name="forward_input")
    x = Dense(
        num_neuron[0],
        name="forward_dense_1",
        kernel_regularizer=regularizers.l2(l2_reg[0]),
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
    )(inputs)
    x = LeakyReLU(negative_slope=act_alpha)(x)

    x = Dense(
        num_neuron[1],
        name="forward_dense_2",
        kernel_regularizer=regularizers.l2(l2_reg[1]),
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
    )(x)
    x = LeakyReLU(negative_slope=act_alpha)(x)
    x = Dropout(dropout_frac[1])(x)

    x = Dense(
        num_neuron[2],
        name="forward_dense_3",
        kernel_regularizer=regularizers.l2(l2_reg[2]),
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
    )(x)
    x = LeakyReLU(negative_slope=act_alpha)(x)
    x = Dropout(dropout_frac[2])(x)

    outputs = Dense(100, name="pmf_output")(x)

    model = keras.Model(inputs, outputs, name="ForwardDNN")
    model.compile(optimizer=optimizer, loss=loss_function, metrics=["mae", "mse"])
    return model


def transformer_block(polymer, surface, num_heads: int = 2, ff_dim: int = 64, dropout: float = 0.1):
    norm_query = LayerNormalization(epsilon=1e-6)(polymer)
    norm_keyval = LayerNormalization(epsilon=1e-6)(surface)

    attn_output = MultiHeadAttention(num_heads=num_heads, key_dim=ff_dim)(
        query=norm_query, key=norm_keyval, value=norm_keyval
    )
    attn_output = Dropout(dropout)(attn_output)
    out1 = Add()([polymer, attn_output])

    x = LayerNormalization(epsilon=1e-6)(out1)
    x = Dense(ff_dim, activation="relu")(x)
    x = Dense(polymer.shape[-1])(x)
    x = Dropout(dropout)(x)
    return Add()([out1, x])


def expand_surface_sequence_fn(x):
    return tf.tile(tf.expand_dims(x, axis=1), [1, 40, 1])


def slice_surface_fn(x):
    return x[:, :400]


def slice_polymer_fn(x):
    return x[:, 400:]


def build_HybridCNN_forward_model() -> keras.Model:
    """Hybrid CNN + GRU + Transformer forward network."""

    kernel_initializer = keras.initializers.he_normal(seed=40)
    inputs = Input(shape=(440,), name="seq_input")

    surface = Lambda(slice_surface_fn, name="lambda_surface_slice")(inputs)
    polymer = Lambda(slice_polymer_fn, name="lambda_polymer_slice")(inputs)

    surface = Reshape((20, 20, 1))(surface)
    polymer = Reshape((40, 1))(polymer)

    x_surface = Conv2D(32, (3, 3), activation="relu", padding="same")(surface)
    x_surface = MaxPooling2D((2, 2))(x_surface)
    x_surface = Conv2D(64, (3, 3), activation="relu", padding="same")(x_surface)
    x_surface = MaxPooling2D((2, 2))(x_surface)
    x_surface = Conv2D(128, (3, 3), activation="relu", padding="same")(x_surface)
    x_surface = Flatten()(x_surface)
    surface_proj = Dense(64, activation="relu")(x_surface)
    surface_seq = Lambda(expand_surface_sequence_fn, name="lambda_expand_surface_sequence")(surface_proj)

    x_polymer = GRU(64, return_sequences=True)(polymer)
    x_polymer = LayerNormalization()(x_polymer)
    x_trans = transformer_block(x_polymer, surface_seq)

    x = Flatten()(x_trans)
    x = Dropout(0.2)(x)
    x = Dense(250, kernel_initializer=kernel_initializer)(x)
    x = LeakyReLU()(x)
    x = Dense(140, kernel_initializer=kernel_initializer, kernel_regularizer=regularizers.l2(1e-6))(x)
    x = LeakyReLU()(x)
    outputs = Dense(100, name="pmf_output")(x)

    model = Model(inputs=inputs, outputs=outputs, name="PolymerSurfaceTransformer")
    model.compile(
        optimizer=optimizers.Nadam(learning_rate=1e-4, beta_1=0.97, beta_2=0.97),
        loss="mae",
        metrics=["mae", "mse"],
    )
    return model
