import tensorflow as tf
from tensorflow import keras
from keras import backend as K
from keras.initializers import Constant
from keras.models import Sequential, Model
from keras import Input, layers, optimizers, regularizers
from keras.layers import (
    Dense, Dropout, Flatten, Reshape, Lambda,
    BatchNormalization, LayerNormalization, GlobalAveragePooling1D,
    Conv2D, Conv2DTranspose, MaxPooling2D,
    GRU, LSTM, Add, Concatenate, 
    LeakyReLU, ELU, PReLU, MultiHeadAttention
)

def transformer_block(polymer, surface, num_heads=2, ff_dim=64, dropout=0.1):
    # Normalize
    norm_query = LayerNormalization(epsilon=1e-6)(polymer)
    norm_keyval = LayerNormalization(epsilon=1e-6)(surface)

    # Attention: query=polymer, key/value=surface
    attn_output = MultiHeadAttention(num_heads=num_heads, key_dim=ff_dim)(
        query=norm_query, key=norm_keyval, value=norm_keyval
    )
    attn_output = Dropout(dropout)(attn_output)
    out1 = Add()([polymer, attn_output])

    # Feedforward part
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


def build_HybridCNN_forward_model():
    kernel_initializer = keras.initializers.he_normal(seed=40)

    input_seq = Input(shape=(440,), name="seq_input")

    # Split surface and polymer
    surface = Lambda(slice_surface_fn, name="lambda_surface_slice")(input_seq)
    polymer = Lambda(slice_polymer_fn, name="lambda_polymer_slice")(input_seq)

    surface = Reshape((20, 20, 1))(surface)  # (None, 20, 20, 1)
    polymer = Reshape((40, 1))(polymer)  # (None, 40, 1)

    # CNN for surface
    x_surface = Conv2D(32, (3, 3), activation="relu", padding="same")(surface)
    x_surface = MaxPooling2D((2, 2))(x_surface)
    x_surface = Conv2D(64, (3, 3), activation="relu", padding="same")(x_surface)
    x_surface = MaxPooling2D((2, 2))(x_surface)
    x_surface = Conv2D(128, (3, 3), activation="relu", padding="same")(x_surface)
    x_surface = Flatten()(x_surface)
    
    surface_proj = Dense(64, activation="relu")(x_surface)  # (None, 64)

    # Expand surface to sequence: (None, 40, features)
    surface_seq = Lambda(expand_surface_sequence_fn, name="lambda_expand_surface_sequence")(surface_proj)

    # Polymer encoding with GRU
    x_polymer = GRU(64, return_sequences=True)(polymer)  # (None, 40, 64)
    x_polymer = LayerNormalization()(x_polymer)

    # Apply Transformer block between polymer as query & surface as key, values
    x_trans = transformer_block(x_polymer, surface_seq)

    # Pooling + Dense
    x = Flatten()(x_trans)
    # x = GlobalAveragePooling1D()(x_trans)
    x = Dropout(0.2)(x)
    x = Dense(250, kernel_initializer=kernel_initializer)(x)
    x = LeakyReLU()(x)
    x = Dense(140, kernel_initializer=kernel_initializer, kernel_regularizer=keras.regularizers.l2(1e-6))(x)
    x = LeakyReLU()(x)
    # x = Dropout(0.1)(x)
    x = Dense(100, name="pmf_output")(x)

    model = Model(inputs=input_seq, outputs=x, name="PolymerSurfaceTransformer")

    model.compile(optimizer=optimizers.Nadam(learning_rate=0.0001, beta_1=0.97, beta_2=0.97), loss="mae", metrics=["mae", "mse"])

    return model
