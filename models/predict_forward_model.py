#!/usr/bin/env python3
"""
Load a trained polymer-surface forward model (DNN or HybridCNN) and predict PMFs.
"""

import os
import argparse
import numpy as np
from tensorflow import keras
from keras.layers import Lambda
from keras import Input, Model
import tensorflow as tf

# -----------------------
# Helper Lambda Functions
# -----------------------
def slice_surface_fn(x):
    return x[:, :400]

def slice_polymer_fn(x):
    return x[:, 400:]

def expand_surface_sequence_fn(x):
    return tf.tile(tf.expand_dims(x, axis=1), [1, 40, 1])

# -----------------------
# Transformer Block
# -----------------------
from keras.layers import LayerNormalization, GRU, MultiHeadAttention, Add, Dense, Dropout, Flatten, Reshape

def transformer_block(polymer, surface, num_heads=2, ff_dim=64, dropout=0.1):
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

# -----------------------
# HybridCNN Forward Model Definition (same as training)
# -----------------------
def build_HybridCNN_forward_model():
    kernel_initializer = keras.initializers.he_normal(seed=40)
    input_seq = Input(shape=(440,), name="seq_input")

    # Split input
    surface = Lambda(slice_surface_fn, name="lambda_surface_slice")(input_seq)
    polymer = Lambda(slice_polymer_fn, name="lambda_polymer_slice")(input_seq)
    surface = Reshape((20, 20, 1))(surface)
    polymer = Reshape((40, 1))(polymer)

    # CNN for surface
    x_surface = keras.layers.Conv2D(32, (3,3), activation="relu", padding="same")(surface)
    x_surface = keras.layers.MaxPooling2D((2,2))(x_surface)
    x_surface = keras.layers.Conv2D(64, (3,3), activation="relu", padding="same")(x_surface)
    x_surface = keras.layers.MaxPooling2D((2,2))(x_surface)
    x_surface = keras.layers.Conv2D(128, (3,3), activation="relu", padding="same")(x_surface)
    x_surface = Flatten()(x_surface)
    surface_proj = Dense(64, activation="relu")(x_surface)
    surface_seq = Lambda(expand_surface_sequence_fn, name="lambda_expand_surface_sequence")(surface_proj)

    # Polymer GRU
    x_polymer = GRU(64, return_sequences=True)(polymer)
    x_polymer = LayerNormalization()(x_polymer)

    # Transformer
    x_trans = transformer_block(x_polymer, surface_seq)

    # Pooling + Dense
    x = Flatten()(x_trans)
    x = Dropout(0.2)(x)
    x = Dense(250, kernel_initializer=kernel_initializer)(x)
    x = keras.layers.LeakyReLU()(x)
    x = Dense(140, kernel_initializer=kernel_initializer, kernel_regularizer=keras.regularizers.l2(1e-6))(x)
    x = keras.layers.LeakyReLU()(x)
    x = Dense(100, name="pmf_output")(x)

    model = Model(inputs=input_seq, outputs=x, name="PolymerSurfaceTransformer")
    model.compile(optimizer=keras.optimizers.Nadam(learning_rate=0.0001, beta_1=0.97, beta_2=0.97),
                  loss="mae", metrics=["mae", "mse"])
    return model

# -----------------------
# Load model & predict
# -----------------------
def load_and_predict(checkpoint_path, input_data_path, output_path, NN_type="HybridCNN"):
    # Load input data
    X = np.load(input_data_path)
    print(f"Loaded input data: {X.shape}")

    # Build model structure
    if NN_type=="HybridCNN":
        model = build_HybridCNN_forward_model()
        custom_objects = {
            "slice_surface_fn": slice_surface_fn,
            "slice_polymer_fn": slice_polymer_fn,
            "expand_surface_sequence_fn": expand_surface_sequence_fn
        }
    elif NN_type=="DNN":
        model = keras.models.load_model(checkpoint_path)  # for DNN, structure already saved
        custom_objects = {}
    else:
        raise ValueError(f"Unknown NN_type: {NN_type}")

    # Load weights
    if NN_type=="HybridCNN":
        model.load_weights(checkpoint_path)
    print(f"Loaded model from {checkpoint_path}")

    # Predict
    predictions = model.predict(X)
    np.save(output_path, predictions)
    print(f"Predictions saved to {output_path}")

# -----------------------
# Command-Line Interface
# -----------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load trained forward model and predict PMF")
    parser.add_argument("--checkpoint_path", type=str, default="./checkpoint/HybridCNN/canonical_forward_model.keras", help="Path to trained .keras checkpoint")
    parser.add_argument("--input_data", type=str, default="../data/processed/fdX_test.npy", help="Path to input .npy file")
    parser.add_argument("--output_data", type=str, default="predictions.npy", help="Path to save predictions")
    parser.add_argument("--NN_type", type=str, default="HybridCNN", choices=["DNN","HybridCNN"])
    args = parser.parse_args()

    load_and_predict(args.checkpoint_path, args.input_data, args.output_data, args.NN_type)

