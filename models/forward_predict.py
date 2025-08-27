#!/usr/bin/env python3
"""
Load pre-trained forward model
"""
import os
import time
import absl.logging
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras import backend as K
from keras.initializers import Constant
from keras.utils import plot_model
from sklearn.model_selection import train_test_split

absl.logging.set_verbosity(absl.logging.INFO)

from keras.models import Sequential, Model
from keras import Input, layers, optimizers, regularizers
from keras.layers import (
    Dense, Dropout, Flatten, Reshape, Lambda,
    BatchNormalization, LayerNormalization, GlobalAveragePooling1D,
    Conv2D, Conv2DTranspose, MaxPooling2D,
    GRU, LSTM, Add, Concatenate, 
    LeakyReLU, ELU, PReLU, MultiHeadAttention
)
from  models.callbacks import CustomCallback, LogsCallbackflex, CustomCallback
from models.forward_model import build_HybridCNN_forward_model, slice_surface_fn, slice_polymer_fn, expand_surface_sequence_fn

use_gpu = input("Do you want to use GPU if available? [y/n]: ").strip().lower() == 'y'

if use_gpu:
    gpus = tf.config.experimental.list_physical_devices("GPU")
    if gpus:
        try:
            tf.config.experimental.set_virtual_device_configuration(
                gpus[0],
                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=15000)]
            )
            print(f"GPU enabled: {gpus[0]}")
        except RuntimeError as e:
            print("GPU configuration failed:", e)
    else:
        print("No GPU found, running on CPU.")
else:
    # Force CPU
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    print("GPU disabled, running on CPU.")

import os
import argparse
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression

from tqdm import tqdm

# ----------------------
# Evaluation Function
# ----------------------
def evaluate_forward_model_performance(y_true, y_pred, y_true_train=None, y_pred_train=None,
                                       plot_scatter=True, compute_sliding=False,
                                       window_size=10, num_scatter_points=100,
                                       save_path=None, use_delf=False):
    import matplotlib.pyplot as plt
    import numpy as np
    from sklearn.metrics import mean_absolute_error, r2_score
    from sklearn.linear_model import LinearRegression

    result = {}
    if use_delf:
        y_true_flat = np.mean(y_true[:, 90:], axis=1) - np.min(y_true, axis=1)
        y_pred_flat = np.mean(y_pred[:, 90:], axis=1) - np.min(y_pred, axis=1)
        result['r2_test'] = r2_score(y_true_flat, y_pred_flat)
        result['mae_test'] = mean_absolute_error(y_true_flat, y_pred_flat)
        if y_true_train is not None:
            y_true_train_flat = np.mean(y_true_train[:, 90:], axis=1) - np.min(y_true_train, axis=1)
            y_pred_train_flat = np.mean(y_pred_train[:, 90:], axis=1) - np.min(y_pred_train, axis=1)
            result['r2_train'] = r2_score(y_true_train_flat, y_pred_train_flat)
            result['mae_train'] = mean_absolute_error(y_true_train_flat, y_pred_train_flat)
        else:
            result['r2_train'] = result['mae_train'] = None
    else:
        y_true_flat = y_true
        y_pred_flat = y_pred
        result['r2_test'] = r2_score(y_true_flat, y_pred_flat)
        result['mae_test'] = mean_absolute_error(y_true_flat, y_pred_flat)
        if y_true_train is not None:
            result['r2_train'] = r2_score(y_true_train.flatten(), y_pred_train.flatten())
            result['mae_train'] = mean_absolute_error(y_true_train.flatten(), y_pred_train.flatten())
        else:
            result['r2_train'] = result['mae_train'] = None

    # Optional scatter plot
    if plot_scatter:
        idx = np.random.choice(len(y_true_flat), size=num_scatter_points, replace=False)
        x_sample = y_true_flat[idx]
        y_sample = y_pred_flat[idx]
        reg = LinearRegression().fit(x_sample.reshape(-1,1), y_sample)
        y_fit = reg.predict(np.sort(x_sample).reshape(-1,1))
        plt.figure(figsize=(4,4))
        plt.scatter(x_sample, y_sample, color='red', label="Test")
        if y_true_train is not None:
            idx_train = np.random.choice(len(y_true_train_flat), size=num_scatter_points, replace=False)
            plt.scatter(y_true_train_flat[idx_train], y_pred_train_flat[idx_train], color='blue', label="Train")
        plt.plot(np.sort(x_sample), y_fit, color='green', label='Fit')
        plt.plot([min(x_sample), max(x_sample)],[min(x_sample), max(x_sample)],'k--')
        plt.xlabel("True ΔF")
        plt.ylabel("Predicted ΔF")
        plt.legend()
        if save_path:
            plt.savefig(save_path,dpi=300,bbox_inches='tight')
        else:
            plt.show()

    return result

def load_and_predict(checkpoint_path, input_data_path, output_path, target_data_path=None, NN_type="HybridCNN"):
    # Load input
    X = np.load(input_data_path)
    print(f"Loaded input: {X.shape}")

    Y_true = np.load(target_data_path) if target_data_path else None

    # Load model
    if NN_type=="HybridCNN":
        custom_objects = {
            "slice_surface_fn": slice_surface_fn,
            "slice_polymer_fn": slice_polymer_fn,
            "expand_surface_sequence_fn": expand_surface_sequence_fn
        }
        model = keras.models.load_model(checkpoint_path, custom_objects=custom_objects)
    else:
        model = keras.models.load_model(checkpoint_path)

    print(f"Loaded model from {checkpoint_path}")

    # Predict all at once
    predictions = model.predict(X, verbose=1)

    # Save predictions
    if output_path:
        np.save(output_path, predictions)
        print(f"Predictions saved to {output_path}")

    # Compute MAE if target available
    if Y_true is not None:
#        metrics = evaluate_forward_model_performance(Y_true, predictions, plot_scatter=False,use_delf=True)
#        print("Test MAE: {:.4f}, R²: {:.4f}".format(metrics['mae_test'], metrics['r2_test']))
        mae = mean_absolute_error(Y_true.flatten(), predictions.flatten())
        print(f"MAE: {mae:.4f}")

    return predictions
# ----------------------
# CLI
# ----------------------
if __name__=="__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Load trained forward model and predict PMF")
    parser.add_argument("--checkpoint_path", type=str, default="models/checkpoint/HybridCNN/canonical_forward_model.keras", help="Path to trained .keras checkpoint")
    parser.add_argument("--input_data", type=str, default="data/processed/fdX_train.npy", help="Path to input .npy file")
    parser.add_argument("--output_data", default="models/predictions.npy")
    parser.add_argument("--target_data", type=str, default="data/processed/fdY_train.npy", help="Path to input .npy file")
    parser.add_argument("--NN_type", default="HybridCNN", choices=["DNN","HybridCNN"])
    args = parser.parse_args()
    load_and_predict(args.checkpoint_path, args.input_data, args.output_data,
                     args.target_data, args.NN_type)
