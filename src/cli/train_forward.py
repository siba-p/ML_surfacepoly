#!/usr/bin/env python3
"""Train the forward (state -> PMF) model from the command line."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import tensorflow as tf

from scripts.prepare_model_data import prepare_model_data
from ..callbacks import LogsCallback
from ..models import build_DNN_forward_model, build_HybridCNN_forward_model
from ..training import train_forward_model


def _configure_device(device: str):
    if device == "cpu":
        tf.config.set_visible_devices([], "GPU")
    elif device == "gpu":
        gpus = tf.config.list_physical_devices("GPU")
        if gpus:
            tf.config.experimental.set_virtual_device_configuration(
                gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=15000)]
            )
    else:
        raise ValueError(f"Unknown device: {device}")


def main():
    parser = argparse.ArgumentParser(description="Train the forward PMF predictor")
    parser.add_argument("--neural-tag", default="canonical", choices=["canonical", "augmented", "mixed"], help="Dataset variant")
    parser.add_argument("--model", default="hybrid", choices=["dnn", "hybrid"], help="Architecture choice")
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="gpu", choices=["cpu", "gpu"])
    parser.add_argument("--checkpoint", type=Path, default=Path("models/checkpoint/HybridCNN/canonical_forward_model.keras"))
    parser.add_argument("--history-out", type=Path, default=Path("models/history/forward_history.npy"))
    args = parser.parse_args()

    _configure_device(args.device)

    fdX_train, fdY_train, fdX_valid, fdY_valid, _, _, _ = prepare_model_data(neural_tag=args.neural_tag)

    model = build_DNN_forward_model() if args.model == "dnn" else build_HybridCNN_forward_model()
    checkpoint_path = args.checkpoint
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    history = train_forward_model(
        model,
        train_inputs=fdX_train,
        train_targets=fdY_train,
        valid_inputs=fdX_valid,
        valid_targets=fdY_valid,
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=[LogsCallback(skip_epochs=50, monitor=["loss", "val_loss", "val_mae"])],
        checkpoint_path=checkpoint_path,
    )

    args.history_out.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.history_out, history.history)
    model.save(checkpoint_path.with_name(checkpoint_path.stem + "_final.keras"))


if __name__ == "__main__":
    main()
