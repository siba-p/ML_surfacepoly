#!/usr/bin/env python3
"""Train the backward/inverse model using the scripted pipeline."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from tensorflow import keras

from ..callbacks import LogsCallback, UpdateSharpnessCallback, AnnealedSmoothBinary
from ..models import (
    build_DNN_backward_model,
    build_HybridCNN_backward_model,
    expand_surface_sequence_fn,
    slice_surface_fn,
    slice_polymer_fn,
)
from ..training import train_backward_model


def _load_forward_model(path: Path):
    custom_objects = {
        "slice_surface_fn": slice_surface_fn,
        "slice_polymer_fn": slice_polymer_fn,
        "expand_surface_sequence_fn": expand_surface_sequence_fn,
    }
    return keras.models.load_model(path, custom_objects=custom_objects)


def main():
    parser = argparse.ArgumentParser(description="Train the backward (PMF -> state) model")
    parser.add_argument("--forward-checkpoint", type=Path, required=True, help="Path to the trained forward model")
    parser.add_argument("--train-seq", type=Path, required=True, help="Numpy file with training sequences (surface+polymer)")
    parser.add_argument("--train-pmf", type=Path, required=True, help="Numpy file with training PMFs")
    parser.add_argument("--valid-seq", type=Path, required=True)
    parser.add_argument("--valid-pmf", type=Path, required=True)
    parser.add_argument("--model", default="hybrid", choices=["dnn", "hybrid"], help="Backward architecture")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--checkpoint", type=Path, default=Path("models/checkpoint/backward_model.keras"))
    parser.add_argument("--history-out", type=Path, default=Path("models/history/backward_history.npy"))
    args = parser.parse_args()

    forward_model = _load_forward_model(args.forward_checkpoint)
    backward_model = build_DNN_backward_model() if args.model == "dnn" else build_HybridCNN_backward_model()

    train_seq = np.load(args.train_seq)
    train_pmf = np.load(args.train_pmf)
    valid_seq = np.load(args.valid_seq)
    valid_pmf = np.load(args.valid_pmf)

    callbacks = [
        LogsCallback(skip_epochs=25, monitor=["trn_loss", "val_loss", "val_surface_bce"]),
        UpdateSharpnessCallback(AnnealedSmoothBinary, verbose=0),
    ]

    history = train_backward_model(
        backward_model,
        forward_model,
        train_seq=train_seq,
        train_pmf=train_pmf,
        valid_seq=valid_seq,
        valid_pmf=valid_pmf,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        callbacks_list=callbacks,
    )

    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    backward_model.save(args.checkpoint)
    args.history_out.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.history_out, history.to_dict())


if __name__ == "__main__":
    main()
