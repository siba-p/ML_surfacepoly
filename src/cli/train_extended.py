#!/usr/bin/env python3
"""Train the tandem + extension inverse design pipeline."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from tensorflow import keras

from ..callbacks import LogsCallback, UpdateSharpnessCallback, AnnealedSmoothBinary
from ..models import (
    build_DNN_backward_model,
    build_HybridCNN_backward_model,
    build_backext_model,
    build_extended_tandem_model,
    expand_surface_sequence_fn,
    slice_surface_fn,
    slice_polymer_fn,
)
from ..training import train_tandem_and_extend_model


def _load_forward(path: Path):
    custom = {
        "slice_surface_fn": slice_surface_fn,
        "slice_polymer_fn": slice_polymer_fn,
        "expand_surface_sequence_fn": expand_surface_sequence_fn,
    }
    return keras.models.load_model(path, custom_objects=custom)


def _maybe_load_weights(model: keras.Model, weights: Path | None):
    if weights is not None:
        model.load_weights(weights)
    return model


def main():
    parser = argparse.ArgumentParser(description="Train the extended tandem pipeline")
    parser.add_argument("--forward-checkpoint", type=Path, required=True)
    parser.add_argument("--model", default="hybrid", choices=["dnn", "hybrid"], help="Backward architecture")
    parser.add_argument("--sequence-train", type=Path, required=True)
    parser.add_argument("--pmf-train", type=Path, required=True)
    parser.add_argument("--extra-train", type=Path, required=True, help="Back-extension inputs")
    parser.add_argument("--sequence-valid", type=Path, required=True)
    parser.add_argument("--pmf-valid", type=Path, required=True)
    parser.add_argument("--extra-valid", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--backward-weights", type=Path)
    parser.add_argument("--backext-weights", type=Path)
    parser.add_argument("--checkpoint", type=Path, default=Path("models/checkpoint/extended_model.keras"))
    parser.add_argument("--history-out", type=Path, default=Path("models/history/extended_history.npy"))
    args = parser.parse_args()

    forward_model = _load_forward(args.forward_checkpoint)
    backward_model = build_DNN_backward_model() if args.model == "dnn" else build_HybridCNN_backward_model()
    backext_model = build_backext_model()

    _maybe_load_weights(backward_model, args.backward_weights)
    _maybe_load_weights(backext_model, args.backext_weights)

    extend_model = build_extended_tandem_model(backext_model, backward_model, forward_model)

    seq_train = np.load(args.sequence_train)
    pmf_train = np.load(args.pmf_train)
    extra_train = np.load(args.extra_train)
    seq_valid = np.load(args.sequence_valid)
    pmf_valid = np.load(args.pmf_valid)
    extra_valid = np.load(args.extra_valid)

    callbacks = [
        LogsCallback(skip_epochs=50, monitor=["val_loss", "val_surface_bce", "val_pmf_recon_mae"]),
        UpdateSharpnessCallback(AnnealedSmoothBinary, verbose=0),
    ]

    history = train_tandem_and_extend_model(
        extend_model,
        backext_model,
        backward_model,
        forward_model,
        sequence_train_data=seq_train,
        pmf_train_data=pmf_train,
        extra_train_data=extra_train,
        sequence_valid_data=seq_valid,
        pmf_valid_data=pmf_valid,
        extra_valid_data=extra_valid,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        callbacks_list=callbacks,
    )

    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    extend_model.save(args.checkpoint)
    args.history_out.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.history_out, history.to_dict())


if __name__ == "__main__":
    main()
