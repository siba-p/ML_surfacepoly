"""Training helpers for forward prediction models."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
from tensorflow import keras


def train_forward_model(
    model: keras.Model,
    train_inputs: np.ndarray,
    train_targets: np.ndarray,
    valid_inputs: np.ndarray,
    valid_targets: np.ndarray,
    *,
    epochs: int,
    batch_size: int = 64,
    callbacks: Iterable[keras.callbacks.Callback] | None = None,
    checkpoint_path: str | Path | None = None,
):
    callbacks = list(callbacks or [])
    if checkpoint_path:
        callbacks.append(
            keras.callbacks.ModelCheckpoint(
                filepath=str(checkpoint_path),
                save_best_only=True,
                monitor="val_mae",
                mode="min",
            )
        )

    history = model.fit(
        train_inputs,
        train_targets,
        validation_data=(valid_inputs, valid_targets),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=0,
    )
    return history
