#!/usr/bin/env python3
"""Minimal inference helper for the forward model."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from tensorflow import keras

from ..models import expand_surface_sequence_fn, slice_polymer_fn, slice_surface_fn


def main():
    parser = argparse.ArgumentParser(description="Run forward-model inference on .npy tensors")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--outputs", type=Path, default=Path("models/predictions.npy"))
    args = parser.parse_args()

    custom = {
        "slice_surface_fn": slice_surface_fn,
        "slice_polymer_fn": slice_polymer_fn,
        "expand_surface_sequence_fn": expand_surface_sequence_fn,
    }
    model = keras.models.load_model(args.checkpoint, custom_objects=custom)
    inputs = np.load(args.inputs)
    preds = model.predict(inputs, verbose=1)
    args.outputs.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.outputs, preds)


if __name__ == "__main__":
    main()
