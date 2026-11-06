#!/usr/bin/env python3
"""Build the train/validation/test tensors from the raw PMF data and write them to data/splits/.

Writes fdX_*/fdY_* (forward model inputs and PMF targets), bxX_* (back-extension inputs: target dF followed by the
polymer sequence) and delF.npy.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from ..data import prepare_model_data
from ..paths import SPLITS


def main():
    parser = argparse.ArgumentParser(description="Create the dataset splits used for training")
    parser.add_argument("--neural-tag", default="canonical", choices=["canonical", "augmented", "mixed"])
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--symmetry", default="d4", choices=["d4", "rot"], help="surface canonicalisation: rotations+reflections or rotations only")
    parser.add_argument("--pmf-reference", default="plateau", choices=["plateau", "raw"], help="reference level of the PMF")
    parser.add_argument("--out-dir", type=Path, default=SPLITS)
    args = parser.parse_args()

    fdX_train, fdY_train, fdX_valid, fdY_valid, fdX_test, fdY_test, delF = prepare_model_data(
        neural_tag=args.neural_tag, random_state=args.seed, symmetry=args.symmetry, pmf_reference=args.pmf_reference
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name, arr in {
        "fdX_train": fdX_train, "fdY_train": fdY_train, "fdX_valid": fdX_valid,
        "fdY_valid": fdY_valid, "fdX_test": fdX_test, "fdY_test": fdY_test, "delF": delF,
    }.items():
        np.save(args.out_dir / f"{name}.npy", arr)

    def backext_inputs(x, y):
        delta_f = y[:, 90:].mean(axis=1, keepdims=True) - y.min(axis=1, keepdims=True)
        return np.hstack([delta_f, x[:, 400:]])

    for split, (x, y) in {"train": (fdX_train, fdY_train), "valid": (fdX_valid, fdY_valid), "test": (fdX_test, fdY_test)}.items():
        np.save(args.out_dir / f"bxX_{split}.npy", backext_inputs(x, y))
    print(f"Wrote splits to {args.out_dir}")


if __name__ == "__main__":
    main()
