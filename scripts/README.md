M## Scripts Overview

This folder hosts every utility needed to turn raw polymer/surface simulations into the tidy tensors consumed by the hybrid CNN forward model, plus diagnostics for understanding the PMF statistics.

| Script | Purpose |
| --- | --- |
| `prepare_canonical_data.py` | Loads Monte Carlo–generated polymers/surfaces and PMF grids, returning canonical numpy arrays. |
| `preprocessing.py` | Implements augmentation routines (surface/polymers flips, rotations, etc.) and helper transforms. |
| `prepare_data.py` | Orchestrates canonical loading + augmentation controls (flags for reshape, polymer/surface augmentation). |
| `pre_data.py` | Provides utilities for building compound inputs (e.g., concatenating ΔF with polymer descriptors) and split helpers. |
| `pmf_analysis.py` | Extracts physics-aware metrics from PMF profiles (ΔF, curvature, gradients, regions of interest). |
| `prepare_model_data.py` | High-level entry point that produces train/valid/test tensors for canonical, augmented, or mixed datasets. |
| `save_data.py` | Example CLI that calls `prepare_model_data` and persists the returned arrays into `data/processed/`. |
| `plotting_utils.py` | Visualization helpers for composition histograms, t-SNE embeddings, PMF overlays, and metric curves. |

## Typical Workflow

1. **Canonical load** – `prepare_canonical_data.py` reads raw numpy assets from `data/raw/` and returns `xdata` / `ydata` tensors along with associated metadata.
2. **Augmentation & reshaping** – `prepare_data.py` calls into `preprocessing.py` to apply the requested augmentations (surface/polymers) and handles any reshape logic required by downstream models.
3. **Feature extraction** – `pmf_analysis.py` computes ΔF plus other scalar descriptors that can be joined with structure encodings through `pre_data.py` utilities.
4. **Dataset assembly** – `prepare_model_data.py` bundles canonical and augmented variants, performs deterministic splits, and exposes the tensors to training scripts.
5. **Persistence** – `save_data.py` (or a custom adapter) writes the numpy arrays into `data/processed/` or `data/splits/` so training runs are reproducible.

## Quick Start

From the repository root:

```bash
python -m scripts.save_data              # writes canonical tensors into data/processed/
python -m scripts.prepare_model_data     # prints shapes for inspection; pass neural_tag
```

Use the `neural_tag` argument (`canonical`, `augmented`, `mixed`) in `prepare_model_data` to switch among dataset flavors before saving. Update `save_data.py` (or duplicate it) if you need to store multiple variants simultaneously.

## Extending the Pipeline

- Add new augmentation ideas inside `preprocessing.py`, then expose flags in `prepare_data.py`.
- Encode additional PMF statistics in `pmf_analysis.py` to enrich the scalar features saved alongside ΔF.
- Keep plotting helpers in `plotting_utils.py` up to date so exploratory analysis reflects the latest preprocessing steps.
