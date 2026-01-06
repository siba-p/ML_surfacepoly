# Data Catalog

This folder stores every artifact needed to train and evaluate the polymer--surface forward model and extendem neural network for inverse design.

## Directory overview

| Path | Description |
| --- | --- |
| raw/ | Original simulation-derived assets: `pmf_datas/` contains WHAM/US PMF profiles, `polymer_fractions/` and `surface_fractions/` hold Monte Carlo-generated compositions used to seed the datasets. |
| processed/ | Canonicalized numpy arrays produced after preprocessing and augmentation. These are the inputs/targets most scripts expect by default. |
| splits/ | Train/validation/test splits exported for quick experimentation (mirrors the naming in `processed/`). |
| simulations/ | Side data for running or reproducing MD simulations (example setups and reusable input decks). |

## Key numpy blobs

- `processed/fdX_*.npy`: one-hot encoded polymer–surface grids used as model inputs (X). Each file corresponds to a split (train/valid/test).
- `processed/fdY_*.npy`: PMF profiles (Y) aligned with the respective `fdX_*` arrays.
- `processed/xdata.npy` and `processed/ydata.npy`: unsplit canonical datasets before applying augmentations or fold assignments.
- `processed/delF.npy`: scalar work of adhesion (ΔF) derived from `fdY_*`, used for auxiliary regression heads or ranking.
- `splits/...`: mirrored copies of the canonical splits, useful when you want to experiment without overwriting the default `processed/` tensors.

## Re-creating the processed data

1. From the repository root, run the model-data preparation pipeline:

   ```bash
   python -m scripts.save_data
   ```

   - `scripts/prepare_model_data.py` performs canonical loading, augmentation, and deterministic splitting. Use the `neural_tag` argument (`"canonical"`, `"augmented"`, or `"mixed"`) to control what gets returned.
   - `scripts/save_data.py` shows how to persist the arrays to `data/processed/`; uncomment any `np.save` lines you need or adapt the module if you want to store alternative variants (e.g., into `data/splits/`).

2. To write a different variant (say the augmented dataset) you can launch an inline helper:

   ```bash
   python - <<'PY'
   import numpy as np
   from scripts.prepare_model_data import prepare_model_data

   fdX_train, fdY_train, fdX_valid, fdY_valid, fdX_test, fdY_test, delF = (
       prepare_model_data(neural_tag="augmented", random_state=11)
   )

   np.save("data/splits/fdX_train.npy", fdX_train)
   np.save("data/splits/fdY_train.npy", fdY_train)
   np.save("data/splits/fdX_valid.npy", fdX_valid)
   np.save("data/splits/fdY_valid.npy", fdY_valid)
   np.save("data/splits/fdX_test.npy", fdX_test)
   np.save("data/splits/fdY_test.npy", fdY_test)
   np.save("data/splits/delF.npy", delF)
   PY
   ```

3. Downstream scripts such as `scripts/prepare_model_data.py`, `scripts/preprocessing.py`, and `scripts/pmf_analysis.py` document the preprocessing stages (canonicalization, augmentation, PMF feature extraction). Refer to them when extending the dataset or regenerating the raw assets.

## Notes

- Keep raw files immutable so regenerated datasets remain reproducible.
- Whenever you change augmentation settings, re-run the pipeline so that `processed/` and `splits/` stay in sync.
- Large `.npy` blobs are not tracked in git by default; consider using Git LFS if you need to share alternate versions.
