# src Package


## Layout

| Module | Purpose |
| --- | --- |
| `callbacks/`  
| `models/` | Forward, backward, tandem, and back-extension architectures. |
| `losses/` | Helper losses. |
| `training/`
| `cli/` | Command-line utilities |
| `checkpoints/`

## Running the CLIs

Execute every command as a module under `src`, e.g.:

```bash
python -m src.cli.train_forward \
  --neural-tag canonical \
  --model hybrid \
  --epochs 2000 \
  --checkpoint src/checkpoint/forward/HybridCNN/canonical_forward_model.keras
```

Backward and extended training expect explicit `.npy` files so you can plug in any split:

```bash
python -m src.cli.train_backward \
  --forward-checkpoint src/checkpoint/forward/HybridCNN/canonical_forward_model.keras \
  --train-seq data/splits/fdX_train.npy \
  --train-pmf data/splits/fdY_train.npy \
  --valid-seq data/splits/fdX_valid.npy \
  --valid-pmf data/splits/fdY_valid.npy
```

For the tandem + extension pipeline provide the three tensors described in the notebook (sequence, pmf, auxiliary/back-extension features):

```bash
python -m src.cli.train_extended \
  --forward-checkpoint src/checkpoint/forward/HybridCNN/canonical_forward_model.keras \
  --sequence-train data/splits/fdX_train.npy \
  --pmf-train data/splits/fdY_train.npy \
  --extra-train data/splits/bxX_train.npy \
  --sequence-valid data/splits/fdX_valid.npy \
  --pmf-valid data/splits/fdY_valid.npy \
  --extra-valid data/splits/bxX_valid.npy
```

Inference uses the matching module:

```bash
python -m src.cli.predict_forward --checkpoint src/checkpoint/... --inputs data/splits/fdX_test.npy
```

