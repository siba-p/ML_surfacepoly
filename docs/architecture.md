# Model architecture

All networks are defined in [`mlsurfpoly/models/`](../mlsurfpoly/models) (Keras 3, TensorFlow backend).

## Forward model: `build_HybridCNN_forward_model`

```
input (440) ─┬─ surface (400 → 20×20×1) → Conv(32) → MaxPool → Conv(64) → MaxPool → Conv(128) → Flatten → Dense(64)
             │                             └─ the 64-d embedding is repeated for each of the 40 bead positions
             └─ polymer (40 → 40×1) → GRU(64, return sequences) → LayerNorm
cross-attention (2 heads, key dim 64): query = polymer states, key/value = surface embedding
→ residual + LayerNorm + feed-forward (64) + residual → Flatten → Dropout(0.2) → Dense(250) → Dense(140) → Dense(100)  (PMF)
```

Loss: mean absolute error; optimiser Nadam (learning rate 1e-4, β = 0.97). The baseline `build_DNN_forward_model` is a fully
connected network with layers 368 / 248 / 141.

## Inverse design

* **Backward (inverse) network**: PMF (100) → dense layers → 5×5×32 → two up-sampling / convolution stages → 20×20 surface logits →
  `AnnealedSmoothBinary`, a tanh activation whose sharpness `m` is increased during training so that the output approaches binary values.
* **Tandem model**: the inverse network is followed by the frozen forward model, so that the loss is evaluated on the PMF of the
  *generated* surface (reconstruction loss) as well as on the surface itself (weighted binary cross-entropy).
* **Back-extension network** (`build_backext_model`): polymer sequence and target ΔF (41 inputs) → a full PMF (100 points).
* **Extended tandem model** (`build_extended_tandem_model`): backext → inverse network → frozen forward network. Given a polymer and
  a target ΔF it designs a surface. Training combines the losses with weights 0.45 (PMF from backext), 2 (ΔF), 2 (surface, focal
  cross-entropy) and 0.1 (PMF reconstruction); see `mlsurfpoly/training/backward.py`.

```
 polymer + target ΔF ──► backext ──► PMF ──► inverse net ──► surface ──► forward net (frozen) ──► PMF′  ≈ PMF
```

## Pretrained models

| File | Description |
| --- | --- |
| `models/pretrained/forward/HybridCNN/canonical_forward_model.keras` | CNN-GRU-attention forward model |
| `models/pretrained/forward/DNN/canonical_forward_model.keras` | fully connected baseline |
| `models/pretrained/inverse/HybridCNN/canonical_tandem_model.keras` | inverse network + frozen forward model |
| `models/pretrained/inverse/HybridCNN/canonical_backward_tandem_model.keras` | inverse network (tandem training stage) |
| `models/pretrained/inverse/HybridCNN/canonical_extend_model.keras` | back-extension + inverse + forward |
| `models/pretrained/inverse/HybridCNN/canonical_tandem_and_extend_model.keras` | extended tandem model after joint training |
| `models/pretrained/inverse/DNN/canonical_tandem_model.keras` | tandem model with a dense inverse network |

The models contain `Lambda` layers; load them with the helper functions exported by `mlsurfpoly.models`
(`slice_surface_fn`, `slice_polymer_fn`, `expand_surface_sequence_fn`) as `custom_objects`, as done in `mlsurfpoly/cli/predict_forward.py`.
