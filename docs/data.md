# Data

## Encoding

| Object | Array | Convention |
| --- | --- | --- |
| Surface pattern | 20 × 20, flattened row-major to 400 values | `0` strongly interacting site (A<sub>s</sub>), `1` weakly interacting site (B<sub>s</sub>) |
| Polymer sequence | 40 values | `0` strongly interacting bead (A<sub>p</sub>), `1` weakly interacting bead (B<sub>p</sub>) |
| Model input `X` | 440 = 400 (surface) + 40 (polymer) | |
| PMF `Y` | 100 values, z<sub>cm</sub> = 4.73-10.37 σ (bin width 0.057 σ), k<sub>B</sub>T | `0` in the unbound state (mean of the last 10 points) |
| ΔF | scalar | mean of the last 10 PMF points minus the minimum of the PMF (positive) |

The surface patterns and polymer sequences are generated with Ising Monte Carlo (coupling J = 0.4, fixed composition, tolerance 10<sup>-4</sup>);
see [`simulation/ising_generator.py`](../simulation/ising_generator.py). Compositions f<sub>A</sub> = 0, 0.1, ..., 1.0; 10 configurations at each
intermediate composition and one at each homogeneous composition, giving 92 surfaces and 92 polymers.

## Folders

```
data/
├── raw/
│   ├── surface_fractions/surface.npy      (92, 20, 20)  all surface patterns (index = surface number)
│   ├── polymer_fractions/polymer.npy      (92, 40)      all polymer sequences
│   └── pmf_datas/pmf_data.npy             (92, 92, 100) PMF of surface i with polymer j (WHAM output)
├── interim/        xdata.npy (8464, 440), ydata.npy (8464, 100): the full 92 × 92 library, unsplit, raw PMF
└── splits/         tensors used for training and evaluation (below)
```

`raw/pmf_datas/surface_<i>/pmf_out_*.xvg` are the individual WHAM profiles; `raw/pmf_datas/pmf_data.npy` is their collection.
The simulation inputs are in [`simulation/`](../simulation/README.md).

## Splits

The all-weak surface and the all-weak polymer (index 0) are not used for learning, which leaves 91 × 91 = 8281 pairs. The pairs are
reduced to canonical form (surface: lexicographically smallest of the 8 rotations and reflections of the 20 × 20 array; polymer:
smaller of the sequence and its reverse), shuffled with seed 11, and split 70 / 20 / 10 %.

| File | Shape | Content |
| --- | --- | --- |
| `fdX_{train,valid,test}.npy` | (5797 / 1656 / 828, 440) | forward-model inputs |
| `fdY_{train,valid,test}.npy` | (5797 / 1656 / 828, 100) | PMF targets |
| `bxX_{train,valid,test}.npy` | (n, 41) | inputs of the back-extension network: target ΔF followed by the 40-bead polymer |
| `delF.npy` | (8281,) | ΔF of every pair, in the order of the canonical dataset |

The split is random over pairs. All 91 polymers and all 91 surfaces occur in the training set; see "Scope and limitations" in the
main README.

## Regenerating

```bash
python -m mlsurfpoly.cli.prepare_data                       # data/splits, as published
python -m mlsurfpoly.cli.prepare_data --symmetry rot        # canonical form over rotations only
python -m mlsurfpoly.cli.prepare_data --pmf-reference raw   # keep the WHAM offset
```

`raw/` should be treated as immutable; everything else can be rebuilt from it.
