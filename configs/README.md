# Surface-Polymer Configuration

This directory contains the tooling that turns configurations into GROMACS-ready surface patches, polymer chains, and combined simulation boxes.

The workflow combines:
Ising model–based pattern generation (for surface heterogeneity and polymer sequences)
Deterministic geometry construction (GROMACS‐compatible .gro and .itp files)
Packing surface–polymer in to a box 

## Directory Map


| --- | --- |
| [configs/ising_generator.py](configs/ising_generator.py) | Ising Monte Carlo generator for binary surfaces (2D lattices) and polymer sequences (1D vectors). |
| [configs/build_surface.py](configs/build_surface.py) | Converts the saved surface lattices into HCP lattice plus topology files, generating `surface_XXX.(gro|itp)` along with `surface_dim.npy`, `box_dim.npy`, and `posresSurf.itp`. |
| [configs/build_polymer.py](configs/build_polymer.py) | Builds Kremer-Grest-style polymers (`polymer_XXX.(gro|itp)`) from saved spin sequences. |
| [configs/build_box.py](configs/build_box.py) | Combines one surface and one polymer into a single `system.gro` with matching topology assets under [configs/md_systems](configs/md_systems). |
| [configs/generate](configs/generate) | Pipeline CLI that chains the three build stages or executes them individually; logs to [configs/pipeline.log](configs/pipeline.log). |
| [configs/pipeline.yaml](configs/pipeline.yaml) | YAML configuration by the pipeline CLI when running the full workflow. |
| [configs/master_script.sh](configs/master_script.sh) | Shell wrapper for batch runs. |
| [configs/topology](configs/topology) | Stores (`*.npy`) and the GROMACS-input files. Surface and polymer fractions (e.g., `surface_fraction_0.6`). |
| [configs/polymer](configs/polymer) | Scratch space for additional polymer|
| [configs/md_systems](configs/md_systems) | Final gromacs input files for combined systems (`sXXX_pYYY`). |


## Typical Workflow

1. **Pattern Generation.**
The Ising generator produces binary patterns (±1) representing:
    - Surface chemistry patterns (2D lattice)
    - Polymer sequences (1D chain)
  Surface generation 
  ```bash
  cd config
  python ising_generator.py --mode surface --fraction 0.6 --n_systems 10 --out_dir topology/surface_fraction_0.6
  ```
  Polymer generation
  ```bash
  python ising_generator.py --mode polymer --fraction 0.6 --n_systems 10 --out_dir topology/polymer_fraction_0.6
  ```
  Outputs are stored as .npy files inside the corresponding topology/ subdirectories.
2. **Convert to GROMACS-input files.**
   ```bash
   python build_surface.py
   python build_polymer.py
   ```
   These commands generates every `topology/surface_fraction_*` or the single `topology/polymer_fraction_0.6` directory, creating `.gro`, `.itp`, and helper `.npy` files inside each `gromacs/` folder.
3. **Assemble simulation boxes.**
   ```bash
   python build_box.py \
     --surface_dir topology/surface_fraction_0.6 \
     --polymer_dir topology/polymer_fraction_0.6 \
     --surface_id 001 \
     --polymer_id 001 \
     --output_root md_systems
   ```
   Each run creates `md_systems/sXXX_pYYY` with `system.gro`, `surface.itp`, `polymer.itp`, and `posresSurf.itp` ready for downstream GROMACS MD engines.

## Using the Pipeline CLI

Run the orchestrator from this directory for a fully automated pass:

```bash
python generate run --config pipeline.yaml
```

Key commands:

- `python generate surface --fraction 0.6 --n 50` &rarr; calls the Ising generator and then `build_surface.py`.
- `python generate polymer --fraction 0.6 --n 10` &rarr; spawns polymer sequences and builds their topologies.
- `python generate box --surface_fraction 0.6 --polymer_fraction 0.6 --surface_id 001 --polymer_id 010` &rarr; synthesizes a single combined system.

Edit [configs/pipeline.yaml](configs/pipeline.yaml) to change global defaults (Python interpreter, number of systems, target IDs, etc.). All stdout/stderr from the CLI is appended to [configs/pipeline.log](configs/pipeline.log) for reproducibility.

## Data Layout

- **Surface files:** `topology/surface_fraction_X.Y/` contains raw `surface_XXX.npy` lattices alongside the `gromacs/` folder with `.gro`, `.itp`, `surface_dim.npy`, `box_dim.npy`, and shared `posresSurf.itp` restraint files.
- **Polymer files:** `topology/polymer_fraction_0.6/` mirrors the same pattern with lattice sequences and a `gromacs/` directory holding `.gro`/`.itp` pairs.
- **Combined systems:** `md_systems/` stores folders like `s001_p001/` containing fully paired coordinates and topologies.

## Tips & Troubleshooting

- The Ising generator enforces the requested fraction exactly at initialization, then relies on Metropolis moves; if convergence stalls, increase `--mc_steps` or loosen `tol` inside [configs/ising_generator.py](configs/ising_generator.py).
- `build_surface.py` writes shared metadata (`surface_dim.npy`, `box_dim.npy`) once per fraction; delete them if you need a fresh run with modified lattice dimensions.
- `build_polymer.py` currently assumes `polymer_fraction_0.6`; duplicate the directory if other fractions are needed and adjust `base_dir` at the bottom of the script.
- `build_box.py` checks for the presence of all required `.gro/.itp/.npy` files before writing; missing assets trigger a descriptive `FileNotFoundError`.
- Keep the numeric parts of `surface_id` and `polymer_id` zero-padded (e.g., `001`) so the filesystem ordering matches the dataset indices.
