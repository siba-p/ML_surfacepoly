# Simulation inputs and workflow

This directory contains the tooling that turns configurations into GROMACS-ready surface patches, polymer chains, and combined simulation boxes.

The workflow combines:
Ising model–based pattern generation (for surface heterogeneity and polymer sequences)
Deterministic geometry construction (GROMACS‐compatible .gro and .itp files)
Packing surface–polymer in to a box 

## Directory Map


| --- | --- |
| [simulation/ising_generator.py](ising_generator.py) | Ising Monte Carlo generator for binary surfaces (2D lattices) and polymer sequences (1D vectors). |
| [simulation/build_surface.py](build_surface.py) | Converts the saved surface lattices into HCP lattice plus topology files, generating `surface_XXX.(gro|itp)` along with `surface_dim.npy`, `box_dim.npy`, and `posresSurf.itp`. |
| [simulation/build_polymer.py](build_polymer.py) | Builds Kremer-Grest-style polymers (`polymer_XXX.(gro|itp)`) from saved spin sequences. |
| [simulation/build_box.py](build_box.py) | Combines one surface and one polymer into a single `system.gro` with matching topology assets under [simulation/md_systems](md_systems). |
| [simulation/generate](generate) | Pipeline CLI that chains the three build stages or executes them individually; logs to [simulation/pipeline.log](pipeline.log). |
| [simulation/pipeline.yaml](pipeline.yaml) | YAML configuration by the pipeline CLI when running the full workflow. |
| [batch/master_script.sh](batch/master_script.sh) | Shell wrapper that expands the generated inputs into one folder per surface-polymer pair for umbrella sampling (legacy batch script, edit paths before use). |
| [simulation/topology](topology) | Stores (`*.npy`) and the GROMACS-input files. Surface and polymer fractions (e.g., `surface_fraction_0.6`). |
| [simulation/md_systems](md_systems) | Final gromacs input files for combined systems (`sXXX_pYYY`). |


## Typical Workflow

1. **Pattern Generation.**
The Ising generator produces binary patterns (±1) representing:
    - Surface chemistry patterns (2D lattice)
    - Polymer sequences (1D chain)
  Surface generation 
  ```bash
  cd simulation
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

Edit [simulation/pipeline.yaml](pipeline.yaml) to change global defaults (Python interpreter, number of systems, target IDs, etc.). All stdout/stderr from the CLI is appended to [simulation/pipeline.log](pipeline.log) for reproducibility.

## Data Layout

- **Surface files:** `topology/surface_fraction_X.Y/` contains raw `surface_XXX.npy` lattices alongside the `gromacs/` folder with `.gro`, `.itp`, `surface_dim.npy`, `box_dim.npy`, and shared `posresSurf.itp` restraint files.
- **Polymer files:** `topology/polymer_fraction_0.6/` mirrors the same pattern with lattice sequences and a `gromacs/` directory holding `.gro`/`.itp` pairs.
- **Combined systems:** `md_systems/` stores folders like `s001_p001/` containing fully paired coordinates and topologies.

## Tips & Troubleshooting

- The Ising generator enforces the requested fraction exactly at initialization, then relies on Metropolis moves; if convergence stalls, increase `--mc_steps` or loosen `tol` inside [simulation/ising_generator.py](ising_generator.py).
- `build_surface.py` writes shared metadata (`surface_dim.npy`, `box_dim.npy`) once per fraction; delete them if you need a fresh run with modified lattice dimensions.
- `build_polymer.py` currently assumes `polymer_fraction_0.6`; duplicate the directory if other fractions are needed and adjust `base_dir` at the bottom of the script.
- `build_box.py` checks for the presence of all required `.gro/.itp/.npy` files before writing; missing assets trigger a descriptive `FileNotFoundError`.
- Keep the numeric parts of `surface_id` and `polymer_id` zero-padded (e.g., `001`) so the filesystem ordering matches the dataset indices.

## Umbrella sampling and WHAM

* `gromacs/mdp/eqb.mdp`, `gromacs/mdp/prod.mdp`: equilibration and production parameters (stochastic dynamics, Δt = 0.005 τ, 4 × 10⁶ production steps, reduced temperature k<sub>B</sub>T/ε = 1, potential-shifted LJ cut off at 2.5 σ).
* `gromacs/example/`: one complete worked system (`s1p1`): topology with the bead-pair interaction table, position and flat-bottom restraints, and `pmf.sh`, which launches the 58 umbrella windows (z<sub>cm</sub> restraint centres 4.7-10.4 σ, step 0.1 σ, κ = 500 ε/σ²) through PLUMED.
* `wham/`: scripts that write the WHAM metadata files and collect the profiles into `data/raw/pmf_datas`.

Units are reduced Lennard-Jones units; in the GROMACS files σ = 1 nm, ε = 1 kJ/mol, m = 1 u (τ = 1 ps).
