#!/usr/bin/env python3
"""
Ising-based generator for surfaces and polymers
===============================================

This script generates binary surface patterns (2D) and polymer
sequences (1D) using a Ising Monte Carlo model.


Author: Sibasankar Panigrahy, Alen James

Example:
python3 ising_generator.py \
  --mode surface \
  --fraction 0.6 \
  --n_systems 5 \
  --output_dir topology/surface_fraction_0.6
"""

import numpy as np
import os
import argparse
from tqdm import tqdm


lattice_size = 20
coupling = 0.4
mc_steps = 1_000
tol = 1e-4
seed = 42
polymer_length = 40


def header(args):
    print("\n" + "=" * 72)
    print(" Generation Summary")
    print("=" * 72)
    print(f" Generation mode      : {args.mode}")
    print(f" Target fraction      : {args.fraction}")
    print(f" Number of systems    : {args.n_systems}")
    print(f" Lattice size         : {args.lattice_size} x {args.lattice_size}")
    if args.mode == "polymer":
        print(f" Polymer length       : {args.polymer_length}")
    print(f" Neighborhood         : {args.neighborhood}")
    print(f" MC update scheme     : {args.mc_scheme}")
    print(f" Coupling (ns)        : {args.coupling}")
    print(f" Max MC steps         : {args.mc_steps}")
    print(f" Output directory     : {args.out_dir}")
    print("=" * 72 + "\n")


def init_lattice(nx, ny, frac_plus):
    """
    Initialize a lattice with an exact fraction of +1 spins.
    """
    lattice = -np.ones((nx, ny), dtype=int)
    n_plus = int(round(frac_plus * nx * ny))

    chosen = np.random.choice(nx * ny, n_plus, replace=False)
    for idx in chosen:
        i = idx % nx
        j = idx // nx
        lattice[i, j] = +1

    return lattice

# ============================================================
# ------------------- Neighborhood Methods --------------------------
# ============================================================

def neighbors_neumann(lattice, i, j):
    nx, ny = lattice.shape
    return (
        lattice[(i + 1) % nx, j]
        + lattice[(i - 1) % nx, j]
        + lattice[i, (j + 1) % ny]
        + lattice[i, (j - 1) % ny]
    )


def neighbors_diagonal(lattice, i, j):
    nx, ny = lattice.shape
    return (
        lattice[(i + 1) % nx, (j + 1) % ny]
        + lattice[(i + 1) % nx, (j - 1) % ny]
        + lattice[(i - 1) % nx, (j + 1) % ny]
        + lattice[(i - 1) % nx, (j - 1) % ny]
    )


def neighbors_moore(lattice, i, j):
    return neighbors_neumann(lattice, i, j) + neighbors_diagonal(lattice, i, j)


NEIGHBOR_METHODS = {
    "neumann": neighbors_neumann,
    "diagonal": neighbors_diagonal,
    "moore": neighbors_moore,
}


def metropolis_accept(lattice, i, j, neighbor_fn, coupling):
    spin = lattice[i, j]
    del_spin = -2 * spin
    nsum = neighbor_fn(lattice, i, j)
    del_energy = -nsum * del_spin

    if del_energy <= 0.0:
        return True
    if np.random.rand() < np.exp(-coupling * del_energy):
        return True
    return False


def mc_single_flip(lattice, neighbor_fn, coupling):
    nx, ny = lattice.shape
    i = np.random.randint(0, nx)
    j = np.random.randint(0, ny)

    if metropolis_accept(lattice, i, j, neighbor_fn, coupling):
        lattice[i, j] *= -1


def mc_full_sweep(lattice, neighbor_fn, coupling):
    nx, ny = lattice.shape
    updated = lattice.copy()
    for i in range(nx):
        for j in range(ny):
            if metropolis_accept(lattice, i, j, neighbor_fn, coupling):
                updated[i, j] *= -1
    lattice[:, :] = updated


def frac_of_plus(lattice):
    return np.sum(lattice == +1) / lattice.size


def surface(lattice):
    return lattice.copy()


def polymer(lattice, polymer_length):
    flat = lattice.flatten()
    ind = np.random.choice(len(flat), size=polymer_length, replace=False)
    return flat[ind]

# ============================================================
# ------------------- Generator ------------------------------
# ============================================================

def generate_systems(
    mode,
    frac,
    n_systems,
    lattice_size,
    polymer_length,
    neighborhood,
    mc_scheme,
    coupling,
    mc_steps,
    tol,
    out_dir,
):

    nx = ny = lattice_size
    neighbor_fn = NEIGHBOR_METHODS[neighborhood]
    os.makedirs(out_dir, exist_ok=True)

    generated = 0
    idx = 0

    while generated < n_systems:

        lattice = init_lattice(nx, ny, frac)

        with tqdm(
            total=mc_steps,
            desc=f"{mode.capitalize()} {generated+1}",
            ncols=110,
            leave=True
        ) as pbar:

            for step in range(mc_steps):

                if mc_scheme == "single":
                    mc_single_flip(lattice, neighbor_fn, coupling)
                else:
                    mc_full_sweep(lattice, neighbor_fn, coupling)

                curr_frac = frac_of_plus(lattice)

                pbar.set_postfix({
                    "fraction": f"{curr_frac:.4f}",
                    "target": f"{frac:.4f}"
                })
                pbar.update(1)

                if step > 100 and np.isclose(curr_frac, frac, atol=tol):

                    if mode == "surface":
                        data = surface(lattice)
                        fname = f"surface_{idx+1:03d}.npy"
                    else:
                        data = polymer(lattice, polymer_length)
                        fname = f"polymer_{idx+1:03d}.npy"

                    np.save(os.path.join(out_dir, fname), data)
                    generated += 1
                    idx += 1
                    break

            else:
                print(
                    f"Restarting...\n"
                )


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Ising generator for surfaces and polymers"
    )

    parser.add_argument("--mode", choices=["surface", "polymer"], required=True)
    parser.add_argument("--fraction", type=float, required=True)
    parser.add_argument("--n_systems", type=int, required=True)

    parser.add_argument("--lattice_size", type=int, default=lattice_size)
    parser.add_argument("--polymer_length", type=int, default=polymer_length)

    parser.add_argument("--neighborhood",
                        choices=["neumann", "diagonal", "moore"],
                        default="neumann")

    parser.add_argument("--mc_scheme",
                        choices=["single", "sweep"],
                        default="single")

    parser.add_argument("--coupling", type=float, default=coupling)
    parser.add_argument("--mc_steps", type=int, default=mc_steps)
    parser.add_argument("--seed", type=int, default=seed)

    parser.add_argument("--out_dir", type=str, required=True)

    return parser.parse_args()


def main():
    args = parse_arguments()
    np.random.seed(args.seed)

    header(args)

    generate_systems(
        mode=args.mode,
        frac=args.fraction,
        n_systems=args.n_systems,
        lattice_size=args.lattice_size,
        polymer_length=args.polymer_length,
        neighborhood=args.neighborhood,
        mc_scheme=args.mc_scheme,
        coupling=args.coupling,
        mc_steps=args.mc_steps,
        tol=tol,
        out_dir=args.out_dir,
    )

    print("\n Systems generated successfully.\n")


if __name__ == "__main__":
    main()

