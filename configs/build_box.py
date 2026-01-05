#!/usr/bin/env python3

"""
Selects a surface and a polymer, combines them into a single
simulation box, and required topology files.

Inputs:
- surface .gro, .itp, posresSurf.itp
- polymer .gro, .itp
- surface_dim.npy, box_dim.npy

Output:
- system_XXXX/
    ├── system.gro
    ├── surface.itp
    ├── polymer.itp
    ├── posresSurf.itp
"""

import numpy as np
import os
import shutil
import argparse


sgm = 1.0
pol_off = 1.5 * sgm
box_off = 25.0


def parse_gro_line(line):
    idx = 0
    resid = int(line[idx:idx+5]); idx += 5
    resname = line[idx:idx+5]; idx += 5
    atomname = line[idx:idx+5]; idx += 5
    atomid = int(line[idx:idx+5]); idx += 5
    x = float(line[idx:idx+8]); idx += 8
    y = float(line[idx:idx+8]); idx += 8
    z = float(line[idx:idx+8])
    return resid, resname, atomname, atomid, np.array([x, y, z])


def read_gro(fname):
    with open(fname, "r") as f:
        f.readline()
        n_atoms = int(f.readline())

        resid, resname, atomname, atomid, pos = [], [], [], [], []

        for _ in range(n_atoms):
            r, rn, an, ai, p = parse_gro_line(f.readline())
            resid.append(r)
            resname.append(rn)
            atomname.append(an)
            atomid.append(ai)
            pos.append(p)

    return (
        np.array(resid),
        np.array(resname),
        np.array(atomname),
        np.array(atomid),
        np.array(pos),
    )


def build_system(
    surface_dir,
    polymer_dir,
    surface_id,
    polymer_id,
    output_root,
):

    surface_gro = os.path.join(surface_dir, "gromacs", f"surface_{surface_id}.gro")
    surface_itp = os.path.join(surface_dir, "gromacs", f"surface_{surface_id}.itp")
    posres_itp  = os.path.join(surface_dir, "gromacs", "posresSurf.itp")

    polymer_gro = os.path.join(polymer_dir, "gromacs", f"polymer_{polymer_id}.gro")
    polymer_itp = os.path.join(polymer_dir, "gromacs", f"polymer_{polymer_id}.itp")

    surface_dim = os.path.join(surface_dir, "gromacs", "surface_dim.npy")
    box_dim = os.path.join(surface_dir, "gromacs", "box_dim.npy")

    for f in [surface_gro, surface_itp, polymer_gro, polymer_itp, surface_dim, box_dim]:
        if not os.path.isfile(f):
            raise FileNotFoundError(f"Missing required file: {f}")

    system_name = f"s{surface_id}_p{polymer_id}"
    system_dir = os.path.join(output_root, system_name)
    os.makedirs(system_dir, exist_ok=True)

    surface_bounds = np.load(surface_dim)
    box_bounds = np.load(box_dim)

    polymer_com_shift = np.array([
        0.5 * (surface_bounds[0, 0] + surface_bounds[0, 1]),
        0.5 * (surface_bounds[1, 0] + surface_bounds[1, 1]),
        surface_bounds[2, 1] + pol_off,
    ])

    S = read_gro(surface_gro)
    P = read_gro(polymer_gro)

    n_surface = len(S[0])
    n_polymer = len(P[0])

    resid = list(S[0])
    resname = list(S[1])
    atomname = list(S[2])
    atomid = list(S[3])
    pos = list(S[4])

    resid_offset = resid[-1]
    atomid_offset = atomid[-1]

    for i in range(n_polymer):
        resid.append(P[0][i] + resid_offset)
        resname.append(P[1][i])
        atomname.append(P[2][i])
        atomid.append(P[3][i] + atomid_offset)
        pos.append(P[4][i] + polymer_com_shift)

    pos = np.array(pos)

    box_gro = os.path.join(system_dir, "system.gro")
    with open(box_gro, "w") as f:
        f.write("Surface–Polymer System\n")
        f.write(f"{n_surface + n_polymer}\n")

        for i in range(n_surface + n_polymer):
            f.write(
                f"{resid[i]:5d}{resname[i]:5s}{atomname[i]:5s}{atomid[i]:5d}"
                f"{pos[i][0]:8.3f}{pos[i][1]:8.3f}{pos[i][2]:8.3f}\n"
            )

        f.write(
            f"{box_bounds[0,1]:10.3f}"
            f"{box_bounds[1,1]:10.3f}"
            f"{box_off:10.3f}\n"
        )

    shutil.copy(surface_itp, os.path.join(system_dir, "surface.itp"))
    shutil.copy(polymer_itp, os.path.join(system_dir, "polymer.itp"))
    shutil.copy(posres_itp, os.path.join(system_dir, "posresSurf.itp"))



def parse_args():
    parser = argparse.ArgumentParser(
        description="Input files"
    )

    parser.add_argument("--surface_dir", required=True,
                        help="Path to surface_fraction_X directory")
    parser.add_argument("--polymer_dir", required=True,
                        help="Path to polymer_fraction_Y directory")

    parser.add_argument("--surface_id", required=True,
                        help="Surface ID (e.g. 191)")
    parser.add_argument("--polymer_id", required=True,
                        help="Polymer ID (e.g. 090)")

    parser.add_argument("--output_root", default="md_systems",
                        help="Root directory for generated systems")

    return parser.parse_args()


if __name__ == "__main__":

    args = parse_args()

    build_system(
        surface_dir=args.surface_dir,
        polymer_dir=args.polymer_dir,
        surface_id=args.surface_id,
        polymer_id=args.polymer_id,
        output_root=args.output_root,
    )

