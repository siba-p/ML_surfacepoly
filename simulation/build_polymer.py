#!/usr/bin/env python3

import numpy as np
import os


resnm = "KGP"

atmnm = ["ACH", "BCH"]
charge = [0.0, 0.0]

mass = 1.0
sgm = 1.0
bond_len = 1.5 * sgm
bond_con = 30.0

n_mer = 40
offset = np.array([2.0 * sgm, 2.0 * sgm, 2.0 * sgm])


def seq_to_ind(val):
    if val == -1:
        return 0
    elif val == +1:
        return 1
    else:
        raise ValueError("Invalid sequence value")


def generate_coord(n_mer):
    """
    Generate polymer coordinates.
    """
    positions = []
    dirr = np.array([1.0, 0.0, 0.0])

    for i in range(n_mer):
        if i == 0:
            pos = offset.copy()
        else:
            if (i - 1) % 4 == 0 or i % 4 == 0:
                dirr[2] = +1.0
            else:
                dirr[2] = -1.0

            if (i + 1) % 4 == 0 or i % 4 == 0:
                dirr[1] = -1.0
            else:
                dirr[1] = +1.0

            step = 0.333 * bond_len * dirr / np.linalg.norm(dirr)
            pos = positions[-1] + step

        positions.append(pos)

    return np.array(positions)


def center_pos(positions):
    return positions - np.mean(positions, axis=0)


def write_itp(path, seq):
    with open(path, "w") as f:
        f.write("[ moleculetype ]\n")
        f.write("; Name nrexcl\n")
        f.write(f"{resnm} 1\n\n")

        f.write("[ atoms ]\n")
        f.write("; nr type resnr resid atom cgnr charge mass\n")

        for i in range(n_mer):
            idx = seq_to_ind(seq[i])
            aname = atmnm[idx]
            atype = "CH" + aname[0]
            chg = charge[idx]

            f.write(
                f"{i+1:5d} {atype:5s} 1 {resnm:5s} "
                f"{aname:5s} {i+1:5d} {chg:8.3f} {mass:8.3f}\n"
            )

        f.write("\n[ bonds ]\n")
        f.write("; ai aj fu b kb\n")

        for i in range(1, n_mer):
            f.write(
                f"{i:5d} {i+1:5d} 7 {bond_len:8.3f} {bond_con:8.3f}\n"
            )


def write_gro(path, seq, positions):
    positions = center_pos(positions)
    box = np.max(np.abs(positions)) + 2.0 * np.max(offset)

    with open(path, "w") as f:
        f.write("Kremer-Grest polymer\n")
        f.write(f"{n_mer}\n")

        for i in range(n_mer):
            idx = seq_to_ind(seq[i])
            aname = atmnm[idx]
            atype = "CH" + aname[0]
            x, y, z = positions[i]

            f.write(
                f"{1:5d}{resnm:5s}{atype:5s}{i+1:5d}"
                f"{x:8.3f}{y:8.3f}{z:8.3f}\n"
            )

        f.write(f"{box:10.3f}{box:10.3f}{box:10.3f}\n")


def main():

    base_dir = "topology/polymer_fraction_0.6"
    diir = os.path.join(base_dir, "gromacs")

    os.makedirs(diir, exist_ok=True)

    for fname in sorted(os.listdir(base_dir)):
        if not fname.endswith(".npy"):
            continue

        pid = fname.replace("polymer_", "").replace(".npy", "")
        seq_path = os.path.join(base_dir, fname)

        seq = np.load(seq_path)
        positions = generate_coord(n_mer)

        write_itp(
            os.path.join(diir, f"polymer_{pid}.itp"),
            seq
        )

        write_gro(
            os.path.join(diir, f"polymer_{pid}.gro"),
            seq,
            positions
        )

print(f"Polymer input files")


if __name__ == "__main__":
    main()

