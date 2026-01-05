import numpy as np
import os
import glob

# ============================================================
# ------------------- Constants ------------------------------
# ============================================================

layers = 5
xboxn = 20
yboxn = 20

atmnm = ["ASH", "BSH", "KSH"]
resid = ["LJB"] * (layers - 1) + ["LJS"]

sgm = [1.0, 1.0]
mass = 1.0
charge = 0.0

offset1 = [3.5 * sgm[0], 3.5 * sgm[0], 0.5 * sgm[0]]
offset2 = [3.5 * sgm[0], 3.5 * sgm[0], 12.5 * sgm[0]]

POSRES_FORCE = 10000

# ============================================================
# ------------------- Geometry -------------------------------
# ============================================================

pos = np.zeros((layers, xboxn, yboxn, 3))

def genHCPLattice():
    """
    HCP lattice generator.
    GEOMETRY IDENTICAL to original script.
    """
    a = 1.0
    b = 1.0
    c = 1.633

    lv = np.array([a, b, c / 2.0])

    v1 = np.array([1.0, 0.0, 0.0])
    v2 = np.array([0.5, np.sqrt(3) / 2.0, 0.0])
    v3 = np.array([0.0, 0.0, 1.0])

    v = np.array([v1 * lv, v2 * lv, v3 * lv])

    init1 = np.array([0.0, 0.0, 0.0])
    init2 = np.array(v[1])
    init = np.array([init1, init2])

    shift1 = np.array([0.0, 0.0, 0.0])
    shift2 = np.array([a / 2.0, a / (2.0 * np.sqrt(3)), 0.0])
    ABshift = np.array([shift1, shift2])

    for l in range(layers):
        for i in range(xboxn):
            for j in range(yboxn):
                if i == 0:
                    pos[l, 0, j, :] = (
                        np.array([
                            init[j % 2, 0],
                            init[1, 1] * j,
                            init[j % 2, 2]
                        ]) + v[2] * l
                    )
                else:
                    pos[l, i, j, :] = pos[l, i - 1, j, :] + v[0]

    for l in range(layers):
        pos[l] += ABshift[l % 2]


# ============================================================
# ------------------- Patch assignment -----------------------
# ============================================================

def genPatch(lattice):
    atoms = []
    for l in range(layers):
        for i in range(yboxn):
            for j in range(xboxn):

                if l == layers - 1:
                    atomname = atmnm[0] if lattice[j, i] == -1 else atmnm[1]
                else:
                    atomname = atmnm[2]

                atype = "SH" + atomname[0]
                ind = l * yboxn * xboxn + i * xboxn + j

                atoms.append(
                    f"\t{ind+1}\t{atomname}\t{l+1}\t{resid[l]}"
                    f"\t{atype}\t{ind+1}\t{charge:.3f}\t{mass:.3f}\n"
                )
    return atoms


# ============================================================
# ------------------- Writers -------------------------------
# ============================================================

def write_itp(path, atoms):
    with open(path, "w") as f:
        f.write("[ moleculetype ]\n")
        f.write("; Name nrexcl\n")
        f.write("LJQ\t1\n\n")
        f.write("[ atoms ]\n")
        f.write("; nr type resnr resid atom cgnr charge mass\n")
        for line in atoms:
            f.write(line)


def write_gro(path, lattice):
    with open(path, "w") as f:
        f.write("SAM Surface, t= 0.0\n")
        f.write(f"{xboxn * yboxn * layers}\n")

        for l in range(layers):
            for i in range(yboxn):
                for j in range(xboxn):

                    if l == layers - 1:
                        atomname = atmnm[0] if lattice[j, i] == -1 else atmnm[1]
                    else:
                        atomname = atmnm[2]

                    atype = "SH" + atomname[0]
                    ind = l * yboxn * xboxn + i * xboxn + j
                    x, y, z = pos[l, j, i]

                    f.write(
                        f"{l+1:5d}{resid[l]:5s}{atype:5s}{ind+1:5d}"
                        f"{x:8.3f}{y:8.3f}{z:8.3f}\n"
                    )

        ubound = np.max(pos.reshape(-1, 3), axis=0)
        f.write(f" {ubound[0]:.3f} {ubound[1]:.3f} {ubound[2]:.3f}")


def write_posres(path, n_atoms):
    with open(path, "w") as f:
        f.write("; position restraints for LJ block\n\n")
        f.write("[ position_restraints ]\n")
        f.write(";  i funct       fcx        fcy        fcz\n")
        for i in range(n_atoms):
            f.write(f"{i+1}\t1\t{POSRES_FORCE}\t{POSRES_FORCE}\t{POSRES_FORCE}\n")


# ============================================================
# ------------------- Main ----------------------------------
# ============================================================

def main():

    fraction_dirs = sorted(glob.glob("topology/surface_fraction_*"))

    if not fraction_dirs:
        raise RuntimeError("No surface_fraction directories found")

    for frac_dir in fraction_dirs:

        surface_files = sorted(
            glob.glob(os.path.join(frac_dir, "surface_*.npy"))
        )
        if not surface_files:
            continue

        gmx_dir = os.path.join(frac_dir, "gromacs")
        os.makedirs(gmx_dir, exist_ok=True)

        print(f"\nProcessing {frac_dir}")

        for surf_file in surface_files:

            sid = os.path.basename(surf_file).replace("surface_", "").replace(".npy", "")
            lattice = np.load(surf_file)

            genHCPLattice()
            atoms = genPatch(lattice)

            # Apply offsets
            for i in range(3):
                pos[:, :, :, i] += offset1[i]

            ubound = np.max(pos.reshape(-1, 3), axis=0) + offset2
            lbound = np.min(pos.reshape(-1, 3), axis=0)

            write_itp(os.path.join(gmx_dir, f"surface_{sid}.itp"), atoms)
            write_gro(os.path.join(gmx_dir, f"surface_{sid}.gro"), lattice)
            write_posres(os.path.join(gmx_dir, "posresSurf.itp"), len(atoms))

            surface_dim = np.zeros((3, 2))
            box_dim = np.zeros((3, 2))
            for i in range(3):
                surface_dim[i, 0] = np.min(pos[-1, :, :, i])
                surface_dim[i, 1] = np.max(pos[-1, :, :, i])
                box_dim[i, 0] = lbound[i]
                box_dim[i, 1] = ubound[i]

            np.save(os.path.join(gmx_dir, "surface_dim.npy"), surface_dim)
            np.save(os.path.join(gmx_dir, "box_dim.npy"), box_dim)

            print(f"  ✔ Surface {sid} built")

    print("\nAll surfaces processed successfully.")


if __name__ == "__main__":
    main()

