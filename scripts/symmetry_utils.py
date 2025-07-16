import numpy as np

def findUniqueMatrices(listOfMats):
    """
    Given a list of numpy matrices, return only the unique ones.
    """
    unique_list = []
    for mat in listOfMats:
        if not any((mat == x).all() for x in unique_list):
            unique_list.append(mat)
    return unique_list


def RotFlipInvariants(lat, rot=True, flip=True):
    """
    Generate all unique rotational and flip transformations of a matrix.
    For 2D square matrices, applies 90/180/270 degree rotations and flips.
    For 1D arrays or non-square 2D, returns reversed array.

    Args:
        lat (np.ndarray): The input array or matrix.
        rot (bool): Whether to include 90-degree rotations.
        flip (bool): Whether to include horizontal and vertical flips.

    Returns:
        List[np.ndarray]: Unique transformed versions of the input matrix.
    """
    invariants = [lat]
    rot_mats = []
    flip_mats = []

    if lat.ndim == 2 and lat.shape[0] == lat.shape[1]:
        if rot:
            for i in range(1, 4):
                rot_mats.append(np.rot90(lat, k=i))
        invariants.extend(rot_mats)

        if flip:
            for matrix in invariants:
                flip_mats.append(np.fliplr(matrix))
                flip_mats.append(np.flipud(matrix))
        invariants.extend(flip_mats)
    else:
        invariants.append(lat[::-1])  # For 1D or non-square, reverse it

    return findUniqueMatrices(invariants)


def findCanonicalForm(arr_list):
    """
    Given a list of numpy arrays, return the lexicographically smallest one.

    Args:
        arr_list (List[np.ndarray]): List of numpy matrices or vectors.

    Returns:
        List[np.ndarray]: A list with the canonical (smallest) matrix.
    """
    min_arr = min(arr_list, key=lambda x: tuple(x.flatten()))
    return [min_arr]



def generate_augmented_versions(canonical_xdata, canonical_ydata):
    """
    Apply RotFlipInvariants to surface (Nx20x20) and polymer (Nx40).
    Returns:
        - aug_xdata: list of arrays [(k_i, 440), ...]
        - aug_ydata: list of arrays [(k_i, 440), ...] repeated canonical_ydata entries
    """
    surface_seqs = canonical_xdata[:, :400]
    polymer_seqs = canonical_xdata[:, 400:]

    aug_x_list = []
    aug_y_list = []

    for idx, (surface, polymer) in enumerate(zip(surface_seqs, polymer_seqs)):

        # Augment surface and polymer sequences
        surface_aug = np.array(
            RotFlipInvariants(surface.reshape(20, 20))
        )  # (k1, 20, 20)
        polymer_aug = np.array(RotFlipInvariants(polymer))  # (k2, 40)

        k1 = surface_aug.shape[0]
        k2 = polymer_aug.shape[0]

        # Flatten surfaces to (k1, 400)
        surface_flat = surface_aug.reshape(k1, -1)
        # print(surface_flat.shape, polymer_aug.shape)

        # Generate all combinations of surface_flat and polymer_aug (k1*k2, 440)
        combined_aug = []
        for i in range(k1):
            for j in range(k2):
                combined = np.hstack((surface_flat[i], polymer_aug[j]))
                combined_aug.append(combined)

        combined_aug = np.array(combined_aug)  # (k1*k2, 440)
        # print(combined_aug.shape)
        aug_x_list.append(combined_aug)

        # Repeat canonical_ydata[idx] k1*k2 times along axis 0
        repeated_y = np.repeat(
            canonical_ydata[idx : idx + 1], k1 * k2, axis=0
        )  # shape (k1*k2, 440)
        # print(repeated_y.shape)
        aug_y_list.append(repeated_y)

    return aug_x_list, aug_y_list
