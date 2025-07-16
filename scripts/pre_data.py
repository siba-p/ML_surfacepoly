# scripts/utils/prepare_data.py

import numpy as np

def split_indices(num_samples, seed=11):
    np.random.seed(seed)
    indices = np.random.permutation(num_samples)
    test_size = int(0.1 * num_samples)
    valid_size = int(0.2 * num_samples)
    test_idx = indices[:test_size]
    valid_idx = indices[test_size:test_size + valid_size]
    train_idx = indices[test_size + valid_size:]
    return train_idx, valid_idx, test_idx

#def generate_backext_xdata(xdata_augmented, delF):
#    backext_xdata = []
#    count = 0  # to index delF sequentially
#    for i in range(len(xdata_augmented)):
#        group = []
#        for j in range(len(xdata_augmented[i])):
#            delF = [delF[count]]
#            polymer = xdata_augmented[i][j][400:]
#            group.append(np.concatenate([delF, polymer]))
#            count += 1
#        backext_xdata.append(group)
#    return backext_xdata
def generate_backext_xdata(xdata_augmented, delF_flat):
    """
    For each input sample, create [delF, polymer (last 40 entries)]
    xdata_augmented: list of (k,) arrays (augmented) or ndarray (canonical)
    delF_flat: 1D array of same total length as xdata

    Returns:
        np.ndarray of shape (N, 41)
    """
    backext_x = []

    if isinstance(xdata_augmented, list):  # For augmented
        flat_delF_idx = 0
        for i in range(len(xdata_augmented)):
            for j in range(len(xdata_augmented[i])):
                polymer = xdata_augmented[i][j][400:]
                dF = [delF_flat[flat_delF_idx]]
                flat_delF_idx += 1
                backext_x.append(np.concatenate([dF, polymer]))
    else:  # For canonical (np.ndarray)
        for i in range(len(xdata_augmented)):
            polymer = xdata_augmented[i][400:]
            dF = [delF_flat[i]]
            backext_x.append(np.concatenate([dF, polymer]))

    return np.array(backext_x)

