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
