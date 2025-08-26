# scripts/prepare_model_data.py

import numpy as np
from prepare_data import prepare_canonical_data
from preprocessing import generate_augmented_versions
from pre_data import split_indices
from pmf_analysis import analyze_pmf_regions


def prepare_model_data(neural_tag="canonical", random_state=11):
    # Load canonical data and delF
    canonical_default_xdata, canonical_default_ydata = prepare_canonical_data()
    results = analyze_pmf_regions(canonical_default_ydata.reshape(91,91,100))
    delF = results["delF"].reshape(-1)
    
    # Augmentation
    augmented_default_xdata, augmented_default_ydata = generate_augmented_versions(
        canonical_default_xdata, canonical_default_ydata
    )
    # Create backext input: [delF, polymer]


    # Flatten delF for augmented data
    flat_augmented_delF = []
    for i in range(len(augmented_default_xdata)):
        flat_augmented_delF.extend([delF[i]] * len(augmented_default_xdata[i]))
    flat_augmented_delF = np.array(flat_augmented_delF)


    # Split data
    num_canonical = len(canonical_default_xdata)
    train_indices, valid_indices, test_indices = split_indices(num_canonical, seed=random_state)

    # Canonical dataset splits
    canonical_default_train_x = canonical_default_xdata[train_indices]
    canonical_default_train_y = canonical_default_ydata[train_indices]
    canonical_default_valid_x = canonical_default_xdata[valid_indices]
    canonical_default_valid_y = canonical_default_ydata[valid_indices]
    canonical_default_test_x = canonical_default_xdata[test_indices]
    canonical_default_test_y = canonical_default_ydata[test_indices]

    # Augmented dataset splits
    augmented_default_train_x = np.concatenate([augmented_default_xdata[i] for i in train_indices], axis=0)
    augmented_default_train_y = np.concatenate([augmented_default_ydata[i] for i in train_indices], axis=0)
    augmented_default_valid_x = np.concatenate([augmented_default_xdata[i] for i in valid_indices], axis=0)
    augmented_default_valid_y = np.concatenate([augmented_default_ydata[i] for i in valid_indices], axis=0)
    augmented_default_test_x = np.concatenate([augmented_default_xdata[i] for i in test_indices], axis=0)
    augmented_default_test_y = np.concatenate([augmented_default_ydata[i] for i in test_indices], axis=0)


    # Dataset selector
    if neural_tag == "canonical":
        fdX_train, fdX_test, fdX_valid = canonical_default_train_x, canonical_default_test_x, canonical_default_valid_x
        fdY_train, fdY_test, fdY_valid = canonical_default_train_y, canonical_default_test_y, canonical_default_valid_y

    elif neural_tag == "augmented":
        fdX_train, fdX_test, fdX_valid = augmented_default_train_x, augmented_default_test_x, augmented_default_valid_x
        fdY_train, fdY_test, fdY_valid = augmented_default_train_y, augmented_default_test_y, augmented_default_valid_y

    elif neural_tag == "mixed":
        fdX_train, fdX_test, fdX_valid = augmented_default_train_x, augmented_default_test_x, augmented_default_valid_x
        fdY_train, fdY_test, fdY_valid = augmented_default_train_y, augmented_default_test_y, augmented_default_valid_y

    else:
        raise Exception("Invalid neural_tag!!!")

    return (
        fdX_train, fdY_train, fdX_valid, fdY_valid, fdX_test, fdY_test,delF
    )

