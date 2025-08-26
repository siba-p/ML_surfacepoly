import numpy as np
from prepare_model_data import prepare_model_data

# Get all data splits
(
    fdX_train, fdY_train, fdX_valid, fdY_valid, fdX_test, fdY_test,delF
) = prepare_model_data(neural_tag="canonical")

# Save to disk
# np.save("../data/processed/fdX_train.npy", fdX_train)
# np.save("../data/processed/fdY_train.npy", fdY_train)
# np.save("../data/processed/fdX_valid.npy", fdX_valid)
# np.save("../data/processed/fdY_valid.npy", fdY_valid)
# np.save("../data/processed/fdX_test.npy",  fdX_test)
# np.save("../data/processed/fdY_test.npy",  fdY_test)

np.save("../data/processed/delF.npy", delF)

