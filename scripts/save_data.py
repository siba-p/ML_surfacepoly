import numpy as np
from prepare_model_data import prepare_model_data

# Get all data splits
(
    fdX_train, fdY_train, fdX_valid, fdY_valid, fdX_test, fdY_test,
    tdX_train, tdY_train, tdX_valid, tdY_valid, tdX_test, tdY_test,
    bxX_train, bxY_train, bxX_valid, bxY_valid, bxX_test, bxY_test,
    delF
) = prepare_model_data(neural_tag="canonical")

# Save to disk
# np.save("../data/processed/fdX_train.npy", fdX_train)
# np.save("../data/processed/fdY_train.npy", fdY_train)
# np.save("../data/processed/fdX_valid.npy", fdX_valid)
# np.save("../data/processed/fdY_valid.npy", fdY_valid)
# np.save("../data/processed/fdX_test.npy",  fdX_test)
# np.save("../data/processed/fdY_test.npy",  fdY_test)
# 
# np.save("../data/processed/tdX_train.npy", tdX_train)
# np.save("../data/processed/tdY_train.npy", tdY_train)
# np.save("../data/processed/tdX_valid.npy", tdX_valid)
# np.save("../data/processed/tdY_valid.npy", tdY_valid)
# np.save("../data/processed/tdX_test.npy",  tdX_test)
# np.save("../data/processed/tdY_test.npy",  tdY_test)
# 
# np.save("../data/processed/bxX_train.npy", bxX_train)
# np.save("../data/processed/bxY_train.npy", bxY_train)
# np.save("../data/processed/bxX_valid.npy", bxX_valid)
# np.save("../data/processed/bxY_valid.npy", bxY_valid)
# np.save("../data/processed/bxX_test.npy",  bxX_test)
# np.save("../data/processed/bxY_test.npy",  bxY_test)

np.save("../data/processed/delF.npy", delF)

