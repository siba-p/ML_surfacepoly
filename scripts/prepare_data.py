import numpy as np
from pathlib import Path
import logging
import argparse
from preprocessing import reshape_data, augment_polymer, augment_surface

#parser = argparse.ArgumentParser(description="Preprocess dataset with optional augmentation steps.")
logging.basicConfig(level=logging.INFO,format = "%(asctime)s - %(levelname)s - %(message)s")

#parser.add_argument("--augment_surface")
#parser.add_argument("--augment_polymer")
#arg = parser.parse_args()

try:
    logging.info("Loading datasets...")
    pmf_data = np.load("../data_final/target/pmf_surface.npy")
    surface = np.load("../data_final/feature/surface.npy")
    polymer = np.load("../data_final/feature/polymer.npy")
except Exception as e:
    loggin.info(f"Failed to load dataset: {str(e)}")
    raise
## Reshape the data
try:
    reshaped_surface, reshaped_polymer, reshaped_pmf = reshape_data(surface,polymer,pmf_data)
except Exception as e:
    loggin.info("Error in reshaping...")
    raise


xdata = np.hstack((reshaped_surface.reshape(len(reshaped_pmf),20*20), reshaped_polymer))
ydata = reshaped_pmf

logging.info(f"Final X Data shape: {xdata.shape}")
logging.info(f"Final Y Data shape: {ydata.shape}")

np.save("xdata_original.npy", xdata)
np.save("ydata_original.npy",ydata)

surfacetemp, polymertemp, pmftemp = augment_surface(reshaped_surface, reshaped_polymer, reshaped_pmf)
surface_augment,polymer_augment,pmf_augment = augment_polymer(surfacetemp, polymertemp, pmftemp)
#surface_augment,polymer_augment,pmf_augment = augment_polymer(augment_surface(reshaped_surface, reshaped_polymer, reshaped_pmf))
logging.info(f"Augmented Surface shape: {surface_augment.shape}, augmented Polymer shape: {polymer_augment.shape}, augmented PMF shape: {pmf_augment.shape}")
np.save("xdata_augment.npy",np.hstack((surface_augment.reshape(len(pmf_augment),20*20),polymer_augment)))
np.save("ydata_augment.npy",pmf_augment)

