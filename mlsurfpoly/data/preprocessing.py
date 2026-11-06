import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def reshape_data(surface, polymer, pmf_data, num_samples=92):
    """
    Reshapes 3D surface, polymer, and PMF data into flat training samples.
    
    Args:
        surface (np.ndarray): Array of shape (num_samples, 20, 20)
        polymer (np.ndarray): Array of shape (num_samples, 40)
        pmf_data (np.ndarray): Array of shape (num_samples, num_samples, 100)
        num_samples (int): Default 92; number of surfaces/polymers

    Returns:
        reshaped_surface: (num_samples, 20, 20)
        reshaped_polymer: (num_samples, 40)
        reshaped_pmf:     (num_samples, 100)
    """
    try:
        logging.info("Starting reshape of input data...")

        # Expand dims to match for tiling
        polymer_data = np.tile(polymer[np.newaxis,:, :], (num_samples, 1, 1))  # (92, 92, 40)
        surface_data = np.tile(surface[:,np.newaxis, :, :], (1, num_samples, 1, 1))  # (92, 92, 20, 20)

        # Reshape for ML training
        reshaped_polymer = polymer_data.reshape(num_samples * num_samples, 40)
        reshaped_surface = surface_data.reshape(num_samples * num_samples, 20, 20)
        reshaped_pmf = pmf_data.reshape(num_samples * num_samples, 100)

        logging.info(f"Surface reshaped to {reshaped_surface.shape}")
        logging.info(f"Polymer reshaped to {reshaped_polymer.shape}")
        logging.info(f"PMF reshaped to {reshaped_pmf.shape}")

        return reshaped_surface, reshaped_polymer, reshaped_pmf

    except Exception as e:
        logging.error("Error in reshaping data:", exc_info=True)
        raise e


from .symmetry import findUniqueMatrices, RotFlipInvariants, findCanonicalForm  # noqa: E402,F401


def surface_canononilize(surface, polymer, pmf, flip=True):
    """Reduce every surface to its canonical orientation (lexicographically smallest variant).

    flip=True uses all 8 symmetries of the square array (rotations and reflections); flip=False uses the 4 rotations only.
    """

    surfacetemp, polymertemp, pmftemp = [], [], []

    for i in range(len(pmf)):  # Iterate through the dataset
        xsample = surface[i,:,:]  # Extract 20×20 surface matrix
        ysample = polymer[i,:]    # Extract 40-dimensional polymer sequence
        zsample = pmf[i,:]        # Extract 100-dimensional PMF

        xsample = RotFlipInvariants(xsample, rot=True, flip=flip)  # rotations (+ reflections if flip)
        xsample = findCanonicalForm(xsample)  # Reduce to canonical form

        # Duplicate polymer and PMF data for all generated surface variations
        ysample = np.tile(ysample, (len(xsample), 1))  # Tile polymer sequence
        zsample = np.tile(zsample, (len(xsample), 1))  # Tile PMF values

        # Store augmented data
        surfacetemp.extend(xsample)
        polymertemp.extend(ysample)
        pmftemp.extend(zsample)

        # Convert lists back to numpy arrays
    return np.array(surfacetemp),np.array(polymertemp),np.array(pmftemp)


def polymer_canononilize(surface,polymer,pmf):
    crude_surface, crude_polymer, crude_pmf = [], [], []

    for i in range(len(pmf)):  # Iterate through the new augmented dataset
        xsample = surface[i,:,:]  # Get the 20×20 surface matrix
        ysample = polymer[i,:]    # Get the 40-dimensional polymer sequence
        zsample = pmf[i,:]        # Get the 100-dimensional PMF

        ysample = RotFlipInvariants(ysample, rot=False, flip=False)  # Apply transformations to polymer
        ysample = findCanonicalForm(ysample)  # Reduce to canonical form

       # Duplicate surface and PMF data for all generated polymer variations
        xsample = np.tile(xsample, (len(ysample), 1, 1))  # Tile surface matrix
        zsample = np.tile(zsample, (len(ysample), 1))     # Tile PMF values

        crude_surface.extend(xsample)
        crude_polymer.extend(ysample)
        crude_pmf.extend(zsample)
    return np.array(crude_surface), np.array(crude_polymer), np.array(crude_pmf)


from .symmetry import generate_augmented_versions  # noqa: E402,F401
