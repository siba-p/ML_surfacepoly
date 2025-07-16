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


def findUniqueMatrices(listOfMats):
  """Finds unique matrices from a given list."""
  uniquelist = []
  for mat in listOfMats:
    if not any((mat == x).all() for x in uniquelist):  # Check if element is already in unique list
      uniquelist.append(mat)
  logging.info(f"Found {len(uniquelist)} unique matrices.")
  return uniquelist

def RotFlipInvariants(lat, rot=True, flip=True):
  """Generates invariants for a given polymer/surface sequence using a series of rotation and flip operations """
  invariants = []
  invariants.append(lat)
  rotMats = []
  flipMats = []
  if(lat.ndim == 2 and lat.shape[0] == lat.shape[1]): # Applying rotation and flip operations for a 2D square surface matrix
    if(rot==True):
      for i in range(1,4):
        temp = np.rot90(lat, k=i) # Perform three 90° counterclockwise rotations (90°, 180°, 270°)
        rotMats.append(temp)
    invariants.extend(rotMats)
    if(flip==True):
      for matrix in invariants:
        temp = np.fliplr(matrix)
        flipMats.append(temp)
        temp = np.flipud(matrix)
        flipMats.append(temp)
      invariants.extend(flipMats)
  elif lat.ndim==1: # Applying sequence reversal for a 1D polymer sequence
    invariants.append(lat[::-1])
  logging.info(f"Generated {len(findUniqueMatrices(invariants))} unique rotational/flip invariants.")
  return findUniqueMatrices(invariants)

def findCanonicalForm(arr_list):
    """
    Given a list of numpy arrays, return the lexicographically smallest one.
    """
    min_arr = min(arr_list, key=lambda x: tuple(x.flatten()))
    return [min_arr]


def surface_canononilize(surface,polymer,pmf):

    surfacetemp, polymertemp, pmftemp = [], [], []

    for i in range(len(pmf)):  # Iterate through the dataset
        xsample = surface[i,:,:]  # Extract 20×20 surface matrix
        ysample = polymer[i,:]    # Extract 40-dimensional polymer sequence
        zsample = pmf[i,:]        # Extract 100-dimensional PMF

        xsample = RotFlipInvariants(xsample, rot=True, flip=False) # Apply rotation and flip to surface
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
