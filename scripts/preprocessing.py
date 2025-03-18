import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def reshape_data(surface, polymer, pmf_data, num_samples=92):
    """Reshapes surface, polymer, and PMF data for training."""
    polymer_data = np.tile(polymer[:, np.newaxis, :], (1, num_samples, 1))
    surface_data = np.tile(surface[np.newaxis, :, :, :], (num_samples, 1, 1, 1))

    reshaped_pmf = pmf_data.transpose(1, 0, 2).reshape(num_samples * num_samples, 100)
    reshaped_polymer = polymer_data.reshape(num_samples * num_samples, 40)
    reshaped_surface = surface_data.reshape(num_samples * num_samples, 20 , 20)
    logging.info(f"Reshaped surface data to {reshaped_surface.shape}")
    logging.info(f"Reshaped polymer data to {reshaped_polymer.shape}")
    logging.info(f"Reshaped PMF data to {reshaped_pmf.shape}")
    return reshaped_surface, reshaped_polymer, reshaped_pmf

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

def canonical_form(matrices):
    """Finds the lexicographically minimal sequence in a list of matrices."""
    canonical = min(matrices, key=lambda x: tuple(x.flatten()))
    logging.info("Canonical form found.")
    return [canonical]

def augment_surface(surface,polymer,pmf,rot=True: bool,flip=True: bool):

    surfacetemp, polymertemp, pmftemp = [], [], []

    for i in range(len(pmf)):  # Iterate through the dataset
        xsample = surface[i,:,:]  # Extract 20×20 surface matrix
        ysample = polymer[i,:]    # Extract 40-dimensional polymer sequence
        zsample = pmf[i,:]        # Extract 100-dimensional PMF
    
        xsample = RotFlipInvariants(xsample, rot=rot, flip=flip) # Apply rotation and flip to surface
      # xsample = findCanonicalForm(xsample)  # Reduce to canonical form
    
        # Duplicate polymer and PMF data for all generated surface variations
        ysample = np.tile(ysample, (len(xsample), 1))  # Tile polymer sequence
        zsample = np.tile(zsample, (len(xsample), 1))  # Tile PMF values
    
        # Store augmented data
        surfacetemp.extend(xsample)
        polymertemp.extend(ysample)
        pmftemp.extend(zsample)
    
        # Convert lists back to numpy arrays
    return np.array(surfacetemp),np.array(polymertemp),np.array(pmftemp)


def augment_polymer(surface,polymer,pmf):
    crude_surface, crude_polymer, crude_pmf = [], [], []

    for i in range(len(pmf)):  # Iterate through the new augmented dataset
        xsample = surface[i,:,:]  # Get the 20×20 surface matrix
        ysample = polymer[i,:]    # Get the 40-dimensional polymer sequence
        zsample = pmf[i,:]        # Get the 100-dimensional PMF

        ysample = RotFlipInvariants(ysample, rot=False, flip=False)  # Apply transformations to polymer
        # ysample = findCanonicalForm(ysample)  # Reduce to canonical form

       # Duplicate surface and PMF data for all generated polymer variations
        xsample = np.tile(xsample, (len(ysample), 1, 1))  # Tile surface matrix
        zsample = np.tile(zsample, (len(ysample), 1))     # Tile PMF values

        crude_surface.extend(xsample)
        crude_polymer.extend(ysample)
        crude_pmf.extend(zsample)
    return np.array(crude_surface), np.array(crude_polymer), np.array(crude_pmf)  

