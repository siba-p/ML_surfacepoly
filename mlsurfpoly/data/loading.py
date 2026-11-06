import numpy as np
import logging

from ..paths import RAW
from .preprocessing import reshape_data, polymer_canononilize, surface_canononilize
#from pmf_analysis import analyze_pmf_regions                            

logging.basicConfig(level=logging.INFO,format = "%(asctime)s - %(levelname)s - %(message)s")


def prepare_canonical_data(symmetry="d4", pmf_reference="plateau"):
    """Load the raw library and return canonical (X, Y) arrays for the 91 x 91 polymer-surface pairs.

    symmetry      'd4' (rotations and reflections of the surface, default, used for the published models) or 'rot'.
    pmf_reference 'plateau' subtracts the mean of the last 10 PMF points so the unbound state is at 0 (default),
                  'raw' keeps the WHAM output as is.
    """
    try:
        logging.info("Loading datasets...")
        pmf_data = np.load(RAW / "pmf_datas" / "pmf_data.npy")[1:, 1:]
        surface = np.load(RAW / "surface_fractions" / "surface.npy")[1:]
        polymer = np.load(RAW / "polymer_fractions" / "polymer.npy")[1:]
    except Exception as e:
        logging.info(f"Failed to load dataset: {str(e)}")
        raise
    ## Reshape the data
    try:
        reshaped_surface, reshaped_polymer, reshaped_pmf = reshape_data(surface, polymer, pmf_data, num_samples=91)
    except Exception:
        logging.info("Error in reshaping...")
        raise
    
    
    xdata = np.hstack((reshaped_surface.reshape(len(reshaped_pmf),20*20), reshaped_polymer))
    ydata = reshaped_pmf
    
    logging.info(f"Final X Data shape: {xdata.shape}")
    logging.info(f"Final Y Data shape: {ydata.shape}")
    
    #np.save("xdata_original.npy", xdata)
    #np.save("ydata_original.npy",ydata)
    
    if pmf_reference == "plateau":
        reshaped_pmf = reshaped_pmf - reshaped_pmf[:, -10:].mean(axis=1, keepdims=True)
    elif pmf_reference != "raw":
        raise ValueError("pmf_reference must be 'plateau' or 'raw'")
    surfacetemp, polymertemp, pmftemp = surface_canononilize(reshaped_surface, reshaped_polymer, reshaped_pmf, flip=symmetry == "d4")
    surface_canon,polymer_canon,pmf_canon = polymer_canononilize(surfacetemp, polymertemp, pmftemp)
    logging.info(f"Canononilized Surface shape: {surface_canon.shape}, Polymer shape: {polymer_canon.shape},\
                  PMF shape: {pmf_canon.shape}")
    #np.save("xdata_canonical.npy",np.hstack((surface_canon.reshape(len(pmf_canon),20*20),polymer_canon)))
    #np.save("ydata_canonical.npy",pmf_canon)
    canonical_default_xdata = np.hstack((surface_canon.reshape(len(pmf_canon),20*20),polymer_canon)) 
    canonical_default_ydata = pmf_canon
    #results = analyze_pmf_regions(pmf_canon.reshape(91,91,100))
    return canonical_default_xdata, canonical_default_ydata 
