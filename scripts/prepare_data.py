import numpy as np
from pathlib import Path
import logging
import argparse
from preprocessing import reshape_data, polymer_canononilize, surface_canononilize
#from pmf_analysis import analyze_pmf_regions                            

logging.basicConfig(level=logging.INFO,format = "%(asctime)s - %(levelname)s - %(message)s")


def prepare_canonical_data():
    try:
        logging.info("Loading datasets...")
        pmf_data = np.load("../data/target/pmf_surface.npy")[1:, 1:]
        surface = np.load("../data/feature/surface.npy")[1:]
        polymer = np.load("../data/feature/polymer.npy")[1:]
    except Exception as e:
        logging.info(f"Failed to load dataset: {str(e)}")
        raise
    ## Reshape the data
    try:
        reshaped_surface, reshaped_polymer, reshaped_pmf = reshape_data(surface, polymer, pmf_data, num_samples=91)
    except Exception as e:
        logging.info("Error in reshaping...")
        raise
    
    
    xdata = np.hstack((reshaped_surface.reshape(len(reshaped_pmf),20*20), reshaped_polymer))
    ydata = reshaped_pmf
    
    logging.info(f"Final X Data shape: {xdata.shape}")
    logging.info(f"Final Y Data shape: {ydata.shape}")
    
    #np.save("xdata_original.npy", xdata)
    #np.save("ydata_original.npy",ydata)
    
    surfacetemp, polymertemp, pmftemp = surface_canononilize(reshaped_surface, reshaped_polymer, reshaped_pmf)
    surface_canon,polymer_canon,pmf_canon = polymer_canononilize(surfacetemp, polymertemp, pmftemp)
    logging.info(f"Canononilized Surface shape: {surface_canon.shape}, Polymer shape: {polymer_canon.shape},\
                  PMF shape: {pmf_canon.shape}")
    #np.save("xdata_canonical.npy",np.hstack((surface_canon.reshape(len(pmf_canon),20*20),polymer_canon)))
    #np.save("ydata_canonical.npy",pmf_canon)
    canonical_default_xdata = np.hstack((surface_canon.reshape(len(pmf_canon),20*20),polymer_canon)) 
    canonical_default_ydata = pmf_canon
    #results = analyze_pmf_regions(pmf_canon.reshape(91,91,100))
    return canonical_default_xdata, canonical_default_ydata 
