# scripts/prepare_canonical_data.py

import numpy as np
from preprocessing import reshape_data, surface_canononilize, polymer_canononilize
from pmf_analysis import analyze_pmf_regions

def prepare_canonical_data():
    pmf_data = np.load("../data/target/pmf_surface.npy")[1:, 1:]
    surface = np.load("../data/feature/surface.npy")[1:]
    polymer = np.load("../data/feature/polymer.npy")[1:]

    reshaped_surface, reshaped_polymer, reshaped_pmf = reshape_data(surface, polymer, pmf_data, num_samples=91)
    canonical_default_xdata = np.hstack((reshaped_surface.reshape(len(reshaped_pmf), 400), reshaped_polymer))
    canonical_default_ydata = reshaped_pmf

    # Canonicalize
    surfacetemp, polymertemp, pmftemp = surface_canononilize(reshaped_surface, reshaped_polymer, reshaped_pmf)
    surface_canon, polymer_canon, pmf_canon = polymer_canononilize(surfacetemp, polymertemp, pmftemp)

    # Get ΔF and other features
    pmf_canon_reshaped = pmf_canon.reshape(91, 91, 100)
    region_results = analyze_pmf_regions(pmf_canon_reshaped)
    delF = region_results["delF"]

    return canonical_default_xdata, canonical_default_ydata, delF

