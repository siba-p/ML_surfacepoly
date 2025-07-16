├── plotting_utils.py # Plotting utilities (e.g., t-SNE, composition plots, metrics)
├── pmf_analysis.py # Extracts PMF features: ΔF, gradients, curvature, etc.
├── pre_data.py # Helper functions to prepare backext input from delF + polymer
├── prepare_canonical_data.py # Loads and reshapes canonical surface, polymer, and PMF data
├── prepare_data.py # Defines data augmentation, canonicalization, and splitting
├── prepare_model_data.py # Main wrapper: calls canonical prep, augmentation, backext prep
├── preprocessing.py # Surface & polymer preprocessing and augmentation functions
├── save_data.py # Optionally saves all datasets (train/val/test) to .npy files
├── symmetry_utils.py # Symmetry utilities: flip, rotate, canonical forms

prepare_model_data.py is the final file to generate canonical/augmented data
