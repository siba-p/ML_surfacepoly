import numpy as np

def analyze_pmf_regions(pmf_data, window_size=3, tol_ratio=0.1):
    """
    Analyze different regions of PMF profile for each surface-polymer pair.

    Parameters:
    -----------
    pmf_data : np.ndarray
        3D array of shape (num_surfaces, num_polymers, pmf_length)
    window_size : int
        Number of points to consider for flat/plateau detection.
    tol_ratio : float
        Tolerance as a fraction of the window mean to identify plateau regions.

    Returns:
    --------
    result : dict of np.ndarray
        Dictionary containing:
        - delF
        - positive_gradient
        - negative_gradient
        - positive_curvature
        - negative_curvature
        - wm_positive_curvature
        - wm_negative_curvature
        - min_pmf
        - min_index
        - min_index_value
    """
    n_surface, n_polymer, _ = pmf_data.shape

    delF = np.zeros((n_surface, n_polymer))
    positive_gradient = np.zeros((n_surface, n_polymer))
    negative_gradient = np.zeros((n_surface, n_polymer))
    positive_curvature = np.zeros((n_surface, n_polymer))
    negative_curvature = np.zeros((n_surface, n_polymer))
    wm_positive_curvature = np.zeros((n_surface, n_polymer))
    wm_negative_curvature = np.zeros((n_surface, n_polymer))
    min_pmf = np.zeros((n_surface, n_polymer))
    min_index = np.zeros((n_surface, n_polymer))
    min_index_value = np.zeros((n_surface, n_polymer))

    for i in range(n_surface):
        for j in range(n_polymer):
            pmf_profile = pmf_data[i, j, :]
            gradients = np.gradient(pmf_profile)
            second_derivative = np.gradient(gradients)

            min_pmf[i, j] = np.min(pmf_profile)
            min_index[i, j] = np.argmin(pmf_profile)
            min_index_value[i, j] = 4.7 + 0.1 * min_index[i, j]

            # Plateau detection
            plat = None
            prev_std = float("inf")

            for start in range(len(pmf_profile) - window_size):
                window = pmf_profile[start : start + window_size]
                current_std = np.std(window)
                tol = tol_ratio * np.mean(window)

                if np.abs(current_std - prev_std) < tol:
                    plat = (start, start + window_size)
                prev_std = current_std

            if plat:
                pmf_plat = np.mean(pmf_profile[plat[0]:plat[1]])
                delF[i, j] = pmf_plat - min_pmf[i, j]

                positive_id = gradients > 0
                negative_id = gradients < 0

                pos_w = np.abs(gradients[positive_id])
                neg_w = np.abs(gradients[negative_id])

                if np.any(positive_id):
                    positive_gradient[i, j] = np.mean(gradients[positive_id])
                    positive_curvature[i, j] = np.mean(second_derivative[positive_id])
                    wm_positive_curvature[i, j] = np.sum(
                        second_derivative[positive_id] * pos_w
                    ) / np.sum(pos_w)

                if np.any(negative_id):
                    negative_gradient[i, j] = np.mean(gradients[negative_id])
                    negative_curvature[i, j] = np.mean(second_derivative[negative_id])
                    wm_negative_curvature[i, j] = np.sum(
                        second_derivative[negative_id] * neg_w
                    ) / np.sum(neg_w)

    return {
        "delF": delF,
        "positive_gradient": positive_gradient,
        "negative_gradient": negative_gradient,
        "positive_curvature": positive_curvature,
        "negative_curvature": negative_curvature,
        "wm_positive_curvature": wm_positive_curvature,
        "wm_negative_curvature": wm_negative_curvature,
        "min_pmf": min_pmf,
        "min_index": min_index,
        "min_index_value": min_index_value,
    }

##results = analyze_pmf_regions(pmf_data)

# Example access
#print("Delta F shape:", results["delF"].shape)
#print("Positive curvature:", results["positive_curvature"][0, 0])
