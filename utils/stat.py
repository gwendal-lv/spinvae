"""
Statistics-related utility functions.
"""
import copy

import numpy as np


def get_outliers_bounds(x: np.array, IQR_factor=1.5):
    # 25th and 75th percentiles (1st and 3rd quartiles)
    quantiles = np.quantile(x, [0.25, 0.75])
    Q1, Q3 = quantiles[0], quantiles[1]
    return Q1 - (Q3 - Q1) * IQR_factor, Q3 + (Q3 - Q1) * IQR_factor


def remove_outliers(x: np.ndarray, IQR_factor=1.5):
    x_limits = get_outliers_bounds(x, IQR_factor)
    return x[(x_limits[0] <= x) & (x <= x_limits[1])]


def get_random_subset_keep_minmax(x: np.array, subset_len: int):
    if len(x.shape) > 1:
        raise ValueError("Only 1D arrays can be provided to this function.")
    if subset_len > (x.shape[0] - 2):
        return x
    x = copy.deepcopy(x)  # needed because shuffling will happen in-place
    np.random.shuffle(x)
    min_max_indices = [np.argmin(x), np.argmax(x)]
    for i, idx in enumerate(min_max_indices):  # If they were to be removed: we'll copy the min and max values back into the subset
        if idx >= subset_len:
            x[i] = x[idx]
    return x[0:subset_len]
