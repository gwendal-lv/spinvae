"""
Statistics-related utility functions.
"""

import numpy as np


def get_outliers_bounds(x: np.array, IQR_factor=1.5):
    # 25th and 75th percentiles (1st and 3rd quartiles)
    quantiles = np.quantile(x, [0.25, 0.75])
    Q1, Q3 = quantiles[0], quantiles[1]
    return Q1 - (Q3 - Q1) * IQR_factor, Q3 + (Q3 - Q1) * IQR_factor


def remove_outliers(x: np.ndarray, IQR_factor=1.5):
    x_limits = get_outliers_bounds(x, IQR_factor)
    return x[(x_limits[0] <= x) & (x <= x_limits[1])]


