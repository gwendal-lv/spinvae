"""
Statistics-related utility functions.
"""
import copy

import numpy as np
import pandas as pd
import scipy.stats


def get_outliers_bounds(x: np.array, IQR_factor=1.5):
    # 25th and 75th percentiles (1st and 3rd quartiles)
    quantiles = np.quantile(x, [0.25, 0.75])
    Q1, Q3 = quantiles[0], quantiles[1]
    return Q1 - (Q3 - Q1) * IQR_factor, Q3 + (Q3 - Q1) * IQR_factor


def remove_outliers(x: np.ndarray, IQR_factor=1.5):
    x_limits = get_outliers_bounds(x, IQR_factor)
    return x[(x_limits[0] <= x) & (x <= x_limits[1])]


def means_without_outliers(df: pd.DataFrame, IQR_factor=1.5):
    """ Returns a Pandas Series containing the "no-outlier" mean of each column of the input DataFrame, i.e.
    means are computed after outliers of each column have been removed. """
    d = {col: remove_outliers(df[col].values, IQR_factor).mean() for col in df.columns}
    return pd.Series(d)  # build a dict, return it as a Series


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


def wilcoxon_test(x: pd.DataFrame, y: pd.DataFrame, improved_if="y<x", p_value_th=0.05):
    """ Computes the wilcoxon test for DataFrames of paired samples, for each column of the x dataframe
     vs. each col of the y dataframe.

    :returns: a Pandas Series containing the test's p-value for each column, and a Pandas Series indicating whether
        y features are improved compared to x
    """
    p_values = dict()
    if improved_if == "y<x":
        alternative = "greater"  # d = x - y "tends to be > 0"  ===> H0 : "x < y" (we'll try to reject that)
    else:
        raise NotImplementedError()
    for col in x:
        test_result = scipy.stats.wilcoxon(x[col].values, y[col].values, alternative=alternative)
        p_values[col] = test_result.pvalue
    p_values = pd.Series(p_values)
    return p_values, p_values < p_value_th


