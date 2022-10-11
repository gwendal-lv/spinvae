"""
Custom exception classes and functions to perform checks and throw errors if required.
"""

import math

import torch


class ModelConvergenceError(ValueError):
    pass


def check_nan_values(epoch, *args):
    """
    Raises a ModelConvergenceError is any Tensor (in kwargs) contains a nan value.

    :param epoch: Current training epoch (used in the error message)
    :param args: Tensors to be tested - floats are accepted
    """
    for i, t in enumerate(args):
        if isinstance(t, float):
            if math.isnan(t):
                raise ModelConvergenceError("Epoch {}: float from *args (#{}) contains a nan item".format(epoch, i))
        elif isinstance(t, torch.Tensor):
            if torch.isnan(t).any():
                raise ModelConvergenceError("Epoch {}: Tensor from *args (#{}) contains a nan item".format(epoch, i))
        else:
            raise ValueError("Unexpected type {} in *args".format(type(t)))


