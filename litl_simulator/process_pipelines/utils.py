"""Utility functions for the process pipeline submodule."""
import numpy as np


def concatenate_arrays(*args):
    """Concatenate multiple arrays."""
    to_return = []
    for arr in args:
        to_return.append(np.concatenate(arr))
    if len(to_return) == 1:
        return to_return[0]
    return to_return


def multiply_arrays(*args):
    """Multiply multiple arrays element wise. They should all have the same shape."""
    if len(args) < 2:
        return args[0]  # nothing to do lol
    shape = args[0].shape
    for i, arg in enumerate(args[1:], start=1):
        if arg.shape != shape:
            raise ValueError(f"Shape mismatch {shape} != {arg.shape} for item {i}")
    mul = np.multiply(args[0], args[1])
    for arg in args[2:]:
        mul = np.multiply(mul, arg)
    return mul


def get_subray_data(arr, i, j, upsampling_ratio):
    """Return a subray data from an array containing all data."""
    upsampled_n_lasers = arr.shape[0]
    npts_per_laser = arr.shape[1]
    if arr.ndim == 3:
        return arr[i:upsampled_n_lasers - upsampling_ratio + 1 + i:upsampling_ratio,
                   j:npts_per_laser - upsampling_ratio + 1 + j:upsampling_ratio, :]
    elif arr.ndim == 2:
        return arr[i:upsampled_n_lasers - upsampling_ratio + 1 + i:upsampling_ratio,
                   j:npts_per_laser - upsampling_ratio + 1 + j:upsampling_ratio]
    raise TypeError("wrong ndim...")


def is_list_like(obj):
    """Return True if obj is list-like."""
    if isinstance(obj, list):
        return True
    if isinstance(obj, tuple):
        return True
    if isinstance(obj, np.ndarray):
        return True
    return False
