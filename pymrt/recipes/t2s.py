import collections

import numpy as np

import pymrt.utils as pmu

from pymrt import msg
from pymrt.computation import voxel_curve_fit


# ======================================================================
def _pre_exp_loglin(arr, exp_factor=0, zero_cutoff=np.spacing(1)):
    arr = np.abs(arr)
    log_arr = np.zeros_like(arr)
    # calculate logarithm only of strictly positive values
    log_arr[arr > zero_cutoff] = \
        np.log(arr[arr > zero_cutoff] * np.exp(exp_factor))
    return log_arr


# ======================================================================
def _post_exp_loglin(arr, exp_factor=0, zero_cutoff=np.spacing(1)):
    # tau = p_arr[..., 0]
    # s_0 = p_arr[..., 1]
    for i in range(arr.shape[-1]):
        if i < arr.shape[-1] - 1:
            mask = np.abs(arr[..., i]) > zero_cutoff
            arr[..., i][mask] = -1.0 / arr[..., i][mask]
        else:
            arr[..., i] = np.exp(arr[..., i] - exp_factor)
    return arr


# ======================================================================
def fit_exp_loglin(
        arr,
        tis,
        num=1,
        full=False,
        exp_factor=None,
        zero_cutoff=None):
    """
    Fit monoexponential decay to images using the log-linear method.

    Args:
        arr (np.ndarray): The input array in arb.units.
            The sampling time Ti varies in the last dimension.
        tis (iterable): The sampling times Ti in time units.
            The number of points must match the last shape size of arr.
        num (int): The degree of the polynomial to fit.
            For monoexponential fits, use num=1.
        full (bool): Calculate additional information on the fit performance.
            If True, more information is given.
            If False, only the optimized parameters are returned.
        exp_

    Returns:
        results (dict): The calculated information.
            If full is True, more information is available.
            `s0` contains the amplitude of the exponential.
            `tau_{i}` for i=1,...,num contain the higher order terms of the fit.
    """
    # 0: untouched, other values might improve numerical stability
    if exp_factor is None:
        exp_factor = 0
    if zero_cutoff is None:
        zero_cutoff = np.spacing(1)

    y_arr = np.array(arr).astype(float)
    x_arr = np.array(tis).astype(float)

    assert (x_arr.size == arr.shape[-1])

    p_arr = voxel_curve_fit(
        y_arr, x_arr,
        None, (np.mean(y_arr),) + (np.mean(x_arr),) * num,
        _pre_exp_loglin, [exp_factor, zero_cutoff], {},
        _post_exp_loglin, [exp_factor, zero_cutoff], {},
        method='poly')
    p_arrs = np.split(p_arr, num + 1, -1)

    results = dict(
        ('s0' if i == 0 else 'tau_{i}'.format(i=i), x)
        for i, x in enumerate(p_arrs[::-1]))

    if full:
        msg('E: Not implemented yet!')

    return results
