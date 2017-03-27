#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
pymrt.recipes.t2s: T2* computation algorithms.
"""

# ======================================================================
# :: Future Imports
from __future__ import (
    division, absolute_import, print_function, unicode_literals)

# ======================================================================
# :: Python Standard Library Imports
# import itertools  # Functions creating iterators for efficient looping
import warnings  # Warning control
import collections  # Container datatypes

# :: External Imports
import numpy as np  # NumPy (multidimensional numerical arrays library)

# :: Local Imports
# import pymrt.utils as pmu

# from pymrt import VERB_LVL, D_VERB_LVL, VERB_LVL_NAMES
# from pymrt import elapsed, print_elapsed
# from pymrt import msg, dbg

from pymrt.recipes import generic

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
        tis_mask=None,
        poly_deg=1,
        full=False,
        exp_factor=0,
        zero_cutoff=np.spacing(1)):
    """
    Fit exponential decay to data using the log-linear method.

    Args:
        arr (np.ndarray): The input array in arb.units.
            The sampling time T_i varies in the last dimension.
        tis (iterable): The sampling times T_i in time units.
            The number of points must match the last shape size of arr.
        tis_mask (iterable[bool]|None): Determine the sampling times Ti to use.
            If None, all will be used.
        poly_deg (int): The degree of the polynomial to fit.
            For monoexponential fits, use num=1.
        full (bool): Calculate additional information on the fit performance.
            If True, more information is given.
            If False, only the optimized parameters are returned.
        exp_factor (float|None):
        zero_cutoff (float|None): The threshold value for masking zero values.

    Returns:
        results (dict): The calculated information.
            If full is True, more information is available.
            `s0` contains the amplitude of the exponential.
            `tau_{i}` for i=1,...,num contain the higher order terms of the fit.
    """
    # 0: untouched, other values might improve numerical stability
    y_arr = np.array(arr).astype(float)
    x_arr = np.array(tis).astype(float)

    if tis_mask:
        y_arr = y_arr[..., tis_mask]
        x_arr = x_arr[tis_mask]

    assert (x_arr.size == arr.shape[-1])

    p_arr = generic.voxel_curve_fit(
        y_arr, x_arr,
        None, (np.mean(y_arr),) + (np.mean(x_arr),) * poly_deg,
        _pre_exp_loglin, [exp_factor, zero_cutoff], {},
        _post_exp_loglin, [exp_factor, zero_cutoff], {},
        method='poly')
    p_arrs = np.split(p_arr, poly_deg + 1, -1)

    results = collections.OrderedDict(
        ('s0' if i == 0 else 'tau_{i}'.format(i=i), x)
        for i, x in enumerate(p_arrs[::-1]))

    if full:
        warnings.warn('E: Not implemented yet!')

    return results
