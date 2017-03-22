#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
pymrt.recipes.b0: dB0 computation algorithms.
"""

# ======================================================================
# :: Future Imports
from __future__ import (
    division, absolute_import, print_function, unicode_literals)

# ======================================================================
# :: Python Standard Library Imports
# import itertools  # Functions creating iterators for efficient looping
# import collections  # Container datatypes
# import warnings  # Warning control
import multiprocessing  # Process-based parallelism

# :: External Imports
import numpy as np  # NumPy (multidimensional numerical arrays library)

# :: Local Imports
import pymrt.utils as pmu


# from pymrt import VERB_LVL, D_VERB_LVL, VERB_LVL_NAMES
# from pymrt import elapsed, print_elapsed
# from pymrt import msg, dbg


# ======================================================================
def time_to_rate(
        arr,
        in_units='ms',
        out_units='Hz'):
    k = 1.0
    if in_units == 'ms':
        k *= 1.0e3
    if out_units == 'kHz':
        k *= 1.0e-3
    arr[arr != 0.0] = k / arr[arr != 0.0]
    return arr


# ======================================================================
def rate_to_time(
        arr,
        in_units='Hz',
        out_units='ms'):
    k = 1.0
    if in_units == 'kHz':
        k *= 1.0e3
    if out_units == 'ms':
        k *= 1.0e-3
    arr[arr != 0.0] = k / arr[arr != 0.0]
    return arr


# ======================================================================
def mag_phs_to_complex(mag_arr, phs_arr=None, fix_phase=True):
    """
    Convert magnitude and phase arrays into a complex array.

    It can automatically correct for arb.units in the phase.

    Args:
        mag_arr (np.ndarray): The magnitude image array in arb.units.
        phs_arr (np.ndarray): The phase image array in rad or arb.units.
            The values range is automatically corrected to radians.
            The wrapped data is expected.
            If units are radians, i.e. data is in the [-π, π) range,
            no conversion is performed.
            If None, only magnitude data is used.
        fix_phase (bool): Fix the phase interval / units.
            If True, `phs_arr` is corrected with `fix_phase_interval()`.

    Returns:
        cx_arr (np.ndarray): The complex image array in arb.units.

    See Also:
        pymrt.computation.fix_phase_interval
    """
    if phs_arr is not None:
        if fix_phase and not pmu.is_in_range(phs_arr, (-np.pi, np.pi)):
            phs_arr = pmu.scale(phs_arr.astype(float), (-np.pi, np.pi))
        cx_arr = pmu.polar2complex(mag_arr.astype(float), phs_arr.astype(float))
    else:
        cx_arr = mag_arr.astype(float)
    return cx_arr


# ======================================================================
def voxel_curve_fit(
        y_arr,
        x_arr,
        fit_func=None,
        fit_params=None,
        pre_func=None,
        pre_args=None,
        pre_kwargs=None,
        post_func=None,
        post_args=None,
        post_kwargs=None,
        method='curve_fit'):
    """
    Curve fitting for y = F(x, p)

    Args:
        y_arr (np.ndarray): Dependent variable with x dependence in the n-th dim
        x_arr (np.ndarray): Independent variable with same size as n-th dim of y
        fit_func (func):
        fit_params (iterable): The initial value(s) of the parameters to fit.
        pre_func (func):
        pre_args (list):
        pre_kwargs (dict):
        post_func (func):
        post_args (list):
        post_kwargs (dict):
        method (str): Method to use for the curve fitting procedure.

    Returns:
        p_arr (np.ndarray) :
    """
    # TODO: finish documentation

    # y_arr : ndarray ???
    #    Dependent variable (x dependence in the n-th dimension).
    # x_arr : ndarray ???
    #    Independent variable (same number of elements as the n-th dimension).

    # reshape to linearize the independent dimensions of the array
    support_axis = -1
    shape = y_arr.shape
    support_size = shape[support_axis]
    y_arr = y_arr.reshape((-1, support_size))
    num_voxels = y_arr.shape[0]
    p_arr = np.zeros((num_voxels, len(fit_params)))
    # preprocessing
    if pre_func is not None:
        if pre_args is None:
            pre_args = []
        if pre_kwargs is None:
            pre_kwargs = {}
        y_arr = pre_func(y_arr, *pre_args, **pre_kwargs)

    if method == 'curve_fit':
        iter_param_list = [
            (fit_func, x_arr, y_i_arr, fit_params)
            for y_i_arr in np.split(y_arr, support_size, support_axis)]
        pool = multiprocessing.Pool(multiprocessing.cpu_count())
        for i, (par_opt, par_cov) in \
                enumerate(pool.imap(pmu.curve_fit, iter_param_list)):
            p_arr[i] = par_opt

    elif method == 'poly':
        # polyfit requires to change matrix orientation using transpose
        p_arr = np.polyfit(x_arr, y_arr.transpose(), len(fit_params) - 1)
        # transpose the results back
        p_arr = np.transpose(p_arr)

    else:
        try:
            p_arr = fit_func(y_arr, x_arr, fit_params)
        except Exception as ex:
            print('WW: Exception "{}" in ndarray_fit() method "{}"'.format(
                ex, method))

    # revert to original shape
    p_arr = p_arr.reshape(list(shape[:support_axis]) + [len(fit_params)])
    # post process
    if post_func is not None:
        if post_args is None:
            post_args = []
        if post_kwargs is None:
            post_kwargs = {}
        p_arr = post_func(p_arr, *post_args, **post_kwargs)
    return p_arr
