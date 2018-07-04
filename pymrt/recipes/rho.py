#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pymrt.recipes.rho: spin density computation.
"""

# ======================================================================
# :: Future Imports
from __future__ import (
    division, absolute_import, print_function, unicode_literals, )

# ======================================================================
# :: Python Standard Library Imports
# import itertools  # Functions creating iterators for efficient looping
import warnings  # Warning control
import collections  # Container datatypes

# :: External Imports
import numpy as np  # NumPy (multidimensional numerical arrays library)

# :: Local Imports
import pymrt as mrt
# import pymrt.utils
import pymrt.correction

# from pymrt import VERB_LVL, D_VERB_LVL, VERB_LVL_NAMES
# from pymrt import elapsed, report
# from pymrt import msg, dbg

from pymrt.recipes import generic
from pymrt.recipes import quality
from pymrt.recipes.generic import (
    fix_phase_interval, rate_to_time, time_to_rate,
    func_exp_decay, fit_exp_tau, fit_exp_loglin, fit_exp_curve_fit,
    fit_exp_tau_quad, fit_exp_tau_diff, fit_exp_tau_quadr, fit_exp_tau_arlo,
    fit_exp_tau_loglin)


# ======================================================================
def fit_multiecho_mono(
        arr,
        echo_times,
        echo_times_mask=None,
        method='quadr',
        prepare=mrt.correction.fix_bias_rician):
    """
    Calculate the mono-exponential fit for T2 data.

    This is also suitable for T2* data from multi-echo FLASH acquisitions.

    Args:
        arr (np.ndarray): The input array in arb. units.
            The echo time must vary in the last dimension and must match the
            length of `echo_times`.
        echo_times (Iterable): The echo times in time units.
            The number of points must match the last shape size of arr.
        echo_times_mask (Iterable[bool]|None): Determine the echo times to use.
            If None, all will be used.
        method (str): Determine the fitting method to use.
            Accepted values are:
             - 'auto': determine an optimal method by inspecting the data.
             - 'loglin': use a log-linear fit, fast but inaccurate and fragile.
             - 'curve_fit': use non-linear least square curve fitting, slow
               but accurate.
             - 'diff': closed-form solution using the differential properties
               of the exponential, very fast but inaccurate.
             - 'quad': closed-form solution using the quadrature properties
               of the exponential, very fast and moderately accurate.
             - 'arlo': closed-form solution using the `Auto-Regression on
               Linear Operations (ARLO)` method (similar to 'quad'),
               very fast and accurate.
             - 'quadr': closed-form solution using the quadrature properties
               of the exponential and optimal noise regression,
               very fast and very accurate (extends both 'quad' and 'arlo').
        prepare (callable|None): Input array preparation.
            Must have the signature: f(np.ndarray) -> np.ndarray.
            Useful for data pre-whitening, including for example the
            correction of magnitude data from Rician mean bias.

    Returns:
        t2s_arr (np.ndarray): The output array.
    """
    methods = ('auto', 'loglin', 'leasq', 'diff', 'quad', 'arlo', 'quadr')

    # data pre-whitening
    arr = prepare(arr) if prepare else arr.astype(float)

    if method == 'auto':
        t2s_arr = fit_exp_tau(arr, echo_times, echo_times_mask)
    elif method == 'loglin':
        t2s_arr = fit_exp_tau_loglin(arr, echo_times, echo_times_mask)
    elif method == 'curve_fit':
        t2s_arr = fit_exp_curve_fit(arr, echo_times, echo_times_mask)['tau']
    elif method == 'quad':
        t2s_arr = fit_exp_tau_quad(arr, echo_times, echo_times_mask)
    elif method == 'diff':
        t2s_arr = fit_exp_tau_diff(arr, echo_times, echo_times_mask)
    elif method == 'quadr':
        t2s_arr = fit_exp_tau_quadr(arr, echo_times, echo_times_mask)
    elif method == 'arlo':
        t2s_arr = fit_exp_tau_arlo(arr, echo_times, echo_times_mask)
    else:
        raise ValueError(
            'valid methods are: {} (given: {})'.format(methods, method))
    return rho_arr


def b1r_correction(rho_arr, b1r_arr):

    return rho_arr / b1r_arr
