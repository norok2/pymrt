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
    fix_phase_interval, rate_to_time, time_to_rate, referencing,
    func_exp_decay, fit_exp, fit_exp_loglin, fit_exp_curve_fit,
    fit_exp_quad, fit_exp_diff, fit_exp_quadr, fit_exp_arlo)


# ======================================================================
def fit_multiecho_mono(
        arr,
        echo_times,
        echo_times_mask=None,
        method='quadr',
        method_kws=None,
        b1r_arr=None,
        ref_mask=None,
        ref_val=None,
        prepare=mrt.correction.fix_bias_rician):
    """
    Calculate the mono-exponential amplitude from fit of multi-echo data.

    This is suitable for both spin echo and gradient echo sequences.
    Note that stimulated echoes effects are ignored.

    Args:
        arr (np.ndarray): The input array in arb. units.
            The echo time must vary in the last dimension and must match the
            length of `echo_times`.
        echo_times (Iterable): The echo times in time units.
            The number of points must match the last shape size of arr.
        echo_times_mask (Iterable[bool]|None): Determine the echo times to use.
            If None, all will be used.
        method (str): Determine the fitting method to use.
            See `recipes.generic.fit_exp()` for more info.
        method_kws (dict|tuple|None): Keyword arguments to pass to `method`.
        b1r_arr (np.ndarray): The receive profile in arb. units.
            Assumes that units of time is ms and units of rates is Hz.
        ref_mask (tuple[int]|np.ndarray|None): The reference mask.
            Values are scaled so that the average inside the mask has
            the value specified in `ref_val`.
        ref_val (int|float|None): The external reference value.
            This is typically set to 100.0 so that the map is in percent.
            If None, no referencing is performed.
        prepare (callable|None): Input array preparation.
            Must have the signature: f(np.ndarray) -> np.ndarray.
            Useful for data pre-whitening, including for example the
            correction of magnitude data from Rician mean bias.

    Returns:
        rho_arr (np.ndarray): The output array.
    """
    # data pre-whitening
    arr = prepare(arr) if prepare else arr.astype(float)
    # compute rho
    rho_arr = fit_exp(
        arr, echo_times, echo_times_mask, method, method_kws)['s0']
    # perform receive profile correction
    if b1r_arr is not None:
        rho_arr = b1r_correction(rho_arr, b1r_arr)
    # perform referencing
    if ref_val is not None:
        rho_arr = referencing(
            arr, [ref_mask], [ref_val], np.mean,
            lambda x, int_refs, ext_refs: x / int_refs[0] * ext_refs[0])
    return rho_arr


# ======================================================================
def b1r_correction(
        rho_arr,
        b1r_arr):
    """
    Compute the receive profile correction for spin density mapping.

    The input arrays must be already registered.

    Args:
        rho_arr (np.ndarray): The uncorrected spin density map in arb. units.
        b1r_arr (np.ndarray):

    Returns:

    """
    return rho_arr / b1r_arr
