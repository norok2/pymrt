#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pymrt.recipes.t2s: T2* reduced transverse relaxation computation.
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
import scipy as sp  # SciPy (signal and image processing library)

import scipy.integrate  # SciPy: Integration and ODEs
import scipy.optimize  # SciPy: Optimization and root finding

# :: Local Imports
import pymrt as mrt
# import pymrt.utils

from pymrt import INFO, DIRS
from pymrt import VERB_LVL, D_VERB_LVL, VERB_LVL_NAMES
from pymrt import elapsed, print_elapsed
from pymrt import msg, dbg

from pymrt.recipes import generic
from pymrt.recipes import quality
from pymrt.recipes.generic import (
    fix_magnitude_bias,
    func_exp_decay, fit_exp_tau, fit_exp_loglin, fit_exp_leasq,
    fit_exp_tau_quad, fit_exp_tau_diff, fit_exp_tau_quadr, fit_exp_tau_arlo,
    fit_exp_tau_loglin)


# ======================================================================
def fit_monoexp(
        arr,
        echo_times,
        echo_times_mask=None,
        method='quadr',
        prepare=fix_magnitude_bias):
    """
    Calculate the mono-exponential fit for T2* data.

    Args:
        arr (np.ndarray): The input array in arb.units.
            The echo time must vary in the last dimension and must match the
            length of `echo_times`.
        echo_times (iterable): The echo times in time units.
            The number of points must match the last shape size of arr.
        echo_times_mask (iterable[bool]|None): Determine the echo times to use.
            If None, all will be used.
        method (str): Determine the fitting method to use.
            Available options are:
             - 'auto': determine an optimal method by inspecting the data.
             - 'loglin': use a log-linear fit, fast but inaccurate and fragile.
             - 'leasq': use non-linear least square fit, slow but accurate.
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
    elif method == 'leasq':
        t2s_arr = fit_exp_leasq(arr, echo_times, echo_times_mask)['tau']
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
    return t2s_arr


# ======================================================================
def _test(use_cache=True):
    # x = np.linspace(1, 40, 5)
    x = np.array([2, 5, 7, 20, 40])
    tau_arr = np.linspace(2, 1000, 4000)
    a_arr = np.linspace(500, 4000, 4000)

    import pymrt.utils
    import os
    base_dir = mrt.utils.realpath('~/hd1/TEMP')
    filepath = os.path.join(base_dir, 'tau_arr.npz')
    if os.path.isfile(filepath) and use_cache:
        y = np.load(filepath)['y']
    else:
        y = np.zeros((len(tau_arr), len(a_arr), len(x)))
        for i, a in enumerate(a_arr):
            for j, tau in enumerate(tau_arr):
                y[j, i] = func_exp_decay(x, tau, a)
        np.savez(filepath, y=y)

    def eval_dist(a, b, axis=-1):
        mu = np.nanmean(a, axis) - b
        std = np.nanstd(a, axis)
        return np.mean(mu), np.mean(std)

    elapsed('gen_tau_phantom')

    snr = 20
    p = 1 / snr
    n = np.max(a_arr) * p * (np.random.random(y.shape) - 0.5)

    m = [True, True, False, False, False]

    # print(fit_exp_loglin(y + n, x)['tau'])
    # print(fit_exp_loglin(y + n, x, weighted=False)['tau'])
    # print(fit_exp_tau_quadr(y + n, x))

    print('quad', eval_dist(fit_exp_tau_quad(y + n, x, m), tau_arr))
    elapsed('quad')

    print('diff', eval_dist(fit_exp_tau_diff(y + n, x, m), tau_arr))
    elapsed('diff')

    print('quadr', eval_dist(fit_exp_tau_quadr(y + n, x, m), tau_arr))
    elapsed('quadr')

    print('quadr_w2',
          eval_dist(fit_exp_tau_quadr(y + n, x, m, window_size=2), tau_arr))
    elapsed('quadr_w2')

    print('quadr_w3',
          eval_dist(fit_exp_tau_quadr(y + n, x, m, window_size=3), tau_arr))
    elapsed('quadr_w3')

    print('arlo', eval_dist(fit_exp_tau_arlo(y + n, x, m), tau_arr))
    elapsed('arlo')

    print('loglin', eval_dist(fit_exp_loglin(y + n, x, m)['tau'], tau_arr))
    elapsed('loglin')

    print(
        'loglin_w',
        eval_dist(fit_exp_loglin(y + n, x, m, variant='weighted_reverse')['tau'],
                  tau_arr))
    elapsed('loglin_w')

    # print('leasq',
    #       eval_dist(fit_exp_leasq(y + n, x, init=[5, 4000])['tau'], tau_arr))
    # elapsed('leasq')

    print_elapsed()


# _test()

# ======================================================================
if __name__ == '__main__':
    import doctest

    msg(__doc__.strip())
    doctest.testmod()

else:
    elapsed()
