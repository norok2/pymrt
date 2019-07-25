#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pymrt.recipes.t2: T2 transverse relaxation computation.
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
import scipy as sp  # SciPy (signal and image processing library)
import flyingcircus as fc  # Everything you always wanted to have in Python.*

# :: External Imports Submodules
import scipy.integrate  # SciPy: Integration and ODEs
import scipy.optimize  # SciPy: Optimization and root finding

# :: Local Imports
import pymrt as mrt
import pymrt.correction

from pymrt import INFO, PATH
from pymrt import VERB_LVL, D_VERB_LVL, VERB_LVL_NAMES
from pymrt import elapsed, report, run_doctests
from pymrt import msg, dbg, fmt, fmtm

from pymrt.recipes import generic
from pymrt.recipes import quality
from pymrt.recipes.generic import (
    fix_phase_interval, rate_to_time, time_to_rate,
    func_exp_decay, fit_exp, fit_exp_loglin, fit_exp_curve_fit,
    fit_exp_quad, fit_exp_diff, fit_exp_quadr, fit_exp_arlo)


# ======================================================================
def fit_multiecho_mono(
        arr,
        echo_times,
        echo_times_mask=None,
        method='quadr',
        method_kws=None,
        invert_tau=False,
        prepare=mrt.correction.fix_bias_rician):
    """
    Calculate the mono-exponential time constant from fit of multi-echo data.

    This is also suitable for computing T2 and T2*:
    - T2 from a multi-echo spin echo sequence (ignoring stimulated echoes)
    - T2* from multi-echo FLASH acquisitions.

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
        invert_tau (bool): Invert tau results to convert times to rates.
            Assumes that units of time is ms and units of rates is Hz.
        prepare (callable|None): Input array preparation.
            Must have the signature: f(np.ndarray) -> np.ndarray.
            Useful for data pre-whitening, including for example the
            correction of magnitude data from Rician mean bias.

    Returns:
        t2_arr (np.ndarray): The output array.
    """
    # data pre-whitening
    arr = prepare(arr) if prepare else arr.astype(float)
    # compute t2
    t2_arr = fit_exp(
        arr, echo_times, echo_times_mask, method, method_kws)['tau']
    # compute relaxation rates instead of times.
    if invert_tau:
        t2_arr = time_to_rate(t2_arr, 'ms', 'Hz')
    return t2_arr


# ======================================================================
def _test(use_cache=True):
    # x = np.linspace(1, 40, 5)
    x = np.array([2, 5, 7, 20, 40])
    tau_arr = np.linspace(2, 1000, 4000)
    a_arr = np.linspace(500, 4000, 4000)

    import pymrt.utils
    import os


    base_dir = fc.base.realpath('~/hd1/TEMP')
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

    print('quad', eval_dist(fit_exp_quad(y + n, x, m)['tau'], tau_arr))
    elapsed('quad')

    print('diff', eval_dist(fit_exp_diff(y + n, x, m)['tau'], tau_arr))
    elapsed('diff')

    print('quadr', eval_dist(fit_exp_quadr(y + n, x, m)['tau'], tau_arr))
    elapsed('quadr')

    print('quadr_w2',
          eval_dist(fit_exp_quadr(y + n, x, m, window_size=2)['tau'], tau_arr))
    elapsed('quadr_w2')

    print('quadr_w3',
          eval_dist(fit_exp_quadr(y + n, x, m, window_size=3)['tau'], tau_arr))
    elapsed('quadr_w3')

    print('arlo', eval_dist(fit_exp_arlo(y + n, x, m)['tau'], tau_arr))
    elapsed('arlo')

    print('loglin', eval_dist(fit_exp_loglin(y + n, x, m)['tau'], tau_arr))
    elapsed('loglin')

    print(
        'loglin_w',
        eval_dist(
            fit_exp_loglin(y + n, x, m, variant='weighted_reverse')['tau'],
            tau_arr))
    elapsed('loglin_w')

    # print('leasq',
    #       eval_dist(fit_exp_curve_fit(y + n, x, init=[5, 4000])['tau'],
    # tau_arr))
    # elapsed('leasq')

    msg(report())


# ======================================================================
elapsed(__file__[len(PATH['base']) + 1:])

# ======================================================================
if __name__ == '__main__':
    run_doctests(__doc__)
