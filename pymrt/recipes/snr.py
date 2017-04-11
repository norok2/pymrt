#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
pymrt.recipes.snr: signal-to-noise ratio (SNR) computation.
"""

# ======================================================================
# :: Future Imports
from __future__ import (
    division, absolute_import, print_function, unicode_literals)

# ======================================================================
# :: Python Standard Library Imports
# import itertools  # Functions creating iterators for efficient looping
import collections  # Container datatypes
import warnings  # Warning control
import multiprocessing  # Process-based parallelism

# :: External Imports
import numpy as np  # NumPy (multidimensional numerical arrays library)
import scipy as sp  # SciPy (signal and image processing library)

import scipy.integrate  # SciPy: Integration and ODEs
import scipy.optimize  # SciPy: Optimization and root finding

# :: Local Imports
import pymrt as mrt
import pymrt.utils

from pymrt import INFO, DIRS
# from pymrt import VERB_LVL, D_VERB_LVL, VERB_LVL_NAMES
from pymrt import elapsed, print_elapsed
from pymrt import msg, dbg


# ======================================================================
def gaussian_autocorrelate(arr):
    snr_val, signal_peak, noise_mean, noise_std = 0, 0, 0, 0
    return snr_val, signal_peak, noise_mean, noise_std


# ======================================================================
def rician_percentile(
        arr,
        peak_percentile=99,
        weak_percentile=1, ):
    if np.issubdtype(arr.dtype, np.complex_):
        arr = np.abs(arr)
    signal_peak = np.percentile(arr, peak_percentile)
    weak_threshold = np.percentile(arr, weak_percentile)
    weak_mask = arr < weak_threshold
    noise_mean = np.mean(arr[weak_mask])
    noise_std = np.std(arr[weak_mask])
    snr_val = signal_peak / noise_std
    return snr_val, signal_peak, noise_mean, noise_std


# ======================================================================
if __name__ == '__main__':
    import doctest

    msg(__doc__.strip())
    doctest.testmod()

else:
    elapsed()
