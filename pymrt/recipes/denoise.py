#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pymrt.recipes.denoise: denoising computation.
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
import pywt as pw  # PyWavelets - Wavelet Transforms in Python

import scipy.interpolate  # Scipy: Interpolation
import scipy.ndimage  # SciPy: ND-image Manipulation

from skimage.restoration import (
    denoise_bilateral, denoise_nl_means, denoise_wavelet,
    denoise_tv_bregman, denoise_tv_chambolle)

# :: Local Imports
import pymrt as mrt
import pymrt.utils

from pymrt.recipes.generic import (
    fix_magnitude_bias,
    mag_phase_2_combine, cx_2_combine)
from pymrt.sequences import mp2rage


# from pymrt import VERB_LVL, D_VERB_LVL, VERB_LVL_NAMES
# from pymrt import elapsed, report
# from pymrt import msg, dbg


# ======================================================================
def standard(
        arr,
        method='wavelet',
        filter_kws=None,
        mode='cartesian'):
    """
    Perform standard single-data de-noising algorithms.

    Can be applied to complex data (see `cx_mode` for the exact behavior).

    Exposes several algorithms from `scipy.ndimage.filters` and
    `skimage.restoration`.

    Args:
        arr:
        method (str): Denoising method.
            Accepted values are:
             - 'gaussian': `scipy.ndimage.gaussian_filter()`
             - 'uniform': `scipy.ndimage.uniform_filter()`
             - 'median': `scipy.ndimage.median_filter()`
             - 'minimum': `scipy.ndimage.minimum_filter()`
             - 'maximum': `scipy.ndimage.maximum_filter()`
             - 'rank': `scipy.ndimage.rank_filter()`
             - 'percentile': `scipy.ndimage.percentile_filter()`
             - 'bilateral': `skimage.restoration.denoise_bilateral()`
             - 'nl_means': `skimage.restoration.denoise_nl_means()`
             - 'wavelet': `skimage.restoration.denoise_wavelet()`
             - 'tv_bregman': `skimage.restoration.denoise_tv_bregman()`
             - 'tv_chambolle': `skimage.restoration.denoise_tv_chambolle()`
        filter_kws:
        mode (str): Complex calculation mode.
            See `mode` parameter of `utils.filter_cx()` for more information.

    Returns:
        arr (np.ndarray): The denoised array.
    """
    method = method.lower()
    if filter_kws is None:
        filter_kws = {}
    if method == 'gaussian':
        if 'sigma' not in filter_kws:
            filter_kws['sigma'] = 1.0
        arr = mrt.utils.filter_cx(
            arr, sp.ndimage.gaussian_filter, (), filter_kws)
    elif method == 'uniform':
        arr = mrt.utils.filter_cx(
            arr, sp.ndimage.uniform_filter, (), filter_kws)
    elif method == 'median':
        arr = mrt.utils.filter_cx(
            arr, sp.ndimage.median_filter, (), filter_kws)
    elif method == 'minimum':
        arr = mrt.utils.filter_cx(
            arr, sp.ndimage.minimum_filter, (), filter_kws)
    elif method == 'maximum':
        arr = mrt.utils.filter_cx(
            arr, sp.ndimage.maximum_filter, (), filter_kws)
    elif method == 'rank':
        if 'rank' not in filter_kws:
            filter_kws['rank'] = 1
        arr = mrt.utils.filter_cx(
            arr, sp.ndimage.rank_filter, (), filter_kws)
    elif method == 'percentile':
        if 'percentile' not in filter_kws:
            filter_kws['percentile'] = 50
        arr = mrt.utils.filter_cx(
            arr, sp.ndimage.percentile_filter, (), filter_kws)
    elif method == 'bilateral':
        arr = mrt.utils.filter_cx(arr, denoise_bilateral, (), filter_kws)
    elif method == 'nl_means':
        arr = mrt.utils.filter_cx(arr, denoise_nl_means, (), filter_kws)
    elif method == 'wavelet':
        arr = mrt.utils.filter_cx(arr, denoise_wavelet, (), filter_kws)
    elif method == 'tv_bregman':
        if 'weight' not in filter_kws:
            filter_kws['weight'] = 1
        arr = mrt.utils.filter_cx(arr, denoise_tv_bregman, (), filter_kws)
    elif method == 'tv_chambolle':
        arr = mrt.utils.filter_cx(arr, denoise_tv_chambolle, (), filter_kws)
    return arr
