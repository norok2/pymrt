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

    Raises:
        ValueError: If `method` is unknown.

    Returns:
        arr (np.ndarray): The denoised array.
    """
    method = method.lower()
    if filter_kws is None:
        filter_kws = {}

    if method == 'gaussian':
        if 'sigma' not in filter_kws:
            filter_kws['sigma'] = 1.0
        filter_func = sp.ndimage.gaussian_filter
    elif method == 'uniform':
        filter_func = sp.ndimage.uniform_filter
    elif method == 'median':
        filter_func = sp.ndimage.median_filter
    elif method == 'minimum':
        filter_func = sp.ndimage.minimum_filter
    elif method == 'maximum':
        filter_func = sp.ndimage.maximum_filter
    elif method == 'rank':
        if 'rank' not in filter_kws:
            filter_kws['rank'] = 1
        filter_func = sp.ndimage.rank_filter
    elif method == 'percentile':
        if 'percentile' not in filter_kws:
            filter_kws['percentile'] = 50
        filter_func = sp.ndimage.percentile_filter
    elif method == 'bilateral':
        filter_func = denoise_bilateral
    elif method == 'nl_means':
        filter_func = denoise_nl_means
    elif method == 'wavelet':
        filter_func = denoise_wavelet
    elif method == 'tv_bregman':
        if 'weight' not in filter_kws:
            filter_kws['weight'] = 1
        filter_func = denoise_tv_bregman
    elif method == 'tv_chambolle':
        filter_func = denoise_tv_chambolle
    else:
        text = 'Unknown method `{}` in `denoise.standard()`.'.format(method)
        raise ValueError(text)

    if np.iscomplex(arr):
        arr = mrt.utils.filter_cx(arr, filter_func, (), filter_kws)
    else:
        arr = filter_func(arr, **filter_kws)
    return arr
