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

# from skimage.restoration import (
#     denoise_bilateral, denoise_nl_means, denoise_wavelet,
#     denoise_tv_bregman, denoise_tv_chambolle)


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


# NOTES:
# - Several filters from `scipy.ndimage` can be used for denoising:
#   - 'scipy.ndimage.gaussian_filter()`
#   - 'scipy.ndimage.uniform_filter()`
#   - 'scipy.ndimage.median_filter()`
#   - 'scipy.ndimage.minimum_filter()`
#   - 'scipy.ndimage.maximum_filter()`
#   - 'scipy.ndimage.rank_filter()`
#   - 'scipy.ndimage.percentile_filter()`
# - Specialized denoising filter exist also in `scikit.image`:
#   - 'skimage.restoration.denoise_bilateral()`
#   - 'skimage.restoration.denoise_nl_means()`
#   - 'skimage.restoration.denoise_wavelet()`
#   - 'skimage.restoration.denoise_tv_bregman()`
#   - 'skimage.restoration.denoise_tv_chambolle()`
