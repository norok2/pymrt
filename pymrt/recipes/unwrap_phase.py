#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
pymrt.recipes.unwrap_phase: phase unwrapping algorithms.
"""

# ======================================================================
# :: Future Imports
from __future__ import (
    division, absolute_import, print_function, unicode_literals)

# ======================================================================
# :: Python Standard Library Imports


# :: External Imports
import numpy as np  # NumPy (multidimensional numerical arrays library)

from numpy.fft import fftshift, ifftshift
from scipy.fftpack import fftn, ifftn

# :: Local Imports
import pymrt.base as pmb

from pymrt import VERB_LVL, D_VERB_LVL, VERB_LVL_NAMES
from pymrt import elapsed, print_elapsed
from pymrt import msg, dbg


# ======================================================================
def unwrap_phase_laplacian(
        arr,
        correction=lambda x: x - np.median(x),
        pad_width=0):
    """
    Super-fast multi-dimensional Laplacian-based Fourier unwrapping.

    Phase unwrapping by using the following equality:

    L = (d / dx)^2

    L(phi) = cos(phi) * L(sin(phi)) - sin(phi) * L(cos(phi))

    phi = IL(L(phi)) = IL(cos(phi) * L(sin(phi)) - sin(phi) * L(cos(phi)))

    Args:
        arr (np.ndarray): The multi-dimensional array to unwrap.
        correction (callable): A correction function for improved accuracy.
        pad_width (float|int): Size of the padding to use.
            This is useful for mitigating border effects.
            If int, it is interpreted as absolute size.
            If float, it is interpreted as relative to the maximum size.

    Returns:
        arr (np.ndarray): The multi-dimensional unwrapped array.

    See Also:
        Schofield, M. A. and Y. Zhu (2003). Optics Letters 28(14): 1194-1196.
    """
    if pad_width:
        shape = arr.shape
        pad_width = pmb.auto_pad_width(pad_width, shape)
        mask = [slice(lower, -upper) for (lower, upper) in pad_width]
        arr = np.pad(arr, pad_width, 'constant', constant_values=0)
    else:
        mask = [slice(None)] * arr.ndim

    # from pymrt.base import laplacian, inv_laplacian
    # from numpy import real, sin, cos
    # arr = real(inv_laplacian(
    #     cos(arr) * laplacian(sin(arr)) - sin(arr) * laplacian(cos(arr))))

    cos_arr = np.cos(arr)
    sin_arr = np.sin(arr)
    kk_2 = fftshift(pmb._kk_2(arr.shape))
    arr = fftn(cos_arr * ifftn(kk_2 * fftn(sin_arr)) -
               sin_arr * ifftn(kk_2 * fftn(cos_arr)))
    kk_2[kk_2 != 0] = 1.0 / kk_2[kk_2 != 0]
    arr *= kk_2
    del cos_arr, sin_arr, kk_2
    arr = np.real(ifftn(arr))

    arr = arr[mask]
    if correction:
        arr = correction(arr)
    return arr


# ======================================================================
def unwrap_phase_sorting_path(
        arr,
        correction=lambda x: x - np.median(x[x != 0.0]),
        wrap_around=False,
        seed=0):
    """
    2D/3D unwrap using sorting by reliability following a non-continous path.

    This is a wrapper around the function skimage.restoration.unwrap_phase

    Args:
        arr (np.ndarray): The multi-dimensional array to unwrap.
        correction (callable): A correction function for improved accuracy.
        wrap_around (bool|iterable[bool]|None): Circular unwrapping.
            See also: skimage.restoration.unwrap_phase.
        seed (int|None): Randomization seed.
            See also: skimage.restoration.unwrap_phase.

    Returns:
        arr (np.ndarray): The multi-dimensional unwrapped array.

    See Also:
        skimage.restoration.unwrap_phase
        Herraez, M. A. et al. (2002). Journal Applied Optics 41(35): 7437.
    """
    from skimage.restoration import unwrap_phase
    arr = unwrap_phase(arr, wrap_around, seed)
    if correction:
        arr = correction(arr)
    return arr
