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
import itertools  # Functions creating iterators for efficient looping

# :: External Imports
import numpy as np  # NumPy (multidimensional numerical arrays library)

from numpy.fft import fftshift, ifftshift
from scipy.fftpack import fftn, ifftn

# :: Local Imports
import pymrt.utils as pmu
import pymrt.computation as pmc

from pymrt import VERB_LVL, D_VERB_LVL, VERB_LVL_NAMES
from pymrt import elapsed, print_elapsed
from pymrt import msg, dbg


# ======================================================================
def unwrap_phase_laplacian(
        arr,
        preprocess=pmc.fix_phase_interval,
        preprocess_args=None,
        preprocess_kws=None,
        postprocess=lambda x: x - np.median(x[x != 0.0]),
        postprocess_args=None,
        postprocess_kws=None,
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
    if preprocess:
        if not preprocess_args:
            preprocess_args = ()
        if not preprocess_kws:
            preprocess_kws = {}
        arr = preprocess(arr, *preprocess_args, **preprocess_kws)

    if pad_width:
        shape = arr.shape
        pad_width = pmu.auto_pad_width(pad_width, shape)
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
    kk_2 = fftshift(pmu._kk_2(arr.shape))
    arr = fftn(cos_arr * ifftn(kk_2 * fftn(sin_arr)) -
               sin_arr * ifftn(kk_2 * fftn(cos_arr)))
    kk_2[kk_2 != 0] = 1.0 / kk_2[kk_2 != 0]
    arr *= kk_2
    del cos_arr, sin_arr, kk_2
    arr = np.real(ifftn(arr))

    arr = arr[mask]
    if postprocess:
        if not postprocess_args:
            postprocess_args = ()
        if not postprocess_kws:
            postprocess_kws = {}
        arr = postprocess(arr, *postprocess_args, **postprocess_kws)
    return arr


# ======================================================================
def unwrap_phase_sorting_path(
        arr,
        preprocess=pmc.fix_phase_interval,
        preprocess_args=None,
        preprocess_kws=None,
        postprocess=lambda x: x - np.median(x[x != 0.0]),
        postprocess_args=None,
        postprocess_kws=None,
        unwrap_axes=(0, 1, 2),
        wrap_around=False,
        seed=0):
    """
    2D/3D unwrap using sorting by reliability following a non-continous path.

    This is a wrapper around the function skimage.restoration.unwrap_phase

    Args:
        arr (np.ndarray): The multi-dimensional array to unwrap.`
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
    if preprocess:
        if not preprocess_args:
            preprocess_args = ()
        if not preprocess_kws:
            preprocess_kws = {}
        arr = preprocess(arr, *preprocess_args, **preprocess_kws)
    if unwrap_axes:
        loop_gen = [[slice(None)] if j in unwrap_axes else range(dim)
            for j, dim in enumerate(arr.shape)]
    else:
        loop_gen = [slice(None)] * arr.ndim
    for indexes in itertools.product(*loop_gen):
        arr[indexes] = unwrap_phase(arr[indexes], wrap_around, seed)
    if postprocess:
        if not postprocess_args:
            postprocess_args = ()
        if not postprocess_kws:
            postprocess_kws = {}
        arr = postprocess(arr, *postprocess_args, **postprocess_kws)
    return arr
