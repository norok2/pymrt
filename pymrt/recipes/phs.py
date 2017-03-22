#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
pymrt.recipes.phs: phase manipulation algorithms.
"""

# ======================================================================
# :: Future Imports
from __future__ import (
    division, absolute_import, print_function, unicode_literals)

# ======================================================================
# :: Python Standard Library Imports
import itertools  # Functions creating iterators for efficient looping
import warnings  # Warning control
import collections  # Container datatypes

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

from pymrt.recipes import generic


# ======================================================================
def fix_phase_interval(arr):
    """
    Ensure that the range of values is interpreted as valid phase information.

    This is useful for DICOM-converted images (without post-processing).

    Args:
        arr (np.ndarray): Array to be processed.

    Returns:
        array (np.ndarray): An array scaled to (-pi,pi).

    Examples:
        >>> fix_phase_interval(np.arange(8))
        array([-3.14159265, -2.24399475, -1.34639685, -0.44879895,  0.44879895,
                1.34639685,  2.24399475,  3.14159265])
        >>> fix_phase_interval(np.array([-10, -5, 0, 5, 10]))
        array([-3.14159265, -1.57079633,  0.        ,  1.57079633,  3.14159265])
        >>> fix_phase_interval(np.array([-10, 10, 1, -3]))
        array([-3.14159265,  3.14159265,  0.31415927, -0.9424778 ])
    """
    if not pmu.is_in_range(arr, (-np.pi, np.pi)):
        arr = pmu.scale(arr.astype(float), (-np.pi, np.pi))
    return arr


# ======================================================================
def phs_to_dphs_multi(
        phs_arr,
        tis,
        num=1,
        full=False,
        exp_factor=None,
        zero_cutoff=None):
    """
    Calculate the polynomial components of the phase variation from phase data.

    Args:
        phs_arr (np.ndarray): The input array in arb.units.
            The sampling time Ti varies in the last dimension.
        tis (iterable): The sampling times Ti in time units.
            The number of points must match the last shape size of arr.
        num (int): The degree of the polynomial to fit.
            For monoexponential fits, use num=1.
        full (bool): Calculate additional information on the fit performance.
            If True, more information is given.
            If False, only the optimized parameters are returned.
        exp_factor (float|None):
        zero_cutoff (float|None):

    Returns:
        results (dict):
    """
    y_arr = np.array(phs_arr).astype(float)
    x_arr = np.array(tis).astype(float)

    assert (x_arr.size == y_arr.shape[-1])

    p_arr = generic.voxel_curve_fit(
        y_arr, x_arr,
        None, (np.mean(y_arr),) + (np.mean(x_arr),) * num, method='poly')

    p_arrs = np.split(p_arr, num + 1, -1)

    results = collections.OrderedDict(
        ('s0' if i == 0 else 'dphs_{i}'.format(i=i), x)
        for i, x in enumerate(p_arrs[::-1]))

    if full:
        warnings.warn('E: Not implemented yet!')

    return results


# ======================================================================
def phs_to_dphs(
        phs_arr,
        tis):
    """
    Calculate the phase variation from phase data.

    Args:
        phs_arr (np.ndarray): The input array in arb.units.
            The sampling time Ti varies in the last dimension.
        tis (iterable): The sampling times Ti in time units.
            The number of points must match the last shape size of arr.

    Returns:
        dphs_arr (np.ndarray): The phase variation in rad/s.
    """
    return phs_to_dphs_multi(phs_arr, tis)['dphs_1']


# ======================================================================
def unwrap_laplacian(
        arr,
        pre_func=fix_phase_interval,
        pre_args=None,
        pre_kws=None,
        post_func=lambda x: x - np.median(x[x != 0.0]),
        post_args=None,
        post_kws=None,
        pad_width=0):
    """
    Super-fast multi-dimensional Laplacian-based Fourier unwrapping.

    Phase unwrapping by using the following equality:

    L = (d / dx)^2

    L(phi) = cos(phi) * L(sin(phi)) - sin(phi) * L(cos(phi))

    phi = IL(L(phi)) = IL(cos(phi) * L(sin(phi)) - sin(phi) * L(cos(phi)))

    Args:
        arr (np.ndarray): The multi-dimensional array to unwrap.
        pre_func (callable): A correction function for improved accuracy.
        pad_width (float|int): Size of the padding to use.
            This is useful for mitigating border effects.
            If int, it is interpreted as absolute size.
            If float, it is interpreted as relative to the maximum size.

    Returns:
        arr (np.ndarray): The multi-dimensional unwrapped array.

    See Also:
        Schofield, M. A. and Y. Zhu (2003). Optics Letters 28(14): 1194-1196.
    """
    if pre_func:
        arr = pre_func(
            arr,
            *(pre_args if pre_args else ()),
            **(pre_kws if pre_kws else {}))

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
    if post_func:
        arr = post_func(
            arr,
            *(post_args if post_args else ()),
            **(post_kws if post_kws else {}))
    return arr


# ======================================================================
def unwrap_sorting_path(
        arr,
        pre_func=fix_phase_interval,
        pre_args=None,
        pre_kws=None,
        post_func=lambda x: x - np.median(x[x != 0.0]),
        post_args=None,
        post_kws=None,
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

    if pre_func:
        arr = pre_func(
            arr,
            *(pre_args if pre_args else ()),
            **(pre_kws if pre_kws else {}))

    if unwrap_axes:
        loop_gen = [[slice(None)] if j in unwrap_axes else range(dim)
                    for j, dim in enumerate(arr.shape)]
    else:
        loop_gen = [slice(None)] * arr.ndim
    for indexes in itertools.product(*loop_gen):
        arr[indexes] = unwrap_phase(arr[indexes], wrap_around, seed)

    if post_func:
        arr = post_func(
            arr,
            *(post_args if post_args else ()),
            **(post_kws if post_kws else {}))
    return arr
