#!/usr/bin/env python3
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
import pymrt as mrt
import pymrt.utils
import pymrt.computation as pmc

from pymrt import VERB_LVL, D_VERB_LVL, VERB_LVL_NAMES
from pymrt import elapsed, print_elapsed
from pymrt import msg, dbg

from pymrt.recipes import generic
from pymrt.recipes.generic import fix_phase_interval as fix_interval


# ======================================================================
def fix_offset(
        arr,
        offset_estimator=np.median):
    """
    Remove the constant phase offset from the phase data.
    
    By default, the constant phase offset is estimated with the median.
    
    Args:
        arr (np.ndarray): The input array.
        offset_estimator (callable): The function to estimate the offset.

    Returns:
        arr (np.ndarray): The output array.
    """
    return arr - offset_estimator(arr)


# ======================================================================
def phs_to_dphs_multi(
        phs_arr,
        tis,
        tis_mask=None,
        unwrap=None,
        poly_deg=1,
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
        tis_mask (iterable[bool]|None): Determine the sampling times Ti to use.
            If None, all will be used.
        unwrap (bool|callable|None): Determine unwrapping method.
            If None, no unwrapping or fixing is performed (assume units is rad).
            If False, data is not unwrapped but values range fix is performed.
            If True, both N-dim unwrapping of data and values range fix are
            performed.
            If callable, the data is preprocessed through.
            If values range fix is required, please consider
            including it in the callable through `fix_interval()`.
        poly_deg (int): The degree of the polynomial to fit.
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

    if tis_mask:
        y_arr = y_arr[..., tis_mask]
        x_arr = x_arr[tis_mask]

    # unwrap along the time evolution
    if isinstance(unwrap, callable):
        y_arr = unwrap(y_arr)
    elif unwrap is None:
        pass
    elif unwrap:
        y_arr = fix_interval(y_arr)
        y_arr = unwrap_laplacian(y_arr, post_func=None)
    else:
        y_arr = fix_interval(y_arr)

    assert (x_arr.size == y_arr.shape[-1])

    p_arr = generic.voxel_curve_fit(
        y_arr, x_arr, fit_params=1 + poly_deg, method='poly')

    p_arrs = np.split(p_arr, poly_deg + 1, -1)

    results = collections.OrderedDict(
        ('s0' if i == 0 else 'dphs_{i}'.format(i=i), x)
        for i, x in enumerate(p_arrs[::-1]))

    if full:
        warnings.warn('E: Not implemented yet!')

    return results


# ======================================================================
def phs_to_dphs(
        phs_arr,
        tis,
        tis_mask,
        units='ms'):
    """
    Calculate the phase variation from phase data.

    Args:
        phs_arr (np.ndarray): The input array in arb.units.
            The sampling time Ti varies in the last dimension.
            Arbitrary units are accepted, will be automatically converted to
            radians under the assumption that data is wrapped.
            Do not provide unwrapped data.
        tis (iterable|int|float): The sampling times Ti in time units.
            The number of points must match the last shape size of arr.
        tis_mask (iterable[bool]|None): Determine the sampling times Ti to use.
            If None, all will be used.
        units (str|float|int): Units of measurement of Ti.
            If str, the following will be accepted: 'ms'
            If int or float, the conversion factor will be multiplied to `ti`.

    Returns:
        dphs_arr (np.ndarray): The phase variation in rad/s.
    """
    if isinstance(units, str):
        if units == 'ms':
            units = 1e-3
        else:
            warnings.warn('Invalid units `{units}`'.format_map(locals()))
            units = 1
    tis = np.array(mrt.utils.auto_repeat(tis)) * units
    if len(tis) > 1:
        dphs_arr = \
            phs_to_dphs_multi(phs_arr, tis, tis_mask, poly_deg=1)['dphs_1']
    else:
        dphs_arr = phs_arr / tis[0]
    return dphs_arr


# ======================================================================
def dphs_to_phs(
        dphs_arr,
        tis,
        phs0_arr=None):
    """
    Calculate the phase variation from phase data.

    Args:
        phs_arr (np.ndarray): The input array in arb.units.
            The sampling time Ti varies in the last dimension.
        tis (iterable|int|float): The sampling times Ti in time units.
            The number of points must match the last shape size of arr.
        tis_mask (iterable[bool]|None): Determine the sampling times Ti to use.
            If None, all will be used.
        units (str|float|int): Units of measurement of Ti.
            If str, the following will be accepted: 'ms'
            If int or float, the conversion factor will be multiplied to `ti`.

    Returns:
        phs_arr (np.ndarray): The phase array in rad.
    """
    tis = np.array(mrt.utils.auto_repeat(tis))
    phs_arr = np.zeros(dphs_arr.shape + (len(tis),))
    for i, ti in enumerate(tis):
        phs_arr[..., i] = dphs_arr * ti


# ======================================================================
def unwrap_laplacian(
        arr,
        pre_func=fix_interval,
        pre_args=None,
        pre_kws=None,
        post_func=fix_offset,
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
        pre_func (callable|None): Preprocessing function for the input.
        pre_args (iterable|None): Positional arguments of preprocess function.
        pre_kws (iterable|None): Keyword arguments of preprocess function.
        post_func (callable|None): Postprocessing function for the output.
        post_args (iterable|None): Positional arguments of postprocess function.
        post_kws (iterable|None): Keyword arguments of postprocess function.
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
        pad_width = mrt.utils.auto_pad_width(pad_width, shape)
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
    kk_2 = fftshift(mrt.utils._kk_2(arr.shape))
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
        pre_func=fix_interval,
        pre_args=None,
        pre_kws=None,
        post_func=fix_offset,
        post_args=None,
        post_kws=None,
        unwrap_axes=(0, 1, 2),
        wrap_around=False,
        seed=0):
    """
    2D/3D unwrap using sorting by reliability following a non-continous path.

    This is a wrapper around the function `skimage.restoration.unwrap_phase`
    
    If higher dimensionality input, loop through extra dimensions.

    Args:
        arr (np.ndarray): The multi-dimensional array to unwrap.`
        pre_func (callable|None): Preprocessing function for the input.
        pre_args (iterable|None): Positional arguments of preprocess function.
        pre_kws (iterable|None): Keyword arguments of preprocess function.
        post_func (callable|None): Postprocessing function for the output.
        post_args (iterable|None): Positional arguments of postprocess function.
        post_kws (iterable|None): Keyword arguments of postprocess function.
        unwrap_axes (tuple[int]): Axes along which unwrapping is performed.
            Must have length 2 or 3.
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
