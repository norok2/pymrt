#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pymrt.recipes.phs: phase manipulation algorithms.
"""

# ======================================================================
# :: Future Imports
from __future__ import (
    division, absolute_import, print_function, unicode_literals, )

# ======================================================================
# :: Python Standard Library Imports
import itertools  # Functions creating iterators for efficient looping
import warnings  # Warning control
import collections  # Container datatypes

# :: External Imports
import scipy as sp  # SciPy (signal and image processing library)
import numpy as np  # NumPy (multidimensional numerical arrays library)
import flyingcircus as fc  # Everything you always wanted to have in Python.*

# :: External Imports Submodules
import scipy.ndimage  # SciPy: ND-image Manipulation
from numpy.fft import fftshift, ifftshift
from scipy.fftpack import fftn, ifftn
import flyingcircus.util  # FlyingCircus: generic basic utilities
import flyingcircus.num  # FlyingCircus: generic numerical utilities

# :: Local Imports
import pymrt as mrt
import pymrt.utils

from pymrt import INFO, PATH
from pymrt import VERB_LVL, D_VERB_LVL, VERB_LVL_NAMES
from pymrt import elapsed, report
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
        poly_deg=1,
        full=False):
    """
    Calculate the polynomial components of the phase variation from phase data.

    Args:
        phs_arr (np.ndarray): The input array in rad.
            The sampling time Ti varies in the last dimension.
            The data is assumed to be already unwrapped.
        tis (Iterable): The sampling times Ti in time units.
            The number of points must match the last shape size of arr.
        tis_mask (Iterable[bool]|None): Determine the sampling times Ti to use.
            If None, all will be used.
        poly_deg (int): The degree of the polynomial to fit.
            For monoexponential fits, use num=1.
        full (bool): Calculate additional information on the fit performance.
            If True, more information is given.
            If False, only the optimized parameters are returned.

    Returns:
        results (dict):
    """
    y_arr = np.array(phs_arr).astype(float)
    x_arr = np.array(tis).astype(float)

    if tis_mask is not None:
        y_arr = y_arr[..., tis_mask]
        x_arr = x_arr[tis_mask]

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
        tis_mask=None,
        unwrapping='laplacian',
        unwrapping_kws=None,
        units='ms'):
    """
    Calculate the phase variation from phase data.

    Args:
        phs_arr (np.ndarray): The input array in arb. units.
            The sampling time Ti varies in the last dimension.
            Arbitrary units are accepted, will be automatically converted to
            radians under the assumption that data is wrapped.
            Do not provide unwrapped data.
        tis (Iterable|int|float): The sampling times Ti in time units.
            The number of points must match the last shape size of arr.
        tis_mask (Iterable[bool]|None): Determine the sampling times Ti to use.
            If None, all will be used.
        unwrapping (str|None): The unwrap method.
            If None, no unwrap is performed.
            If str, perform unwrap using `pymrt.recipes.phs.unwrap()`.
            Accepted values are:
             - 'auto': uses the best method for the given input.
               If `len(tis) == 1`
               use `pymrt.recipes.phs.unwrap_sorting_path()`, otherwise
               use `pymrt.recipes.phs.unwrap_laplacian()`.
             - 'laplacian': use `pymrt.recipes.phs.unwrap_laplacian()`.
             - 'sorting_path': use `pymrt.recipes.phs.unwrap_sorting_path()`.
        unwrapping_kws (dict|tuple|None): Additional keyword arguments.
            These are passed to `pymrt.recipes.phs.unwrap()`.
        units (str|float|int): Units of measurement of Ti.
            If str, the following will be accepted: 'ms'.
            If int or float, the conversion factor will be multiplied to `ti`.

    Returns:
        dphs_arr (np.ndarray): The phase variation in rad/s.
    """
    # todo: fix documentation
    if isinstance(units, str):
        if units == 'ms':
            units = 1e-3
        else:
            warnings.warn('Invalid units `{units}`'.format(**locals()))
            units = 1
    tis = np.array(fc.util.auto_repeat(tis, 1)) * units

    if unwrapping is not None:
        phs_arr = unwrap(phs_arr, unwrapping, unwrapping_kws)

    if len(tis) == 1:
        dphs_arr = phs_arr / tis[0]
    else:
        dphs_arr = \
            phs_to_dphs_multi(
                phs_arr, tis, tis_mask, poly_deg=1)['dphs_1'][..., 0]

    return dphs_arr


# ======================================================================
def cx2_to_dphs(
        arr1,
        arr2,
        d_ti,
        unwrap='laplacian',
        unwrap_kws=None,
        units='ms'):
    """
    Calculate the phase variation from two complex data.

    Args:
        phs_arr (np.ndarray): The input array in arb. units.
            The sampling time Ti varies in the last dimension.
            Arbitrary units are accepted, will be automatically converted to
            radians under the assumption that data is wrapped.
            Do not provide unwrapped data.
        d_ti (Iterable|int|float): The sampling times Ti in time units.
            The number of points must match the last shape size of arr.
        tis_mask (Iterable[bool]|None): Determine the sampling times Ti to use.
            If None, all will be used.
        units (str|float|int): Units of measurement of Ti.
            If str, the following will be accepted: 'ms'
            If int or float, the conversion factor will be multiplied to `ti`.

    Returns:
        dphs_arr (np.ndarray): The phase variation in rad/s.
    """
    # todo: fix documentation
    dphs_arr = arr1 * arr2.conj()
    dphs_arr = np.arctan2(np.imag(dphs_arr), np.real(dphs_arr))

    if unwrap is not None:
        dphs_arr = unwrap(dphs_arr, unwrap, unwrap_kws)

    return dphs_arr / d_ti


# ======================================================================
def dphs_to_phs(
        dphs_arr,
        tis,
        phs0_arr=0):
    """
    Calculate the phase variation from phase data.

    Args:
        dphs_arr (np.ndarray): The input array in rad.
            The sampling time Ti varies in the last dimension.
        tis (Iterable|int|float): The sampling times Ti in time units.
            The number of points will match the last shape size of `phs_arr`.
        phs0_arr (np.ndarray|int|float): The initial phase offset.
            If int or float, a constant offset is used.

    Returns:
        phs_arr (np.ndarray): The phase array in rad.
    """
    shape = dphs_arr.shape
    tis = np.array(
        fc.util.auto_repeat(tis, 1)).reshape((1,) * len(shape) + (-1,))
    dphs_arr = dphs_arr.reshape(shape + (1,))
    return dphs_arr * tis + phs0_arr


# ======================================================================
def unwrap_laplacian(
        arr,
        pad_width=0):
    """
    Super-fast multi-dimensional Laplacian-based Fourier unwrap.

    Phase unwrap by using the following equality:

    L = (d / dx)^2

    L(phi) = cos(phi) * L(sin(phi)) - sin(phi) * L(cos(phi))

    phi = IL(L(phi)) = IL(cos(phi) * L(sin(phi)) - sin(phi) * L(cos(phi)))

    Args:
        arr (np.ndarray): The input array.
        pad_width (float|int): Size of the padding to use.
            This is useful for mitigating border effects.
            If int, it is interpreted as absolute size.
            If float, it is interpreted as relative to the maximum size.

    Returns:
        arr (np.ndarray): The unwrapped array.

    See Also:
        Schofield, M. A. and Y. Zhu (2003). Optics Letters 28(14): 1194-1196.
    """
    arr, mask = fc.num.padding(arr, pad_width)

    # from pymrt.base import laplacian, inv_laplacian
    # from numpy import real, sin, cos
    # arr = real(inv_laplacian(
    #     cos(arr) * laplacian(sin(arr)) - sin(arr) * laplacian(cos(arr))))

    cos_arr = np.cos(arr)
    sin_arr = np.sin(arr)
    kk_2 = fftshift(fc.num.laplace_kernel(arr.shape))
    arr = fftn(cos_arr * ifftn(kk_2 * fftn(sin_arr)) -
               sin_arr * ifftn(kk_2 * fftn(cos_arr)))
    kk_2[kk_2 != 0] = 1.0 / kk_2[kk_2 != 0]
    arr *= kk_2
    del cos_arr, sin_arr, kk_2
    arr = np.real(ifftn(arr))

    arr = arr[mask]
    return arr


# ======================================================================
def unwrap_laplacian_corrected(
        arr,
        pad_width=0):
    """
    Super-fast multi-dimensional corrected Laplacian-based Fourier unwrap.

    EXPERIMENTAL!

    Phase unwrap by using the following equality:

    L = (d / dx)^2

    L(phi) = cos(phi) * L(sin(phi)) - sin(phi) * L(cos(phi))

    phi = IL(L(phi)) = IL(cos(phi) * L(sin(phi)) - sin(phi) * L(cos(phi)))

    Args:
        arr (np.ndarray): The input array.
        pad_width (float|int): Size of the padding to use.
            This is useful for mitigating border effects.
            If int, it is interpreted as absolute size.
            If float, it is interpreted as relative to the maximum size.

    Returns:
        arr (np.ndarray): The unwrapped array.

    See Also:
        Schofield, M. A. and Y. Zhu (2003). Optics Letters 28(14): 1194-1196.
    """
    raise NotImplementedError
    if pad_width:
        shape = arr.shape
        pad_width = fc.util.auto_pad_width(pad_width, shape)
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
    kk_2 = fftshift(fc.num.laplace_kernel(arr.shape))
    arr = fftn(cos_arr * ifftn(kk_2 * fftn(sin_arr)) -
               sin_arr * ifftn(kk_2 * fftn(cos_arr)))
    kk_2[kk_2 != 0] = 1.0 / kk_2[kk_2 != 0]
    arr *= kk_2
    del cos_arr, sin_arr, kk_2
    arr = np.real(ifftn(arr))

    arr = arr[mask]
    return arr


# ======================================================================
def unwrap_region_merging(
        arr,
        split=6,
        select=min,
        step=2 * np.pi,
        threshold=None):
    """
    Accurate unwrap using a region-merging approach.

    EXPERIMENTAL!

    Args:
        arr (np.ndarray): The input array.
        split (int): The number of bins for splitting the regions.
        select (callable|None): The function for selecting the optimal region.
            If callable, must have the signature:
            select(Iterable) -> float
            The input of the `select` function are merging costs of the
            other regions.
        step (float): The size of the wrap discontinuity.
            For phase this is 2 * PI.
        threshold (float|None): The threshold above which regions are matched.
            If None, this is computed as `step / 2`.

    Returns:
        arr (np.ndarray): The output array.
    """
    if not threshold:
        threshold = step / 2
    arr_min, arr_max = arr.min(), arr.max()
    if abs(arr_max - arr_min) > step:
        warnings.warn('The input is not properly wrapped')
    num_labels = 0
    labels_arr = np.zeros_like(arr, dtype=int)
    splits = np.linspace(arr_min, arr_max, split + 1, endpoint=True)
    for i, (split_min, split_max) in enumerate(zip(splits[:-1], splits[1:])):
        mask = (arr >= split_min) if i == 0 else (arr > split_min)
        mask *= (arr <= split_max)
        label_arr, num_label = sp.ndimage.label(mask)
        offset = (label_arr > 0) * (num_labels)
        labels_arr = labels_arr + offset + label_arr
        num_labels += num_label
    u_arr = arr.copy()
    u_mask = (labels_arr == 1)
    u_arr[~u_mask] = 0.0
    unprocessed = set(range(2, num_labels + 1))
    for i in range(2, num_labels + 1):
        costs = {}
        for j in unprocessed:
            t_mask = labels_arr == j
            t_f_mask = sp.ndimage.binary_dilation(u_mask) * t_mask
            if sum(t_f_mask):
                u_f_mask = u_mask * sp.ndimage.binary_dilation(t_mask)
                t_f_val = np.mean(arr[t_f_mask])  # value at target frontier
                u_f_val = np.mean(u_arr[u_f_mask])  # value at unwrap frontier
                cost = (u_f_val - t_f_val)
                if abs(cost) > threshold:
                    t_step = np.sign(cost) * ((abs(cost) // step + 1) * step)
                else:
                    t_step = 0
                if select:
                    costs[cost] = (j, t_step)
                else:
                    break
        if select:
            cost_key = select(costs.keys())
            j, t_step = costs[cost_key]
            t_mask = labels_arr == j
        u_arr[t_mask] += arr[t_mask] + t_step
        u_mask += t_mask
        unprocessed.remove(j)
    return u_arr


# ======================================================================
def unwrap_sorting_path(
        arr,
        unwrap_axes=(0, 1, 2),
        wrap_around=False,
        seed=0):
    """
    2D/3D unwrap using sorting by reliability following a non-continous path.

    This is a wrapper around the function `skimage.restoration.unwrap_phase`
    
    If higher dimensionality input, loop through extra dimensions.

    Args:
        arr (np.ndarray): The input array.
        unwrap_axes (tuple[int]): Axes along which unwrap is performed.
            Must have length 2 or 3.
        wrap_around (bool|Iterable[bool]|None): Circular unwrap.
            See also: skimage.restoration.unwrap_phase.
        seed (int|None): Randomization seed.
            See also: skimage.restoration.unwrap_phase.

    Returns:
        arr (np.ndarray): The unwrapped array.

    See Also:
        skimage.restoration.unwrap_phase
        Herraez, M. A. et al. (2002). Journal Applied Optics 41(35): 7437.
    """
    from skimage.restoration import unwrap_phase


    if unwrap_axes:
        loop_gen = [
            (slice(None),) if j in unwrap_axes else range(dim)
            for j, dim in enumerate(arr.shape)]
    else:
        loop_gen = (slice(None),) * arr.ndim
    arr = arr.copy()
    for indexes in itertools.product(*loop_gen):
        arr[indexes] = unwrap_phase(arr[indexes], wrap_around, seed)
    return arr


# ======================================================================
def unwrap_iter(arr):
    """
    Iterate one-dimensional unwrapping over all directions.

    Args:
        arr (np.ndarray): The input array.

    Returns:
        arr (np.ndarray): The unwrapped array.
    """
    for i in range(arr.ndim):
        arr = np.unwrap(arr, axis=i)
    return arr


# ======================================================================
def unwrap_cnn(
        arr):
    """
    Fast unwrap using Convolutional Neural Networks.

    EXPERIMENTAL!

    Args:
        arr (np.ndarray): The input array.

    Returns:
        arr (np.ndarray): The output array.
    """
    raise NotImplementedError


# ======================================================================
def unwrap(
        arr,
        method='laplacian',
        method_kws=None):
    """
    Perform phase unwrap.

    Multi-dimensional inputs are supported.

    Args:
        arr (np.ndarray): The input array.  
        method (str): The unwrap method.
            Accepted values are:
             - 'laplacian': use `pymrt.recipes.phs.unwrap_laplacian()`.
             - 'sorting_path': use `pymrt.recipes.phs.unwrap_sorting_path()`.
        method_kws (dict|tuple|None): Keyword arguments to pass to `method`.

    Returns:
        arr (np.ndarray): The unwrapped array.
    """
    method = method.lower()
    method_kws = {} if method_kws is None else dict(method_kws)
    if method == 'laplacian':
        method = unwrap_laplacian
    elif method == 'sorting_path':
        method = unwrap_sorting_path
    elif method == 'region_merging':
        method = unwrap_region_merging
    elif method == 'laplacian_corrected':
        method = unwrap_laplacian_corrected
    elif method == 'cnn':
        method = unwrap_cnn
    else:
        text = 'Unknown unwrap method `{}`'.format(method)
        warnings.warn(text)
    if callable(method):
        arr = method(arr, **method_kws)
    return arr


# ======================================================================
elapsed(__file__[len(PATH['base']) + 1:])

# ======================================================================
if __name__ == '__main__':
    import doctest  # Test interactive Python examples

    msg(__doc__.strip())
    doctest.testmod()
    msg(report())
