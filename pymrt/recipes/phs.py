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
import numpy as np  # NumPy (multidimensional numerical arrays library)
import scipy as sp  # SciPy (signal and image processing library)
import flyingcircus as fc  # Everything you always wanted to have in Python*

# :: External Imports Submodules
import scipy.ndimage  # SciPy: ND-image Manipulation
import scipy.sparse  # SciPy: Sparse Matrices
# import scipy.sparse.linalg  # SciPy: Sparse Matrices - Linear Algebra
from numpy.fft import fftshift, ifftshift
from scipy.fftpack import fftn, ifftn

try:
    import skimage.restoration
except ImportError:
    class skimage(object):
        restoration = None

# :: Local Imports
import pymrt as mrt  # Python Magnetic Resonance Tools: the multi-tool of MRI.
import pymrt.util
import pymrt.correction

from pymrt import INFO, PATH
from pymrt import VERB_LVL, D_VERB_LVL, VERB_LVL_NAMES
from pymrt import elapsed, report, run_doctests
from pymrt import msg, dbg, fmt, fmtm
from pymrt import HAS_JIT, jit

from pymrt.recipes import generic
from pymrt.recipes.generic import fix_phase_interval as fix_interval
from flyingcircus.extra import wrap_cyclic as wrap


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
        full=False,
        time_units='s'):
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
        time_units (str|float|int): Units of measurement of Ti.
            If str, any valid SI time unit (e.g. `ms`, `ns`) will be accepted.
            If int or float, the conversion factor will be multiplied to `ti`.

    Returns:
        results (OrderedDict): The multiple components of the polynomial fit.
            The following terms are present:
             - `s0` (np.ndarray): The constant term.
             - `dphs_{i}` (np.ndarray): The monomial term(s); `i` is between
               1 and `poly_deg` (both included).
    """
    units_factor = 1
    if isinstance(time_units, str) and 's' in time_units:
        prefix, _ = time_units.split('s', 1)
        units_factor = fc.prefix_to_factor(prefix)
    elif isinstance(time_units, (int, float)):
        units_factor = time_units
    else:
        warnings.warn(fmtm('Invalid units `{time_units}`. Ignored.'))

    tis = np.array(fc.auto_repeat(tis, 1)) * units_factor
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
        raise NotImplementedError

    return results


# ======================================================================
def phs_to_dphs(
        phs_arr,
        tis,
        tis_mask=None,
        unwrapping='laplacian',
        unwrapping_kws=None,
        time_units='ms'):
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
        unwrapping_kws (Mappable|None): Additional keyword arguments.
            These are passed to `pymrt.recipes.phs.unwrap()`.
        time_units (str|float|int): Units of measurement of Ti.
            If str, any valid SI time unit (e.g. `ms`, `ns`) will be accepted.
            If int or float, the conversion factor will be multiplied to `ti`.

    Returns:
        dphs_arr (np.ndarray): The phase variation in rad/s.
    """
    units_factor = 1
    if isinstance(time_units, str) and 's' in time_units:
        prefix, _ = time_units.split('s', 1)
        units_factor = fc.prefix_to_factor(prefix)
    elif isinstance(time_units, (int, float)):
        units_factor = time_units
    else:
        warnings.warn(fmtm('Invalid units `{time_units}`. Ignored.'))

    tis = np.array(fc.auto_repeat(tis, 1)) * units_factor

    if unwrapping is not None:
        phs_arr = unwrap(phs_arr, unwrapping, unwrapping_kws)

    if len(tis) == 1:
        dphs_arr = phs_arr / tis[0]
    else:
        dphs_arr = \
            phs_to_dphs_multi(
                phs_arr, tis, tis_mask, poly_deg=1, time_units='s') \
                ['dphs_1'][..., 0]

    return dphs_arr


# ======================================================================
def cx2_to_dphs(
        arr1,
        arr2,
        d_ti,
        unwrap_method='laplacian_c',
        unwrap_kws=None,
        time_units='ms'):
    """
    Calculate the phase variation from two complex data.

    Args:
        arr1 (np.ndarray): The phase array for sampling time 1 in arb. units.
            Arbitrary units are accepted, will be automatically converted to
            radians under the assumption that data is wrapped.
            Do not provide unwrapped data.
            Shape must match that of `arr2`.
        arr2 (np.ndarray): The phase array for sampling time 2 in arb. units.
            Arbitrary units are accepted, will be automatically converted to
            radians under the assumption that data is wrapped.
            Do not provide unwrapped data.
            Shape must match that of `arr1`.
        d_ti (int|float): The sampling time differenct in time units.
        unwrap_method (str|None): The unwrap method.
            See `pymrt.recipes.phs.unwrap()` for details.
        unwrap_kws (Mappable|None): Keyword arguments for `unwrap()`.
        time_units (str|float|int): Units of measurement of Ti.
            If str, any valid SI time unit (e.g. `ms`, `ns`) will be accepted.
            If int or float, the conversion factor will be multiplied to `ti`.

    Returns:
        dphs_arr (np.ndarray): The phase variation in rad/s.
    """
    units_factor = 1
    if isinstance(time_units, str) and 's' in time_units:
        prefix, _ = time_units.split('s', 1)
        units_factor = fc.prefix_to_factor(prefix)
    elif isinstance(time_units, (int, float)):
        units_factor = time_units
    else:
        warnings.warn(fmtm('Invalid units `{time_units}`. Ignored.'))

    dphs_arr = arr1 * arr2.conj()
    dphs_arr = np.arctan2(np.imag(dphs_arr), np.real(dphs_arr))

    if unwrap is not None:
        dphs_arr = unwrap(dphs_arr, unwrap_method, unwrap_kws)

    return dphs_arr / (d_ti * units_factor)


# ======================================================================
def dphs_to_phs(
        dphs_arr,
        tis,
        phs0_arr=0,
        time_units='ms'):
    """
    Calculate the phase variation from phase data.

    Args:
        dphs_arr (np.ndarray): The input array in rad.
            The sampling time Ti varies in the last dimension.
        tis (Iterable|int|float): The sampling times Ti in time units.
            The number of points will match the last shape size of `phs_arr`.
        phs0_arr (np.ndarray|int|float): The initial phase offset.
            If int or float, a constant offset is used.
        time_units (str|float|int): Units of measurement of Ti.
            If str, any valid SI time unit (e.g. `ms`, `ns`) will be accepted.
            If int or float, the conversion factor will be multiplied to `ti`.

    Returns:
        phs_arr (np.ndarray): The phase array in rad.
    """
    units_factor = 1
    if isinstance(time_units, str) and 's' in time_units:
        prefix, _ = time_units.split('s', 1)
        units_factor = fc.prefix_to_factor(prefix)
    elif isinstance(time_units, (int, float)):
        units_factor = time_units
    else:
        warnings.warn(fmtm('Invalid units `{time_units}`. Ignored.'))

    shape = dphs_arr.shape
    tis = np.array(fc.auto_repeat(tis, 1)) * units_factor
    tis = tis.reshape((1,) * len(shape) + (-1,))
    dphs_arr = dphs_arr.reshape(shape + (1,))
    return dphs_arr * tis + phs0_arr


# ======================================================================
def fix_rounding(
        w_arr,
        u_arr,
        step=2 * np.pi):
    """
    Compute the rounding-correction for wrapped data.

    This computes the closest integer such that the differences between the
    wrapped and the unwrapped data are multiples of `step`.

    This is typically used for correcting unwrapped data from Fourier-based
    methods, like e.g. Laplacian.

    For phase data the step must be 2π.

    Note: `pymrt.recipes.phs.congruence_correction()` offers a more robust
    approach against noisy data.

    Args:
        w_arr (np.ndarray): The wrapped phase.
        u_arr (np.ndarray): The approximate unwrapped phase.
        step (float): The size of the wrap discontinuity.
            For phase data this is 2π.

    Returns:
        u_arr (np.ndarray): The corrected unwrapped phase.

    See Also:
        - Robinson, Simon Daniel, Kristian Bredies, Diana Khabipova,
          Barbara Dymerska, José P. Marques, and Ferdinand Schweser.
          “An Illustrated Comparison of Processing Methods for MR Phase
          Imaging and QSM: Combining Array Coil Signals and Phase Unwrapping.”
          NMR in Biomedicine, January 1, 2016, n/a-n/a.
          https://doi.org/10.1002/nbm.3601.
    """
    k = np.round(0.5 * (u_arr - w_arr) / (step / 2))
    return step * k + w_arr


# ======================================================================
def fix_congruence(
        w_arr,
        u_arr,
        step=2 * np.pi,
        congruences=16,
        discont=lambda x: x >= 3 * np.pi / 2,
        discont_mask=None):
    """
    Compute the congruence-correction for wrapped data.

    This performs a simple congruence correction, where the unwrapped data
    is summed with an offset that compensate for the small numerical errors,
    so that the differences between the wrapped and the unwrapped data
    are multiples of `step`.

    This is typically used for correcting unwrapped data from Fourier-based
    methods, like e.g. Laplacian.

    For phase data the step must be 2π.

    The congruence step is chosen to be the one that minimizes the number
    of discontinuities. The discontinuities are computed by summing up all the
    the points where the absolute of the gradient along a given dimension
    is larger than π.

    Note that phase wraps may still be present if there are gradient
    components (whose Laplacian is 0) that are larger than 2π.

    This approach is numerically more stable than the approximation by
    rounding to 2π multiples, especially in the presence of noise.

    Args:
        w_arr (np.ndarray): The wrapped phase.
        u_arr (np.ndarray): The approximate unwrapped phase.
        step (float): The size of the wrap discontinuity.
            For phase data this is 2π.
        congruences (int|None): The number of congruence values to test.
            If None, no correction is performed.
        discont (callable): The discontinuity condition.
            This is computed on the absolute of the gradient for all axis.
            Must have the signature:
            is_discont(np.ndarray[float]) -> np.ndarray[bool].
        discont_mask (np.ndarray[bool]|None): The discontinuity mask.
            This serves to exclude portions of the array where discontinuities
            are irrelevant (e.g. due to noise).
            If None, all array is considered.

    Returns:
        u_arr (np.ndarray): The corrected unwrapped phase.

    See Also:
        - Robinson, Simon Daniel, Kristian Bredies, Diana Khabipova,
          Barbara Dymerska, José P. Marques, and Ferdinand Schweser.
          “An Illustrated Comparison of Processing Methods for MR Phase
          Imaging and QSM: Combining Array Coil Signals and Phase Unwrapping.”
          NMR in Biomedicine, January 1, 2016, n/a-n/a.
          https://doi.org/10.1002/nbm.3601.
    """
    if congruences is not None:
        if discont_mask is None:
            discont_mask = tuple(slice(None) for _ in range(w_arr.ndim))
        num_discont = []
        for congruence in range(congruences):
            phs_step = step * congruence / congruences
            c_arr = u_arr + phs_step + wrap(w_arr - u_arr - phs_step)
            num_discont.append(np.sum([
                discont(np.abs(np.gradient(c_arr[discont_mask], axis=j)))
                for j in range(w_arr.ndim)]))
        congruence = np.argmin(num_discont)
        phs_step = step * congruence / congruences
        u_arr = u_arr + phs_step + wrap(w_arr - u_arr - phs_step)
    return u_arr


# ======================================================================
def reliab_diff2(
        arr,
        step=2 * np.pi):
    """
    Compute the reliability weighting for unwrapping.

    This is based on second differences of neighboring voxels.

    Args:
        arr (np.ndarray): The wrapped phase array.
        step (float): The size of the wrap discontinuity.
            For phase data this is 2π.

    Returns:
        reliab_arr (np.ndarray): The reliability weighting.

    See Also:
        - Herráez, Miguel Arevallilo, David R. Burton, Michael J. Lalor, and
          Munther A. Gdeisat. “Fast Two-Dimensional Phase-Unwrapping Algorithm
          Based on Sorting by Reliability Following a Noncontinuous Path.”
          Applied Optics 41, no. 35 (December 10, 2002): 7437.
          https://doi.org/10.1364/AO.41.007437.
    """
    reliab_arr = np.zeros(arr.shape)
    windows = (slice(None, -2), slice(1, -1), slice(2, None))
    slices = list(itertools.product(*[windows for _ in range(arr.ndim)]))
    mid_point = len(slices) // 2
    mid_slice = slices[mid_point]
    # the slices are paired: `j` is in the "opposite" direction as `-j -1`
    # `wrap` implements `gamma` from the paper
    diffs = [
        wrap(arr[slices[j]] - arr[mid_slice], step, step / 2) -
        wrap(arr[mid_slice] - arr[slices[-j - 1]], step, step / 2)
        for j in range(mid_point)]
    diff2_arr = np.sqrt(sum(x ** 2 for x in diffs))
    reliab_arr[mid_slice][diff2_arr != 0] = 1 / diff2_arr[diff2_arr != 0]
    reliab_arr[mid_slice][diff2_arr == 0] = np.inf
    reliab_arr[np.isnan(arr)] = np.nan
    return reliab_arr


# ======================================================================
def unwrap_laplacian(
        arr,
        pad_width=0,
        denoising=None,
        denoising_kws=None):
    """
    Multi-dimensional Laplacian-based Fourier unwrap.

    Given the Laplacian operator:

    L = (d / dx)^2

    phase unwrap by using the following equality:

    L(phi_u) = im(exp(-i * phi) * L(exp(i * phi))

    which can be solved for `phi_u` (phase unwrapped) by:

    L(phi) = cos(phi) * L(sin(phi)) - sin(phi) * L(cos(phi))

    phi_u = IL(L(phi_u)) = IL(im(exp(-i * phi) * L(exp(i * phi))) =
          = IL(cos(phi) * L(sin(phi)) - sin(phi) * L(cos(phi)))

    Args:
        arr (np.ndarray): The wrapped phase array.
        pad_width (float|int): Size of the padding to use.
            This is useful for mitigating border effects.
            If int, it is interpreted as absolute size.
            If float, it is interpreted as relative to the maximum size.
        denoising (callable|None): The denoising function.
            If callable, must have the following signature:
            denoising(np.ndarray, ...) -> np.ndarray.
            It is applied to `np.cos(arr)` and `np.sin(arr)` separately.
        denoising_kws (Mappable|None): Keyword arguments.
            These are passed to the function specified in `denoising`.
            If Iterable, must be convertible to a dictionary.
            If None, no keyword arguments will be passed.

    Returns:
        arr (np.ndarray): The unwrapped phase array.

    See Also:
        - Schofield, Marvin A., and Yimei Zhu. “Fast Phase Unwrapping
          Algorithm for Interferometric Applications.” Optics Letters 28,
          no. 14 (July 15, 2003): 1194–96.
          https://doi.org/10.1364/OL.28.001194.
    """
    arr, mask = fc.extra.padding(arr, pad_width)

    # from pymrt.base import laplacian, inv_laplacian
    # from numpy import real, sin, cos
    # arr = real(inv_laplacian(
    #     cos(arr) * laplacian(sin(arr)) - sin(arr) * laplacian(cos(arr))))

    cos_arr = np.cos(arr)
    sin_arr = np.sin(arr)
    if callable(denoising):
        denoising_kws = dict(denoising_kws) \
            if denoising_kws is not None else {}
        cos_arr = denoising(cos_arr, **denoising_kws)
        sin_arr = denoising(sin_arr, **denoising_kws)
    kk2 = fftshift(fc.extra.laplace_kernel(arr.shape, factors=arr.shape))
    arr = fftn(cos_arr * ifftn(kk2 * fftn(sin_arr)) -
               sin_arr * ifftn(kk2 * fftn(cos_arr)))
    fc.extra.apply_at(kk2, lambda x: 1 / x, kk2 != 0, in_place=True)
    arr *= kk2
    del cos_arr, sin_arr, kk2
    arr = np.real(ifftn(arr))

    return arr[mask]


# ======================================================================
def unwrap_laplacian_c(
        arr,
        pad_width=0.0,
        denoising=None,
        denoising_kws=None,
        congruences=16,
        discont=lambda x: x >= 3 * np.pi / 2,
        discont_mask=None):
    """
    Multi-dimensional congruence-corrected Laplacian-based Fourier unwrap.

    The unwrapped phase is computed by: `pymrt.recipes.phs.unwrap_laplacian()`.
    Then, the congruence correction is performed using:
    `pymrt.recipes.phs.congruence_correction()`

    Args:
        arr (np.ndarray): The wrapped phase array.
        pad_width (float|int): Size of the padding to use.
            See `pymrt.recipes.phs.unwrap_laplacian()` for more info.
        denoising (callable|None): The denoising function.
            See `pymrt.recipes.phs.unwrap_laplacian()` for more info.
        denoising_kws (Mappable|None): Keyword arguments.
            See `pymrt.recipes.phs.unwrap_laplacian()` for more info.
        congruences (int): The number of congruence values to test.
            See `pymrt.recipes.phs.congruence_correction()` for more info.
        discont (callable): The discontinuity condition.
            See `pymrt.recipes.phs.congruence_correction()` for more info.
        discont_mask (np.ndarray[bool]|None): The discontinuity mask.
            See `pymrt.recipes.phs.congruence_correction()` for more info.

    Returns:
        arr (np.ndarray): The unwrapped phase array.

    See Also:

        - Schofield, Marvin A., and Yimei Zhu. “Fast Phase Unwrapping
          Algorithm for Interferometric Applications.” Optics Letters 28,
          no. 14 (July 15, 2003): 1194–96.
          https://doi.org/10.1364/OL.28.001194.
        - Robinson, Simon Daniel, Kristian Bredies, Diana Khabipova,
          Barbara Dymerska, José P. Marques, and Ferdinand Schweser.
          “An Illustrated Comparison of Processing Methods for MR Phase
          Imaging and QSM: Combining Array Coil Signals and Phase Unwrapping.”
          NMR in Biomedicine, January 1, 2016, n/a-n/a.
          https://doi.org/10.1002/nbm.3601.
    """
    u_arr = unwrap_laplacian(arr, pad_width, denoising, denoising_kws)
    return fix_congruence(
        arr, u_arr, 2 * np.pi, congruences, discont, discont_mask)


# ======================================================================
def unwrap_gradient(
        arr,
        pad_width=0,
        denoising=None,
        denoising_kws=None):
    """
    Multi-dimensional Gradient-based Fourier unwrap.

    Args:
        arr (np.ndarray): The wrapped phase array.
        pad_width (float|int): Size of the padding to use.
            This is useful for mitigating border effects.
            If int, it is interpreted as absolute size.
            If float, it is interpreted as relative to the maximum size.
        denoising (callable|None): The denoising function.
            If callable, must have the following signature:
            denoising(np.ndarray, ...) -> np.ndarray.
            It is applied to the real and imaginary part of `np.exp(1j * arr)`
            separately, using `fc.extra.filter_cx()`.
        denoising_kws (Mappable|None): Keyword arguments.
            These are passed to the function specified in `denoising`.
            If Iterable, must be convertible to a dictionary.
            If None, no keyword arguments will be passed.

    Returns:
        arr (np.ndarray): The unwrapped phase array.

    See Also:
        - Volkov, Vyacheslav V., and Yimei Zhu. “Deterministic Phase
          Unwrapping in the Presence of Noise.” Optics Letters 28, no. 22
          (November 15, 2003): 2156–58. https://doi.org/10.1364/OL.28.002156.
    """
    arr, mask = fc.extra.padding(arr, pad_width)

    arr = np.exp(1j * arr)
    if callable(denoising):
        denoising_kws = dict(denoising_kws) \
            if denoising_kws is not None else {}
        arr = fc.extra.filter_cx(arr, denoising, (), denoising_kws)
    kks = [
        fftshift(kk)
        for kk in fc.extra.gradient_kernels(arr.shape, factors=arr.shape)]
    grads = np.gradient(arr)

    u_arr = np.zeros(arr.shape, dtype=complex)
    kk2 = np.zeros(arr.shape, dtype=complex)
    for kk, grad in zip(kks, grads):
        u_arr += -1j * kk * fftn(np.real(-1j * grad / arr))
        kk2 += kk ** 2
    fc.extra.apply_at(kk2, lambda x: 1 / x, kk2 != 0, in_place=True)
    arr = np.real(ifftn(kk2 * u_arr)) / (2 * np.pi)
    return arr[mask]


# ======================================================================
def unwrap_gradient_c(
        arr,
        pad_width=0.0,
        denoising=None,
        denoising_kws=None,
        congruences=16,
        discont=lambda x: x >= 3 * np.pi / 2,
        discont_mask=None):
    """
    Multi-dimensional congruence-corrected Gradient-based Fourier unwrap.

    The unwrapped phase is computed by: `pymrt.recipes.phs.unwrap_gradient()`.
    Then, the congruence correction is performed using:
    `pymrt.recipes.phs.congruence_correction()`

    Args:
        arr (np.ndarray): The wrapped phase array.
        pad_width (float|int): Size of the padding to use.
            See `pymrt.recipes.phs.unwrap_gradient()` for more info.
        denoising (callable|None): The denoising function.
            See `pymrt.recipes.phs.unwrap_gradient()` for more info.
        denoising_kws (Mappable|None): Keyword arguments.
            See `pymrt.recipes.phs.unwrap_gradient()` for more info.
        congruences (int): The number of congruence values to test.
            See `pymrt.recipes.phs.congruence_correction()` for more info.
        discont (callable): The discontinuity condition.
            See `pymrt.recipes.phs.congruence_correction()` for more info.
        discont_mask (np.ndarray[bool]|None): The discontinuity mask.
            See `pymrt.recipes.phs.congruence_correction()` for more info.

    Returns:
        arr (np.ndarray): The unwrapped phase array.

    See Also:
        - Volkov, Vyacheslav V., and Yimei Zhu. “Deterministic Phase
          Unwrapping in the Presence of Noise.” Optics Letters 28, no. 22
          (November 15, 2003): 2156–58. https://doi.org/10.1364/OL.28.002156.
        - Robinson, Simon Daniel, Kristian Bredies, Diana Khabipova,
          Barbara Dymerska, José P. Marques, and Ferdinand Schweser.
          “An Illustrated Comparison of Processing Methods for MR Phase
          Imaging and QSM: Combining Array Coil Signals and Phase Unwrapping.”
          NMR in Biomedicine, January 1, 2016, n/a-n/a.
          https://doi.org/10.1002/nbm.3601.
    """
    u_arr = unwrap_gradient(arr, pad_width, denoising, denoising_kws)
    return fix_congruence(
        arr, u_arr, 2 * np.pi, congruences, discont, discont_mask)


# ======================================================================
def unwrap_sorting_path(
        arr,
        step=2 * np.pi,
        reliab=reliab_diff2,
        reliab_kws=None):
    """
    Unwrap using sorting by reliability following a non-continuous path.

    This is an n-dimensional implementation.

    A faster implementation is available in:
    `pymrt.recipes.phs.unwrap_sorting_path_()`.

    Args:
        arr (np.ndarray): The wrapped phase array.
        step (float): The size of the wrap discontinuity.
            For phase data this is 2π.
        reliab (callable): The function for computing reliability.
        reliab_kws (Mappable|None): Keyword arguments.
            These are passed to the function specified in `reliab`.
            If Iterable, must be convertible to a dictionary.
            If None, no keyword arguments will be passed.

    Returns:
        arr (np.ndarray): The unwrapped phase array.

    See Also:
        - Herráez, Miguel Arevallilo, David R. Burton, Michael J. Lalor, and
          Munther A. Gdeisat. “Fast Two-Dimensional Phase-Unwrapping Algorithm
          Based on Sorting by Reliability Following a Noncontinuous Path.”
          Applied Optics 41, no. 35 (December 10, 2002): 7437.
          https://doi.org/10.1364/AO.41.007437.
    """
    shape = arr.shape
    reliab_kws = dict(reliab_kws) if reliab_kws is not None else {}
    reliab_arr = reliab(arr, step, **reliab_kws)
    edges_arr, orig_idx_arr, dest_idx_arr = \
        fc.extra.compute_edge_weights(reliab_arr)
    del reliab_arr
    sorted_edges_indices = np.argsort(edges_arr.ravel())[::-1]
    orig_idx_arr = orig_idx_arr.ravel()
    dest_idx_arr = dest_idx_arr.ravel()
    arr = arr.ravel().copy()
    group_arr = np.arange(arr.size)
    group_sizes = np.ones(group_arr.shape, dtype=int)
    num_nan = np.count_nonzero(np.isnan(edges_arr))
    for i in range(num_nan, len(sorted_edges_indices)):
        move_idx = orig_idx_arr[sorted_edges_indices[i]]
        keep_idx = dest_idx_arr[sorted_edges_indices[i]]
        if group_arr[move_idx] == group_arr[keep_idx] or \
                group_sizes[group_arr[move_idx]] < 1 or \
                group_sizes[group_arr[keep_idx]] < 1:
            continue
        # : ensure that the origin group is updated
        if group_sizes[group_arr[move_idx]] > group_sizes[group_arr[keep_idx]]:
            move_idx, keep_idx = keep_idx, move_idx
        # : perform unwrapping
        diff_val = np.floor(
            (arr[keep_idx] - arr[move_idx] + (step / 2)) / step) * step
        # separate case for improved performances
        move_mask = (group_arr == group_arr[move_idx]) \
            if group_sizes[group_arr[move_idx]] > 1 else move_idx
        if diff_val != 0.0:
            arr[move_mask] += diff_val
        # : bookkeeping of modified indexes
        group_sizes[group_arr[keep_idx]] += group_sizes[group_arr[move_idx]]
        group_sizes[group_arr[move_idx]] -= group_sizes[group_arr[move_idx]]
        group_arr[move_mask] = group_arr[keep_idx]
    return arr.reshape(shape)


# ======================================================================
def unwrap_sorting_path_(
        arr,
        step=2 * np.pi):
    """
    Unwrap using sorting by reliability following a non-continuous path.

    This is identical to `pymrt.recipes.phs.unwrap_sorting_path()`, except
    that it can be decorated with Numba JIT for (hopefully) faster execution.

    Args:
        arr (np.ndarray): The wrapped phase array.
        step (float): The size of the wrap discontinuity.
            For phase data this is 2π.

    Returns:
        arr (np.ndarray): The unwrapped phase array.

    See Also:
        - Herráez, Miguel Arevallilo, David R. Burton, Michael J. Lalor, and
          Munther A. Gdeisat. “Fast Two-Dimensional Phase-Unwrapping Algorithm
          Based on Sorting by Reliability Following a Noncontinuous Path.”
          Applied Optics 41, no. 35 (December 10, 2002): 7437.
          https://doi.org/10.1364/AO.41.007437.
    """
    shape = arr.shape
    reliab_arr = reliab_diff2(arr, step)
    edges_arr, orig_idx_arr, dest_idx_arr = \
        fc.extra.compute_edge_weights(reliab_arr)
    sorted_edges_indices = np.argsort(edges_arr.ravel())[::-1]
    orig_idx_arr = orig_idx_arr.ravel()
    dest_idx_arr = dest_idx_arr.ravel()
    arr = arr.ravel().copy()
    group_arr = np.arange(arr.size)
    group_sizes = np.ones(group_arr.shape, dtype=int)
    num_nan = np.count_nonzero(np.isnan(edges_arr))
    for i in range(num_nan, len(sorted_edges_indices)):
        move_idx = orig_idx_arr[sorted_edges_indices[i]]
        keep_idx = dest_idx_arr[sorted_edges_indices[i]]
        if group_arr[move_idx] == group_arr[keep_idx] or \
                group_sizes[group_arr[move_idx]] < 1 or \
                group_sizes[group_arr[keep_idx]] < 1:
            continue
        # : ensure that the origin group is updated
        if group_sizes[group_arr[move_idx]] > group_sizes[group_arr[keep_idx]]:
            move_idx, keep_idx = keep_idx, move_idx
        # : perform unwrapping
        diff_val = np.floor(
            (arr[keep_idx] - arr[move_idx] + (step / 2)) / step) * step
        # separate case for improved performances
        move_mask = (group_arr == group_arr[move_idx]) \
            if group_sizes[group_arr[move_idx]] > 1 else move_idx
        if diff_val != 0.0:
            arr[move_mask] += diff_val
        # : bookkeeping of modified indexes
        group_sizes[group_arr[keep_idx]] += group_sizes[group_arr[move_idx]]
        group_sizes[group_arr[move_idx]] -= group_sizes[group_arr[move_idx]]
        group_arr[move_mask] = group_arr[keep_idx]
    return arr.reshape(shape)


# ======================================================================
def unwrap_region_merging(
        arr,
        mask=None,
        split=6,
        select=min,
        step=2 * np.pi,
        threshold=None):
    """
    Accurate unwrap using a region-merging approach.

    Note this can be EXTREMELY slow.

    EXPERIMENTAL!

    Args:
        arr (np.ndarray): The wrapped phase array.
        mask (np.ndarray[bool]|None): The mask array.
            This serves to restrict the computation only to the masked values
            of `arr`, e.g. to remove noise-dominated regions.
            If None, all array is considered.
        split (int): The number of bins for splitting the regions.
        select (callable|None): The function for selecting the optimal region.
            If callable, must have the signature:
            select(Iterable) -> float
            The input of the `select` function are merging costs of the
            other regions.
        step (float): The size of the wrap discontinuity.
            For phase data this is 2π.
        threshold (float|None): The threshold above which regions are matched.
            If None, this is computed as `step / 2`.

    Returns:
        arr (np.ndarray): The output array.

    See Also:
        - Jenkinson, Mark. “Fast, Automated, N-Dimensional Phase-Unwrapping
          Algorithm.” Magnetic Resonance in Medicine 49, no. 1
          (January 1, 2003): 193–97. https://doi.org/10.1002/mrm.10354.
    """
    arr = fc.extra.apply_mask(arr, mask)
    if not threshold:
        threshold = step / 2
    arr_min, arr_max = arr.min(), arr.max()
    if abs(arr_max - arr_min) > step:
        warnings.warn('The input is not properly wrapped')
    num_labels = 0
    labels_arr = np.zeros_like(arr, dtype=int)
    splits = np.linspace(arr_min, arr_max, split + 1, endpoint=True)
    for i, (split_min, split_max) in enumerate(zip(splits[:-1], splits[1:])):
        s_mask = (arr >= split_min) if i == 0 else (arr > split_min)
        s_mask *= (arr <= split_max)
        label_arr, num_label = sp.ndimage.label(
            fc.extra.apply_mask(s_mask, mask))
        offset = fc.extra.apply_mask(label_arr > 0, mask) * (num_labels)
        labels_arr = labels_arr + offset + label_arr
        num_labels += num_label
    u_arr = arr.copy()
    u_mask = (labels_arr == 1)
    u_arr[~u_mask] = 0.0
    unprocessed = set(range(2, num_labels + 1))
    for i in range(2, num_labels + 1):
        costs = {}
        u_d_mask = sp.ndimage.binary_dilation(u_mask)  # dilated unwrap mask
        # `t_mask`, `t_step` will be defined after the loop (by construction)
        for j in unprocessed:
            t_mask = labels_arr == j
            t_f_mask = u_d_mask * t_mask
            if np.sum(t_f_mask) > 0:
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
def unwrap_sorting_path_2d_3d(
        arr,
        unwrap_axes=(0, 1, 2),
        wrap_around=False,
        seed=0):
    """
    2D/3D unwrap using sorting by reliability following a non-continous path.

    This is a wrapper around the function `skimage.restoration.unwrap_phase()`
    which can only handle 2D and 3D data.
    If higher dimensionality input, loop through extra dimensions.
    If `scikit.image` (which provides `skimage`) cannot be found,
    uses `unwrap_sorting_path()` to 2D / 3D data, according to the
    `unwrap_axes` parameter, but the `wrap_around` and `seed` parameters are
    ignored.

    Args:
        arr (np.ndarray): The wrapped phase array.
        unwrap_axes (Iterable[int]): Axes along which unwrap is performed.
            Must have length 2 or 3.
        wrap_around (bool|Iterable[bool]|None): Circular unwrap.
            See also: skimage.restoration.unwrap_phase.
        seed (int|None): Randomization seed.
            See also: skimage.restoration.unwrap_phase.

    Returns:
        arr (np.ndarray): The unwrapped phase array.

    See Also:
        - skimage.restoration.unwrap_phase()
        - Herráez, Miguel Arevallilo, David R. Burton, Michael J. Lalor, and
          Munther A. Gdeisat. “Fast Two-Dimensional Phase-Unwrapping Algorithm
          Based on Sorting by Reliability Following a Noncontinuous Path.”
          Applied Optics 41, no. 35 (December 10, 2002): 7437.
          https://doi.org/10.1364/AO.41.007437.
        - Abdul-Rahman, Hussein, Munther Gdeisat, David Burton, and Michael
          Lalor. “Fast Three-Dimensional Phase-Unwrapping Algorithm Based on
          Sorting by Reliability Following a Non-Continuous Path.”
          In SPIE 5856, 5856:32–40. 3 Aug 2005, 2005.
          https://doi.org/10.1117/12.611415.
    """
    if unwrap_axes:
        loop_gen = [
            (slice(None),) if j in unwrap_axes else range(dim)
            for j, dim in enumerate(arr.shape)]
    else:
        loop_gen = (slice(None),) * arr.ndim
    arr = arr.copy()
    for indexes in itertools.product(*loop_gen):
        if skimage.restoration:
            arr[indexes] = skimage.restoration.unwrap_phase(
                arr[indexes], wrap_around, seed)
        else:
            arr[indexes] = unwrap_sorting_path(arr[indexes])
    return arr


# ======================================================================
def unwrap_1d_iter(
        arr,
        axes=None,
        denoising=sp.ndimage.gaussian_filter,
        denoising_kws=(('sigma', 1.0),),
        congruences=16,
        discont=lambda x: x >= 3 * np.pi / 2,
        discont_mask=None):
    """
    Iterate one-dimensional unwrapping over all directions.

    This is effective in multi-dimensional unwrapping provided that
    the image is sufficiently smooth.

    This can be achieved by numerically achieved by up-sampling followed by
    down-sampling.

    Args:
        arr (np.ndarray): The wrapped phase array.
        axes (Iterable[int]|int|None): The dimensions along which to unwrap.
            If Int, unwrapping in a single dimension is performed.
            If None, unwrapping is performed in all dimensions from 0 to -1.
        denoising (callable|None): The denoising function.
            If callable, must have the following signature:
            denoising(np.ndarray, ...) -> np.ndarray.
            It is applied to the real and imaginary part of `np.exp(1j * arr)`
            separately, using `fc.extra.filter_cx()` and then
            converted back to a phase with `np.angle()` before applying the
            unwrapping.
        denoising_kws (Mappable|None): Keyword arguments.
            These are passed to the function specified in `denoising`.
            If Iterable, must be convertible to a dictionary.
            If None, no keyword arguments will be passed.
        congruences (int): The number of congruence values to test.
            See `pymrt.recipes.phs.congruence_correction()` for more info.
        discont (callable): The discontinuity condition.
            See `pymrt.recipes.phs.congruence_correction()` for more info.
        discont_mask (np.ndarray[bool]|None): The discontinuity mask.
            See `pymrt.recipes.phs.congruence_correction()` for more info.

    Returns:
        arr (np.ndarray): The unwrapped phase array.
    """
    u_arr = arr.copy()
    if callable(denoising):
        denoising_kws = dict(denoising_kws) \
            if denoising_kws is not None else {}
        u_arr = np.angle(
            fc.extra.filter_cx(np.exp(1j * u_arr), denoising, (),
                               denoising_kws))
    if axes is None:
        axes = tuple(range(arr.ndim))
    axes = fc.auto_repeat(axes, 1)
    for i in axes:
        u_arr = np.unwrap(u_arr, axis=i)
    u_arr = fix_congruence(
        arr, u_arr, 2 * np.pi, congruences, discont, discont_mask)
    return u_arr


# ======================================================================
def unwrap(
        arr,
        method='laplacian_c',
        method_kws=None):
    """
    Perform phase unwrap.

    Multi-dimensional inputs are supported.

    Args:
        arr (np.ndarray): The input array.  
        method (str): The unwrap method.
            Accepted values are:
             - 'laplacian_c': use `pymrt.recipes.phs.unwrap_laplacian_c()`.
             - 'laplacian': use `pymrt.recipes.phs.unwrap_laplacian()`.
             - 'gradient_c': use `pymrt.recipes.phs.unwrap_gradient_c()`.
             - 'gradient': use `pymrt.recipes.phs.unwrap_gradient()`.
             - 'sorting_path': use `pymrt.recipes.phs.unwrap_sorting_path()`.
             - 'sorting_path_': use `pymrt.recipes.phs.unwrap_sorting_path_()`.
             - 'region_merging': use `pymrt.recipes.phs.unwrap_gradient()`.

             - '1d_iter': use `pymrt.recipes.phs.unwrap_1d_iter()`.
        method_kws (Mappable|None): Keyword arguments to pass to `method`.

    Returns:
        arr (np.ndarray): The unwrapped phase array.
    """
    methods = (
        'laplacian_c', 'laplacian', 'gradient', 'gradient_c',
        'sorting_path', 'sorting_path_', 'region_merging',
        'sorting_path_2d_3d', '1d_iter')
    method = method.lower()
    method_kws = {} if method_kws is None else dict(method_kws)
    if method in methods:
        method = exec('unwrap_' + method)
    else:
        text = 'Unknown unwrap method `{}`. Using default `{}`.'.format(
            method, methods[0])
        warnings.warn(text)
    if callable(method):
        arr = method(arr, **method_kws)
    return arr


# ======================================================================
elapsed(__file__[len(PATH['base']) + 1:])

# ======================================================================
if __name__ == '__main__':
    run_doctests(__doc__)
