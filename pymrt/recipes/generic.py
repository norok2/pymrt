#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pymrt.recipes.generic: generic computation algorithms.
"""

# ======================================================================
# :: Future Imports
from __future__ import (
    division, absolute_import, print_function, unicode_literals, )

# ======================================================================
# :: Python Standard Library Imports
import os  # Miscellaneous operating system interfaces
# import itertools  # Functions creating iterators for efficient looping
import collections  # Container datatypes
import warnings  # Warning control
import multiprocessing  # Process-based parallelism

# :: External Imports
import numpy as np  # NumPy (multidimensional numerical arrays library)
import scipy as sp  # SciPy (signal and image processing library)
import pywt as pw  # PyWavelets - Wavelet Transforms in Python
import flyingcircus as fc  # Everything you always wanted to have in Python*

# :: External Imports Submodules
import scipy.integrate  # SciPy: Integration and ODEs
import scipy.optimize  # SciPy: Optimization and root finding
import scipy.stats  # SciPy: Statistical functions
import scipy.sparse  # SciPy: Sparse Matrices
import scipy.sparse.linalg  # SciPy: Sparse Matrices - Linear Algebra

# :: Local Imports
import pymrt as mrt  # Python Magnetic Resonance Tools: the multi-tool of MRI
import pymrt.util
import pymrt.segmentation

from pymrt import INFO, PATH
from pymrt import VERB_LVL, D_VERB_LVL, VERB_LVL_NAMES
from pymrt import elapsed, report, run_doctests
from pymrt import msg, dbg, fmt, fmtm

from pymrt.constants import GAMMA, GAMMA_BAR


# ======================================================================
def func_exp_recovery(t_arr, tau, s_0, eff=1.0, const=0.0):
    """
    s(t)= s_0 * (1 - 2 * eff * exp(-t/tau)) + const

    [s_0 > 0, tau > 0, eff > 0]
    """
    s_t_arr = s_0 * (1.0 - 2.0 * eff * np.exp(-t_arr / tau)) + const
    # if s_0 > 0.0 and tau > 0.0 and eff > 0.0:
    #     s_t_arr = s_0 * (1.0 - 2.0 * eff * exp(-t_arr / tau)) + const
    # else:
    #     s_t_arr = np.tile(np.inf, len(t_arr))
    return s_t_arr


# ======================================================================
def func_exp_decay(t_arr, tau, s_0, const=0.0):
    """
    s(t)= s_0 * exp(-t/tau) + const

    [s_0 > 0, tau > 0]
    """
    s_t_arr = s_0 * np.exp(-t_arr / tau) + const
    #    if s_0 > 0.0 and tau > 0.0:
    #        s_t_arr = s_0 * exp(-t_arr / tau) + const
    #    else:
    #        s_t_arr = np.tile(np.inf, len((t_arr)))
    return s_t_arr


# ======================================================================
def time_to_rate(
        arr,
        in_units='ms',
        out_units='Hz'):
    """
    Convert array from time to rate units.
    
    Args:
        arr (np.ndarray): The input array.
        in_units (str|int|float): The input units.
            If numeric, input array is divided by this value.
            If str, use predefined numeric constant accordingly.
        out_units (str|int|float): The output units.
            If numeric, output array is divided by this value.
            If str, use predefined numeric constant accordingly.

    Returns:
        arr (np.ndarray): The output array.
    """
    if isinstance(in_units, (float, int)) \
            and isinstance(out_units, (float, int)):
        k = in_units / out_units
    else:
        k = 1.0
        if in_units == 'ms':
            k *= 1.0e3
        if out_units == 'kHz':
            k *= 1.0e-3
    arr[arr != 0.0] = k / arr[arr != 0.0]
    return arr


# ======================================================================
def rate_to_time(
        arr,
        in_units='Hz',
        out_units='ms'):
    """
    Convert array from rate to time units.
    
    Args:
        arr (np.ndarray): The input array.
        in_units (str|int|float): The input units.
            If numeric, input array is divided by this value.
            If str, use predefined numeric constant accordingly.
        out_units (str|int|float): The output units.
            If numeric, output array is divided by this value.
            If str, use predefined numeric constant accordingly.

    Returns:
        arr (np.ndarray): The output array.
    """
    if isinstance(in_units, (float, int)) \
            and isinstance(out_units, (float, int)):
        k = in_units / out_units
    else:
        k = 1.0
        if in_units == 'kHz':
            k *= 1.0e3
        if out_units == 'ms':
            k *= 1.0e-3
    arr[arr != 0.0] = k / arr[arr != 0.0]
    return arr


# ======================================================================
def is_linear(arr, axis=-1):
    """
    Check if an array contains a linear sequence of values.
    
    This is useful, for example, to check if sampling points are equidistant.
    
    Args:
        arr (np.ndarray): The input array.
        axis (int): The axis along which the sequence is evaluated.

    Returns:
        result (bool): The result of the comparison.
            If True the array contains a linear sequence.
            False otherwise.
            
    Examples:
        >>> arr = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29]
        >>> is_linear(arr)
        True
        >>> arr = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
        >>> is_linear(arr)
        False
    """
    arr = np.diff(arr, axis=axis)  # first derivative
    arr = np.diff(arr, axis=axis)  # second derivative
    return np.allclose(arr, 0)


# ======================================================================
def cx_div(
        arr1,
        arr2,
        regularization=np.spacing(1.0),
        values_interval=None):
    """
    Calculate the pseudo-ratio expression: s1 * s2 / (s1^2 + s2^2)

    This is an SNR optimal expression for (s1 / s2) or (s2 / s1).

    Equivalent expressions are:

    .. math::
        f(s_1, s_2) = \frac{s_1 s_2}{(s_1^2 + s_2^2}
        = \frac{1}{\frac{s_1}{s_2}+\frac{s_2}{s_1}}

    which is either the inverse of the arithmetic mean of the two ratios or
    half
    the harmonic mean of the two ratios.

    Resulting values are in the [-0.5, 0.5] interval.

    Args:
        arr1 (float|np.ndarray): Complex image of the first inversion.
        arr2 (float|np.ndarray): Complex image of the second inversion.
        regularization (float|int): Parameter for the regularization.
            This parameter is added to the denominator of the expression
            for preventing undefined values when both s1 and s2 vanish.
        values_interval (Any): The output values interval.
            The standard values are linearly converted to this range.
            If None, the natural [-0.5, 0.5] interval will be used.
            See `flyingcircus.extra.valid_interval()` for details on the
            accepted input.

    Returns:
        result (float|complex|np.ndarray): The pseud-ratio array.
    """
    result = arr1 * arr2 / (
            np.abs(arr1) ** 2 + np.abs(arr2) ** 2 + regularization)
    if values_interval:
        values_interval = fc.extra.valid_interval(values_interval)
        result = fc.extra.scale(result, values_interval, (-0.5, 0.5))
    return result


# ======================================================================
def fix_phase_interval(arr):
    """
    Ensure that the range of values is interpreted as valid phase information.

    This is useful for DICOM-converted images (without post-processing).

    Args:
        arr (np.ndarray): Array to be processed.

    Returns:
        arr (np.ndarray): An array scaled to (-π,π).

    Examples:
        >>> fix_phase_interval(np.arange(8))
        array([-3.14159265, -2.24399475, -1.34639685, -0.44879895,  0.44879895,
                1.34639685,  2.24399475,  3.14159265])
        >>> fix_phase_interval(np.array([-10, -5, 0, 5, 10]))
        array([-3.14159265, -1.57079633,  0.        ,  1.57079633,\
  3.14159265])
        >>> fix_phase_interval(np.array([-10, 10, 1, -3]))
        array([-3.14159265,  3.14159265,  0.31415927, -0.9424778 ])
    """
    if not fc.extra.is_in_range(arr, (-np.pi, np.pi)):
        arr = fc.extra.scale(arr.astype(float), (-np.pi, np.pi))
    return arr


# ======================================================================
def mag_phs_to_complex(mag_arr, phs_arr=None, fix_phase=True):
    """
    Convert magnitude and phase arrays into a complex array.

    It can automatically correct for arb. units in the phase.

    Args:
        mag_arr (np.ndarray): The magnitude image array in arb. units.
        phs_arr (np.ndarray): The phase image array in rad or arb. units.
            The values range is automatically corrected to radians.
            The wrapped data is expected.
            If units are radians, i.e. data is in the [-π, π) range,
            no conversion is performed.
            If None, only magnitude data is used.
        fix_phase (bool): Fix the phase interval / units.
            If True, `phs_arr` is corrected with `fix_phase_interval()`.

    Returns:
        cx_arr (np.ndarray): The complex image array in arb. units.

    See Also:
        pymrt.computation.fix_phase_interval
    """
    if phs_arr is not None:
        if fix_phase:
            phs_arr = fix_phase_interval(phs_arr)
        cx_arr = fc.extra.polar2complex(
            mag_arr.astype(float), phs_arr.astype(float))
    else:
        cx_arr = mag_arr.astype(float)
    return cx_arr


# ======================================================================
def referencing(
        arr,
        masks,
        ext_refs=None,
        combines=np.mean,
        ref_operation=(
                lambda arr, int_refs, ext_refs:
                arr - int_refs[0] + ext_refs[0])):
    """
    Reference a measurement to an arbitrary set of values.

    Values within the data are matched to external reference values.
    If any of the masks is defined incorrectly, no referencing is performed.

    Args:
        arr (np.ndarray): The input array. 
        masks (Iterable[Iterable[Iterable[int]]|np.ndarray]|None]):
            The reference masks.
            Values corresponding to the specified mask are used to compute
            the internal reference value.
            Each item can be:
                - np.ndarray[bool]: An array mask with same shape as `arr`.
                - Iterable[Iterable[int]]: The extrema (min_index, max_index)
                  defining an hyperrectangle.
                  The hyperrectangle must be contained within `arr`.
                - None: all elements of the array will be used.
        ext_refs (Iterable[int|float]|None): The external references.
            Units depend on input and referencing operation.
            If Iterable, its length must match that of masks.
            If None, no referencing is performed.
        combines (Iterable[callable]|callable):
            Computation of internal references.
            Each of the callable must have the following signature:
            func(np.ndarray) -> int|float
            If Iterable, its length must match that of masks.
            If callable, the same function will be used for all masks.
        ref_operation (callable): Computation of the referenced array.
            Must have the following signature:
            func(np.ndarray, Iterable[int|float], Iterable[int|float]) ->
            np.ndarray

    Returns:
        res_arr (np.ndarray): The referenced array.

    Raises:
        AssertionError: if length of masks, ext_refs and combines do not match.
    """
    res_arr = arr
    if ext_refs is not None:
        assert (len(masks) == len(ext_refs))
        num_refs = len(ext_refs)
        combines = fc.auto_repeat(combines, num_refs, True, True)
        are_mask_arr = [
            isinstance(mask, np.array) and (
                    np.issubdtype(mask.dtype,
                                  np.bool) and arr.shape == mask.shape)
            for mask in masks]
        are_mask_borders = [
            len(mask) == len(arr.shape) and all(
                [upper > lower >= 0 and lower < upper <= max_i
                 for (lower, upper), max_i in zip(mask, arr.shape)])
            for mask in masks]
        if all(is_mask_arr or is_mask_borders or mask is None
                for is_mask_arr, is_mask_borders, mask in zip(
                are_mask_arr, are_mask_borders, masks)):
            int_refs = [
                combine(arr[mask]) for combine, mask in zip(combines, masks)]
            res_arr = ref_operation(arr, int_refs, ext_refs)
        else:
            text = 'Incorrect `masks` in `pymrt.recipes.generic.warnings().'
            warnings.warn(text)
    return res_arr


# ======================================================================
def _pre_exp_loglin(arr, exp_factor=0, zero_cutoff=np.spacing(1.0)):
    arr = np.abs(arr)
    log_arr = fc.extra.apply_at(
        arr, lambda x: np.log(x) / np.exp(exp_factor), arr > zero_cutoff)
    return log_arr


# ======================================================================
def _post_exp_loglin(arr, exp_factor=0, zero_cutoff=np.spacing(1.0)):
    # tau = p_arr[..., 0]
    # s_0 = p_arr[..., 1]
    axis = -1
    for i in range(arr.shape[axis]):
        if i < arr.shape[axis] - 1:
            mask = tuple(slice(None) for d in arr[..., i].shape) \
                if zero_cutoff is None else np.abs(arr[..., i]) > zero_cutoff
            arr[..., i][mask] = -1.0 / arr[..., i][mask]
        else:
            arr[..., i] = np.exp(arr[..., i] - exp_factor)
    return arr


# ======================================================================
def _exp_s0_from_tau(arr, tau_arr, tis_arr):
    return np.mean(arr / np.exp(-tis_arr / tau_arr[..., None]))


# ======================================================================
def fit_exp_loglin(
        arr,
        tis,
        tis_mask=None,
        poly_deg=1,
        variant=None,
        full=False,
        exp_factor=0,
        zero_cutoff=np.spacing(1.0)):
    """
    Fit exponential decay to data using the log-linear method.

    Args:
        arr (np.ndarray): The input array in arb. units.
            The sampling time T_i varies in the last dimension.
        tis (Iterable): The sampling times T_i in time units.
            The number of points must match the last shape size of arr.
        tis_mask (Iterable[bool]|None): Determine the sampling times Ti to use.
            If None, all will be used.
        poly_deg (int): The degree of the polynomial to fit.
            For monoexponential fits, use num=1.
        variant (str|None): Specify a variant of the algorithm.
            A valid Python expression is expected and used as keyword argument
            of the `numpy.polyfit()` function.
            Most notably can be used to specify (global) data weighting, e.g.:
            `w=1/np.sqrt(x_arr)`.
        full (bool): Calculate additional information on the fit performance.
            If True, more information is given.
            If False, only the optimized parameters are returned.
        exp_factor (float|None): The data pre-whitening factor.
            A value different from zero, may improve numerical stability
            for very large or very small data.
        zero_cutoff (float|None): The threshold value for masking zero values.
            If None, no cut-off is performed.

    Returns:
        results (dict): The calculated information.
            If full is True, more information is available.
            Standard content:
            - `s0`: the amplitude of the exponential in arb. units.
            - `tau`: the exponential decay time constant in time units.
              Units are determined by the units of `tis`.
            - `tau_{i}` (for i=2,...,poly_deg-2): the higher order fit terms.
              Units are determined by the units of `tis`.
    """
    axis = -1

    y_arr = np.array(arr).astype(float)
    x_arr = np.array(tis).astype(float)

    if tis_mask is not None:
        y_arr = y_arr[..., tis_mask]
        x_arr = x_arr[tis_mask]

    assert (x_arr.size == y_arr.shape[axis])

    try:
        method_kws = dict(eval(variant))
    except Exception as e:  # avoid crashing for invalid variant
        text = fmtm(
            'While evaluating `variant`: {variant},' +
            'the following exception occurred: {e}.')
        warnings.warn(text)
        method_kws = {}

    p_arr = voxel_curve_fit(
        y_arr, x_arr, fit_params=1 + poly_deg,
        pre_func=_pre_exp_loglin, pre_args=[exp_factor, zero_cutoff],
        post_func=_post_exp_loglin, post_args=[exp_factor, zero_cutoff],
        method='poly', method_kws=method_kws)

    shape = p_arr.shape[:axis]
    p_arrs = [arr.reshape(shape) for arr in np.split(p_arr, 1 + poly_deg, -1)]

    results = collections.OrderedDict(
        ('s0' if i == 0 else 'tau' + ('_{i}'.format(i=i) if i > 1 else ''), x)
        for i, x in enumerate(p_arrs[::-1]))

    if full:
        raise NotImplementedError

    return results


# ======================================================================
def fit_exp_curve_fit(
        arr,
        tis,
        tis_mask=None,
        optim='lm',
        init=None,
        full=False,
        num_proc=0,
        exp_factor=0,
        zero_cutoff=np.spacing(1.0)):
    """
    Fit exponential decay to data using the log-linear method.

    Args:
        arr (np.ndarray): The input array in arb. units.
            The sampling time T_i varies in the last dimension.
        tis (Iterable): The sampling times T_i in time units.
            The number of points must match the last shape size of arr.
        tis_mask (Iterable[bool]|None): Determine the sampling times Ti to use.
            If None, all will be used.
        optim (str):
        init (np.ndarray|Iterable): The parameters to fit.
            If np.ndarray, must have all dims except the last same as `y_arr`;
            the last dim dictate the number of fit parameters; must have the
            same shape as the output.
            If Iterable, specify the initial value(s) of the parameters to fit.
        full (bool): Calculate additional information on the fit performance.
            If True, more information is given.
            If False, only the optimized parameters are returned.
        num_proc (int|None): The number of parallel processes.
            If 1, the execution is sequential.
            If 0 or None, the number of workers is determined automatically.
        exp_factor (float|None):
        zero_cutoff (float|None): The threshold value for masking zero values.
            If None, no cut-off is performed.

    Returns:
        results (dict): The calculated information.
            If full is True, more information is available.
            Standard content:
            - `s0`: the amplitude of the exponential in arb. units.
            - `tau`: the exponential decay time constant in time units.
              Units are determined by the units of `tis`.
    """
    axis = -1

    y_arr = np.array(arr).astype(float)
    x_arr = np.array(tis).astype(float)

    if tis_mask is not None:
        y_arr = y_arr[..., tis_mask]
        x_arr = x_arr[tis_mask]

    assert (x_arr.size == y_arr.shape[axis])

    num_params = 2

    if not init:
        init = [1] * num_params

    method_kws = dict(method=optim)
    if num_proc > 1:
        method = 'curve_fit_parallel'
        method_kws['num_proc'] = num_proc
    else:
        method = 'curve_fit_sequential'

    p_arr = voxel_curve_fit(
        y_arr, x_arr, func_exp_decay, init,
        method=method, method_kws=method_kws)

    shape = p_arr.shape[:axis]
    p_arrs = [arr.reshape(shape) for arr in np.split(p_arr, num_params, axis)]

    results = collections.OrderedDict((('tau', p_arrs[0]), ('s0', p_arrs[1])))

    if full:
        raise NotImplementedError

    return results


# ======================================================================
def fit_exp_quad(
        arr,
        tis,
        tis_mask=None,
        integrate=sp.integrate.simps,
        combine=np.nanmedian):
    """
    Mono-exponential decay fit using integral properties.
    
    The function to fit is: :math:`s(t) = A e^{-t / \\tau}`
    
    The value of :math:`\\tau` is estimated using the following formula:
    :math:`\int_{t_i}^{t_{i+j}} s(t) dt = \\tau [s(t_i) - s(t_{i+j})]`
    
    The integral is estimated numerically.
    
    All possible combination of i and j are calculated and combined together.
    
    This is a closed-form solution.
    
    Args:
        arr (np.ndarray): The input array in arb. units.
            The sampling time T_i varies in the last dimension.
        tis (Iterable): The sampling times T_i in time units.
            The number of points must match the last shape size of arr.
        tis_mask (Iterable[bool]|None): Determine the sampling times Ti to use.
            If None, all will be used.
        integrate (callable): The numerical integration function to use.
            Must accept an `axis=-1` keyword argument, which defines the
            dimension over which integration is done.
        combine (callable): The combination method to use.
            Must accept an `axis=-1` keyword argument, which defines the
            dimension over which combination is done.

    Returns:
        results (dict): The calculated information.
            If full is True, more information is available.
            Standard content:
            - `s0`: the amplitude of the exponential in arb. units.
            - `tau`: the exponential decay time constant in time units.
              Units are determined by the units of `tis`.
    """
    axis = -1

    y_arr = np.array(arr).astype(float)
    x_arr = np.array(tis).astype(float)

    if tis_mask is not None:
        y_arr = y_arr[..., tis_mask]
        x_arr = x_arr[tis_mask]

    assert (x_arr.size == y_arr.shape[axis])

    num = (x_arr.size * (x_arr.size - 1)) // 2
    s_arr = np.zeros((y_arr.shape[:axis]) + (num,))
    d_arr = np.zeros((y_arr.shape[:axis]) + (num,))
    n = 0
    for j in range(1, x_arr.size):
        for i in range(x_arr.size - j):
            s_arr[..., n] = integrate(
                y_arr[..., i:i + j + 1], x_arr[i:i + j + 1], axis=axis)
            d_arr[..., n] = y_arr[..., i] - y_arr[..., i + j]
            n += 1
    with np.errstate(divide='ignore', invalid='ignore'):
        tau_arr = s_arr / d_arr
    tau_arr = combine(tau_arr, axis=axis)
    del s_arr, d_arr, y_arr
    s0_arr = _exp_s0_from_tau(arr, tau_arr, x_arr)
    result = {'s0': s0_arr, 'tau': tau_arr}
    return result


# ======================================================================
def fit_exp_quadr(
        arr,
        tis,
        tis_mask=None,
        integrate=sp.integrate.simps,
        window_size=None):
    """
    Mono-exponential decay fit using integral properties (revisited).

    The function to fit is: :math:`s(t) = A e^{-t / \\tau}`

    The value of :math:`\\tau` is estimated using the following formula:
    :math:`\int_{t_i}^{t_{i+j}} s(t) dt = \\tau [s(t_i) - s(t_{i+j})]`

    The integral is estimated numerically.

    The value of `j` is determined by the `window size` parameters.
    All valid combination of i and j are calculated (from the number of
    support points).

    The partial terms are combined together using the formula:

    :math:`\\tau = \\frac{s_{ss} + s_{ds}}{s_{dd} + s_{ds}}`

    with:
        - :math:`s_{ss} = \sum_j \sum_{i=0}^{N-j} s_i^2`
        - :math:`s_{sd} = \sum_j \sum_{i=0}^{N-j} s_i d_i`
        - :math:`s_{dd} = \sum_j \sum_{i=0}^{N-j} d_i^2`

    where:
        - :math:`t_i` are the sampling points;
        - :math:`\Delta t_i` is the sampling point spacing (must be constant);
        - :math:`s_i` are the numerical integrals over the window size;
        - :math:`d_i` are the signal extrema over the window size;
        - :math:`j` is the numerical window size;
        - :math:`N` is the number of sampling points.

    The value of the window size `j` can be fixed, and the method reduces
    to the `Auto-Regression on Linear Operations (ARLO)` method.

    This is a closed-form solution.

    Args:
        arr (np.ndarray): The input array in arb. units.
            The sampling time T_i varies in the last dimension.
        tis (Iterable): The sampling times T_i in time units.
            The number of points must match the last shape size of arr.
        tis_mask (Iterable[bool]|None): Determine the sampling times Ti to use.
            If None, all will be used.
        integrate (callable): The numerical integration function to use.
            Must accept an `axis=-1` keyword argument, which defines the
            dimension over which integration is done.
        window_size (int|None): The window over which calculating the integral.
            If None, all valid window sizes are considered.

    Returns:
        results (dict): The calculated information.
            If full is True, more information is available.
            Standard content:
            - `s0`: the amplitude of the exponential in arb. units.
            - `tau`: the exponential decay time constant in time units.
              Units are determined by the units of `tis`.

    Notes:
        This method can be understood as a multi-window extension of the
        `Auto-Regression on Linear Operations (ARLO)` method.

    References:
        - Pei, M., Nguyen, T.D., Thimmappa, N.D., Salustri, C., Dong, F.,
          Cooper, M.A., Li, J., Prince, M.R., Wang, Y., 2015. Algorithm for
          fast monoexponential fitting based on Auto-Regression on Linear
          Operations (ARLO) of data. Magn. Reson. Med. 73, 843–850.
          doi:10.1002/mrm.25137
    """
    axis = -1

    y_arr = np.array(arr).astype(float)
    x_arr = np.array(tis).astype(float)

    if tis_mask is not None:
        y_arr = y_arr[..., tis_mask]
        x_arr = x_arr[tis_mask]

    assert (x_arr.size == y_arr.shape[axis])

    axis = -1

    if window_size:
        num = x_arr.size - window_size
        s_arr = np.zeros((y_arr.shape[:axis]) + (num,))
        d_arr = np.zeros((y_arr.shape[:axis]) + (num,))
        for i in range(num):
            s_arr[..., i] = integrate(
                y_arr[..., i:i + window_size + 1],
                x_arr[i:i + window_size + 1], axis=axis)
            d_arr[..., i] = y_arr[..., i] - y_arr[..., i + window_size]
    else:
        num = (x_arr.size * (x_arr.size - 1)) // 2
        s_arr = np.zeros((y_arr.shape[:axis]) + (num,))
        d_arr = np.zeros((y_arr.shape[:axis]) + (num,))
        n = 0
        for j in range(1, x_arr.size):
            for i in range(x_arr.size - j):
                s_arr[..., n] = integrate(
                    y_arr[..., i:i + j + 1], x_arr[i:i + j + 1], axis=axis)
                d_arr[..., n] = y_arr[..., i] - y_arr[..., i + j]
                n += 1

    sum_ss = np.nansum(s_arr * s_arr, axis=axis)
    sum_sd = np.nansum(s_arr * d_arr, axis=axis)
    sum_dd = np.nansum(d_arr * d_arr, axis=axis)
    with np.errstate(divide='ignore', invalid='ignore'):
        tau_arr = (sum_ss + sum_sd) / (sum_sd + sum_dd)

    del s_arr, d_arr, sum_ss, sum_sd, sum_dd, y_arr
    s0_arr = _exp_s0_from_tau(arr, tau_arr, x_arr)
    result = {'s0': s0_arr, 'tau': tau_arr}
    return result


# ======================================================================
def fit_exp_diff(
        arr,
        tis,
        tis_mask=None,
        differentiate=np.gradient,
        combine=np.nanmedian):
    """
    Mono-exponential decay fit using differential properties.
    
    The function to fit is: :math:`s(t) = A e^{-t / \\tau}`
    
    The value of :math:`\\tau` is estimated using the following formula:
    :math:`\\tau = -\\frac{s(t)}{s'(t)}`
    
    where:
    :math:`s'(t) = \\frac{d}{dt}s(t) = -\\frac{A}{\\tau} e^{-t / \\tau}`
    i.e.:
    :math:`s'(t) = -\\frac{1}{\\tau} s(t)`
    
    The derivative is estimated numerically for each sampling point and
    the result is combined together.
    
    This is a closed-form solution.
    
    Args:
        arr (np.ndarray): The input array in arb. units.
            The sampling time T_i varies in the last dimension.
        tis (Iterable): The sampling times T_i in time units.
            The number of points must match the last shape size of arr.
        tis_mask (Iterable[bool]|None): Determine the sampling times Ti to use.
            If None, all will be used.
        differentiate (callable): The numerical differentiation function to
        use.
            Must accept an `axis=-1` keyword argument, which defines the
            dimension over which integration is done.
        combine (callable): The combination method to use.
            Must accept an `axis=-1` keyword argument, which defines the
            dimension over which combination is done.

    Returns:
        results (dict): The calculated information.
            If full is True, more information is available.
            Standard content:
            - `s0`: the amplitude of the exponential in arb. units.
            - `tau`: the exponential decay time constant in time units.
              Units are determined by the units of `tis`.
    """
    axis = -1

    y_arr = np.array(arr).astype(float)
    x_arr = np.array(tis).astype(float)

    if tis_mask is not None:
        y_arr = y_arr[..., tis_mask]
        x_arr = x_arr[tis_mask]

    assert (x_arr.size == y_arr.shape[axis])

    with np.errstate(divide='ignore', invalid='ignore'):
        dy_arr = differentiate(y_arr, axis=axis)
        dx_arr = differentiate(x_arr, axis=axis)
        tau_arr = -y_arr * dx_arr / dy_arr

    tau_arr = combine(tau_arr, axis=axis)
    del dy_arr, dx_arr, y_arr
    s0_arr = _exp_s0_from_tau(arr, tau_arr, x_arr)
    result = {'s0': s0_arr, 'tau': tau_arr}
    return result


# ======================================================================
def fit_exp_arlo(
        arr,
        tis,
        tis_mask=None,
        window_size=2):
    """
    Mono-exponential decay fit using 'Auto-Regression on Linear Operations'.

    The function to fit is: :math:`s(t) = A e^{-t / \\tau}`
    
    The value of :math:`\\tau` is estimated using the following formula:
    
    :math:`\\tau = f \\frac{s_{ss} + f s_{ds}}{s_{dd} + f s_{ds}}`

    with:
        - :math:`s_{ss} = \sum_{i=0}^{N-W} s_i^2`
        - :math:`s_{sd} = \sum_{i=0}^{N-W} s_i d_i`
        - :math:`s_{dd} = \sum_{i=0}^{N-W} d_i^2`
        - :math:`f = \Delta t_i / W`
    
    where:
        - :math:`t_i` are the sampling points;
        - :math:`\Delta t_i` is the sampling point spacing (must be constant);
        - :math:`s_i` are the numerical integrals over the window size;
        - :math:`d_i` are the signal extrema over the window size;
        - :math:`W` is the window size;
        - :math:`N` is the number of sampling points.

    This is a closed-form solution.

    Args:
        arr (np.ndarray): The input array in arb. units.
            The sampling time T_i varies in the last dimension.
        tis (Iterable): The sampling times T_i in time units.
            The number of points must match the last shape size of arr.
        tis_mask (Iterable[bool]|None): Determine the sampling times Ti to use.
            If None, all will be used.
        window_size (int): The window over which calculating the integral. 

    Returns:
        results (dict): The calculated information.
            If full is True, more information is available.
            Standard content:
            - `s0`: the amplitude of the exponential in arb. units.
            - `tau`: the exponential decay time constant in time units.
              Units are determined by the units of `tis`.

    Notes:
        This is an independent implementation of the monoexponential fitting
        method "Auto-Regression on Linear Operations (ARLO)".
        However, some of the formulae derived in the original paper,
        particularly the final one may follow a different notation.
        This software implements the formula specified in this documentation.

    References:
        - Pei, M., Nguyen, T.D., Thimmappa, N.D., Salustri, C., Dong, F.,
          Cooper, M.A., Li, J., Prince, M.R., Wang, Y., 2015. Algorithm for
          fast monoexponential fitting based on Auto-Regression on Linear
          Operations (ARLO) of data. Magn. Reson. Med. 73, 843–850.
          doi:10.1002/mrm.25137
    """
    axis = -1

    y_arr = np.array(arr).astype(float)
    x_arr = np.array(tis).astype(float)

    if tis_mask is not None:
        y_arr = y_arr[..., tis_mask]
        x_arr = x_arr[tis_mask]

    assert (x_arr.size == y_arr.shape[axis])  # dimensions must match

    axis = -1

    if not is_linear(x_arr):
        warnings.warn('Formula not exact for non-linear sampling')

    dti = np.mean(np.diff(x_arr))
    s_arr = np.zeros((y_arr.shape[:axis]) + (x_arr.size - window_size,))
    d_arr = np.zeros((y_arr.shape[:axis]) + (x_arr.size - window_size,))
    for i in range(x_arr.size - window_size):
        s_arr[..., i] = sp.integrate.simps(
            y_arr[..., i:i + window_size + 1],
            x_arr[i:i + window_size + 1], axis=axis)
        d_arr[..., i] = y_arr[..., i] - y_arr[..., i + window_size]

    sum_ss = np.nansum(s_arr * s_arr, axis=axis)
    sum_sd = np.nansum(s_arr * d_arr, axis=axis)
    sum_dd = np.nansum(d_arr * d_arr, axis=axis)
    with np.errstate(divide='ignore', invalid='ignore'):
        tau_arr = (dti / window_size) \
                  * (sum_ss + dti / window_size * sum_sd) \
                  / (sum_dd + dti / window_size * sum_sd)

    del s_arr, d_arr, sum_ss, sum_sd, sum_dd, y_arr
    s0_arr = _exp_s0_from_tau(arr, tau_arr, x_arr)
    result = {'s0': s0_arr, 'tau': tau_arr}
    return result


# ======================================================================
def fit_exp(
        arr,
        tis,
        tis_mask=None,
        method='quadr',
        method_kws=None):
    """
    Mono-exponential decay fit.

    Args:
        arr (np.ndarray): The input array in arb. units.
            The sampling time T_i varies in the last dimension.
        tis (Iterable): The sampling times T_i in time units.
            The number of points must match the last shape size of arr.
        tis_mask (Iterable[bool]|None): Determine the sampling times Ti to use.
            If None, all will be used.
        method (str): Determine the fitting method to use.
            Accepted values are:
             - 'auto': determine an optimal method by inspecting the data.
             - 'loglin': use a log-linear fit, fast but fragile.
             - 'curve_fit': use non-linear least square curve fitting, slow
               but accurate.
             - 'diff': closed-form solution using the differential properties
               of the exponential, very fast but fragile.
             - 'quad': closed-form solution using the quadrature properties
               of the exponential, very fast and moderately robust.
             - 'arlo': closed-form solution using the `Auto-Regression on
               Linear Operations (ARLO)` method (similar to 'quad'),
               very fast and robust.
             - 'quadr': closed-form solution using the quadrature properties
               of the exponential and optimal noise regression,
               very fast and very robust (extends both 'quad' and 'arlo').
        method_kws (Mappable|None): Keyword arguments to pass to `method`.

    Returns:
        results (dict): The calculated information.
            If full is True, more information is available.
            Standard content:
            - `s0`: the amplitude of the exponential in arb. units.
            - `tau`: the exponential decay time constant in time units.
              Units are determined by the units of `tis`.
            Additional content may be available depending on the method used.
    """
    method = method.lower()
    method_kws = {} if method_kws is None else dict(method_kws)
    methods = ('loglin', 'curve_fit', 'diff', 'quad', 'arlo', 'quadr')

    if method == 'auto':
        raise NotImplementedError

    if method in methods:
        method = eval(method)
    if not callable(method):
        text = (
                'Unknown method `{}` in `recipes.generic.fit_exp(). ' +
                'Using fallback `{}`.'.format(method, methods[0]))
        warnings.warn(text)
        method = eval(methods[0])

    return method(arr, tis, tis_mask, **method_kws)


# ======================================================================
def cx_2_combine(
        cx1_arr,
        cx2_arr,
        func='ratio',
        regularization=np.spacing(1.0)):
    """
    Calculate the combination of two arrays.

    Args:
        cx1_arr (float|np.ndarray): First complex array.
        cx2_arr (float|np.ndarray): Second complex array.
        func (str|callable): Determine the combination function to use.
            If str, must be any of:
             - 'ratio', 'div': :math:`\\frac{s_1}{s_2}`
             - 'i-ratio', 'i-div', 'inverse-ratio': :math:`\\frac{s_2}{s_1}`
             - 'p-ratio', 'pseudo-ratio', 'uni', 'uniform':
               :math:`\\frac{s_1 s_2}{s_1^2+s_2^2}`.
             - 'mp2rage': The MP2RAGE rho:
               :math:`\\frac{s_1^* s_2}{s_1^2+s_2^2}`
            If callable, its signature must be:
            `func(np.ndarray, np.ndarray) -> np.ndarray`
        regularization (float|int): Parameter for the regularization.
            This parameter is added to the denominator of the fractional
            expressions for normalization purposes, therefore should be much
            smaller than the average of the magnitude arrays.
            Larger values of this parameter may have as side effect the
            denoising the background.

    Returns:
        result (np.ndarray): The combined  array.
    """
    if callable(func):
        result = func(cx1_arr, cx2_arr)
    else:
        func = func.lower()
        if func in ('ratio', 'div', 'inverse-ratio'):
            result = cx1_arr / (cx2_arr + regularization)
        elif func in ('i-ratio', 'i-div'):
            result = cx2_arr / (cx1_arr + regularization)
        elif func in ('p-ratio', 'pseudo-ratio', 'uni', 'uniform'):
            result = cx_div(cx1_arr, cx2_arr, regularization)
        elif func == 'mp2rage':
            result = np.real(
                cx1_arr.conj() * cx2_arr /
                (np.abs(cx1_arr) + np.abs(cx2_arr) + regularization))
        else:
            raise ValueError('Unknown value `{}` for `func`.'.format(func))
    return result


# ======================================================================
def mag_phase_2_combine(
        mag1_arr,
        phs1_arr,
        mag2_arr,
        phs2_arr,
        func='ratio',
        regularization=np.spacing(1.0)):
    """
    Calculate the combination of two arrays.

    This is also referred to as the uniform arrays, because it should be free
    from low-spatial frequency biases.

    Args:
        cx1_arr (float|np.ndarray): First complex array.
        cx2_arr (float|np.ndarray): Second complex array.
        func (str|callable): Determine the combination function to use.
            If str, must be any of:
             - 'ratio': :math:`\frac{s_1}{s_2}`
             - 'pseudo-ratio': :math:`\frac{s_1 s_2}{s_1^2+s_2^2}`.
             - 'mp2rage': The MP2RAGE rho:
               :math:`\frac{s_1^* s_2}{s_1^2+s_2^2}`
            If callable, its signature must be:
            `func(np.ndarray, np.ndarray) -> np.ndarray`
        regularization (float|int): Parameter for the regularization.
            This parameter is added to the denominator of the fractional
            expressions for normalization purposes, therefore should be much
            smaller than the average of the magnitude arrays.
            Larger values of this parameter may have as side effect the
            denoising the background.

    Returns:
        result (np.ndarray): The rho (uniform) array.
    """
    mag1_arr = mag1_arr.astype(float)
    mag2_arr = mag2_arr.astype(float)
    phs1_arr = fix_phase_interval(phs1_arr)
    phs2_arr = fix_phase_interval(phs2_arr)
    inv1_arr = fc.extra.polar2complex(mag1_arr, phs1_arr)
    inv2_arr = fc.extra.polar2complex(mag2_arr, phs2_arr)
    return cx_2_combine(inv1_arr, inv2_arr, regularization, func)


# ======================================================================
def _curve_fit(args):
    """
    Interface to use `scipy.optimize.curve_fit` with multiprocessing.
    
    If one of the following exception is encountered:
     
    The resulting parameters and their covariance are set to NaN.

    Args:
        args (list): Positional parameters.
            These are passed to `scipy.optimize.curve_fit()`.

    Returns:
        p_val (np.ndarray): Optimized parameters.
        p_cov (np.ndarray): The covariance of the optimized parameters.
            The diagonals provide the variance of the parameter estimate
    """
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            warnings.simplefilter('error', sp.optimize.OptimizeWarning)
            p_val, p_cov = sp.optimize.curve_fit(*args)
    except (RuntimeError, ValueError, sp.optimize.OptimizeWarning):
        err_val = np.nan
        # number of fitting parameters
        num_params = len(args[3])
        p_val = np.full(num_params, err_val)
        p_cov = np.full((num_params, num_params), err_val)
    return p_val, p_cov


# ======================================================================
def voxel_curve_fit(
        y_arr,
        x_arr,
        fit_func=None,
        fit_params=None,
        pre_func=None,
        pre_args=None,
        pre_kws=None,
        post_func=None,
        post_args=None,
        post_kws=None,
        method=None,
        method_kws=None):
    """
    Curve fitting for y = F(x, p)

    Args:
        y_arr (np.ndarray): Dependent variable array.
            The shape of this array is unrestricted.
            The last axis depends on the independent variable.
        x_arr (np.ndarray): Independent variable array.
            Must be 1D, with length equal to the size of last `y_arr` axis.
        fit_func (func): The function to fit.
        fit_params (int|np.ndarray|Iterable): The parameters to fit.
            If int, specify the number of parameters to fit.
            If np.ndarray, must have all dims except the last same as `y_arr`;
            the last dim dictate the number of fit parameters; must have the
            same shape as the output.
            If Iterable, specify the initial value(s) of the parameters to fit.
            If the method requires initial values, but only the number of
            parameters to fit is specified, initial value(s) set to one.
        pre_func (func):
        pre_args (list):
        pre_kws (Mappable|None):
        post_func (func):
        post_args (list):
        post_kws (Mappable|None):
        method (str): Method to use for the curve fitting procedure.
        method_kws (Mappable|None): Keyword arguments to pass to `method`.

    Returns:
        p_arr (np.ndarray) :
    """
    axis = -1

    # reshape to linearize the independent dimensions of the array
    shape = y_arr.shape
    support_size = shape[axis] if len(shape) > 1 else 1
    if len(shape) > 1:
        y_arr = y_arr.reshape((-1, support_size))
    num_voxels = y_arr.shape[0] if len(shape) > 1 else 1

    # p_arr = np.zeros((num_voxels, num_params))
    if isinstance(fit_params, int):
        num_params = fit_params
        fit_params = (1.0,) * num_params
        p_arr = np.ones((num_voxels, num_params))
    elif isinstance(fit_params, np.ndarray):
        num_params = fit_params.shape[axis]
        p_arr = fit_params
        fit_params = tuple(np.mean(p_arr, axis=0) if len(shape) > 1 else p_arr)
    else:
        num_params = len(fit_params)
        p_arr = np.tile(fit_params, num_voxels).astype(float)

    p_arr = p_arr.reshape((num_voxels, num_params))

    # preprocessing
    if pre_func is not None:
        if pre_args is None:
            pre_args = []
        if pre_kws is None:
            pre_kws = {}
        y_arr = pre_func(y_arr, *pre_args, **pre_kws)

    if not method:
        if fit_func:
            method = 'curve_fit_parallel_map'
        elif fit_params:
            method = 'poly'
    method_kws = {} if method_kws is None else dict(method_kws)

    if method == 'curve_fit_sequential':
        for i in range(num_voxels):
            y_i_arr = y_arr[i] if len(shape) > 1 else y_arr
            p_i_arr = p_arr[i]
            try:
                tmp = sp.optimize.curve_fit(
                    fit_func, x_arr, y_i_arr, p_i_arr, **method_kws)
            except RuntimeError:
                p_arr[i] = p_i_arr
            else:
                p_arr[i] = tmp[0]

    elif method == 'curve_fit_parallel':
        if 'num_proc' in method_kws:
            num_proc = method_kws['num_proc']
            method_kws.pop('num_proc')
        else:
            num_proc = multiprocessing.cpu_count() + 1
        pool = multiprocessing.Pool()
        mp_results = []
        for i in range(num_voxels):
            y_i_arr = y_arr[i] if len(shape) > 1 else y_arr
            p_i_arr = p_arr[i]
            kws = dict(f=fit_func, xdata=x_arr, ydata=y_i_arr, p0=p_i_arr)
            kws.update(method_kws)
            mp_results.append(
                pool.apply_async(sp.optimize.curve_fit, kwds=kws))
        for i, mp_result in enumerate(mp_results):
            try:
                p_val, p_cov = mp_result.get()
            except sp.optimize.OptimizeWarning:
                pass
            else:
                p_arr[i] = p_val

    elif method == 'curve_fit_parallel_map':
        num_proc = method_kws['num_proc'] \
            if 'num_proc' in method_kws else multiprocessing.cpu_count() + 1
        chunksize = method_kws['chunksize'] \
            if 'chunksize' in method_kws else 2 * multiprocessing.cpu_count()
        bounds = method_kws['bounds'] \
            if 'bounds' in method_kws else (-np.inf, np.inf)
        method = method_kws['method'] \
            if 'method' in method_kws else None
        iter_voxels = [
            (fit_func, x_arr, y_arr[i] if len(shape) > 1 else y_arr, p_arr[i],
             None, False, True, bounds, method)
            for i in range(num_voxels)]
        pool = multiprocessing.Pool(num_proc)
        for i, res in enumerate(pool.imap(_curve_fit, iter_voxels, chunksize)):
            p_val, p_cov = res
            if not np.isnan(np.sum(p_val)):
                p_arr[i] = p_val

    elif method == 'poly':
        # polyfit requires to change matrix orientation using transpose
        p_arr = np.polyfit(
            x_arr, y_arr.transpose(), num_params - 1, **method_kws)
        # transpose the results back
        p_arr = np.transpose(p_arr)

    else:
        try:
            p_arr = fit_func(y_arr, x_arr, fit_params)
        except Exception as e:
            warnings.warn(fmtm(
                'W: Exception `{e}` in ndarray_fit() method `method}`'))

    # revert to original shape
    p_arr = p_arr.reshape((shape[:axis]) + (num_params,))

    # post process
    if post_func is not None:
        if post_args is None:
            post_args = []
        if post_kws is None:
            post_kws = {}
        p_arr = post_func(p_arr, *post_args, **post_kws)

    return p_arr


# ======================================================================
def linsolve_iter(
        linear_operator,
        const_term,
        max_iter=None,
        tol=1e-8,
        x0_arr=None,
        preconditioner=None,
        callback=None,
        method=None,
        method_kws=None,
        verbose=D_VERB_LVL):
    """
    Iterative solve for x the linear system: Ax = b

    More in general, this will look for:

    .. math::
        argmin_x ||Ax - b||_2^2

    Eventual regularization terms must be included in the definition of the
    linear operator :math:`A` and the constant term :math:`b`.

    Args:
        linear_operator (np.ndarray|LinearOperator): The linear operator A.
            The `scipy.sparse.linalg.LinearOperator()` must be used when
            the linear operator :math:`A` is not explicitly known.
        const_term (np.ndarray): The constant term b.
        max_iter (int|None): The maximum number of iterations.
            If None, the default value for each method is used.
        tol (float): The iteration tolerance.
            The exact use of this parameter varies from method to method.
        x0_arr (np.ndarray|None): The initial guess.
            If None, the default value for each method is used.
        preconditioner (np.ndarray|LinearOperator): The preconditioner.
            This is a linear operator that improves the numerical conditioning.
            It should approximate :math:`A^{-1}`.
        callback (callable|None): Function called at each iteration.
            Must have the following signature: f(np.ndarray) -> Any|None
            The first argument is the approximate solution for `x_arr` at
            the given iteration step.
        method (str|None): Iterative algorithm to use.
            If None, this is determined automatically based on the problem.
            Accepted values are:
             - 'lsmr': use `scipy.sparse.linalg.lsmr()`.
               Requires computing :math:`Ax` and :math:`A^Hb`.
             - 'lsqr': use `scipy.sparse.linalg.lsqr()`.
               Requires computing :math:`Ax` and :math:`A^Hb`.
             - 'bicg': use `scipy.sparse.linalg.bicg()`.
               Requires an endomorphism (square matrix), and
               computing :math:`Ax` and :math:`A^Hb`.
             - 'bicgstab': use `scipy.sparse.linalg.bicgstab()`.
               Requires an endomorphism (square matrix).
             - 'cg': use `scipy.sparse.linalg.cg()`.
               Requires a hermitian (:math:`A=A^H`), positive definite
               endomorphism (square matrix).
             - 'cgs': use `scipy.sparse.linalg.cgs()`.
               Requires an endomorphism (square matrix).
             - 'gmres': use `scipy.sparse.linalg.gmres()`.
               Requires an endomorphism (square matrix).
             - 'lgmres': use `scipy.sparse.linalg.lgmres()`.
               Requires an endomorphism (square matrix).
             - 'qmr': use `scipy.sparse.linalg.qmr()`.
               Requires an endomorphism (square matrix), and
               computing :math:`Ax` and :math:`A^Hb`.
             - 'minres': use `scipy.sparse.linalg.minres()`.
               Requires an endomorphism (square matrix).
        method_kws (Mappable|None): Keyword arguments to pass to `method`.
        verbose (int): Set level of verbosity.

    Returns:
        x_arr (np.ndarray): The output array.
    """
    # TODO: Add support for stopping condition
    show = verbose >= VERB_LVL['high']
    if method is None:
        if linear_operator.shape[0] != linear_operator.shape[1]:
            method = 'lsqr'
        else:
            try:
                linear_operator.dot(const_term)
            except AttributeError:
                method = 'lgmres'
            else:
                method = 'lsmr'
    method = method.lower()
    method_kws = {} if method_kws is None else dict(method_kws)

    if method == 'lsmr':
        if x0_arr is not None:
            text = 'Initial guess not used.'
            warnings.warn(text)
        if preconditioner is not None:
            text = 'Preconditionr operator `preconditioner` not used.'
            warnings.warn(text)
        if callback is not None:
            text = 'Callback function `callback` not used.'
            warnings.warn(text)

        res = sp.sparse.linalg.lsmr(
            linear_operator, const_term,
            atol=tol, btol=tol,
            maxiter=max_iter, show=show, **method_kws)

    elif method == 'lsqr':
        if x0_arr is not None:
            text = 'Initial guess `x0_arr` not used.'
            warnings.warn(text)
        if preconditioner is not None:
            text = 'Preconditionr operator `preconditioner` not used.'
            warnings.warn(text)
        if callback is not None:
            text = 'Callback function `callback` not used.'
            warnings.warn(text)

        res = sp.sparse.linalg.lsqr(
            linear_operator, const_term,
            atol=tol, btol=tol,
            iter_lim=max_iter, show=show, **method_kws)

    elif method == 'bicg':
        res = sp.sparse.linalg.bicg(
            linear_operator, const_term,
            tol=tol, x0=x0_arr, callback=callback, M=preconditioner,
            maxiter=max_iter, **method_kws)

    elif method == 'bicgstab':
        res = sp.sparse.linalg.bicgstab(
            linear_operator, const_term,
            tol=tol, x0=x0_arr, callback=callback, M=preconditioner,
            maxiter=max_iter, **method_kws)

    elif method == 'cg':
        res = sp.sparse.linalg.cg(
            linear_operator, const_term,
            tol=tol, x0=x0_arr, callback=callback, M=preconditioner,
            maxiter=max_iter, **method_kws)

    elif method == 'cgs':
        res = sp.sparse.linalg.cgs(
            linear_operator, const_term,
            tol=tol, x0=x0_arr, M=preconditioner,
            maxiter=max_iter, callback=callback, **method_kws)

    elif method == 'gmres':
        res = sp.sparse.linalg.gmres(
            linear_operator, const_term,
            tol=tol, x0=x0_arr, callback=callback, M=preconditioner,
            maxiter=max_iter, **method_kws)

    elif method == 'lgmres':
        res = sp.sparse.linalg.lgmres(
            linear_operator, const_term,
            tol=tol, x0=x0_arr, callback=callback, M=preconditioner,
            maxiter=max_iter, **method_kws)

    elif method == 'qmr':
        if preconditioner is not None:
            text = 'Preconditionr operator `preconditioner` not used.'
            warnings.warn(text)
        res = sp.sparse.linalg.qmr(
            linear_operator, const_term,
            tol=tol, x0=x0_arr, callback=callback,
            maxiter=max_iter, **method_kws)

    elif method == 'minres':
        res = sp.sparse.linalg.minres(
            linear_operator, const_term,
            tol=tol, x0=x0_arr, callback=callback, M=preconditioner,
            maxiter=max_iter, **method_kws)

    else:
        text = 'Unknown iterative solver method `{}`'.format(method)
        raise ValueError(text)

    return res[0]


# ======================================================================
elapsed(__file__[len(PATH['base']) + 1:])

# ======================================================================
if __name__ == '__main__':
    run_doctests(__doc__)
