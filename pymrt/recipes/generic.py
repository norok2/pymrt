#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pymrt.recipes.generic: generic computation algorithms.
"""

# ======================================================================
# :: Future Imports
from __future__ import (
    division, absolute_import, print_function, unicode_literals)

# ======================================================================
# :: Python Standard Library Imports
# import itertools  # Functions creating iterators for efficient looping
import collections  # Container datatypes
import warnings  # Warning control
import multiprocessing  # Process-based parallelism

# :: External Imports
import numpy as np  # NumPy (multidimensional numerical arrays library)
import scipy as sp  # SciPy (signal and image processing library)

import scipy.integrate  # SciPy: Integration and ODEs
import scipy.optimize  # SciPy: Optimization and root finding

# :: Local Imports
import pymrt as mrt
import pymrt.utils
import pymrt.segmentation

from pymrt import INFO, DIRS
# from pymrt import VERB_LVL, D_VERB_LVL, VERB_LVL_NAMES
from pymrt import elapsed, print_elapsed
from pymrt import msg, dbg

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
        regularization=np.spacing(1),
        values_interval=None):
    """
    Calculate the expression: s1 * s2 / (s1^2 + s2^2)

    This is an SNR optimal expression for (s1 / s2) or (s2 / s1).
    Resulting values are in the [-0.5, 0.5] interval.

    Args:
        arr1 (float|np.ndarray): Complex image of the first inversion.
        arr2 (float|np.ndarray): Complex image of the second inversion.
        regularization (float|int): Parameter for the regularization.
            This parameter is added to the denominator of the expression
            for preventing undefined values when both s1 and s2 vanish.
        values_interval (tuple[float|int]|None): The output values interval.
            The standard values are linearly converted to this range.
            If None, the natural [-0.5, 0.5] interval will be used.

    Returns:
        rho_arr (float|np.ndarray): The calculated rho (uniform) image.
    """
    rho_arr = arr1 * arr2 / (np.abs(arr1) + np.abs(arr2) + regularization)
    if values_interval:
        rho_arr = mrt.utils.scale(rho_arr, values_interval, (-0.5, 0.5))
    return rho_arr


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
        array([-3.14159265, -1.57079633,  0.        ,  1.57079633,
        3.14159265])
        >>> fix_phase_interval(np.array([-10, 10, 1, -3]))
        array([-3.14159265,  3.14159265,  0.31415927, -0.9424778 ])
    """
    if not mrt.utils.is_in_range(arr, (-np.pi, np.pi)):
        arr = mrt.utils.scale(arr.astype(float), (-np.pi, np.pi))
    return arr


# ======================================================================
def mag_phs_to_complex(mag_arr, phs_arr=None, fix_phase=True):
    """
    Convert magnitude and phase arrays into a complex array.

    It can automatically correct for arb.units in the phase.

    Args:
        mag_arr (np.ndarray): The magnitude image array in arb.units.
        phs_arr (np.ndarray): The phase image array in rad or arb.units.
            The values range is automatically corrected to radians.
            The wrapped data is expected.
            If units are radians, i.e. data is in the [-π, π) range,
            no conversion is performed.
            If None, only magnitude data is used.
        fix_phase (bool): Fix the phase interval / units.
            If True, `phs_arr` is corrected with `fix_phase_interval()`.

    Returns:
        cx_arr (np.ndarray): The complex image array in arb.units.

    See Also:
        pymrt.computation.fix_phase_interval
    """
    if phs_arr is not None:
        if fix_phase:
            phs_arr = fix_phase_interval(phs_arr)
        cx_arr = mrt.utils.polar2complex(mag_arr.astype(float),
                                         phs_arr.astype(float))
    else:
        cx_arr = mag_arr.astype(float)
    return cx_arr


# ======================================================================
def fix_noise_mean(arr):
    """
    Fix magnitude level to remove the mean of the Rician noise.

    The noise of the resulting data should now have zero mean and be
    approximately Gaussian.

    Args:
        arr (np.ndarray): The input array.

    Returns:
        arr (np.ndarray): The output array.
    """
    noise_mask = arr < mrt.segmentation.threshold_otsu(arr) / 2
    return arr - np.mean(arr[noise_mask])


# ======================================================================
def _pre_exp_loglin(arr, exp_factor=0, zero_cutoff=np.spacing(1)):
    arr = np.abs(arr)
    log_arr = np.zeros_like(arr)
    # calculate logarithm only of strictly positive values
    log_arr[arr > zero_cutoff] = (
        np.log(arr[arr > zero_cutoff] * np.exp(exp_factor)))
    return log_arr


# ======================================================================
def _post_exp_loglin(arr, exp_factor=0, zero_cutoff=np.spacing(1)):
    # tau = p_arr[..., 0]
    # s_0 = p_arr[..., 1]
    axis = -1
    for i in range(arr.shape[axis]):
        if i < arr.shape[axis] - 1:
            mask = np.abs(arr[..., i]) > zero_cutoff
            arr[..., i][mask] = -1.0 / arr[..., i][mask]
        else:
            arr[..., i] = np.exp(arr[..., i] - exp_factor)
    return arr


# ======================================================================
def fit_exp_loglin(
        arr,
        tis,
        tis_mask=None,
        poly_deg=1,
        variant=None,
        full=False,
        exp_factor=0,
        zero_cutoff=np.spacing(1)):
    """
    Fit exponential decay to data using the log-linear method.

    Args:
        arr (np.ndarray): The input array in arb.units.
            The sampling time T_i varies in the last dimension.
        tis (iterable): The sampling times T_i in time units.
            The number of points must match the last shape size of arr.
        tis_mask (iterable[bool]|None): Determine the sampling times Ti to use.
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

    Returns:
        results (dict): The calculated information.
            If full is True, more information is available.
            Standard content:
            - `s0` contains the amplitude of the exponential in arb.units.
            - `tau_{i}` for i=1,...,num contain the higher order fit terms.
              Units are determined by the units of `tis`.
    """
    axis = -1

    y_arr = np.array(arr).astype(float)
    x_arr = np.array(tis).astype(float)

    if tis_mask:
        y_arr = y_arr[..., tis_mask]
        x_arr = x_arr[tis_mask]

    assert (x_arr.size == y_arr.shape[axis])

    try:
        method_kws = dict(eval(variant))
    except Exception:
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
        warnings.warn('E: Not implemented yet!')

    return results


# ======================================================================
def fit_exp_leasq(
        arr,
        tis,
        tis_mask=None,
        optim='lm',
        init=None,
        full=False,
        exp_factor=0,
        zero_cutoff=np.spacing(1)):
    """
    Fit exponential decay to data using the log-linear method.

    Args:
        arr (np.ndarray): The input array in arb.units.
            The sampling time T_i varies in the last dimension.
        tis (iterable): The sampling times T_i in time units.
            The number of points must match the last shape size of arr.
        tis_mask (iterable[bool]|None): Determine the sampling times Ti to use.
            If None, all will be used.
        optim (str):
        init (np.ndarray|iterable): The parameters to fit.
            If np.ndarray, must have all dims except the last same as `y_arr`;
            the last dim dictate the number of fit parameters; must have the
            same shape as the output.
            If iterable, specify the initial value(s) of the parameters to fit.
        full (bool): Calculate additional information on the fit performance.
            If True, more information is given.
            If False, only the optimized parameters are returned.
        exp_factor (float|None):
        zero_cutoff (float|None): The threshold value for masking zero values.

    Returns:
        results (dict): The calculated information.
            If full is True, more information is available.
            Standard content:
            - `s0` contains the amplitude of the exponential in arb.units.
            - `tau_{i}` for i=1,...,num contain the higher order fit terms.
              Units are determined by the units of `tis`.
    """
    axis = -1

    y_arr = np.array(arr).astype(float)
    x_arr = np.array(tis).astype(float)

    if tis_mask:
        y_arr = y_arr[..., tis_mask]
        x_arr = x_arr[tis_mask]

    assert (x_arr.size == y_arr.shape[axis])

    num_params = 2

    if not init:
        init = [1] * num_params

    p_arr = voxel_curve_fit(
        y_arr, x_arr, func_exp_decay, init, method='curve_fit_parallel_map',
        method_kws=dict(method=optim))

    shape = p_arr.shape[:axis]
    p_arrs = [arr.reshape(shape) for arr in np.split(p_arr, num_params, axis)]

    results = collections.OrderedDict((('tau', p_arrs[0]), ('s0', p_arrs[1])))

    if full:
        warnings.warn('E: Not implemented yet!')

    return results


# ======================================================================
def fit_exp_tau_quad(
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
        arr (np.ndarray): The input array in arb.units.
            The sampling time T_i varies in the last dimension.
        tis (iterable): The sampling times T_i in time units.
            The number of points must match the last shape size of arr.
        tis_mask (iterable[bool]|None): Determine the sampling times Ti to use.
            If None, all will be used.
        integrate (callable): The numerical integration function to use.
            Must accept an `axis=-1` keyword argument, which defines the
            dimension over which integration is done.
        combine (callable): The combination method to use.
            Must accept an `axis=-1` keyword argument, which defines the
            dimension over which combination is done.

    Returns:
        tau_arr (np.ndarray): The exponential time constant in time units.
            Units are determined by the units of `tis`.
    """
    axis = -1

    y_arr = np.array(arr).astype(float)
    x_arr = np.array(tis).astype(float)

    if tis_mask:
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
    tau_arr = s_arr / d_arr
    return combine(tau_arr, axis=axis)


# ======================================================================
def fit_exp_tau_quadr(
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
        arr (np.ndarray): The input array in arb.units.
            The sampling time T_i varies in the last dimension.
        tis (iterable): The sampling times T_i in time units.
            The number of points must match the last shape size of arr.
        tis_mask (iterable[bool]|None): Determine the sampling times Ti to use.
            If None, all will be used.
        integrate (callable): The numerical integration function to use.
            Must accept an `axis=-1` keyword argument, which defines the
            dimension over which integration is done.
        window_size (int|None): The window over which calculating the integral.
            If None, all valid window sizes are considered.
    Returns:
        tau_arr (np.ndarray): The exponential time constant in time units.
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

    if tis_mask:
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
    tau_arr = (sum_ss + sum_sd) / (sum_sd + sum_dd)

    return tau_arr


# ======================================================================
def fit_exp_tau_diff(
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
        arr (np.ndarray): The input array in arb.units.
            The sampling time T_i varies in the last dimension.
        tis (iterable): The sampling times T_i in time units.
            The number of points must match the last shape size of arr.
        tis_mask (iterable[bool]|None): Determine the sampling times Ti to use.
            If None, all will be used.
        differentiate (callable): The numerical differentiation function to
        use.
            Must accept an `axis=-1` keyword argument, which defines the
            dimension over which integration is done.
        combine (callable): The combination method to use.
            Must accept an `axis=-1` keyword argument, which defines the
            dimension over which combination is done.

    Returns:
        tau_arr (np.ndarray): The exponential time constant in time units.
            Units are determined by the units of `tis`.
    """
    axis = -1

    y_arr = np.array(arr).astype(float)
    x_arr = np.array(tis).astype(float)

    if tis_mask:
        y_arr = y_arr[..., tis_mask]
        x_arr = x_arr[tis_mask]

    assert (x_arr.size == y_arr.shape[axis])

    dy_arr = differentiate(y_arr, axis=axis)
    dx_arr = differentiate(x_arr, axis=axis)
    tau_arr = -y_arr * dx_arr / dy_arr

    return combine(tau_arr, axis=axis)


# ======================================================================
def fit_exp_tau_arlo(
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
        arr (np.ndarray): The input array in arb.units.
            The sampling time T_i varies in the last dimension.
        tis (iterable): The sampling times T_i in time units.
            The number of points must match the last shape size of arr.
        tis_mask (iterable[bool]|None): Determine the sampling times Ti to use.
            If None, all will be used.
        window_size (int): The window over which calculating the integral. 

    Returns:
        tau_arr (np.ndarray): The exponential time constant in time units.
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

    if tis_mask:
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
    tau_arr = (dti / window_size) \
              * (sum_ss + dti / window_size * sum_sd) \
              / (sum_dd + dti / window_size * sum_sd)
    return tau_arr


# ======================================================================
def fit_exp_tau_loglin(
        arr,
        tis,
        tis_mask=None,
        variant='w=1/np.sqrt(x_arr)',
        exp_factor=0,
        zero_cutoff=np.spacing(1)):
    """
    Fit exponential decay to data using the log-linear method.

    Args:
        arr (np.ndarray): The input array in arb.units.
            The sampling time T_i varies in the last dimension.
        tis (iterable): The sampling times T_i in time units.
            The number of points must match the last shape size of arr.
        tis_mask (iterable[bool]|None): Determine the sampling times Ti to use.
            If None, all will be used.
        variant (str): Specify a variant of the algorithm.
            A valid Python expression is expected and used as keyword argument
            of the `numpy.polyfit()` function.
            Most notably can be used to specify (global) data weighting, e.g.:
            `w=1/np.sqrt(x_arr)`.
        exp_factor (float|None): The data pre-whitening factor.
            A value different from zero, may improve numerical stability
            for very large or very small data.
        zero_cutoff (float|None): The threshold value for masking zero values.

    Returns:
        tau_arr (np.ndarray): The exponential time constant in time units.
            Units are determined by the units of `tis`.
    """
    results = fit_exp_loglin(
        arr, tis, tis_mask, poly_deg=1,
        variant=variant, full=False, exp_factor=exp_factor,
        zero_cutoff=zero_cutoff)
    return results['tau']


# ======================================================================
def fit_exp_tau(
        arr,
        tis,
        tis_mask=None):
    """
    Mono-exponential decay fit.
    
    The actual algorithm to be used is determined from the data itself.
    
    Args:
        arr (np.ndarray): The input array in arb.units.
            The sampling time T_i varies in the last dimension.
        tis (iterable): The sampling times T_i in time units.
            The number of points must match the last shape size of arr.
        tis_mask (iterable[bool]|None): Determine the sampling times Ti to use.
            If None, all will be used.

    Returns:
        tau_arr (np.ndarray): The exponential time constant in time units.
            Units are determined by the units of `tis`.

    """
    raise NotImplementedError


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
    from scipy.optimize import OptimizeWarning


    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            warnings.simplefilter('error', OptimizeWarning)
            p_val, p_cov = sp.optimize.curve_fit(*args)
    except (RuntimeError, ValueError, OptimizeWarning):
        err_val = np.nan
        # number of fitting parameters
        num_params = len(args[3])
        p_val = np.ones(num_params) * err_val
        p_cov = np.ones((num_params, num_params)) * err_val
    return p_val, p_cov


# ======================================================================
def voxel_curve_fit(
        y_arr,
        x_arr,
        fit_func=None,
        fit_params=None,
        pre_func=None,
        pre_args=None,
        pre_kwargs=None,
        post_func=None,
        post_args=None,
        post_kwargs=None,
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
        fit_params (int|np.ndarray|iterable): The parameters to fit.
            If int, specify the number of parameters to fit.
            If np.ndarray, must have all dims except the last same as `y_arr`;
            the last dim dictate the number of fit parameters; must have the
            same shape as the output.
            If iterable, specify the initial value(s) of the parameters to fit.
            If the method requires initial values, but only the number of
            parameters to fit is specified, initial value(s) set to one.
        pre_func (func):
        pre_args (list):
        pre_kwargs (dict):
        post_func (func):
        post_args (list):
        post_kwargs (dict):
        method (str): Method to use for the curve fitting procedure.
        method_kws (dict): Keyword parameters passed to the fitting procedure.

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
        if pre_kwargs is None:
            pre_kwargs = {}
        y_arr = pre_func(y_arr, *pre_args, **pre_kwargs)

    if not method:
        if fit_func:
            method = 'curve_fit_parallel_map'
        elif fit_params:
            method = 'poly'

    if not method_kws:
        method_kws = {}

    if method == 'curve_fit_sequential':
        for i in range(num_voxels):
            y_i_arr = y_arr[i] if len(shape) > 1 else y_arr
            p_i_arr = p_arr[i]
            tmp = sp.optimize.curve_fit(
                fit_func, x_arr, y_i_arr, p_i_arr, **method_kws)
            p_arr[i] = tmp[0]

    elif method == 'curve_fit_parallel':
        if 'num_proc' in method_kws:
            num_proc = method_kws['num_proc']
            method_kws.pop('num_proc')
        else:
            num_proc = multiprocessing.cpu_count()
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
            if not any(np.isnan(p_val)):
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
        except Exception as ex:
            warnings.warn(
                'W: Exception `{ex}` in ndarray_fit() method'
                ' `method}`'.format_map(locals()))

    # revert to original shape
    p_arr = p_arr.reshape((shape[:axis]) + (num_params,))

    # post process
    if post_func is not None:
        if post_args is None:
            post_args = []
        if post_kwargs is None:
            post_kwargs = {}
        p_arr = post_func(p_arr, *post_args, **post_kwargs)

    return p_arr


# ======================================================================
if __name__ == '__main__':
    import doctest

    msg(__doc__.strip())
    doctest.testmod()

else:
    elapsed()
