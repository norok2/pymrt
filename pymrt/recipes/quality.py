#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pymrt.recipes.quality: quality assurance (QA) computations.
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

# :: Local Imports
import pymrt as mrt
import pymrt.segmentation
import pymrt.utils
from pymrt import correction
# from pymrt import VERB_LVL, D_VERB_LVL, VERB_LVL_NAMES
from pymrt import elapsed
from pymrt import msg


# import scipy.integrate  # SciPy: Integration and ODEs
# import scipy.optimize  # SciPy: Optimization and root finding
# import scipy.signal  # SciPy: Signal Processing


# ======================================================================
def _snr(
        signal_arr,
        noise_arr):
    """
    Define the Signal-to-Noise Ratio (SNR).

    .. math::
        \\mathrm{SNR}\\equiv\\frac{P_s}{P_n}

    where :math:`s` is the signal, :math:`n` is the noise and
    :math:`P` denotes the power.

    Args:
        signal_arr (np.ndarray): The array containing the signal.
        noise_arr (np.ndarray): The array containing the noise.

    Returns:
        _snr (float): The Signal-to-Noise Ratio (SNR).
    """
    return np.mean(np.abs(signal_arr)) / np.std(noise_arr)


# ======================================================================
def _psnr(
        signal_arr,
        noise_arr):
    """
    Define the peak Signal-to-Noise Ratio (pSNR).

    .. math::
        \\mathrm{pSNR}\\equiv\\frac{\\max(s + n) - \\min(s + n)}{P_n}

    where :math:`s` is the signal, :math:`n` is the noise and
    :math:`P` denotes the power.

    Args:
        signal_arr (np.ndarray): The array containing the signal.
        noise_arr (np.ndarray): The array containing the noise.

    Returns:
        _snr (float): The peak Signal-to-Noise Ratio (pSNR).
    """
    return np.ptp(np.concatenate([signal_arr, noise_arr])) / np.std(noise_arr)


# ======================================================================
def _cnr(
        contrast_val,
        noise_arr):
    """
    Define the Contrast-to-Noise Ratio (CNR).

    .. math::
        \\mathrm{CNR}\\equiv\\frac{C}{P_n}

    where :math:`C` is the contrast value, :math:`n` is the noise,
    :math:`P` denotes the power.

    Args:
        contrast_val (float): The contrast value.
        noise_arr (np.ndarray): The array containing the noise.

    Returns:
        _snr (float): The Signal-to-Noise Ratio (SNR).
    """
    return contrast_val / np.std(noise_arr)


# ======================================================================
def _pcnr(
        signal_arr,
        noise_arr):
    """
    Define the peak Contrast-to-Noise Ratio (pCNR).

    .. math::
        \\mathrm{CNR}\\equiv\\frac{\\max(s) - \\min(s)}{P_n}

    where :math:`s` is the signal, :math:`n` is the noise,
    :math:`P` denotes the power.

    Args:
        contrast_val (float): The contrast value.
        noise_arr (np.ndarray): The array containing the noise.

    Returns:
        _snr (float): The Signal-to-Noise Ratio (SNR).
    """
    return np.ptp(signal_arr) / np.std(noise_arr)


# ======================================================================
def snr_multi_acq(
        arrs,
        remove_bias=True):
    """
    Calculate the

    Args:
        arrs (iterable[np.ndarray]): The input arrays.
        remove_bias (bool): Remove bias in the signal from the noise mean.

    Returns:
        result (float): The Signal-to-Noise Ratio (SNR).

    Examples:
        >>> np.random.seed(0)
        >>> k_signal, k_noise = 10, 3
        >>> signal_arr = np.array((0, 0, 1, 1, 1, 1, 0, 0)) * k_signal
        >>> noise1_arr = k_noise * np.random.random(len(signal_arr))
        >>> noise2_arr = k_noise * np.random.random(len(signal_arr))
        >>> test_arr = signal_arr + noise1_arr
        >>> retest_arr = signal_arr + noise2_arr
        >>> val = snr_multi_acq([test_arr, retest_arr])
        >>> round(val)
        12.0
        >>> n_acq = 100
        >>> arrs = [
        ...     signal_arr + k_noise * np.random.random(len(signal_arr))
        ...     for _ in range(n_acq)]
        >>> val = snr_multi_acq(arrs)
        >>> round(val)
        153.0
    """
    if len(arrs) == 2:
        signal_arr, noise_arr = correction.test_retest(*arrs)
    else:
        signal_arr, noise_arr = correction.multi_acq(
            arrs, remove_bias=remove_bias)
    return _snr(signal_arr, noise_arr)


# ======================================================================
def psnr_multi_acq(
        arrs,
        remove_bias=True):
    """
    Calculate the

    Args:
        arrs (iterable[np.ndarray]): The input test array.
        remove_bias (bool): Remove bias in the signal from the noise mean.

    Returns:
        result (float): The Signal-to-Noise Ratio (SNR).

    Examples:
        >>> np.random.seed(0)
        >>> k_signal, k_noise = 10, 3
        >>> signal_arr = np.array((0, 0, 1, 1, 1, 1, 0, 0)) * k_signal
        >>> noise1_arr = k_noise * np.random.random(len(signal_arr))
        >>> noise2_arr = k_noise * np.random.random(len(signal_arr))
        >>> test_arr = signal_arr + noise1_arr
        >>> retest_arr = signal_arr + noise2_arr
        >>> val = psnr_multi_acq([test_arr, retest_arr])
        >>> round(val)
        23.0
        >>> n_acq = 100
        >>> arrs = [
        ...     signal_arr + k_noise * np.random.random(len(signal_arr))
        ...     for _ in range(n_acq)]
        >>> val = psnr_multi_acq(arrs)
        >>> round(val)
        255.0
    """
    if len(arrs) == 2:
        signal_arr, noise_arr = correction.test_retest(*arrs)
    else:
        signal_arr, noise_arr = correction.multi_acq(
            arrs, remove_bias=remove_bias)
    return _psnr(signal_arr, noise_arr)


# ======================================================================
def snr(
        arr,
        method='auto',
        *args,
        **kwargs):
    """
    Calculate the Signal-to-Noise Ratio (SNR).

    .. math::
        \\mathrm{SNR}\\equiv\\frac{P_s}{P_n}

    where :math:`s` is the signal, :math:`n` is the noise and
    :math:`P` denotes the power.

    The approximation is based on the specified estimation method.

    Args:
        arr (np.ndarray): The input array.
        method (str|callable): The signal/noise estimation method.
            See `signal_noise()` for more details.
        *args: Positional arguments passed to `method()`.
        **kwargs: Keyword arguments passed to `method()`.

    Returns:
        result (float): The Signal-to-Noise Ratio (SNR).

    Examples:
        >>> arr = np.array((1, 2, 100, 100, 100, 101, 1, 2))
        >>> val = snr(arr)
        >>> round(val)
        200.0
        >>> arr = np.array((1, 2, 10, 10, 10, 10, 1, 2))
        >>> val = snr(arr, 'otsu')
        >>> round(val)
        20.0
        >>> arr = np.array((1, 2, 10, 10, 10, 10, 1, 2))
        >>> val = snr(arr, 'relative', (0.75, 0.25))
        >>> round(val)
        20.0
        >>> arr = np.array((1, 2, 10, 10, 10, 10, 1, 2))
        >>> val = snr(arr, 'percentile', 0.5)
        >>> round(val)
        20.0
        >>> arr = np.array((1, 2, 10, 10, 10, 10, 1, 2))
        >>> val = snr(arr, 'thresholds', 'inv_hist_peaks')
        >>> round(val)
        20.0
    """
    signal_arr, noise_arr = correction(arr, method, *args, **kwargs)
    return _snr(signal_arr, noise_arr)


# ======================================================================
def psnr(
        arr,
        method='auto',
        *args,
        **kwargs):
    """
    Calculate the peak Signal-to-Noise Ratio (pSNR).

    .. math::
        \\mathrm{pSNR}\\equiv\\frac{\\max(s)}{P_n}

    where :math:`s` is the signal, :math:`n` is the noise and
    :math:`P` denotes the power.

    The approximation is based on the specified method.

    Args:
        arr (np.ndarray): The input array.
        method (str|callable): The signal/noise estimation method.
            See `signal_noise()` for more details.
        *args: Positional arguments passed to `method()`.
        **kwargs: Keyword arguments passed to `method()`.

    Returns:
        result (float): The peak Signal-to-Noise Ratio (pSNR).

    Examples:
        >>> arr = np.array((1, 2, 100, 100, 100, 101, 1, 2))
        >>> val = psnr(arr)
        >>> round(val)
        200.0
        >>> arr = np.array((1, 2, 10, 10, 10, 10, 1, 2))
        >>> val = psnr(arr, 'otsu')
        >>> round(val)
        18.0
        >>> arr = np.array((1, 2, 10, 10, 10, 10, 1, 2))
        >>> val = psnr(arr, 'relative', (0.75, 0.25))
        >>> round(val)
        18.0
        >>> arr = np.array((1, 2, 10, 10, 10, 10, 1, 2))
        >>> val = psnr(arr, 'percentile', 0.5)
        >>> round(val)
        18.0
        >>> arr = np.array((1, 2, 10, 10, 10, 10, 1, 2))
        >>> val = psnr(arr, 'thresholds', 'inv_hist_peaks')
        >>> round(val)
        18.0
    """
    signal_arr, noise_arr = correction(arr, method, *args, **kwargs)
    return _psnr(signal_arr, noise_arr)


# ======================================================================
def separate_signals(
        arr,
        method='otsu',
        *args,
        **kwargs):
    """
    Separate N signal components according to threshold(s).

    Args:
        arr (np.ndarray): The input array.
        method (iterable[float]|str|callable): The separation method.
            If iterable[float], the specified thresholds value are used.
            If str, the thresholds are estimated using
            `segmentation.auto_thresholds()` with its `method` parameter set
            to `method`.
            Additional accepted values:
             - 'mean': use the mean value of the signal.
             - 'midval': use the middle of the values range.
             - 'median': use the median value of the signal.
             - 'otsu': use the Otsu threshold.
            If callable, the signature must be:
            f(np.ndarray, *args, **kwargs) -> iterable[float]
        *args: Positional arguments passed to `method()`.
        **kwargs: Keyword arguments passed to `method()`.

    Returns:
        result (tuple[np.ndarray]): The tuple
            contains:
                - signal1_arr: The first signal component array.
                - signal2_arr: The second signal component array.

    Examples:
        >>> arr = np.array((0, 0, 1, 1, 1, 1, 0, 0))
        >>> separate_signals(arr)
        (array([0, 0, 0, 0]), array([1, 1, 1, 1]))
        >>> arr = np.arange(10)
        >>> separate_signals(arr, method=(2, 6))
        (array([0, 1]), array([2, 3, 4, 5]), array([6, 7, 8, 9]))
    """
    if isinstance(method, str):
        if method == 'mean':
            thresholds = np.mean(arr)
        elif method == 'midval':
            thresholds = mrt.segmentation.threshold_relative(arr, 0.5)
        elif method == 'median':
            thresholds = mrt.segmentation.threshold_percentile(arr, 0.5)
        else:
            thresholds = mrt.segmentation.auto_thresholds(
                arr, method, dict(kwargs))
    elif callable(method):
        thresholds = method(arr, *args, **kwargs)
    else:
        thresholds = tuple(method)

    thresholds = mrt.utils.auto_repeat(thresholds, 1)

    masks = []
    full_mask = np.ones(arr.shape, dtype=bool)
    last_threshold = np.min(arr)
    for i, threshold in enumerate(sorted(thresholds)):
        mask = arr < threshold
        mask *= arr >= last_threshold
        masks.append(mask)
        full_mask -= mask
        last_threshold = threshold

    return tuple(arr[mask] for mask in masks) + (arr[full_mask],)


# ======================================================================
def contrasts(arrs):
    """
    Calculate pair-wise contrasts from arrays:

    .. math::
        C_{ij} = \\frac{|P_i - P_j|}{P_i + P_j}

    where :math:`C` is the contrast value, :math:`P` is the signal power,
    and :math:`i,j` indexes run through all possible arrays combinations.

    Args:
        arrs (iterable[np.ndarray]): The input arrays.

    Returns:
        results (tuple[float]): The pair-wise contrast values.
    """
    return tuple(
        (np.abs(np.mean(np.abs(arr1)) - np.mean(np.abs(arr2))) /
         (np.mean(np.abs(arr1)) + np.mean(np.abs(arr2))))
        for arr1, arr2 in itertools.combinations(arrs, 2))


# ======================================================================
def combine_contrasts(
        contrast_values,
        method=np.mean,
        *args,
        **kwargs):
    """
    Calculate the contrast by combining the pair-wise contrast values.

    Args:
        contrast_values (iterable[float]): The pair-wise contrast values.
        method (callable): The pair-wise contrasts combination method.
            The signature must be:
            f(iterable, *args, **kwargs) -> float
        *args: Positional arguments passed to `method()`.
        **kwargs: Keyword arguments passed to `method()`.

    Returns:
        result (float): The contrast.
    """
    return method(contrast_values, *args, **kwargs)


# ======================================================================
def contrast(
        arrs,
        method=np.mean,
        *args,
        **kwargs):
    """
    Calculate the contrast by combining the pair-wise contrast values.

    Args:
        arrs (iterable[np.ndarray]): The input arrays
        method (callable): The pair-wise contrasts combination method.
            The signature must be:
            f(iterable, *args, **kwargs) -> float
        *args: Positional arguments passed to `method()`.
        **kwargs: Keyword arguments passed to `method()`.

    Returns:
        result (float): The contrast.
    """
    return method(contrasts(arrs), *args, **kwargs)


# ======================================================================
def cnr(
        arr,
        ss_method='otsu',
        sn_method='otsu',
        ss_kws=None,
        sn_kws=None):
    """
    Calculate the Contrast-to-Noise Ratio (CNR).

    .. math::
        \\mathrm{CNR}\\equiv\\frac{C}{P_n}

    where :math:`C` is the contrast value, :math:`n` is the noise,
    :math:`P` denotes the power.

    The approximation is based on the specified estimation methods.

    Firstly, the noise is separate from the signal(s).
    Then, different signals are separated and their combined pair-wise
    contrast is calculated.

    Args:
        arr (np.ndarray): The input array.
        ss_method (float|str|callable): Signal sources separation.
            See `separate_signals()` for more details.
        sn_method (str|callable): The signal/noise estimation method.
            See `signal_noise()` for more details.
        ss_kws: Keyword arguments passed to `ss_method()`.
        sn_kws: Keyword arguments passed to `sn_method()`.

    Returns:
        result (float): The Contrast-to-Noise Ratio (CNR).

    Examples:
        >>> arr = np.array((1, 2, 100, 100, 100, 101, 1, 2))
        >>> val = cnr(arr)
        >>> round(val * 100)
        1.0
        >>> arr = np.array((1, 2, 1000, 3000, 1000, 3000, 1, 2))
        >>> val = cnr(arr)
        >>> round(val * 100)
        100.0
        >>> arr = np.array((1, 2, 2000, 3000, 1800, 3000, 1, 2))
        >>> val = cnr(arr)
        >>> round(val * 100)
        45.0
    """
    signal_arr, noise_arr = correction(arr, sn_method, **sn_kws)
    signal_arrs = separate_signals(signal_arr, ss_method, **ss_kws)
    return _cnr(contrast(signal_arrs), noise_arr)


# ======================================================================
def pcnr(
        arr,
        method='otsu',
        *args,
        **kwargs):
    """
    Calculate the peak Contrast-to-Noise Ratio (pCNR).

    .. math::
        \\mathrm{CNR}\\equiv\\frac{\\max(s) - \\min(s)}{P_n}

    where :math:`s` is the signal, :math:`n` is the noise,
    :math:`P` denotes the power.
    The approximation is based on the specified method.

    Args:
        arr (np.ndarray): The input array.
        method (str|callable): The signal/noise estimation method.
            See `signal_noise()` for more details.
        *args: Positional arguments passed to `method()`.
        **kwargs: Keyword arguments passed to `method()`.

    Returns:
        result (float): The peak Signal-to-Noise Ratio (pSNR).

    Examples:
        >>> arr = np.array((1, 2, 100, 100, 100, 101, 1, 2))
        >>> val = pcnr(arr)
        >>> round(val)
        2.0
        >>> arr = np.array((1, 2, 11, 10, 10, 10, 1, 2))
        >>> val = pcnr(arr, 'otsu')
        >>> round(val)
        2.0
        >>> arr = np.array((1, 2, 10, 15, 10, 10, 1, 2))
        >>> val = pcnr(arr, 'relative', (0.75, 0.25))
        >>> round(val)
        0.0
        >>> arr = np.array((1, 2, 10, 10, 10, 10, 1, 2))
        >>> val = pcnr(arr, 'percentile', 0.5)
        >>> round(val)
        0.0
        >>> arr = np.array((1, 2, 9, 11, 10, 10, 1, 2))
        >>> val = pcnr(arr, 'thresholds', 'inv_hist_peaks')
        >>> round(val)
        4.0
    """
    signal_arr, noise_arr = correction(arr, method, *args, **kwargs)
    return _pcnr(signal_arr, noise_arr)


# ======================================================================
def quick_check(arr):
    """
    Calculate all quality metrics that are fast to compute.

    Args:
        arr:

    Returns:

    """
    raise NotImplementedError


# ======================================================================
if __name__ == '__main__':
    import doctest

    msg(__doc__.strip())
    doctest.testmod()

else:
    elapsed()
