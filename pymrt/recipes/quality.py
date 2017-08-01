#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
pymrt.recipes._snr: quality assurance (QA) computations.
"""

# ======================================================================
# :: Future Imports
from __future__ import (
    division, absolute_import, print_function, unicode_literals)

# ======================================================================
# :: Python Standard Library Imports
import itertools  # Functions creating iterators for efficient looping
import collections  # Container datatypes
import warnings  # Warning control
import multiprocessing  # Process-based parallelism

# :: External Imports
import numpy as np  # NumPy (multidimensional numerical arrays library)
import scipy as sp  # SciPy (signal and image processing library)

# import scipy.integrate  # SciPy: Integration and ODEs
# import scipy.optimize  # SciPy: Optimization and root finding
# import scipy.signal  # SciPy: Signal Processing
import scipy.ndimage  # SciPy: ND-image Manipulation
import scipy.special  # SciPy: Special functions

# :: Local Imports
import pymrt as mrt
import pymrt.utils
import pymrt.segmentation

from pymrt import INFO, DIRS
# from pymrt import VERB_LVL, D_VERB_LVL, VERB_LVL_NAMES
from pymrt import elapsed, print_elapsed
from pymrt import msg, dbg


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
    return np.ptp(np.stack([signal_arr, noise_arr])) / np.std(noise_arr)


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
def signal_noise_test_retest(
        test_arr,
        retest_arr):
    """
    Separate signal from noise using test-retest data.

    Test-retest data refer to two instances of the same acquisition.
    Assumes the measured signal is the same, but the noise, while different,
    has zero mean.

    test = signal + noise_test
    retest = signal + noise_retest

    signal = (test + retest) / 2
    noise = (test - retest) / 2

    Args:
        test_arr (np.ndarray): The input test array.
        retest_arr (np.ndarray): The input retest array.

    Returns:
        result (tuple[np.ndarray]): The tuple
            contains:
                - signal_arr: The signal array.
                - noise_arr: The noise array.
    """
    signal_arr = (test_arr + retest_arr) / 2
    noise_arr = (test_arr - retest_arr) / 2
    return signal_arr, noise_arr


# ======================================================================
def signal_noise_multi_acq(
        arrs,
        remove_bias=True):
    """
    Separate signal from noise using multiple test-retest data.

    Assumes the measured signal is the same, but the noise, while different,
    has zero (or constant) mean.

    Args:
        arrs (iterable[np.ndarray]): The input test array.
        remove_bias (bool): Remove bias in the signal from the noise mean.

    Returns:
        result (tuple[np.ndarray]): The tuple
            contains:
                - signal_arr: The signal array.
                - noise_arr: The noise array.
    """
    axis = -1
    num = len(arrs)
    arr = np.stack(arrs, axis)
    signal_arr = np.mean(arr, axis)
    noise_arr = np.std(arr, axis)
    if num > 2 and remove_bias:
        bias_arr = np.zeros_like(signal_arr)
        for i, j in itertools.combinations(range(num), 2):
            bias_arr += arrs[i] - arrs[j]
        bias_arr /= sp.special.binom(num, 2)
        signal_arr -= bias_arr
    return signal_arr, noise_arr


# ======================================================================
def signal_noise_otsu(arr):
    """
    Separate signal from noise using the Otsu threshold.

    Args:
        arr (np.ndarray): The input array.

    Returns:
        result (tuple[np.ndarray]): The tuple
            contains:
                - signal_arr: The signal array.
                - noise_arr: The noise array.
    """
    signal_mask = arr > mrt.segmentation.threshold_otsu(arr)
    signal_arr = arr[signal_mask]
    noise_arr = arr[~signal_mask]
    return signal_arr, noise_arr


# ======================================================================
def signal_noise_relative(
        arr,
        thresholds=(0.75, 0.25)):
    """
    Separate signal from noise using the relative threshold(s).

    Args:
        arr (np.ndarray): The input array.
        thresholds (int|float|iterable[int|float]: The percentile values.
            Values must be in the [0, 1] range.
            If int or float, values above are considered signal,
            and below or equal ar considered noise.
            If iterable, values above the first percentile threshold are
            considered signals, while values below the second percentile
            threshold are considered noise.
            Other values are ignored.

    Returns:
        result (tuple[np.ndarray]): The tuple
            contains:
                - signal_arr: The signal array.
                - noise_arr: The noise array.

    See Also:
        segmentation.threshold_relative(),
    """
    thresholds = mrt.utils.auto_repeat(thresholds, 2, check=True)
    signal_threshold, noise_threshold = \
        mrt.segmentation.threshold_relative(arr, thresholds)
    signal_mask = arr > signal_threshold
    noise_mask = arr <= noise_threshold
    signal_arr = arr[signal_mask]
    noise_arr = arr[noise_mask]
    return signal_arr, noise_arr


# ======================================================================
def signal_noise_percentile(
        arr,
        thresholds=(0.75, 0.25)):
    """
    Separate signal from noise using the percentile threshold(s).

    Args:
        arr (np.ndarray): The input array.
        thresholds (int|float|iterable[int|float]: The percentile values.
            Values must be in the [0, 1] range.
            If int or float, values above are considered signal,
            and below or equal ar considered noise.
            If iterable, values above the first percentile threshold are
            considered signals, while values below the second percentile
            threshold are considered noise.
            At most two values are accepted.

    Returns:
        result (tuple[np.ndarray]): The tuple
            contains:
                - signal_arr: The signal array.
                - noise_arr: The noise array.

    See Also:
        segmentation.threshold_percentile()
    """
    thresholds = mrt.utils.auto_repeat(thresholds, 2, check=True)
    signal_threshold, noise_threshold = \
        mrt.segmentation.threshold_percentile(arr, thresholds)
    signal_mask = arr > signal_threshold
    noise_mask = arr <= noise_threshold
    signal_arr = arr[signal_mask]
    noise_arr = arr[noise_mask]
    return signal_arr, noise_arr


# ======================================================================
def signal_noise_mean_std(
        arr,
        mean_factor=1,
        std_factor=2,
        symmetric=False):
    """
    Separate signal from noise using a threshold combining mean and std.dev.

    Thresholds are calculated as:
    :math:`K_{+,-} = k_\\mu * \\mu \\pm k_\\sigma * \\sigma`

    Signal/noise values interval depend on the `symmetric` parameter.

    Args:
        arr (np.ndarray): The input array.
        mean_factor (int|float): The mean multiplication factor.
            This is usually set to 1.
        std_factor (int|float): The standard deviation multiplication factor.
            This is usually set to a number between 1 and 2.
        symmetric (bool): Use symmetric thresholds.
            If symmetric, signal values are inside the [K-,K+] range.
            Otherwise, signal values are above K-.

    Returns:
        result (tuple[np.ndarray]): The tuple
            contains:
                - signal_arr: The signal array.
                - noise_arr: The noise array.
    """
    signal_mask = arr > np.mean(arr) * mean_factor - np.std(arr) * std_factor
    if symmetric:
        signal_mask *= (
            arr < np.mean(arr) * mean_factor + np.std(arr) * std_factor)
    signal_arr = arr[signal_mask]
    noise_arr = arr[~signal_mask]
    return signal_arr, noise_arr


# ======================================================================
def signal_noise_thresholds(
        arr,
        signal_threshold='otsu',
        noise_threshold=None,
        signal_kws=None,
        noise_kws=None,
        signal_index=None,
        noise_index=None):
    """
    Separate signal from noise using value thresholds.

    Assumes that a significant portion of the data contains no signal.

    Args:
        arr (np.ndarray): The input array.
        signal_threshold (int|float|str|None): The noise threshold.
            If str, the threshold is estimated using
            `segmentation.auto_thresholds()` with its `method` parameter set
            to `signal_threshold`.
        noise_threshold (int|float|str|None): The noise threshold.
            If str, the threshold is estimated using
            `segmentation.auto_thresholds()` with its `method` parameter set
            to `noise_threshold`.
            If None, `noise_threshold` is set to `signal_threshold`.
        signal_kws (dict|None): Keyword parameters.
            If `signal_threshold` is str, the parameters are passed to
            `segmentation.auto_thresholds()` for `signal_threshold`.
        noise_kws (dict|None): Keyword parameters.
            If `noisel_threshold` is str, the parameters are passed to
            `segmentation.auto_thresholds()` for `noise_threshold`.
        signal_index (int|None): Select a specific threshold.
            The index is applied to the iterable obtained from
            `segmentation.auto_thresholds()` for `signal_threshold`.
            If None, the first value is selected.
        noise_index (int|None): Select a specific threshold.
            The index is applied to the iterable obtained from
            `segmentation.auto_thresholds()` for `noise_threshold`.
            If None, the first value is selected.

    Returns:
        result (tuple[np.ndarray]): The tuple
            contains:
                - signal_arr: The signal array.
                - noise_arr: The noise array.

    See Also:
        segmentation.auto_thresholds()
    """
    if isinstance(signal_threshold, str):
        if signal_kws is None:
            signal_kws = dict()
        signal_threshold = mrt.segmentation.auto_thresholds(
            arr, signal_threshold, signal_kws)

    if noise_threshold is None:
        noise_threshold = signal_threshold
    elif isinstance(noise_threshold, str):
        if noise_kws is None:
            noise_kws = dict()
        noise_threshold = mrt.segmentation.auto_thresholds(
            arr, noise_threshold, noise_kws)

    if signal_index is None:
        signal_index = 0
    if noise_index is None:
        noise_index = 0

    signal_mask = arr > signal_threshold[signal_index]
    noise_mask = arr <= noise_threshold[noise_index]
    signal_arr = arr[signal_mask]
    noise_arr = arr[noise_mask]
    return signal_arr, noise_arr


# ======================================================================
def signal_noise_denoise(arr, smoothing=2):
    """
    Separate signal from noise using denoisingof the data.

    Args:
        arr (np.ndarray): The input array.
        smoothing (int|float|iterable[int|float]): Smoothing factor.
            Size of the box for the uniform filter.
            If int or float, the box size is the same in all dims.
            If iterable, each value correspond to a dimension of arr and its
            size must match the number of dims of arr.

    Returns:
        result (tuple[np.ndarray]): The tuple
            contains:
                - signal_arr: The signal array.
                - noise_arr: The noise array.
    """
    signal_arr = sp.ndimage.uniform_filter(arr, smoothing)
    noise_arr = arr - signal_arr
    return signal_arr, noise_arr


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
        signal_arr, noise_arr = signal_noise_test_retest(*arrs)
    else:
        signal_arr, noise_arr = signal_noise_multi_acq(
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
        signal_arr, noise_arr = signal_noise_test_retest(*arrs)
    else:
        signal_arr, noise_arr = signal_noise_multi_acq(
            arrs, remove_bias=remove_bias)
    return _psnr(signal_arr, noise_arr)


# ======================================================================
def signal_noise(arr, method, *args, **kwargs):
    """
    Separate signal from noise.

    Args:
        arr (np.ndarray): The input array.
        method (str): The signal/noise estimation method.
            If str, available methods are (recommended: 'otsu'):
             - 'otsu': Uses `signal_noise_otsu()`.
             - 'relative': Uses `signal_noise_relative()`.
             - 'percentile': Uses `signal_noise_percentile()`.
             - 'mean_std': Uses `signal_noise_mean_std()`.
             - 'thresholds': Uses `signal_noise_thresholds()`.
             - 'denoise': Uses `signal_noise_denoise()`.
            If callable, the signature must be:
            f(np.ndarray, *args, **kwargs) -> (np.ndarray, np.ndarray)
            where the input array is `arr` and the two returned arrays are:
             - the signal array
             - the noise array
        *args: Positional arguments passed to `method()`.
        **kwargs: Keyword arguments passed to `method()`.

    Returns:
        method (callable): The selected signal/noise estimation method.

    Raises:
        ValueError: If `method` is unknown.
    """
    methods = (
        'otsu', 'relative', 'percentile', 'mean_std', 'thresholds',
        'denoise')
    method = method.lower()
    if method == 'otsu':
        method = signal_noise_otsu
    elif method == 'relative':
        method = signal_noise_relative
    elif method == 'percentile':
        method = signal_noise_percentile
    elif method == 'mean_std':
        method = signal_noise_mean_std
    elif method == 'thresholds':
        method = signal_noise_thresholds
    elif method == 'denoise':
        method = signal_noise_denoise
    else:
        raise ValueError(
            'valid methods are: {} (given: {})'.format(methods, method))
    return method(arr, *args, **kwargs)


# ======================================================================
def snr(
        arr,
        method='otsu',
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
    signal_arr, noise_arr = signal_noise(arr, method, *args, **kwargs)
    return _snr(signal_arr, noise_arr)


# ======================================================================
def psnr(
        arr,
        method='otsu',
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
    signal_arr, noise_arr = signal_noise(arr, method, *args, **kwargs)
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
    signal_arr, noise_arr = signal_noise(arr, sn_method, **sn_kws)
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
    signal_arr, noise_arr = signal_noise(arr, method, *args, **kwargs)
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
