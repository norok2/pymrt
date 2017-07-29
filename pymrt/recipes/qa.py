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
def signal_noise_spectral(arr):
    raise NotImplementedError
    snr_val, signal_peak, noise_mean, noise_std = 0, 0, 0, 0
    return snr_val, signal_peak, noise_mean, noise_std


# ======================================================================
def signal_noise_test_retest(test_arr, retest_arr):
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
def signal_noise_multi_acq(arrs, remove_bias=True):
    """
    Separate signal from noise using multiple test-retest data.

    Assumes the measured signal is the same, but the noise, while different,
    has zero (or constant) mean.

    Args:
        arrs (iterable[np.ndarray]): The input test array.
        remove_bias (bool): Re

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
    if remove_bias:
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
def signal_noise_threshold(
        arr,
        signal_threshold='inv_hist_peaks',
        noise_threshold='inv_hist_peaks',
        signal_kws=None,
        noise_kws=None):
    """
    Separate signal from noise using value thresholds.

    Assumes that a significant portion of the data contains no signal.

    Args:
        arr (np.ndarray): The input array.
        signal_threshold (int|float|str|None): The noise threshold.
            If str, the threshold is estimated using
            `segmentation.auto_thresholds()` with its `mode` parameter set to
            `noise_threshold`.
        noise_threshold (int|float|str|None): The noise threshold.
            If str, the threshold is estimated using
            `segmentation.auto_thresholds()` with its `mode` parameter set to
            `noise_threshold`.
            If None, `noise_threshold` is set to `signal_threshold`.
        signal_kws (dict|None): Keyword parameters for `auto_thresholds()`.
        noise_kws (dict|None): Keyword parameters for `auto_thresholds()`.

    Returns:
        result (tuple[np.ndarray]): The tuple
            contains:
                - signal_arr: The signal array.
                - noise_arr: The noise array.

    See Also:
        auto_thresholds
    """
    # todo: make it more useful
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

    signal_mask = arr > np.percentile(arr, signal_threshold)
    noise_mask = arr < np.percentile(arr, noise_threshold)
    signal_arr = arr[signal_mask]
    noise_arr = arr[~noise_mask]
    return signal_arr, noise_arr


# ======================================================================
def signal_noise_thresholds(
        arr,
        signal_threshold='inv_hist_peaks',
        noise_threshold='inv_hist_peaks',
        signal_kws=None,
        noise_kws=None):
    """
    Separate signal from noise using value thresholds.

    Assumes that a significant portion of the data contains no signal.

    Args:
        arr (np.ndarray): The input array.
        signal_threshold (int|float|str|None): The noise threshold.
            If str, the threshold is estimated using
            `segmentation.auto_thresholds()` with its `mode` parameter set to
            `noise_threshold`.
        noise_threshold (int|float|str|None): The noise threshold.
            If str, the threshold is estimated using
            `segmentation.auto_thresholds()` with its `mode` parameter set to
            `noise_threshold`.
            If None, `noise_threshold` is set to `signal_threshold`.
        signal_kws (dict|None): Keyword parameters for `auto_thresholds()`.
        noise_kws (dict|None): Keyword parameters for `auto_thresholds()`.

    Returns:
        result (tuple[np.ndarray]): The tuple
            contains:
                - signal_arr: The signal array.
                - noise_arr: The noise array.

    See Also:
        auto_thresholds
    """
    # todo: make it more useful
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

    signal_mask = arr > np.percentile(arr, signal_threshold)
    noise_mask = arr < np.percentile(arr, noise_threshold)
    signal_arr = arr[signal_mask]
    noise_arr = arr[~noise_mask]
    return signal_arr, noise_arr


# ======================================================================
def _psnr(signal_arr, noise_arr):
    """
    Define the peak Signal-to-Noise Ratio (pSNR).

    .. math::
        \\mathrm{pSNR}\\equiv\\frac{\\max(s)}{P_n}

    where :math:`s` is the signal, :math:`n` is the noise and
    :math:`P` denotes the power.

    Args:
        signal_arr (np.ndarray): The array containing the signal.
        noise_arr (np.ndarray): The array containing the noise.

    Returns:
        _snr (float): The peak Signal-to-Noise Ratio (pSNR).
    """
    return np.max(np.abs(signal_arr)) / np.std(noise_arr)


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
def _cnr(
        signal1_arr,
        signal2_arr,
        noise_arr):
    """
    Define the Contrast-to-Noise Ratio (CNR).

    .. math::
        \\mathrm{CNR}\\equiv\\frac{|P_{s,1} - P_{s,2}|}{P_n}

    where :math:`s` is the signal, :math:`n` is the noise,
    :math:`P` denotes the power and :math:`1,2` refer to the signal sources.

    Args:
        signal1_arr (np.ndarray): The array containing the first signal.
        signal2_arr (np.ndarray): The array containing the second signal.
        noise_arr (np.ndarray): The array containing the noise.

    Returns:
        _snr (float): The Signal-to-Noise Ratio (SNR).
    """
    return (
        np.abs(np.mean(np.abs(signal1_arr)) - np.mean(np.abs(signal2_arr))) /
        np.std(noise_arr))


def _signal_noise_prepare(method):
    if isinstance(method, str):
        method = method.lower()
        if method == 'otsu':
            method = signal_noise_otsu
        elif method == 'test_retest':
            method = signal_noise_test_retest
        elif method == 'multi_acq':
            method = signal_noise_multi_acq
        elif method == 'threshold':
            method = signal_noise_thresholds
        else:
            raise ValueError('Invalid `{}` method.'.format(method))
    return method


# ======================================================================
def psnr(
        method,
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
        method (callable): The estimation method.
            The callable must return two arrays:
             - the signal array
             - the noise array
        *args: Positional arguments passed to `method()`.
        **kwargs: Keyword arguments passed to `method()`.

    Returns:
        result (float): The peak Signal-to-Noise Ratio (pSNR).
    """
    method, method_kws = _signal_noise_prepare(method)
    signal_arr, noise_arr = method(*args, **kwargs)
    return _psnr(signal_arr, noise_arr)


# ======================================================================
def snr(
        method,
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
        method (callable): The estimation method.
            The callable must return two arrays:
             - the signal array
             - the noise array
        *args: Positional arguments passed to `method()`.
        **kwargs: Keyword arguments passed to `method()`.

    Returns:
        result (float): The Signal-to-Noise Ratio (SNR).
    """
    method, method_kws = _signal_noise_prepare(method)
    signal_arr, noise_arr = method(*args, **kwargs)
    return _snr(signal_arr, noise_arr)


# ======================================================================
def cnr(
        method,
        *args,
        **kwargs):
    """
    Calculate the Contrast-to-Noise Ratio (CNR).

    .. math::
        \\mathrm{SNR}\\equiv\\frac{P_s}{P_n}

    where :math:`s` is the signal, :math:`n` is the noise and
    :math:`P` denotes the power.

    The approximation is based on the specified estimation method.

    Args:
        method (callable): The estimation method.
            The callable must return two arrays:
             - the signal array
             - the noise array
        *args: Positional arguments passed to `method()`.
        **kwargs: Keyword arguments passed to `method()`.

    Returns:
        result (float): The Signal-to-Noise Ratio (SNR).
    """
    method, method_kws = _signal_noise_prepare(method)
    signal_arr, noise_arr = method(*args, **kwargs)
    return _snr(signal_arr, noise_arr)


# ======================================================================
if __name__ == '__main__':
    import doctest

    msg(__doc__.strip())
    doctest.testmod()

else:
    elapsed()
