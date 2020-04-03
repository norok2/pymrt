#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pymrt.correction: signal/noise/bias-related corrections.
"""

# ======================================================================
# :: Future Imports
from __future__ import (
    division, absolute_import, print_function, unicode_literals, )

# ======================================================================
# :: Python Standard Library Imports
import itertools  # Functions creating iterators for efficient looping
import warnings  # Warning control

# :: External Imports
import numpy as np  # NumPy (multidimensional numerical arrays library)
import scipy as sp  # SciPy (signal and image processing library)
import pywt as pw  # PyWavelets - Wavelet Transforms in Python
import flyingcircus as fc  # Everything you always wanted to have in Python*
import raster_geometry  # Create/manipulate N-dim raster geometric shapes.

# import scipy.integrate  # SciPy: Integration and ODEs
# import scipy.optimize  # SciPy: Optimization and root finding
# import scipy.signal  # SciPy: Signal Processing
import scipy.ndimage  # SciPy: ND-image Manipulation
import scipy.special  # SciPy: Special functions

try:
    from skimage.restoration import (
        denoise_bilateral, denoise_nl_means, denoise_wavelet,
        denoise_tv_bregman, denoise_tv_chambolle)
except ImportError:
    denoise_bilateral = denoise_nl_means = denoise_wavelet \
        = denoise_tv_bregman = denoise_tv_chambolle = None

# :: Local Imports
import pymrt as mrt  # Python Magnetic Resonance Tools: the multi-tool of MRI.
import pymrt.segmentation

from pymrt import INFO, PATH
from pymrt import VERB_LVL, D_VERB_LVL, VERB_LVL_NAMES
from pymrt import elapsed, report, run_doctests
from pymrt import msg, dbg, fmt, fmtm


# ======================================================================
def dwt_filter(
        arr,
        wavelet,
        mode='symmetric',
        axes=None,
        threshold=None,
        sigma=None,
        approximation='soft',
        levels=None):
    """

    Args:
        arr:
        wavelet:
        mode:
        axes:
        threshold:
        sigma:
        approximation:
        levels:

    Returns:

    """
    raise NotImplementedError


# ======================================================================
def denoise(
        arr,
        method='wavelet',
        method_kws=None,
        cx_mode='cartesian'):
    """
    Perform standard single-data de-noising algorithms.

    It can be applied to complex data (see `cx_mode` for the exact behavior).

    Exposes several algorithms from `scipy.ndimage.filters` and
    `skimage.restoration`.

    Args:
        arr (np.ndarray): The input array.
        method (str): Denoising method.
            Accepted values are:
             - 'gaussian': `scipy.ndimage.gaussian_filter()`
             - 'uniform': `scipy.ndimage.uniform_filter()`
             - 'median': `scipy.ndimage.median_filter()`
             - 'minimum': `scipy.ndimage.minimum_filter()`
             - 'maximum': `scipy.ndimage.maximum_filter()`
             - 'rank': `scipy.ndimage.rank_filter()`
             - 'percentile': `scipy.ndimage.percentile_filter()`
             - 'dwt': `pymrt.correction.dwt_filter()`
             - 'nl_means': `skimage.restoration.denoise_nl_means()`
             - 'wavelet': `skimage.restoration.denoise_wavelet()`
             - 'tv_bregman': `skimage.restoration.denoise_tv_bregman()`
             - 'tv_chambolle': `skimage.restoration.denoise_tv_chambolle()`
             - 'bilateral': `skimage.restoration.denoise_bilateral()`
               (only works with 2D images)
        method_kws (Mappable|None): Keyword arguments to pass to `method`.
            These are passed to the corresponding function.
            See the respective documentation for details.
        cx_mode (str): Complex calculation mode.
            If `arr` is not complex, this parameter is ignored.
            See `mode` parameter of `fc.extra.filter_cx()` for more info.

    Raises:
        ValueError: If `method` is unknown.

    Returns:
        arr (np.ndarray): The denoised array.
    """
    method = method.lower()
    method_kws = {} if method_kws is None else dict(method_kws)
    if method == 'gaussian':
        if 'sigma' not in method_kws:
            method_kws['sigma'] = 1.0
        filter_func = sp.ndimage.gaussian_filter
    elif method == 'uniform':
        filter_func = sp.ndimage.uniform_filter
    elif method == 'median':
        if 'size' not in method_kws:
            method_kws['size'] = 5
        filter_func = sp.ndimage.median_filter
    elif method == 'minimum':
        if 'size' not in method_kws:
            method_kws['size'] = 5
        filter_func = sp.ndimage.minimum_filter
    elif method == 'maximum':
        if 'size' not in method_kws:
            method_kws['size'] = 5
        filter_func = sp.ndimage.maximum_filter
    elif method == 'rank':
        if 'size' not in method_kws:
            method_kws['size'] = 5
        if 'rank' not in method_kws:
            method_kws['rank'] = 1
        filter_func = sp.ndimage.rank_filter
    elif method == 'percentile':
        if 'size' not in method_kws:
            method_kws['size'] = 5
        if 'percentile' not in method_kws:
            method_kws['percentile'] = 50
        filter_func = sp.ndimage.percentile_filter
    elif method == 'bilateral':
        method_kws['multichannel'] = False
        filter_func = denoise_bilateral
    elif method == 'nl_means':
        method_kws['multichannel'] = False
        filter_func = denoise_nl_means
    elif method == 'wavelet':
        if np.max(np.abs(arr)) > 1.0:
            text = 'Image will be clipped to unity.'
            warnings.warn(text)
        filter_func = denoise_wavelet
    elif method == 'dwt':
        filter_func = dwt_filter
    elif method == 'tv_bregman':
        if 'weight' not in method_kws:
            method_kws['weight'] = 1
        filter_func = denoise_tv_bregman
    elif method == 'tv_chambolle':
        filter_func = denoise_tv_chambolle
    else:
        text = 'Unknown method `{}`'.format(method)
        raise ValueError(text)
    if not callable(filter_func):
        warnings.warn(
            'Method `{}` available only after installing `scikit.image`' \
                .format(method))

    if np.any(np.iscomplex(arr)):
        arr = fc.extra.filter_cx(arr, filter_func, (), method_kws, cx_mode)
    else:
        arr = filter_func(np.real(arr), **method_kws)
    return arr


# ======================================================================
def denoise_multi(arrs):
    raise NotImplementedError


# ======================================================================
def sn_split_signals(
        arr,
        method='otsu',
        *_args,
        **_kws):
    """
    Separate N signal components according to threshold(s).

    Args:
        arr (np.ndarray): The input array.
        method (Iterable[float]|str|callable): The separation method.
            If Iterable[float], the specified thresholds value are used.
            If str, the thresholds are estimated using
            `pymrt.segmentation.auto_thresholds()` with its `method`
            parameter set
            to `method`.
            Additional accepted values:
             - 'mean': use the mean value of the signal.
             - 'midval': use the middle of the values range.
             - 'median': use the median value of the signal.
             - 'otsu': use the Otsu threshold.
            If callable, the signature must be:
            f(np.ndarray, *_args, **_kws) -> Iterable[float]
        *_args: Positional arguments for `method()`.
        **_kws: Keyword arguments for `method()`.

    Returns:
        result (tuple[np.ndarray]): The tuple
            contains:
             - signal1_arr: The first signal component array.
             - signal2_arr: The second signal component array.

    Examples:
        >>> arr = np.array((0, 0, 1, 1, 1, 1, 0, 0))
        >>> sn_split_signals(arr)
        (array([0, 0, 0, 0]), array([1, 1, 1, 1]))
        >>> arr = np.arange(10)
        >>> sn_split_signals(arr, method=(2, 6))
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
            thresholds = mrt.segmentation.auto_thresholds(arr, method, _kws)
    elif callable(method):
        thresholds = method(arr, *_args, **_kws)
    else:
        thresholds = tuple(method)

    thresholds = fc.auto_repeat(thresholds, 1)

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
def sn_split_test_retest(
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
def sn_split_multi_acq(
        arrs,
        remove_bias=True):
    """
    Separate signal from noise using multiple test-retest data.

    Assumes the measured signal is the same, but the noise, while different,
    has zero (or constant) mean.

    Args:
        arrs (Iterable[np.ndarray]): The input test array.
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
def sn_split_calib_region(
        arr,
        s_region=None,
        n_region=None,
        region_shape=raster_geometry.nd_cuboid,
        region_shape_kws=None):
    """
    Separate signal from noise a calibration region.

    EXPERIMENTAL!

    Use a n-dim superellipsoid or cuboid as calibration regions for signal
    and noise estimation.

    Args:
        arr (np.ndarray): The input array.
        s_region (Iterable[float|Iterable[float]]|None): The signal region.
            If Iterable, it is passed as positional arguments to
            `geometry.extrema_to_semisizes_position()`.
            If None, a suitable region is guessed.
        n_region (Iterable[float|Iterable[float]]|None): The noise region.
            If Iterable, it is passed as positional arguments to
            `geometry.extrema_to_semisizes_position()`.
            If None, a suitable region is guessed.
        region_shape (callable|str): Shape of the calibration region.
            Accepted values are:Function for calculating the region masks.
            The signature of the function must accept three positional
            arguments: shape, semisizes, position.
            The last two arguments can be the output of:
            `geometry.extrema_to_semisizes_position()`.
            Suitable callable are:
             - `geometry.nd_cuboid()`
             - `geometry.nd_superellipsoid()`

    Returns:
        result (tuple[np.ndarray]): The tuple
            contains:
             - signal_arr: The signal array.
             - noise_arr: The noise array.

    """
    # todo: fix implementation
    if not s_region:
        raise NotImplementedError
    if not n_region:
        raise NotImplementedError
    s_semisizes, s_position = raster_geometry.extrema_to_semisizes_position(
        *s_region, num=arr.ndim)
    signal_arr = arr[
        region_shape(arr.shape, s_semisizes, s_position)]
    n_semisizes, n_position = raster_geometry.extrema_to_semisizes_position(
        *n_region, num=arr.ndim)
    noise_arr = arr[
        region_shape(arr.shape, n_semisizes, n_position)]
    return signal_arr, noise_arr


# ======================================================================
def sn_split_optim(arr):
    """
    Separate signal from noise using optimal peak thresholding.

    Uses the first inverted peak after the the first peak of the data
    histogram.

    Args:
        arr (np.ndarray): The input array.

    Returns:
        result (tuple[np.ndarray]): The tuple
            contains:
             - signal_arr: The signal array.
             - noise_arr: The noise array.
    """
    threshold = mrt.segmentation.threshold_optim(arr)
    signal_mask = arr > threshold
    signal_arr = arr[signal_mask]
    noise_arr = arr[~signal_mask]
    return signal_arr, noise_arr


# ======================================================================
def sn_split_otsu(
        arr,
        corrections=(1.0, 0.2)):
    """
    Separate signal from noise using the Otsu threshold.

    Args:
        arr (np.ndarray): The input array.
        corrections (int|float|Iterable[int|float]: The correction factors.
            If value is 1, no correction is performed.
            If int or float, the Otsu threshold is corrected (multiplied)
            by the corresponding factor before thresholding.
            If Iterable, the first correction is used to estimate the signal,
            while the second correction is used to estimate the noise.
            At most two values are accepted.
            When the two values are not identical some values may be ignored
            or counted both in signal and in noise.

    Returns:
        result (tuple[np.ndarray]): The tuple
            contains:
             - signal_arr: The signal array.
             - noise_arr: The noise array.
    """
    corrections = fc.auto_repeat(corrections, 2, check=True)
    otsu = mrt.segmentation.threshold_otsu(arr)
    signal_mask = arr > otsu * corrections[0]
    noise_mask = arr <= otsu * corrections[1]
    signal_arr = arr[signal_mask]
    noise_arr = arr[noise_mask]
    return signal_arr, noise_arr


# ======================================================================
def sn_split_relative(
        arr,
        thresholds=(0.75, 0.25)):
    """
    Separate signal from noise using the relative threshold(s).

    Args:
        arr (np.ndarray): The input array.
        thresholds (int|float|Iterable[int|float]: The percentile values.
            Values must be in the [0, 1] range.
            If int or float, values above are considered signal,
            and below or equal ar considered noise.
            If Iterable, values above the first percentile threshold are
            considered signals, while values below the second percentile
            threshold are considered noise.
            At most two values are accepted.
            When the two values are not identical some values may be ignored
            or counted both in signal and in noise.

    Returns:
        result (tuple[np.ndarray]): The tuple
            contains:
             - signal_arr: The signal array.
             - noise_arr: The noise array.

    See Also:
        segmentation.threshold_relative(),
    """
    thresholds = fc.auto_repeat(thresholds, 2, check=True)
    signal_threshold, noise_threshold = \
        mrt.segmentation.threshold_relative(arr, thresholds)
    signal_mask = arr > signal_threshold
    noise_mask = arr <= noise_threshold
    signal_arr = arr[signal_mask]
    noise_arr = arr[noise_mask]
    return signal_arr, noise_arr


# ======================================================================
def sn_split_percentile(
        arr,
        thresholds=(0.75, 0.25)):
    """
    Separate signal from noise using the percentile threshold(s).

    Args:
        arr (np.ndarray): The input array.
        thresholds (int|float|Iterable[int|float]: The percentile values.
            Values must be in the [0, 1] range.
            If int or float, values above are considered signal,
            and below or equal ar considered noise.
            If Iterable, values above the first percentile threshold are
            considered signals, while values below the second percentile
            threshold are considered noise.
            At most two values are accepted.
            When the two values are not identical some values may be ignored
            or counted both in signal and in noise.

    Returns:
        result (tuple[np.ndarray]): The tuple
            contains:
             - signal_arr: The signal array.
             - noise_arr: The noise array.

    See Also:
        segmentation.threshold_percentile()
    """
    thresholds = fc.auto_repeat(thresholds, 2, check=True)
    signal_threshold, noise_threshold = \
        mrt.segmentation.threshold_percentile(arr, thresholds)
    signal_mask = arr > signal_threshold
    noise_mask = arr <= noise_threshold
    signal_arr = arr[signal_mask]
    noise_arr = arr[noise_mask]
    return signal_arr, noise_arr


# ======================================================================
def sn_split_mean_std(
        arr,
        std_steps=(-1, -2),
        mean_steps=1):
    """
    Separate signal from noise using a threshold combining mean and std.dev.

    Thresholds are calculated using `pymrt.segmentation.threshold_mean_std()`.

    Signal/noise values interval depend on the `symmetric` parameter.

    Args:
        arr (np.ndarray): The input array.
        std_steps (Iterable[int|float]): The st.dev. multiplication step(s).
            These are usually values between -2 and 2.
        mean_steps (Iterable[int|float]): The mean multiplication step(s).
            This is usually set to 1.

    Returns:
        result (tuple[np.ndarray]): The tuple
            contains:
             - signal_arr: The signal array.
             - noise_arr: The noise array.
    """
    thresholds = sorted(
        mrt.segmentation.threshold_mean_std(arr, std_steps, mean_steps),
        reverse=True)
    signal_mask = arr > thresholds[0]
    noise_mask = arr <= thresholds[-1]
    signal_arr = arr[signal_mask]
    noise_arr = arr[noise_mask]
    return signal_arr, noise_arr


# ======================================================================
def sn_split_thresholds(
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
            `pymrt.segmentation.auto_thresholds()` with its `method`
            parameter set
            to `signal_threshold`.
        noise_threshold (int|float|str|None): The noise threshold.
            If str, the threshold is estimated using
            `pymrt.segmentation.auto_thresholds()` with its `method`
            parameter set
            to `noise_threshold`.
            If None, `noise_threshold` is set to `signal_threshold`.
        signal_kws (Mappable|None): Keyword parameters.
            If `signal_threshold` is str, the parameters are passed to
            `pymrt.segmentation.auto_thresholds()` for `signal_threshold`.
        noise_kws (Mappable|None): Keyword parameters.
            If `noisel_threshold` is str, the parameters are passed to
            `pymrt.segmentation.auto_thresholds()` for `noise_threshold`.
        signal_index (int|None): Select a specific threshold.
            The index is applied to the Iterable obtained from
            `pymrt.segmentation.auto_thresholds()` for `signal_threshold`.
            If None, the first value is selected.
        noise_index (int|None): Select a specific threshold.
            The index is applied to the Iterable obtained from
            `pymrt.segmentation.auto_thresholds()` for `noise_threshold`.
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
def sn_split_denoise(
        arr,
        method='gaussian',
        method_kws=None):
    """
    Separate signal from noise using denoising of the data.

    Args:
        arr (np.ndarray): The input array.
        method (str): Denoising method.
            This is passed to `denoise()`
        method_kws (Mappable|None): Keyword arguments to pass to `method`.
            These are passed to the corresponding function.
            See the respective documentation for details.

    Returns:
        result (tuple[np.ndarray]): The tuple
            contains:
             - signal_arr: The signal array.
             - noise_arr: The noise array.
    """
    method_kws = {} if method_kws is None else dict(method_kws)
    signal_arr = denoise(arr, method, method_kws)
    noise_arr = arr - signal_arr
    return signal_arr, noise_arr


# ======================================================================
def sn_split(
        arr,
        method='auto',
        *_args,
        **_kws):
    """
    Separate signal from noise.

    Args:
        arr (np.ndarray): The input array.
        method (str): The signal/noise estimation method.
            If str, uses the `sn_split_` functions from this module.
            Accepted values are:
             - 'auto': Uses 'optim' if positive, 'denoise' otherwise.
             - 'optim': use `pymrt.correction.sn_split_optim()`.
             - 'otsu': use `pymrt.correction.sn_split_otsu()`.
                Only works for positive values.
             - 'relative': use `pymrt.correction.sn_split_relative()`.
                Only works for positive values.
             - 'percentile': use `pymrt.correction.sn_split_percentile()`.
                Only works for positive values.
             - 'mean_std': use `pymrt.correction.sn_split_mean_std()`.
                Only works for positive values.
             - 'thresholds': use `pymrt.correction.sn_split_thresholds()`.
                Only works for positive values.
             - 'calib_region': use `pymrt.correction.sn_split_calib_region()`.
                Specify the calibration regions directly.
             - 'denoise': use `pymrt.correction.sn_split_denoise()`.
                Useful when no noise calibration region is present.
            If callable, the signature must be:
            f(np.ndarray, *_args, **_kws) -> (np.ndarray, np.ndarray)
            where the input array is `arr` and the two returned arrays are:
             - the signal array
             - the noise array
        *_args: Positional arguments for `pymrt.correction.method()`.
        **_kws: Keyword arguments for `pymrt.correction.method()`.

    Returns:
        result (tuple[np.ndarray]): The tuple
            contains:
             - signal_arr: The signal array.
             - noise_arr: The noise array.

    Raises:
        ValueError: If `method` is unknown.
    """
    methods = (
        'auto', 'optim',
        'otsu', 'relative', 'percentile', 'mean_std', 'thresholds',
        'calib_region', 'denoise')
    method = method.lower()
    if method == 'auto':
        if np.all(arr) >= 0.0:
            method = sn_split_optim
        else:
            method = sn_split_denoise
    elif method == 'optim':
        method = sn_split_optim
    elif method == 'otsu':
        method = sn_split_otsu
    elif method == 'relative':
        method = sn_split_relative
    elif method == 'percentile':
        method = sn_split_percentile
    elif method == 'mean_std':
        method = sn_split_mean_std
    elif method == 'thresholds':
        method = sn_split_thresholds
    elif method == 'calib_region':
        method = sn_split_calib_region
    elif method == 'denoise':
        method = sn_split_denoise
    else:
        raise ValueError(
            'valid methods are: {} (given: {})'.format(methods, method))
    return method(arr, *_args, **_kws)


# ======================================================================
def estimate_noise_sigma_dwt(
        arr,
        wavelet='dmey',
        mode='symmetric',
        axes=None,
        estimator=lambda d: np.std(d),
        correction=lambda s, x: s - s / np.std(x)):
    """
    Estimate of the noise standard deviation with discrete wavelet transform.

    This is computed from the single-level discrete wavelete transform (DWT)
    details coefficients.

    Default values are for Gaussian distribution.
    Other distribution may require tweaking.

    Args:
        arr (np.ndarray): The input array.
        wavelet (str|pw.Wavelet): The wavelet to use.
            See `pw.dwtn()` and `pw.wavelist()` for more info.
        mode (str): Signal extension mode.
            See `pw.dwtn()` and `pw.Modes()` for more info.
        axes (Iterable[int]|None): Axes over which to perform the DWT.
            See `pw.dwtn()` for more info.
        estimator (callable): The estimator to use.
            Must accept an `np.ndarray` as its first argument.
            Sensible options may be:
             - `np.std`
             - `lambda x: np.median(np.abs(x)) / sp.stats.norm.ppf(0.75)`
             - `lambda x: np.std(x) / np.mean(np.abs(x)) - 1
        correction (callable|None): Correct for signal bias.

    Returns:
        sigma (float): The estimate of the noise standard deviation.
    """
    coeffs = pw.dwtn(arr, wavelet=wavelet, mode=mode, axes=axes)
    # select details coefficients in all dimensions
    d_coeffs = coeffs['d' * arr.ndim]
    # d_coeffs = d_coeffs[np.nonzero(d_coeffs)]
    sigma = estimator(d_coeffs)
    if correction:
        sigma = correction(sigma, arr)
    return sigma


# ======================================================================
def estimate_noise_sigma_sn_split(
        arr,
        method='auto',
        *_args,
        **_kws):
    """
    Estimate of the noise standard deviation from signal/noise separation.

    Args:
        arr (np.ndarray): The input array.
        method (str): The signal/noise estimation method.
            This is passed to `sn_split()`
        *_args: Positional arguments for `method()`.
        **_kws: Keyword arguments for `method()`.

    Returns:
        sigma (float): The estimate of the noise standard deviation.
    """
    signal_arr, noise_arr = sn_split(arr, method, *_args, **_kws)
    sigma = np.std(noise_arr)
    return sigma


# ======================================================================
def estimate_noise_sigma(
        arr,
        dwt_kws=None,
        method='otsu',
        *_args,
        **_kws):
    """
    Optimal estimate of the noise standard deviation.

    This is obtained by calculating the DWT noise sigma estimate on a region
    containing mostly noise.

    Args:
        arr (np.ndarray): The input array.
        dwt_kws (Mappable|None): Keyword parameters for noise sigma DWT.
            This is passed to `estimate_noise_sigma_dwt()`.
        method (str): The signal/noise estimation method.
            This is passed to `sn_split()`
        *_args: Positional arguments for `method()`.
        **_kws: Keyword arguments for `method()`.

    Returns:
        sigma (float): The estimate of the noise standard deviation.
    """
    signal_arr, noise_arr = sn_split(arr, method, *_args, **_kws)
    if dwt_kws is None:
        dwt_kws = {}
    sigma = estimate_noise_sigma_dwt(noise_arr, **dwt_kws)
    return sigma


# ======================================================================
def fix_bias_rician(
        arr,
        method='best',
        method_kws=None,
        positive=True):
    """
    Fix magnitude level to remove the bias associated with Rician noise.

    The background noise region is estimated from histogram peaks.

    The noise of the resulting data is still definite positive, but much
    closer to a Gaussian distribution, and the magnitude bias is now
    reduced.

    Args:
        arr (np.ndarray): The input array.
        method (str): Sigma noise estimation method.
            Accepted values are:
             - 'best': `estimate_noise_sigma()`
             - 'separated': `estimate_noise_sigma_sn_split()`
             - 'region': `sigma_noise_region()`
        method_kws (Mappable|None): Keyword arguments to pass to `method`.
            These are passed to the corresponding function.
            See the respective documentation for details.
        positive (bool): Force result to be positive.
            If True, the new signal magnitude is given by:
            :math:`s' = \\sqrt{|s^2 - \\sigma^2|}`
            Otherwise, uses the sign of the argument of the square root.
            :math:`s' = sgn({s^2 - \\sigma^2})\\sqrt{|s^2 - \\sigma^2|}`

    Returns:
        arr (np.ndarray): The output array.

    Raises:
        ValueError: If `method` is unknown.

    References:
        - Gudbjartsson, H., Patz, S., 1995. The Rician Distribution of Noisy
          MRI Data. Magn Reson Med 34, 910–914.
    """
    arr = arr.astype(float)

    method_kws = {} if method_kws is None else dict(method_kws)
    sigma = estimate_noise_sigma(arr, **method_kws)

    # sigma *= (np.sqrt(2.0 / (4 - np.pi)))  # correct for Rice factor
    # print('sigma={}, min={}, max= {}, mean={}, std={}, median={}'.format(
    #     sigma, np.min(arr), np.max(arr), np.mean(arr), np.std(arr),
    #     np.median(arr)))  # DEBUG
    arr = arr ** 2 - sigma ** 2
    if positive:
        arr = np.sqrt(np.abs(arr))
    else:
        arr = np.sign(arr) * np.sqrt(np.abs(arr))
    return arr


# ======================================================================
elapsed(__file__[len(PATH['base']) + 1:])

# ======================================================================
if __name__ == '__main__':
    run_doctests(__doc__)
