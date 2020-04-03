#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pymrt.segmentation: generic segmentation.
"""

# ======================================================================
# :: Future Imports
from __future__ import (
    division, absolute_import, print_function, unicode_literals, )

# ======================================================================
# :: Python Standard Library Imports
import os  # Miscellaneous operating system interfaces
# import shutil  # High-level file operations
# import math  # Mathematical functions
# import time  # Time access and conversions
# import datetime  # Basic date and time types
# import operator  # Standard operators as functions
# import collections  # Container datatypes
import itertools  # Functions creating iterators for efficient looping
# import functools  # Higher-order functions and operations on callable objects
# import argparse  # Parser for command-line options, arguments and subcommands
# import re  # Regular expression operations
# import subprocess  # Subprocess management
# import multiprocessing  # Process-based parallelism
# import inspect  # Inspect live objects
# import csv  # CSV File Reading and Writing [CSV: Comma-Separated Values]
# import json  # JSON encoder and decoder [JSON: JavaScript Object Notation]
# import unittest  # Unit testing framework
import doctest  # Test interactive Python examples
import warnings  # Warning control

# :: External Imports
import numpy as np  # NumPy (multidimensional numerical arrays library)
import scipy as sp  # SciPy (signal and image processing library)
# import matplotlib as mpl  # Matplotlib (2D/3D plotting library)
# import sympy as sym  # SymPy (symbolic CAS library)
# import PIL  # Python Image Library (image manipulation toolkit)
# import SimpleITK as sitk  # Image ToolKit Wrapper
# import nibabel as nib  # NiBabel (NeuroImaging I/O Library)
# import nipy  # NiPy (NeuroImaging in Python)
# import nipype  # NiPype (NiPy Pipelines and Interfaces)
import flyingcircus as fc  # Everything you always wanted to have in Python*

# :: External Imports Submodules
# import matplotlib.pyplot as plt  # Matplotlib's pyplot: MATLAB-like syntax
import scipy.ndimage  # SciPy: ND-image Manipulation
# import scipy.optimize  # SciPy: Optimization Algorithms
# import scipy.integrate  # SciPy: Integrations facilities
# import scipy.constants  # SciPy: Mathematal and Physical Constants
# import scipy.stats  # SciPy: Statistical functions
import scipy.signal  # SciPy: Signal Processing

# :: Local Imports
import pymrt as mrt  # Python Magnetic Resonance Tools: the multi-tool of MRI.

from pymrt import INFO, PATH
from pymrt import VERB_LVL, D_VERB_LVL, VERB_LVL_NAMES
from pymrt import elapsed, report, run_doctests
from pymrt import msg, dbg, fmt, fmtm

from pymrt.config import CFG


# TODO: implement other types of segmentations?


# ======================================================================
def threshold_relative(
        arr,
        values=0.5):
    """
    Calculate threshold relative to array values range.

    Args:
        arr (np.ndarray): The input array.
        values (float|Iterable[float]): The relative threshold value(s).
            Values must be in the [0, 1] range.

    Returns:
        result (tuple[float]): the calculated threshold.
    """

    min_val = np.min(arr)
    max_val = np.max(arr)
    values = fc.auto_repeat(values, 1)
    return tuple(
        min_val + (max_val - min_val) * float(value)
        for value in values)


# ======================================================================
def threshold_percentile(
        arr,
        values=0.5):
    """
    Calculate threshold percentile.

    This is the threshold value at which X% (= value) of the data is smaller
    than the threshold.

    Args:
        arr (np.ndarray): The input array.
        values (float|Iterable[float]): The percentile value(s).
            Values must be in the [0, 1] range.

    Returns:
        result (tuple[float]): the calculated thresholds.
    """
    values = fc.auto_repeat(values, 1)
    values = tuple(100.0 * value for value in values)
    return tuple(np.percentile(arr, values))


# ======================================================================
def threshold_mean_std(
        arr,
        std_steps=(-2, -1, 0, 1, 2),
        mean_steps=1):
    """
    Calculate threshold from mean and standard deviations.

    This is the threshold value at which X% (= value) of the data is smaller
    than the threshold.

    Args:
        arr (np.ndarray): The input array.
        std_steps (Iterable[int|float]): The st.dev. multiplication step(s).
            These are usually values between -2 and 2.
        mean_steps (Iterable[int|float]): The mean multiplication step(s).
            This is usually set to 1.


    Returns:
        result (tuple[float]): the calculated thresholds.
    """
    mean = np.nanmean(arr)
    std = np.nanstd(arr)
    min_val = np.min(arr)
    max_val = np.max(arr)
    mean_steps = fc.auto_repeat(mean_steps, 1)
    std_steps = fc.auto_repeat(std_steps, 1)
    return tuple(
        mean * mean_step + std * std_step
        for mean_step, std_step in itertools.product(mean_steps, std_steps)
        if min_val <= mean * mean_step + std * std_step <= max_val)


# ======================================================================
def threshold_otsu(
        arr,
        bins='sqrt'):
    """
    Optimal foreground/background threshold value based on Otsu's method.

    Args:
        arr (np.ndrarray): The input array.
        bins (int|str|None): Number of bins used to calculate histogram.
            If str or None, this is automatically calculated from the data
            using `fc.extra.auto_bin()` with `method` set to `bins` if str,
            and using the default `fc.extra.auto_bin()` method if set to
            None.

    Returns:
        threshold (float): The threshold value.

    Raises:
        ValueError: If `arr` only contains a single value.

    Examples:
        >>> num = 1000
        >>> x = np.linspace(-10, 10, num)
        >>> arr = np.sin(x) ** 2
        >>> threshold = threshold_otsu(arr)
        >>> round(threshold, 1)
        0.5

    References:
        - Otsu, N., 1979. A Threshold Selection Method from Gray-Level
          Histograms. IEEE Transactions on Systems, Man, and Cybernetics 9,
          62–66. doi:10.1109/TSMC.1979.4310076
    """
    # todo: extend to multiple classes
    return fc.extra.otsu_threshold(arr, bins=bins)


# ======================================================================
def threshold_otsu2(
        arr,
        bins='sqrt'):
    """
    Optimal foreground/background threshold value based on 2D Otsu's method.

    EXPERIMENTAL!

    Args:
        arr (np.ndrarray): The input array.
        bins (int|str|None): Number of bins used to calculate histogram.
            If str or None, this is automatically calculated from the data
            using `fc.extra.auto_bin()` with `method` set to `bins` if str,
            and using the default `fc.extra.auto_bin()` method if set to
            None.

    Returns:
        threshold (float): The threshold value.

    References:
        - Otsu, N., 1979. A Threshold Selection Method from Gray-Level
          Histograms. IEEE Transactions on Systems, Man, and Cybernetics 9,
          62–66. doi:10.1109/TSMC.1979.4310076
    """
    raise NotImplementedError


# ======================================================================
def threshold_hist_peaks(
        arr,
        bins='sqrt',
        depth='rice'):
    """
    Calculate threshold from the peaks in the histogram.

    Args:
        arr (np.ndarray): The input array.
        bins (int|str|None): The number of bins for the histogram.
            If str, this is determined using `fc.extra.auto_bins()`.
            If None, the default method in `fc.extra.auto_bins()` is used.
        depth (int|str|None): The peak finding depth.
            This parameter determines the peak finding rate in rapidly varying
            ("noisy") histograms.
            Smaller values correspond to more peaks being found.
            If str, this is determined using `fc.extra.auto_bins()`.
            If None, the default method in `fc.extra.auto_bins()` is used.

    Returns:
        result (tuple[float]): the calculated thresholds.
    """
    if isinstance(bins, str):
        bins = fc.extra.auto_bin(arr, bins)
    elif bins is None:
        bins = fc.extra.auto_bin(arr)
    hist, bin_edges = np.histogram(arr, bins)
    bin_centers = fc.midval(bin_edges)
    # depth determines the dynamic smoothing of the histogram
    if isinstance(depth, str):
        depth = fc.extra.auto_bin(arr, depth)
    elif bins is None:
        depth = fc.extra.auto_bin(arr)
    # at least 1 width value is required
    widths = np.arange(1, max(2, depth))
    with np.errstate(divide='ignore', invalid='ignore'):
        peaks = sp.signal.find_peaks_cwt(hist, widths)
    return tuple(bin_centers[peaks])


# ======================================================================
def threshold_inv_hist_peaks(
        arr,
        bins='sqrt',
        depth='rice'):
    """
    Calculate threshold from the peaks in the inverted histogram.

    The inverted histogram is obtained by subtracting the histogram to its
    maximum value.

    Args:
        arr (np.ndarray): The input array.
        bins (int|str|None): The number of bins for the histogram.
            If str, this is determined using `fc.extra.auto_bins()`.
            If None, the default method in `fc.extra.auto_bins()` is used.
        depth (int|str|None): The peak finding depth.
            This parameter determines the peak finding rate in rapidly varying
            ("noisy") histograms.
            Smaller values correspond to more peaks being found.
            If str, this is determined using `fc.extra.auto_bins()`.
            If None, the default method in `fc.extra.auto_bins()` is used.

    Returns:
        result (tuple[float]): the calculated thresholds.
    """
    if isinstance(bins, str):
        bins = fc.extra.auto_bin(arr, bins)
    elif bins is None:
        bins = fc.extra.auto_bin(arr)
    hist, bin_edges = np.histogram(arr, bins)
    bin_centers = fc.midval(bin_edges)
    # depth determines the dynamic smoothing of the histogram
    if isinstance(depth, str):
        depth = fc.extra.auto_bin(arr, depth)
    elif bins is None:
        depth = fc.extra.auto_bin(arr)
    # at least 1 width value is required
    widths = np.arange(1, max(2, depth))
    with np.errstate(divide='ignore', invalid='ignore'):
        peaks = sp.signal.find_peaks_cwt(np.max(hist) - hist, widths)
    return tuple(bin_centers[peaks])


# ======================================================================
def threshold_hist_peak_edges(
        arr,
        bins='sqrt',
        depth='rice'):
    """
    Calculate threshold from the peak edges in the histogram.

    The peak edges are defined as the mid-values of the peaks and the
    inverse peaks.

    Args:
        arr (np.ndarray): The input array.
        bins (int|str|None): The number of bins for the histogram.
            If str, this is determined using `fc.extra.auto_bins()`.
            If None, the default method in `fc.extra.auto_bins()` is used.
        depth (int|str|None): The peak finding depth.
            This parameter determines the peak finding rate in rapidly varying
            ("noisy") histograms.
            Smaller values correspond to more peaks being found.
            If str, this is determined using `fc.extra.auto_bins()`.
            If None, the default method in `fc.extra.auto_bins()` is used.

    Returns:
        result (tuple[float]): the calculated thresholds.
    """
    if isinstance(bins, str):
        bins = fc.extra.auto_bin(arr, bins)
    elif bins is None:
        bins = fc.extra.auto_bin(arr)
    hist, bin_edges = np.histogram(arr, bins)
    bin_centers = fc.midval(bin_edges)
    # depth determines the dynamic smoothing of the histogram
    if isinstance(depth, str):
        depth = fc.extra.auto_bin(arr, depth)
    elif bins is None:
        depth = fc.extra.auto_bin(arr)
    # at least 1 width value is required
    widths = np.arange(1, max(2, depth))
    with np.errstate(divide='ignore', invalid='ignore'):
        peaks = sp.signal.find_peaks_cwt(hist, widths)
    peak_edges = fc.midval(peaks)
    return tuple(bin_centers[peak_edges])


# ======================================================================
def threshold_inv_hist_peak_edges(
        arr,
        bins='sqrt',
        depth='rice'):
    """
    Calculate threshold from the peak edges in the histogram.

    The peak edges are defined as the mid-values of the peaks and the
    inverse peaks.

    Args:
        arr (np.ndarray): The input array.
        bins (int|str|None): The number of bins for the histogram.
            If str, this is determined using `fc.extra.auto_bins()`.
            If None, the default method in `fc.extra.auto_bins()` is used.
        depth (int|str|None): The peak finding depth.
            This parameter determines the peak finding rate in rapidly varying
            ("noisy") histograms.
            Smaller values correspond to more peaks being found.
            If str, this is determined using `fc.extra.auto_bins()`.
            If None, the default method in `fc.extra.auto_bins()` is used.

    Returns:
        result (tuple[float]): the calculated thresholds.
    """
    if isinstance(bins, str):
        bins = fc.extra.auto_bin(arr, bins)
    elif bins is None:
        bins = fc.extra.auto_bin(arr)
    hist, bin_edges = np.histogram(arr, bins)
    bin_centers = fc.midval(bin_edges)
    # depth determines the dynamic smoothing of the histogram
    if isinstance(depth, str):
        depth = fc.extra.auto_bin(arr, depth)
    elif bins is None:
        depth = fc.extra.auto_bin(arr)
    # at least 1 width value is required
    widths = np.arange(1, max(2, depth))
    with np.errstate(divide='ignore', invalid='ignore'):
        inv_peaks = sp.signal.find_peaks_cwt(np.max(hist) - hist, widths)
    inv_peak_edges = fc.midval(inv_peaks)
    return tuple(bin_centers[inv_peak_edges])


# ======================================================================
def threshold_twice_first_peak(arr):
    """
    Calculate the background threshold from a histogram peaks analysis.

    Assuming that the background is the first peak of the histogram, the
    background threshold is estimated as the minimum of:
     - twice the first peak minus half the minumum value of the array.
     - the first inverted peak after the first peak.

    Args:
        arr (np.ndarray): The input array.

    Returns:
        threshold (float): The threshold value.
    """
    hist_peaks = np.array(threshold_hist_peaks(arr))
    inv_hist_peaks = np.array(threshold_inv_hist_peaks(arr))
    threshold = min(
        inv_hist_peaks[inv_hist_peaks > hist_peaks[0]][0],
        2 * hist_peaks[0] - np.min(arr) / 2)
    return threshold


# ======================================================================
def threshold_cum_hist_elbow(arr):
    """
    Calculate a threshold from the eventual elbow of the cumulative histogram.

    EXPERIMENTAL!

    Args:
        arr (np.ndarray): The input array.

    Returns:
        threshold (float): The threshold value.
    """
    raise NotImplementedError


# ======================================================================
def threshold_rayleigh(
        arr,
        num=64,
        chunks=16,
        tol=1e-1):
    """
    Calculate the optimal background threshold for Rayleigh noise.

    Rayleigh noise is a special case of Rician noise generating from
    zero-mean Gaussian noise.

    Uses the relationship between the mean :math:`\\mu_n` and the standard
    deviation :math:`\\sigma_n` of the distribution from the standard
    deviation :math:`\\sigma_G` of the generating Gaussian distribution:

    .. math::
        \\frac{\\mu_n}{\\sigma_n} =
        \\frac{\\sigma_G\\sqrt{\\frac{\\pi}{2}}}
        {\\sigma_G\\sqrt{\\frac{4 - \\pi}{2}}} =
        \\sqrt{\\frac{\\pi}{4 - \\pi}}

    Args:
        arr (np.ndarray): The input array.
        num (int): The number of value levels.
            This specifies the number of value levels in each chunk.
        chunks (int): The number of chunks for value levels.
            Divide the values range into the specified number of chunks
            and search for the the matching threshold sequentially.
            If the threshold is found in an earlier chunk, subsequent chunks
            do not need to be evaluated.
        tol (float): The tolerance for closeness.

    Returns:
        threshold (float): The threshold value.
    """
    mu_sigma_ratio = (np.pi / (4 - np.pi)) ** 0.5
    min_val = np.min(arr)
    max_val = np.max(arr)
    threshold = min_val
    found = False
    diff = 0
    for j in range(chunks):
        min_chunk = min_val + (max_val - min_val) * j / chunks
        max_chunk = min_val + (max_val - min_val) * (j + 1) / chunks
        for i, threshold in enumerate(np.linspace(min_chunk, max_chunk), num):
            mask = arr <= threshold
            last_diff = diff
            n_arr = arr[mask]
            mu_n, sigma_n = np.nanmean(n_arr), np.nanstd(n_arr)
            if sigma_n > 0:
                diff = (mu_n / sigma_n) - mu_sigma_ratio
                if diff * last_diff < 0 or np.abs(diff) < tol:
                    found = True
                    break
        if found:
            break
    return threshold


# ======================================================================
def threshold_optim(arr):
    """
    Calculate the optimal background threshold.

    Assuming that the background is the first peak of the histogram, the
    background threshold is estimated as the minimum of:
     - twice the first peak minus half the minumum value of the array.
     - the first inverted peak after the first peak.

    Args:
        arr (np.ndarray): The input array.

    Returns:
        threshold (float): The threshold value.
    """
    hist_peaks = np.array(threshold_hist_peaks(arr))
    inv_hist_peaks = np.array(threshold_inv_hist_peaks(arr))
    threshold = min(
        inv_hist_peaks[inv_hist_peaks > hist_peaks[0]][0],
        2 * hist_peaks[0] - np.min(arr) / 2)
    return threshold


# ======================================================================
def auto_thresholds(
        arr,
        method='otsu',
        kws=None):
    """
    Calculate a thresholding value based on the specified method.

    Args:
        arr (np.ndarray): The input array.
        method (str): The threshold method.
            Accepted values are:
             - 'relative': use `pymrt.segmentation.threshold_relative()`.
             - 'percentile': use `pymrt.segmentation.threshold_percentile()`.
             - 'mean_std': use `pymrt.segmentation.threshold_mean_std()`.
             - 'otsu': use `pymrt.segmentation.threshold_otsu()`.
             - 'otsu2': use `pymrt.segmentation.threshold_otsu2()`.
             - 'hist_peaks': use `pymrt.segmentation.threshold_hist_peaks()`.
             - 'inv_hist_peaks': use
             `pymrt.segmentation.threshold_inv_hist_peaks()`.
             - 'hist_peak_edges': use
             `pymrt.segmentation.threshold_hist_peak_edges()`.
             - 'inv_hist_peak_edges': use
             `pymrt.segmentation.threshold_inv_hist_peak_edges()`.
             - 'twice_first_peak': use
             `pymrt.segmentation.threshold_twice_first_peak()`.
             - 'cum_hist_elbow': use
             `pymrt.segmentation.threshold_cum_hist_elbow()`.
             - 'rayleigh': use `pymrt.segmentation.threshold_rayleigh()`.
             - 'optim': use `pymrt.segmentation.threshold_optim()`.
        kws (dict|None): Keyword parameters for the selected method.

    Returns:
        thresholds (tuple[float]): The threshold(s).

    Raises:
        ValueError: If `method` is unknown.
    """
    if method:
        method = method.lower()
    methods = (
        'relative', 'percentile', 'mean_std',
        'otsu', 'otsu2',
        'hist_peaks', 'inv_hist_peaks',
        'hist_peak_edges', 'inv_hist_peak_edges',
        'twice_first_peak',
        # 'cum_hist_elbow', 'cum_hist_quad',
        # 'cum_hist_quad_weight', 'cum_hist_quad_inv_weight',
        'rayleigh', 'optim')
    if kws is None:
        kws = dict()
    if method == 'relative':
        thresholds = threshold_relative(arr, **dict(kws))
    elif method == 'percentile':
        thresholds = threshold_percentile(arr, **dict(kws))
    elif method == 'mean_std':
        thresholds = threshold_mean_std(arr, **dict(kws))
    elif method == 'otsu':
        thresholds = threshold_otsu(arr, **dict(kws))
    elif method == 'otsu2':
        thresholds = threshold_otsu2(arr, **dict(kws))
    elif method == 'hist_peaks':
        thresholds = threshold_hist_peaks(arr, **dict(kws))
    elif method == 'inv_hist_peaks':
        thresholds = threshold_inv_hist_peaks(arr, **dict(kws))
    elif method == 'hist_peak_edges':
        thresholds = threshold_hist_peak_edges(arr, **dict(kws))
    elif method == 'inv_hist_peak_edges':
        thresholds = threshold_inv_hist_peak_edges(arr, **dict(kws))
    elif method == 'twice_first_peak':
        thresholds = threshold_twice_first_peak(arr, **dict(kws))
    elif method == 'cum_hist_elbow':
        thresholds = threshold_cum_hist_elbow(arr, **dict(kws))
    elif method == 'rayleigh':
        thresholds = threshold_rayleigh(arr, **dict(kws))
    elif method == 'optim':
        thresholds = threshold_optim(arr, **dict(kws))
    else:  # if method not in methods:
        raise ValueError(
            'valid methods are: {} (given: {})'.format(methods, method))
    # ensures that the result is Iterable
    thresholds = tuple(fc.auto_repeat(thresholds, 1))
    return thresholds


# ======================================================================
def threshold_to_mask(
        arr,
        threshold,
        comparison='>'):
    """
    Apply the specified threshold to the array.

    Args:
        arr (np.ndarray): Input array for the masking.
        threshold (int|float): Value for the threshold.
        comparison (str): A string representing the numeric relationship
            Accepted values are: ['==', '!=', '>', '<', '>=', '<=']

    Returns:
        mask (np.ndarray[bool]): Mask for which comparison is True.
    """
    # for security reasons comparison is checked before eval
    comparisons = ('==', '!=', '>', '<', '>=', '<=')
    if comparison in comparisons:
        mask = eval('arr {c} threshold'.format(c=comparison))
    else:
        raise ValueError(fmtm(
            'valid comparisons are: {comparisons} (given: {comparison})'))
    return mask


# ======================================================================
def label_thresholds(
        arr,
        thresholds,
        comparison='>'):
    """
    Create labels from an image according to specific thresholds.

    Args:
        arr (np.ndarray): Array from which mask is created.
        thresholds (Iterable[int|float]): Value(s) for the threshold.
        comparison (str): A string representing the numeric relationship
            Accepted values are: ['>', '<', '>=', '<=']

    Returns:
        label (np.ndarray[int]): Labels for each threshold region.
    """
    label = np.zeros_like(arr, dtype=int)
    for threshold in thresholds:
        mask = threshold_to_mask(arr, threshold, comparison)
        label += mask.astype(int)
    return label


# ======================================================================
def find_objects(
        arr,
        structure=None,
        max_label=0,
        reduce_support=False):
    """
    Label contiguous objects from an array and generate corresponding masks.

    The background is assumed to have a value of 0.
    Masks of larger objects are listed first.

    Args:
        arr (np.ndarray): The array to operate with.
        structure (ndarray|None): Definition of feature connections.
            If None, use default.
        max_label (int): Limit the number of labeled to search through.
        reduce_support (bool): Reduce the support of the masks to their size.
            Effectively, the shape of the output is adapted to the content.

    Returns:
        labeled (np.ndarray): The array containing the labeled objects.
        masks (list[np.ndarray]): A list of the objects as mask arrays.
            The list is sorted by decresing size (larger to smaller).
            The shape of each mask is either the same as the input, or it is
            adapted to its content, depending on the `reduce_support` flag.
    """
    labeled, num_labels = sp.ndimage.label(arr, structure)
    masks = []
    if reduce_support:
        containers = sp.ndimage.find_objects(labeled, max_label)
    else:
        containers = [[slice(None)] * len(labeled.shape)] * num_labels
    for i, (label, container) in enumerate(zip(labeled, containers)):
        label_value = i + 1
        mask = labeled[container]
        mask = (mask == label_value)
        masks.append(mask)
    # sort labeled and masks by size (descending)
    masks = sorted(masks, key=lambda x: -np.sum(x))
    labeled = np.zeros_like(labeled).astype(int)
    for value, mask in enumerate(masks):
        labeled += mask.astype(int) * (value + 1)
    return labeled, masks


# ======================================================================
def label_nested_structures(
        arr,
        seed=None):
    """
    Label nested structures incrementally.

    EXPERIMENTAL!

    This is useful for segmenting structures that have the topology of
    nested layers.

    Args:
        arr (np.ndarray): The input array.
        threshold (float): The thresholding.
        seed (Iterable[int]|None): The initial seed for starting the layering.
            If None, the center of `arr` is used.

    Returns:

    """
    raise NotImplementedError


# ======================================================================
def clip_range(
        arr,
        interval,
        out_values=None):
    """
    Set values outside the specified interval to constant.

    Similar masking patters could be obtained with `label_thresholds()`
    or `threshold_to_mask()`.

    Args:
        arr (np.ndarray): The input array.
        interval (Iterable[int|float]): The values interval.
            Must contain 2 items: (t1, t2)
            Values outside this range are set according to `out_values`.
        out_values (int|float|Iterable[int|float]|None): The replacing values.
            If int or float, values outside the (t1, t2) range are replaced
            with `out_values`.
            If Iterable, must contain 2 items: (v1, v2), and values below `t1`
            are replaced with `v1`, while values above `t2` are replaced with
            `v2`.
            If None, uses v1 = t1 and v2 = t2: values below `t1` are replaced
            with `t1`, while values above `t2` are replaced with `t2`.

    Returns:
        arr (np.ndarray): The clipped array.
    """
    t1, t2 = interval
    if out_values is None:
        out_values = interval
    out_values = fc.auto_repeat(out_values, 2, check=True)
    v1, v2 = out_values
    arr[arr < t1] = v1
    arr[arr > t2] = v2
    return arr


# ======================================================================
def auto_mask(
        arr,
        threshold='otsu',
        threshold_kws=None,
        comparison='>',
        smoothing=0.0,
        erosion_iter=0,
        dilation_iter=0):
    """
    Create a compact mask from an image according to specific threshold.

    This is achieved with the following workflow:
        - Gaussian filter smoothing
        - masking values according to threshold
        - binary erosion
        - binary dilation

    Args:
        arr (np.ndarray): Input array for the masking.
        threshold (int|float|str): Value/method for the threshold.
            If str, the threshold is estimated using `auto_thresholds()` with
            its `mode` parameter set to `threshold`.
        threshold_kws (Mappable|None): Keyword parameters for `thresholding()`.
        comparison (str): A string representing the numeric relationship
            Accepted values are: ['==', '!=', '>', '<', '>=', '<=']
        smoothing (float): Sigma to be used for Gaussian smoothing.
            If zero, no filtering done.
        erosion_iter : int (optional)
            Number of binary erosion iteration in mask post-processing.
        dilation_iter : int (optional)
            Number of binary dilation iteration in mask post-processing.
    Returns:
        arr (np.ndarray[bool]): Mask for which comparison is True.
    """
    if smoothing > 0.0:
        arr = sp.ndimage.gaussian_filter(arr, smoothing)

    if isinstance(threshold, str):
        thresholds = auto_thresholds(arr, threshold, threshold_kws)
        index = 0 if len(thresholds) > 1 else len(thresholds) // 2
        threshold = thresholds[index]
    arr = threshold_to_mask(arr, threshold, comparison)

    if erosion_iter > 0:
        arr = sp.ndimage.binary_erosion(arr, iterations=erosion_iter)

    if dilation_iter > 0:
        arr = sp.ndimage.binary_dilation(arr, iterations=dilation_iter)

    return arr


# ======================================================================
elapsed(__file__[len(PATH['base']) + 1:])

# ======================================================================
if __name__ == '__main__':
    run_doctests(__doc__)
