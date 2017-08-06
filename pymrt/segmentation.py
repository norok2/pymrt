#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pymrt.segmentation: generic simple segmentation
"""

# ======================================================================
# :: Future Imports
from __future__ import (
    division, absolute_import, print_function, unicode_literals)

# ======================================================================
# :: Python Standard Library Imports
# import os  # Miscellaneous operating system interfaces
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

# :: External Imports Submodules
# import matplotlib.pyplot as plt  # Matplotlib's pyplot: MATLAB-like syntax
import scipy.ndimage  # SciPy: ND-image Manipulation
# import scipy.optimize  # SciPy: Optimization Algorithms
# import scipy.integrate  # SciPy: Integrations facilities
# import scipy.constants  # SciPy: Mathematal and Physical Constants
# import scipy.stats  # SciPy: Statistical functions
import scipy.signal  # SciPy: Signal Processing

# :: Local Imports
import pymrt as mrt
import pymrt.utils
from pymrt import INFO
from pymrt import VERB_LVL, D_VERB_LVL
from pymrt import msg, dbg


# TODO: implement other types of segmentations?

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
            using `utils.auto_bin()` with `method` set to `bins` if str,
            and using the default `utils.auto_bin()` method if set to None.

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
    # Check if flat-valued array
    if arr.min() == arr.max():
        raise ValueError('The array contains a single value.')

    if isinstance(bins, str):
        bins = mrt.utils.auto_bin(arr, bins)
    elif bins is None:
        bins = mrt.utils.auto_bin(arr)

    hist, bin_edges = np.histogram(arr, bins)
    bin_centers = mrt.utils.midval(bin_edges)
    hist = hist.astype(float)

    # class probabilities for all possible thresholds
    weight1 = np.cumsum(hist)
    weight2 = np.cumsum(hist[::-1])[::-1]
    # class means for all possible thresholds
    mean1 = np.cumsum(hist * bin_centers) / weight1
    mean2 = (np.cumsum((hist * bin_centers)[::-1]) / weight2[::-1])[::-1]
    # calculate the variance for all possible thresholds
    variance12 = weight1[:-1] * weight2[1:] * (mean1[:-1] - mean2[1:]) ** 2

    i_max_variance = np.argmax(variance12)
    threshold = bin_centers[:-1][i_max_variance]
    return threshold


# ======================================================================
def threshold_relative(
        arr,
        values=0.5):
    """
    Calculate threshold relative to array values range.

    Args:
        arr (np.ndarray): The input array.
        values (float|iterable[float]): The relative threshold value(s).
            Values must be in the [0, 1] range.

    Returns:
        result (tuple[float]): the calculated threshold.
    """

    min_val = np.min(arr)
    max_val = np.max(arr)
    values = mrt.utils.auto_repeat(values, 1)
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
        values (float|iterable[float]): The percentile value(s).
            Values must be in the [0, 1] range.

    Returns:
        result (tuple[float]): the calculated thresholds.
    """
    values = mrt.utils.auto_repeat(values, 1)
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
        mean_steps (iterable[int|float]): The mean multiplication step(s).
            This is usually set to 1.
        std_steps (iterable[int|float]): The percentile value(s).
            Values must be in the [0, 1] range.

    Returns:
        result (tuple[float]): the calculated thresholds.
    """
    mean = np.mean(arr)
    std = np.std(arr)
    min_val = np.min(arr)
    max_val = np.max(arr)
    mean_steps = mrt.utils.auto_repeat(mean_steps, 1)
    std_steps = mrt.utils.auto_repeat(std_steps, 1)
    return tuple(
        mean * mean_step + std * std_step
        for mean_step, std_step in itertools.product(mean_steps, std_steps)
        if min_val <= mean * mean_step + std * std_step <= max_val)


# ======================================================================
def threshold_hist_peaks(
        arr,
        bins='sqrt',
        depth=None):
    """
    Calculate threshold from the peaks in the histogram.

    Args:
        arr (np.ndarray): The input array.
        bins (int|str): The number of bins for the histogram.
            If str, this is determined using `utils.auto_bins()`.
        depth (int|None): The peak finding depth.
            This parameter determines the peak finding rate in rapidly varying
            ("noisy") histograms.
            Smaller values correspond to more peaks being found.
            If None, this is the determined using the number of bins.

    Returns:
        result (tuple[float]): the calculated thresholds.
    """
    if isinstance(bins, str):
        bins = mrt.utils.auto_bin(arr, bins)
    elif bins is None:
        bins = mrt.utils.auto_bin(arr)
    hist, bin_edges = np.histogram(arr, bins)
    bin_centers = mrt.utils.midval(bin_edges)
    # depth determines the dynamic smoothing of the histogram
    if depth is None:
        depth = int(bins / 64)
    # at least 1 width value is required
    widths = np.arange(1, max(2, depth))
    with np.errstate(divide='ignore', invalid='ignore'):
        peaks = sp.signal.find_peaks_cwt(hist, widths)
    return tuple(bin_centers[peaks])


# ======================================================================
def threshold_inv_hist_peaks(
        arr,
        bins='sqrt',
        depth=None):
    """
    Calculate threshold from the peaks in the inverted histogram.

    The inverted histogram is obtained by subtracting the histogram to its
    maximum value.

    Args:
        arr (np.ndarray): The input array.
        bins (int|str): The number of bins for the histogram.
            If str, this is determined using `utils.auto_bins()`.
        depth (int|None): The peak finding depth.
            This parameter determines the peak finding rate in rapidly varying
            ("noisy") histograms.
            Smaller values correspond to more peaks being found.
            If None, this is the determined using the number of bins.

    Returns:
        result (tuple[float]): the calculated thresholds.
    """
    if isinstance(bins, str):
        bins = mrt.utils.auto_bin(arr, bins)
    elif bins is None:
        bins = mrt.utils.auto_bin(arr)
    hist, bin_edges = np.histogram(arr, bins)
    bin_centers = mrt.utils.midval(bin_edges)
    # depth determines the dynamic smoothing of the histogram
    if depth is None:
        depth = int(bins / 64)
    # at least 1 width value is required
    widths = np.arange(1, max(2, depth))
    with np.errstate(divide='ignore', invalid='ignore'):
        peaks = sp.signal.find_peaks_cwt(np.max(hist) - hist, widths)
    return tuple(bin_centers[peaks])


# ======================================================================
def threshold_hist_peak_edges(
        arr,
        bins='sqrt',
        depth=None):
    """
    Calculate threshold from the peak edges in the histogram.

    The peak edges are defined as the mid-values of the peaks and the
    inverse peaks.

    Args:
        arr (np.ndarray): The input array.
        bins (int|str): The number of bins for the histogram.
            If str, this is determined using `utils.auto_bins()`.
        depth (int|None): The peak finding depth.
            This parameter determines the peak finding rate in rapidly varying
            ("noisy") histograms.
            Smaller values correspond to more peaks being found.
            If None, this is the determined using the number of bins.

    Returns:
        result (tuple[float]): the calculated thresholds.
    """
    if isinstance(bins, str):
        bins = mrt.utils.auto_bin(arr, bins)
    elif bins is None:
        bins = mrt.utils.auto_bin(arr)
    hist, bin_edges = np.histogram(arr, bins)
    bin_centers = mrt.utils.midval(bin_edges)
    # depth determines the dynamic smoothing of the histogram
    if depth is None:
        depth = int(bins / 64)
    # at least 1 width value is required
    widths = np.arange(1, max(2, depth))
    with np.errstate(divide='ignore', invalid='ignore'):
        peaks = sp.signal.find_peaks_cwt(hist, widths)
        inv_peaks = sp.signal.find_peaks_cwt(np.max(hist) - hist, widths)
    peak_edges = mrt.utils.midval(np.ndarray(
        x for x in itertools.chain.from_iterable(
            itertools.zip_longest(peaks, inv_peaks)
            if peaks[0] < inv_peaks[0] else
            itertools.zip_longest(inv_peaks, peaks))
        if x))
    return tuple(bin_centers[peak_edges])


def threshold_background_peaks(arr):
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
def auto_thresholds(
        arr,
        method='otsu',
        kws=None):
    """
    Calculate a thresholding value based on the specified method.

    Args:
        arr (np.ndarray): The input array.
        method (str): The threshold method.
            Available methods:
             - 'bg_peaks': a histogram foreground/background separation method.
             - 'otsu': the Otsu foreground/background separation method.
             - 'relative': thresholds values relative to the array range.
             - 'percentile': thresholds values from percentile distribution.
             - 'hist_peaks': uses histogram peaks as thresholds.
             - 'inv_hist_peaks': uses inverted histogram peaks as thresholds.
             - 'hist_peak_edges': uses histogram peak edges as thresholds.
        kws (dict|None): Keyword parameters for the selected method.

    Returns:
        thresholds (tuple[float]): The threshold(s).

    Raises:
        ValueError: If `method` is unknown.
    """
    if method:
        method = method.lower()
    methods = (
        'otsu', 'relative', 'percentile', 'mean_std',
        'hist_peaks', 'inv_hist_peaks', 'hist_peak_edges')
    if kws is None:
        kws = dict()
    if method == 'otsu':
        thresholds = threshold_otsu(arr, **dict(kws))
    elif method == 'relative':
        thresholds = threshold_relative(arr, **dict(kws))
    elif method == 'percentile':
        thresholds = threshold_percentile(arr, **dict(kws))
    elif method == 'mean_std':
        thresholds = threshold_mean_std(arr, **dict(kws))
    elif method == 'hist_peaks':
        thresholds = threshold_hist_peaks(arr, **dict(kws))
    elif method == 'inv_hist_peaks':
        thresholds = threshold_inv_hist_peaks(arr, **dict(kws))
    elif method == 'hist_peak_edges':
        thresholds = threshold_hist_peak_edges(arr, **dict(kws))
    else:  # if method not in methods:
        raise ValueError(
            'valid methods are: {} (given: {})'.format(methods, method))
    # ensures that the result is iterable
    thresholds = tuple(mrt.utils.auto_repeat(thresholds, 1))
    return thresholds


# ======================================================================
def _threshold(
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
        raise ValueError(
            'valid comparisons are: {comparisons}'
            ' (given: {comparison})'.format_map(locals()))
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
        thresholds (iterable[int|float]): Value(s) for the threshold.
        comparison (str): A string representing the numeric relationship
            Accepted values are: ['>', '<', '>=', '<=']

    Returns:
        label (np.ndarray[int]): Labels for each threshold region.
    """
    label = np.zeros_like(arr, dtype=int)
    for threshold in thresholds:
        mask = _threshold(arr, threshold, comparison)
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

    This is useful for segmenting self-contained structures.

    Args:
        arr:
        seed:

    Returns:

    """
    raise NotImplementedError


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
        threshold_kws (dict|None): Keyword parameters for `thresholding()`.
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
    arr = _threshold(arr, threshold, comparison)

    if erosion_iter > 0:
        arr = sp.ndimage.binary_erosion(arr, iterations=erosion_iter)

    if dilation_iter > 0:
        arr = sp.ndimage.binary_dilation(arr, iterations=dilation_iter)

    return arr


# ======================================================================
if __name__ == '__main__':
    msg(__doc__.strip())
    doctest.testmod()
