#!/usr/bin/env python
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
# import itertools  # Functions creating iterators for efficient looping
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
        bins='sqrt',
        mask_nan=True,
        mask_inf=True,
        mask_vals=None):
    """
    Optimal foreground/background threshold value based on Otsu's method.

    Args:
        arr (np.ndrarray): The input array.
        bins (int|str|None): Number of bins used to calculate histogram.
            If str or None, this is automatically calculated from the data
            using `utils.auto_bin()` with `method` set to `bins` if str,
            and using the default `utils.auto_bin()` method if set to None.
        mask_nan (bool): Mask NaN values.
        mask_inf (bool): Mask Inf values.
        mask_vals (iterable|None): Values to mask.
            If None, no values are masked.

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
    # Check if flat-valued array
    if arr.min() == arr.max():
        raise ValueError('The array contains a single value.')

    if mask_nan and len(arr) > 0:
        arr = arr[~np.isnan(arr)]
    if mask_inf and len(arr) > 0:
        arr = arr[~np.isinf(arr)]
    if not mask_vals:
        mask_vals = ()
    for val in mask_vals:
        if len(arr) > 0:
            arr = arr[arr != val]

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
        value=0.5):
    """
    Calculate threshold relative to array values range.

    Args:
        arr (np.ndarray): The input array.
        value (float): The relative threshold value.
            Values must be in the [0, 1] range.

    Returns:
        threshold (float): the calculated threshold.
    """

    min_val = np.min(arr)
    max_val = np.max(arr)
    return min_val + (max_val - min_val) * float(value)


# ======================================================================
def threshold_percentile(
        arr,
        value=0.5):
    """
    Calculate threshold percentile.

    This is the threshold value at which X% (= value) of the data is smaller
    than the threshold.

    Args:
        arr (np.ndarray): The input array.
        value (float): The percentile value.
            Values must be in the [0, 1] range.

    Returns:
        threshold (float): the calculated threshold.
    """
    return np.percentile(arr, 100 * value)


# ======================================================================
def threshold_hist_peak(
        arr,
        depth=None,
        index=None):
    """
    Calculate threshold from the peaks in the histogram.

    Args:
        arr (np.ndarray): The input array.
        depth (float): The peak finding depth.
            This parameter determines the peak finding rate in rapidly varying
            histograms.
            Smaller values correspond to more peaks being found.
        index (int|None): Index of the histogram peak to use for thresholding.
            If None, the middle one is used.
            If the first one is desired, use `0`.
            If the last one is desired, use `-1`.

    Returns:
        threshold (float): the calculated threshold.
    """
    # todo: improve definition of `depth`
    hist, bin_edges = np.histogram(arr, mrt.utils.auto_bin(arr))
    bin_centers = mrt.utils.midval(bin_edges)
    widths = np.arange(1, max(2, depth))
    peaks = sp.signal.find_peaks_cwt(hist, widths)
    if index is None:
        index = int(len(peaks) / 2)
    return bin_centers[index]


# ======================================================================
def thresholding(
        arr,
        method='otsu',
        kws=None):
    """
    Calculate a thresholding value based on the specified method.

    Args:
        arr ():
        method ():
        kws ():

    Returns:

    """
    if method:
        method = method.lower()
    modes = ('otsu', 'relative', 'percentile', 'hist_peak', 'inv_hist_peak')
    if kws is None:
        kws = dict()
    if method == 'otsu':
        threshold = threshold_otsu(arr, **dict(kws))
    elif method == 'relative':
        threshold = threshold_relative(arr, **dict(kws))
    elif method == 'percentile':
        threshold = threshold_relative(arr, **dict(kws))
    elif method == 'hist_peak':
        raise NotImplementedError
    elif method == 'inv_hist_peak':
        raise NotImplementedError
    else:  # if mode not in modes:
        raise ValueError(
            'valid modes are: {modes}'
            ' (given: {mode})'.format_map(locals()))
    return threshold


# ======================================================================
def mask_threshold(
        arr,
        threshold='otsu',
        thresholding_kws=None,
        comparison='>'):
    """
    Create a mask from an image according to specific threshold.

    Args:
        arr (np.ndarray): Input array for the masking.
        threshold (int|float|str): Value for the threshold.
            If str, the threshold is estimated using `thresholding()` with
            its `mode` parameter set to `threshold`.
        thresholding_kws (dict|None): Keyword parameters for `thresholding()`.
        comparison (str): A string representing the numeric relationship
            Accepted values are: ['==', '!=', '>', '<', '>=', '<=']

    Returns:
        mask (np.ndarray[bool]): Mask for which comparison is True.
    """
    # for security reasons comparison is checked before eval
    comparisons = ('==', '!=', '>', '<', '>=', '<=')
    if comparison in comparisons:
        if isinstance(threshold, str):
            if thresholding_kws is None:
                thresholding_kws = dict()
            threshold = thresholding(arr, threshold, **dict(thresholding_kws))
        mask = eval('arr {comparison} threshold'.format_map(locals()))
    else:
        raise ValueError(
            'valid comparisons are: {comparisons}'
            ' (given: {comparison})'.format_map(locals()))
    return mask


# ======================================================================
def mask_compact(
        arr,
        threshold=0.0,
        comparison='>',
        mode='absolute',
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
        threshold (int|float|tuple[int|float]): Value(s) for the threshold.
        comparison (str): A string representing the numeric relationship
            Accepted values are: ['==', '!=', '>', '<', '>=', '<=']
        mode (str): Determines how to interpret / process the threshold value.
            Available values are:
             - 'absolute': use the absolute value
             - 'relative': use a value relative to values interval
             - 'percentile': use the value obtained from the percentiles
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

    arr = mask_threshold(arr, threshold, comparison, mode)

    if erosion_iter > 0:
        arr = sp.ndimage.binary_erosion(arr, iterations=erosion_iter)

    if dilation_iter > 0:
        arr = sp.ndimage.binary_dilation(arr, iterations=dilation_iter)

    return arr


# ======================================================================
def label_thresholds(
        arr,
        thresholds=0.0,
        comparison='>',
        mode='absolute'):
    """
    Create labels from an image according to specific thresholds.

    Args:
        arr (np.ndarray): Array from which mask is created.
        thresholds (int|float|tuple[int|float]): Value(s) for the threshold.
        comparison (str): A string representing the numeric relationship
            Accepted values are: ['>', '<', '>=', '<=']
        mode (str): Determines how to interpret / process the threshold value.
            Available values are:
             - 'absolute': use the absolute value
             - 'relative': use a value relative to values interval
             - 'percentile': use the value obtained from the percentiles

    Returns:
        label (np.ndarray[int]): Labels for values
    """
    label = np.zeros_like(arr, dtype=int)
    thresholds = mrt.utils.auto_repeat(thresholds, 1)
    for threshold in thresholds:
        mask = mask_threshold(arr, threshold, comparison, mode)
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
def snr_analysis(
        arr,
        num_samples=200):
    """
    Estimate the SNR of a real-positive-valued array.

    Args:
        arr:

    Returns:
        _snr (float):
        signal_level (float):
        noise_level (float):
    """
    # todo: calculate the SNR of an image
    import matplotlib.pyplot as plt

    print('start...')
    hist, bin_edges = np.histogram(arr, bins=num_samples, normed=True)
    step = bin_edges[1] - bin_edges[0]
    x = np.linspace(0, 100, num_samples)
    y = np.cumsum(hist) * step
    y2 = np.array([sp.percentile(arr, q) for q in x])
    plt.figure()
    plt.plot(hist)
    plt.figure()
    plt.plot(x, y)
    plt.figure()
    plt.plot(x, y2)
    plt.show()
    signal_level = 1
    noise_level = 10
    snr = signal_level / noise_level
    return snr, signal_level, noise_level


# ======================================================================
if __name__ == '__main__':
    msg(__doc__.strip())
    doctest.testmod()

    import nibabel as nib

    src = "~/hd1/TEMP/QSM-PLB" \
          "/P05_d0_S8_FLASH_3D_0p6_multiecho_corrected_magnitude_sum_Te8.16" \
          ".nii"
    array = nib.load(src).get_data()
    print(snr_analysis(array))
