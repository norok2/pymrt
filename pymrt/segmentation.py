#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
pymrt.segmentation: generic simple segmentation
"""

# ======================================================================
# :: Future Imports
from __future__ import division
from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals

# ======================================================================
# :: Python Standard Library Imports
# import os  # Miscellaneous operating system interfaces
# import shutil  # High-level file operations
# import math  # Mathematical functions
# import time  # Time access and conversions
# import datetime  # Basic date and time types
# import operator  # Standard operators as functions
# import collections  # High-performance container datatypes
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

# :: Local Imports
from pymrt import INFO
from pymrt import VERB_LVL
from pymrt import D_VERB_LVL


# TODO: implement other types of segmentations?


# ======================================================================
def mask_threshold(
        arr,
        threshold=0.0,
        comparison='>',
        mode='absolute'):
    """
    Create a mask from an image according to specific threshold.

    Args:
        arr (np.ndarray): Array from which mask is created.
        threshold (int|float|tuple[int|float]): Value(s) for the threshold.
        comparison (str): A string representing the numeric relationship
            Accepted values are: ['==', '!=', '>', '<', '>=', '<=']
        mode (str): Determines how to interpret / process the threshold value.
            Available values are:
             - 'absolute': use the absolute value
             - 'relative': use a value relative to values interval
             - 'percentile': use the value obtained from the percentiles

    Returns:
        mask (np.ndarray[bool]): Mask for which comparison is True.
    """
    # for security reasons comparison is checked before eval
    comparisons = ('==', '!=', '>', '<', '>=', '<=')
    modes = ('absolute', 'relative', 'percentile')
    if comparison in comparisons:
        # warning: security: use of eval
        if mode == 'relative':
            min_val = np.min(arr)
            max_val = np.max(arr)
            threshold = min_val + (max_val - min_val) * float(threshold)
        elif mode == 'percentile':
            threshold = np.percentile(arr, threshold)
        elif mode == 'absolute':
            pass
        else:  # if mode not in modes:
            raise ValueError('mode not known: {}'.format(mode))
        mask = eval('arr {} threshold'.format(comparison))
    else:
        raise ValueError(
            'valid comparison modes are: [{}]'.format(
                ', '.join(comparisons)))
    return mask


# ======================================================================
def label_thresholds(
        arr,
        thresholds=(0.0,),
        comparison='>',
        mode='absolute'):
    """
    Create labels from an image according to specific thresholds.

    Args:
        arr (np.ndarray): Array from which mask is created.
        threshold (int|float|tuple[int|float]): Value(s) for the threshold.
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
        max_label (int): Limit the number of labels to search through.
        reduce_support (bool): Reduce the support of the masks to their size.
            Effectively, the shape of the output is adapted to the content.

    Returns:
        labels (np.ndarray): The array containing the labeled objects.
        masks (list[np.ndarray]): A list of the objects as mask arrays.
            The list is sorted by decresing size (larger to smaller).
            The shape of each mask is either the same as the input, or it is
            adapted to its content, depending on the `reduce_support` flag.
    """
    labels, num_labels = sp.ndimage.label(arr, structure)
    masks = []
    if reduce_support:
        containers = sp.ndimage.find_objects(labels, max_label)
    else:
        containers = [[slice(None)] * len(labels.shape)] * num_labels
    for i, (label, container) in enumerate(zip(labels, containers)):
        label_value = i + 1
        mask = labels[container]
        mask = (mask == label_value)
        masks.append(mask)
    # sort labels and masks by size (descending)
    masks = sorted(masks, key=lambda x: -np.sum(x))
    labels = np.zeros_like(labels).astype(int)
    for value, mask in enumerate(masks):
        labels += mask.astype(int) * (value + 1)
    return labels, masks


# ======================================================================
def snr_analysis(
        arr,
        num_samples=200):
    """
    Estimate the SNR of a real-positive-valued array.

    Args:
        arr:

    Returns:
        snr (float):
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
    print(__doc__)
    doctest.testmod()

    import nibabel as nib

    src = "~/hd1/TEMP/QSM-PLB" \
          "/P05_d0_S8_FLASH_3D_0p6_multiecho_corrected_magnitude_sum_Te8.16" \
          ".nii"
    array = nib.load(src).get_data()
    print(snr_analysis(array))
