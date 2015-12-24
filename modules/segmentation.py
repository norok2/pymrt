#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
mri_tools.segmentation: generic segmentation using scikit
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
# import mayavi.mlab as mlab  # Mayavi's mlab: MATLAB-like syntax
# import scipy.optimize  # SciPy: Optimization Algorithms
# import scipy.integrate  # SciPy: Integrations facilities
# import scipy.constants  # SciPy: Mathematal and Physical Constants
import scipy.ndimage  # SciPy: ND-image Manipulation


# import scipy.stats  # SciPy: Statistical functions

# :: Local Imports


# import mri_tools.modules.nifti as mrn
# from mri_tools import INFO
# from mri_tools import VERB_LVL
# from mri_tools import D_VERB_LVL
# from mri_tools import get_first_line

# TODO: implement other types of segmentations?


# ======================================================================
def mask_threshold(
        array,
        threshold=0.0,
        comparison='>',
        mode='absolute'):
    """
    Create a mask from image according to specific threshold.

    Parameters
    ==========
    array : ndarray
        Array from which mask is created.
    threshold : int, float or tuple
        Value(s) to be used for determining the threshold.
    comparison : str (optional)
        A string representing the numeric relationship: [=, !=, >, <, >=, <=]
    mode : str (optional)
        Determines how to interpret / process the threshold value.
        Available values are:
        | 'absolute': use the absolute value
        | 'relative': use a value relative to dynamic range
        | 'percentile': use the value obtained from the percentiles

    Returns
    =======
    mask : ndarray
        Mask for which comparison is True.

    """
    # for security reasons comparison is checked before eval
    comparisons = ('==', '!=', '>', '<', '>=', '<=')
    modes = ('absolute', 'relative', 'percentile')
    if comparison in comparisons:
        # warning: security: use of eval
        if mode == 'relative':
            min_val = np.min(array)
            max_val = np.max(array)
            threshold = min_val + (max_val - min_val) * float(threshold)
        elif mode == 'percentile':
            threshold = np.percentile(array, threshold)
        elif mode == 'absolute':
            pass
        else:
            raise ValueError('mode not known: {}'.format(mode))
        mask = eval('array {} threshold'.format(comparison))
    else:
        raise ValueError(
                'valid comparison modes are: [{}]'.format(
                        ', '.join(comparisons)))
    return mask


# ======================================================================
def find_objects(
        array,
        structure=None,
        max_label=0,
        reduce_support=False):
    """
    Label contiguous objects from an array and generate corresponding masks.

    The background is assumed to have a value of 0.
    Masks of larger objects are listed first.


    Parameters
    ==========
    array : ndarray
        The array to operate with.
    structure : ndarray or None (optional)
        The array that defines the feature connections. If None, use default.
    max_label : int (optional)
        Limit the number of labels to search through.
    reduce_support : bool
        If True the support of the masks is reduced to their size.

    Returns
    =======
    labels : ndarray
        The array containing the labelled objects.
    masks : ndarray list
        A list of array containing the object in sorted by size (descending).

    """
    labels, num_labels = sp.ndimage.label(array, structure)
    masks = []
    if reduce_support:
        containers = sp.ndimage.find_objects(labels, max_label)
    else:
        containers = [
            [slice(None) for dim in labels.shape]
            for num in range(num_labels)]
    for idx, (label, container) in enumerate(zip(labels, containers)):
        label_value = idx + 1
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
        array,
        num_samples=200):
    """
    Estimate the SNR of a real-positive-valued array.

    Args:
        array:

    Returns:

    """
    import matplotlib.pyplot as plt


    print('start...')
    hist, bin_edges = np.histogram(array, bins=num_samples, normed=True)
    step = bin_edges[1] - bin_edges[0]
    x = np.linspace(0, 100, num_samples)
    y = np.cumsum(hist) * step
    y2 = np.array([sp.percentile(array, q) for q in x])
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


import nibabel as nib

src = "/nobackup/isar1/TEMP/QSM-PLB" \
      "/P05_d0_S8_FLASH_3D_0p6_multiecho_corrected_magnitude_sum_Te8.16.nii"
array = nib.load(src).get_data()
print(snr_analysis(array))
