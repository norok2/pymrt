#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Participation to the QSM 2016 challenge."""

# ======================================================================
# :: Future Imports
from __future__ import division
from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals

# ======================================================================
# :: Python Standard Library Imports
import os  # Miscellaneous operating system interfaces

# :: External Imports
import numpy as np  # NumPy (multidimensional numerical arrays library)
import scipy as sp  # SciPy (signal and image processing library)

# :: External Imports Submodules
import scipy.linalg
import scipy.ndimage
import scipy.signal

import matplotlib.pyplot as plt  # Matplotlib's pyplot: MATLAB-like syntax

from pymrt import elapsed, print_elapsed
import pymrt.input_output as mrio


# ======================================================================
def scale(
        val,
        in_interval=(0.0, 1.0),
        out_interval=(0.0, 1.0)):
    """
    Linear convert the value from input interval to output interval

    Args:
        val (float|np.ndarray): Value(s) to convert
        in_interval (float,float): Interval of the input value
        out_interval (float,float): Interval of the output value.

    Returns:
        val (float): The converted value

    Examples:
        >>> scale(100, (0, 100), (0, 1000))
        1000.0
        >>> scale(50, (-100, 100), (0, 1000))
        750.0
        >>> scale(50, (0, 1), (0, 10))
        500.0
        >>> scale(0.5, (0, 1), (-10, 10))
        0.0
        >>> scale(np.pi / 3, (0, np.pi), (0, 180))
        60.0
    """
    in_min, in_max = in_interval
    out_min, out_max = out_interval
    return (val - in_min) / (in_max - in_min) * (out_max - out_min) + out_min


# ======================================================================
def minmax(arr):
    """
    Calculate the minimum and maximum of an array: (min, max).

    Args:
        arr (np.ndarray): The input array.

    Returns:
        min (float): the minimum value of the array
        max (float): the maximum value of the array

    Examples:
        >>> minmax(np.arange(10))
        (0, 9)
    """
    return np.min(arr), np.max(arr)


# ======================================================================
def auto_replicate(val, n):
    try:
        iter(val)
    except TypeError:
        val = (val,) * n
    return val


# ======================================================================
def gaussian_kernel(shape, sigmas, center=0.0, ndim=1, normalize=True):
    for val in (shape, sigmas, center):
        try:
            iter(val)
        except TypeError:
            pass
        else:
            ndim = max(len(val), ndim)

    shape = auto_replicate(shape, ndim)
    sigmas = auto_replicate(sigmas, ndim)
    center = auto_replicate(center, ndim)

    assert (len(sigmas) == len(shape))
    assert (len(center) == len(shape))

    grid = [slice(-(x - x0) // 2 + 1, (x - x0) // 2 + 1)
            for x, x0 in zip(shape, center)]
    coord = np.ogrid[grid]
    kernel = np.exp(
        -(sum([x ** 2 / (2 * sigma ** 2) for x, sigma in zip(coord, sigmas)])))
    if normalize:
        kernel = kernel / np.sum(kernel)
    return kernel


# ======================================================================
def rmse(
        arr1,
        arr2,
        scaling=100):
    """
    Calculate the root mean squared error of the first vs the second array.

    RMSE = A * ||arr1 - arr2|| / ||arr2||

    Args:
        arr1 (np.ndarray): The first input array.
        arr2 (np.ndarray): The second input array.
        scaling (float): The scaling factor.
            Useful to express the results in percent.

    Returns:
        rmse (float): The root mean squared error
    """
    assert (arr1.shape == arr2.shape)
    norm = scipy.linalg.norm
    rmse = scaling * norm(arr1 - arr2) / norm(arr2)
    return rmse


# ======================================================================
def hfen(
        arr1,
        arr2,
        filter_sizes=15,
        sigmas=1.5):
    """
    Compute the high-frequency error norm.

    The Laplacian of a Gaussian filter is used to get high frequency
    information.

    Args:
        arr1 (np.ndarray): The first input array.
        arr2 (np.ndarray): The second input array.
        filter_sizes (tuple[int]|int): The size of the filter in px.
            If a single value is given, is is assumed to be equal in all dims.
        sigmas (tuple[float]|float): The sigma of the gaussian kernel in px.
            If a single value is given, it is assumed to be equal in all dims.

    Returns:
        hfen (float): The high-frequency error norm.
    """
    assert (arr1.shape == arr2.shape)
    ndim = arr1.ndim

    filter_sizes = auto_replicate(filter_sizes, ndim)
    sigmas = auto_replicate(sigmas, ndim)
    assert (len(sigmas) == len(filter_sizes))

    grid = [slice(-filter_size // 2 + 1, filter_size // 2 + 1)
            for filter_size in filter_sizes]
    coord = np.ogrid[grid]
    gaussian_filter = gaussian_kernel(filter_sizes, sigmas)
    hfen_factor = \
        sum([x ** 2 / sigma ** 4 for x, sigma in zip(coord, sigmas)]) + \
        - sum([1 / sigma ** 2 for sigma in sigmas])
    arr_filter = gaussian_filter * hfen_factor
    arr_filter = arr_filter - np.sum(arr_filter) / np.prod(arr_filter.shape)

    # the filter should be symmetric, therefore: correlate == convolve
    # additionally, fftconvolve much faster than direct convolve or correlate

    # arr1_corr = scipy.ndimage.filters.correlate(arr1, arr_filter)
    arr1_corr = scipy.signal.fftconvolve(arr1, arr_filter, 'same')
    # arr2_corr = scipy.ndimage.filters.correlate(arr2, arr_filter)
    arr2_corr = scipy.signal.fftconvolve(arr2, arr_filter, 'same')

    hfen = rmse(arr1_corr, arr2_corr)
    return hfen


# ======================================================================
def ssim(
        arr1,
        arr2,
        arr_interval=None,
        aa=(1, 1, 1),
        kk=(0.010, 0.030, 0.015)):
    """
    Calculate the structure similarity index, SSIM.

    This is defined as: SSIM = (lum ** alpha) * (con ** beta) * (sti ** gamma)
     - lum is a measure of the luminosity, with exp. weight alpha
     - con is a measure of the contrast, with exp. weight beta
     - sti is a measure of the structural information, with exp. weight gamma

    Args:
        arr1 (np.ndarray): The first input array.
        arr2 (np.ndarray): The second input array.
        arr_interval (tuple[float]): Minimum and maximum allowed values.
            The values of both arr1 and arr2 should be within this interval.
        aa (tuple[float]): The exponentiation weight factors. Must be 3.
            Modulate the relative weight of the three SSIM components
            (luminosity, contrast and structural information).
            If they are all equal to 1, the computation can be simplified.
        kk (tuple[float]): The ratio regularization constant factors. Must be 3.
            Determine the regularization constants as a factors of the total
            interval size (squared) for the three SSIM components
            (luminosity, contrast and structural information).
            Must be numbers much smaller than 1.

    Returns:
        ssim (float): The structure similarity index.

    See Also:
        Wang, Zhou, A. C. Bovik, H. R. Sheikh, and E. P. Simoncelli. “Image
        Quality Assessment: From Error Visibility to Structural Similarity.”
        IEEE Transactions on Image Processing 13, no. 4 (April 2004):
        600–612. doi:10.1109/TIP.2003.819861.
    """
    assert (arr1.shape == arr2.shape)
    if arr_interval is None:
        arr_interval = (
            min(np.min(arr1), np.min(arr2)), max(np.max(arr1), np.max(arr2)))
    interval_size = np.ptp(arr_interval)
    cc = [(k * interval_size) ** 2 for k in kk]
    mu1 = np.mean(arr1)
    mu2 = np.mean(arr2)
    sigma1 = np.std(arr1)
    sigma2 = np.std(arr2)
    sigma12 = np.sum((arr1 - mu1) * (arr2 - mu2)) / (arr1.size - 1)
    ff = [
        (2 * mu1 * mu2 + cc[0]) / (mu1 ** 2 + mu2 ** 2 + cc[0]),
        (2 * sigma1 * sigma2 + cc[1]) / (sigma1 ** 2 + sigma2 ** 2 + cc[1]),
        (sigma12 + cc[2]) / (sigma1 * sigma2 + cc[2])
    ]
    ssim = np.prod(np.array([f ** a for (f, a) in zip(ff, aa)]), 0)
    return ssim


# ======================================================================
def ssim_map(
        arr1,
        arr2,
        filter_sizes=5,
        sigmas=1.5,
        arr_interval=None,
        aa=(1, 1, 1),
        kk=(0.010, 0.030, 0.015)):
    """
    Calculate the local structure similarity index map.

    Args:
        arr1 (np.ndarray): The first input array.
        arr2 (np.ndarray): The second input array.
        filter_sizes (tuple[int]|int): The size of the filter in px.
            If a single value is given, is is assumed to be equal in all dims.
        sigmas (tuple[float]|float): The sigma of the gaussian kernel in px.
            If a single value is given, it is assumed to be equal in all dims.
        arr_interval (tuple[float]): Minimum and maximum allowed values.
            The values of both arr1 and arr2 should be within this interval.
        aa (tuple[float]): The exponentiation weight factors. Must be 3.
            Modulate the relative weight of the three SSIM components
            (luminosity, contrast and structural information).
            If they are all equal to 1, the computation can be simplified.
        kk (tuple[float]): The ratio regularization constant factors. Must be 3.
            Determine the regularization constants as a factors of the total
            interval size (squared) for the three SSIM components
            (luminosity, contrast and structural information).
            Must be numbers much smaller than 1.

    Returns:
        ssim_arr (np.ndarray): The local structure similarity index map
        ssim (float): The global (mean) structure similarity index.

    See Also:
        Wang, Zhou, A. C. Bovik, H. R. Sheikh, and E. P. Simoncelli. “Image
        Quality Assessment: From Error Visibility to Structural Similarity.”
        IEEE Transactions on Image Processing 13, no. 4 (April 2004):
        600–612. doi:10.1109/TIP.2003.819861.
    """
    assert (arr1.shape == arr2.shape)
    if arr_interval is None:
        arr_interval = (
            min(np.min(arr1), np.min(arr2)), max(np.max(arr1), np.max(arr2)))
    interval_size = np.ptp(arr_interval)
    ndim = arr1.ndim
    arr_filter = gaussian_kernel(
        auto_replicate(filter_sizes, ndim), auto_replicate(sigmas, ndim))
    convolve = scipy.signal.fftconvolve
    mu1 = convolve(arr1, arr_filter, 'same')
    mu2 = convolve(arr2, arr_filter, 'same')
    mu1_mu1 = mu1 ** 2
    mu2_mu2 = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    sg1_sg1 = convolve(arr1 ** 2, arr_filter, 'same') - mu1_mu1
    sg2_sg2 = convolve(arr2 ** 2, arr_filter, 'same') - mu2_mu2
    sg12 = convolve(arr1 * arr2, arr_filter, 'same') - mu1_mu2
    cc = [(k * interval_size) ** 2 for k in kk]
    # determine whether to use the simplified expression
    if all(aa) == 1 and 2 * cc[2] == cc[1]:
        ssim_arr = ((2 * mu1_mu2 + cc[0]) * (2 * sg12 + cc[1])) / (
            (mu1_mu1 + mu2_mu2 + cc[0]) * (sg1_sg1 + sg2_sg2 + cc[1]))
    else:
        sg1 = np.sqrt(np.abs(sg1_sg1))
        sg2 = np.sqrt(np.abs(sg2_sg2))
        ff = [
            (2 * mu1_mu2 + cc[0]) / (mu1_mu1 + mu2_mu2 + cc[0]),
            (2 * sg1 * sg2 + cc[1]) / (sg1_sg1 + sg2_sg2 + cc[1]),
            (sg12 + cc[2]) / (sg1 * sg2 + cc[2])
        ]
        ssim_arr = np.prod(np.array([f ** a for (f, a) in zip(ff, aa)]), 0)
    ssim = np.mean(ssim_arr)
    return ssim_arr, ssim


a1 = mrio.load(
    '/nobackup/isar2/cache/qsm_2016_challenge/backup/data/chi_cosmos.nii.gz'
    '').astype(np.float64)
a2 = mrio.load(
    '/nobackup/isar2/cache/qsm_2016_challenge/backup/data/phs_unwrap.nii.gz'
    '').astype(np.float64)

# import profile
# profile.run('compute_hfen(arr1, arr2)', sort=1)

# print(rmse(a1, a2))
# elapsed('rmse')

# print(hfen(a1, a2))
# elapsed('hfen')

msk1 = a1 != 0.0
msk2 = a2 != 0.0
a12 = np.stack((a1, a2))
min12 = np.min(a12)
max12 = np.max(a12)
a1[msk1] = scale(a1[msk1], minmax(a12), (0, 255))
a2[msk2] = scale(a2[msk2], minmax(a12), (0, 255))
elapsed('prepare for ssim')

print(ssim(a1, a2))
elapsed('ssim')

ssim_arr, val = ssim_map(a1, a2)
print(val)
elapsed('ssim_map')
from skimage.measure import compare_ssim

print(compare_ssim(a1, a2, win_size=5, gaussian_weights=True, sigma=1.5,
                   dynamic_range=np.ptp((0, 255))))
elapsed('ssim_scikit')
print_elapsed()
