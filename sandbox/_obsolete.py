#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
pymrt: code that is now deprecated but can still be useful for legacy scripts.
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
import sys  # System-specific parameters and functions
# import shutil  # High-level file operations
# import math  # Mathematical functions
# import time  # Time access and conversions
# import datetime  # Basic date and time types
# import operator  # Standard operators as functions
# import collections  # High-performance container datatypes
# import itertools  # Functions creating iterators for efficient looping
# import functools  # Higher-order functions and operations on callable objects
# import argparse  # Parser for command-line options, arguments and sub-command
# import subprocess  # Subprocess management
# import multiprocessing  # Process-based parallelism
# import fractions  # Rational numbers
# import csv  # CSV File Reading and Writing [CSV: Comma-Separated Values]
# import json  # JSON encoder and decoder [JSON: JavaScript Object Notation]
# import inspect  # Inspect live objects
# import stat  # Interpreting stat() results
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
import scipy.optimize  # SciPy: Optimization Algorithms
# import scipy.integrate  # SciPy: Integrations facilities
# import scipy.constants  # SciPy: Mathematal and Physical Constants
# import scipy.ndimage  # SciPy: ND-image Manipulation
import scipy.signal  # SciPy: Signal Processing


# ======================================================================
def tty_colorify(
        text,
        color=None):
    """
    Add color TTY-compatible color code to a string, for pretty-printing.

    Args:
        text (str): The text to color.
        color (str|int|None): Identifier for the color coding.
            Lowercase letters modify the forground color.
            Uppercase letters modify the background color.
            Available colors:
             - r/R: red
             - g/G: green
             - b/B: blue
             - c/C: cyan
             - m/M: magenta
             - y/Y: yellow (brown)
             - k/K: black (gray)
             - w/W: white (gray)

    Returns:
        text (str): The colored text.

    See also:
        tty_colors
    """
    tty_colors = {
        'r': 31, 'g': 32, 'b': 34, 'c': 36, 'm': 35, 'y': 33, 'w': 37, 'k': 30,
        'R': 41, 'G': 42, 'B': 44, 'C': 46, 'M': 45, 'Y': 43, 'W': 47, 'K': 40,
    }

    if color in tty_colors:
        tty_color = tty_colors[color]
    elif color in tty_colors.values():
        tty_color = color
    else:
        tty_color = None
    if tty_color and sys.stdout.isatty():
        return '\x1b[1;{color}m{}\x1b[1;m'.format(text, color=tty_color)
    else:
        return text


# ======================================================================
def interval_size(interval):
    """
    Calculate the (signed) size of an interval given as a 2-tuple (A,B)

    Deprecated: use numpy.ptp instead

    Args:
        interval (float,float): Interval for computation

    Returns:
        val (float): The converted value

    Examples:

    """
    return interval[1] - interval[0]


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


'''
# ======================================================================
def calc_averages(
       filepath_list,
       out_dirpath,
       threshold=0.05,
       rephasing=True,
       registration=False,
       limit_num=None,
       force=False,
       verbose=D_VERB_LVL):
   """
   Calculate the average of MR complex images.

   TODO: clean up code / fix documentation

   Parameters
   ==========
   """
   def _compute_regmat(par):
       """Multiprocessing-friendly version of 'compute_affine_fsl()'."""
       return compute_affine_fsl(*par)

   tmp_dirpath = os.path.join(out_dirpath, 'tmp')
   if not os.path.exists(tmp_dirpath):
       os.makedirs(tmp_dirpath)
   # sort by scan number
   get_num = lambda filepath: parse_filename(filepath)['num']
   filepath_list.sort(key=get_num)
   # generate output name
   sum_num, sum_avg = 0, 0
   for filepath in filepath_list:
       info = parse_filename(filepath)
       base, params = parse_series_name(info['name'])
       sum_num += info['num']
       if PARAM_ID['avg'] in params:
           sum_avg += params[PARAM_ID['avg']]
       else:
           sum_avg += 1
   params[PARAM_ID['avg']] = sum_avg // 2
   name = to_series_name(base, params)
   new_info = {
       'num': sum_num,
       'name': name,
       'img_type': TYPE_ID['temp'],
       'te_val': info['te_val']}
   out_filename = to_filename(new_info)
   out_tmp_filepath = os.path.join(out_dirpath, out_filename)
   out_mag_filepath = change_img_type(out_tmp_filepath, TYPE_ID['mag'])
   out_phs_filepath = change_img_type(out_tmp_filepath, TYPE_ID['phs'])
   out_filepath_list = [out_tmp_filepath, out_mag_filepath, out_phs_filepath]
   # perform calculation
  if mrb.check_redo(filepath_list, out_filepath_list, force) and sum_avg > 1:
       # stack multiple images together
       # assume every other file is a phase image, starting with magnitude
       img_tuple_list = []
       mag_filepath = phs_filepath = None
       for filepath in filepath_list:
           if verbose > VERB_LVL['none']:
               print('Source:\t{}'.format(os.path.basename(filepath)))
           img_type = parse_filename(filepath)['img_type']
           if img_type == TYPE_ID['mag'] or not mag_filepath:
               mag_filepath = filepath
           elif img_type == TYPE_ID['phs'] or not phs_filepath:
               phs_filepath = filepath
           else:
               raise RuntimeWarning('Filepath list not valid for averaging.')
           if mag_filepath and phs_filepath:
               img_tuple_list.append([mag_filepath, phs_filepath])
               mag_filepath = phs_filepath = None

#        # register images
#        regmat_filepath_list = [
#            os.path.join(
#            tmp_dirpath,
#            mrio.del_ext(os.path.basename(img_tuple[0])) +
#            mrb.add_extsep(mrb.EXT['txt']))
#            for img_tuple in img_tuple_list]
#        iter_param_list = [
#            (img_tuple[0], img_tuple_list[0][0], regmat)
#            for img_tuple, regmat in
#            zip(img_tuple_list, regmat_filepath_list)]
#        pool = multiprocessing.Pool(multiprocessing.cpu_count())
#        pool.map(_compute_regmat, iter_param_list)
#        reg_filepath_list = []
#        for idx, img_tuple in enumerate(img_tuple_list):
#            regmat = regmat_filepath_list[idx]
#            for filepath in img_tuple:
#                out_filepath = os.path.join(
#                    tmp_dirpath, os.path.basename(filepath))
#                apply_affine_fsl(
#                    filepath, img_tuple_list[0][0], out_filepath, regmat)
#                reg_filepath_list.append(out_filepath)
#        # combine all registered images together
#        img_tuple_list = []
#        for filepath in reg_filepath_list:
#            if img_type == TYPE_ID['mag'] or not mag_filepath:
#                mag_filepath = filepath
#            elif img_type == TYPE_ID['phs'] or not phs_filepath:
#                phs_filepath = filepath
#            else:
#               raise RuntimeWarning('Filepath list not valid for averaging.')
#            if mag_filepath and phs_filepath:
#                img_tuple_list.append([mag_filepath, phs_filepath])
#                mag_filepath = phs_filepath = None

       # create complex images and disregard inappropriate
       img_list = []
       avg_power = 0.0
       num = 0
       shape = None
       for img_tuple in img_tuple_list:
           mag_filepath, phs_filepath = img_tuple
           img_mag_nii = nib.load(mag_filepath)
           img_mag = img_mag_nii.get_data()
           img_phs_nii = nib.load(mag_filepath)
           img_phs = img_phs_nii.get_data()
           affine_nii = img_mag_nii.get_affine()
           if not shape:
               shape = img_mag.shape
           if avg_power:
               rel_power = np.abs(avg_power - np.sum(img_mag)) / avg_power
           if (not avg_power or rel_power < threshold) \
                   and shape == img_mag.shape:
               img_list.append(mrb.polar2complex(img_mag, img_phs))
               num += 1
               avg_power = (avg_power * (num - 1) + np.sum(img_mag)) / num
       out_mag_filepath = change_param_val(
           out_mag_filepath, PARAM_ID['avg'], num)

       # k-space constant phase correction
       img0 = img_list[0]
       ft_img0 = np.fft.fftshift(np.fft.fftn(img0))
       k0_max = np.unravel_index(np.argmax(ft_img0), ft_img0.shape)
       for idx, img in enumerate(img_list):
           ft_img = np.fft.fftshift(np.fft.fftn(img))
           k_max = np.unravel_index(np.argmax(ft_img), ft_img.shape)
           dephs = np.angle(ft_img0[k0_max] / ft_img[k_max])
           img = np.fft.ifftn(np.fft.ifftshift(ft_img * np.exp(1j * dephs)))
           img_list[idx] = img

       img = mrb.ndstack(img_list, -1)
       img = np.mean(img, -1)
       mrio.save(out_mag_filepath, np.abs(img), affine_nii)
#        mrio.save(out_phs_filepath, np.angle(img), affine_nii)

#        fixed = np.abs(img_list[0])
#        for idx, img in enumerate(img_list):
#            affine = mrb.affine_registration(np.abs(img), fixed, 'rigid')
#            img_list[idx] = mrb.apply_affine(img_list[idx], affine)
#        mrio.save(out_filepath, np.abs(img), affine_nii)
#        print(img.shape, img.nbytes / 1024 / 1024)  # DEBUG
#        # calculate the Fourier transform
#        for img in img_list:
#            fft_list.append(np.fft.fftshift(np.fft.fftn(img)))
#        fixed = np.abs(img[:, :, :, 0])
#        mrb.sample2d(fixed, -1)
#        tmp = tmp * np.exp(1j*0.5)
#        moving = sp.ndimage.shift(fixed, [1.0, 5.0, 0.0])
#        mrb.sample2d(moving, -1)

#        print(linear, shift)
#        moved = sp.ndimage.affine_transform(moving, linear, offset=-shift)
#        mrb.sample2d(moved, -1)
#        mrio.save(out_filepath, moving, affine)
#        mrio.save(mag_filepath, fixed, affine)
#        mrio.save(phs_filepath, moved-fixed, affine)
#        for idx in range(len(img_list)):
#            tmp_img = img[:, :, :, idx]
#            tmp_fft = fft[:, :, :, idx]
#            mrb.sample2d(np.real(tmp_fft), -1)
#            mrb.sample2d(np.imag(tmp_fft), -1)
#            mrb.sample2d(np.abs(img[:, :, :, idx]), -1)
#            mrb.sample2d(np.angle(img[:, :, :, idx]), -1)

       # calculate output
       if verbose > VERB_LVL['none']:
           print('Target:\t{}'.format(os.path.basename(out_mag_filepath)))
           print('Target:\t{}'.format(os.path.basename(out_phs_filepath)))
   return mag_filepath, phs_filepath
'''

# ======================================================================
if __name__ == '__main__':
    print(__doc__)
    doctest.testmod()
