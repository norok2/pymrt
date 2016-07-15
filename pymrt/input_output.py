#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
pymrt: useful I/O utilities.

TODO:
- improve affine support
- improve header support
"""

# ======================================================================
# :: Future Imports
from __future__ import division
from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals

# ======================================================================
# :: Python Standard Library Imports
import os  # Miscellaneous operating system interfaces
# import shutil  # High-level file operations
# import math  # Mathematical functions
# import time  # Time access and conversions
# import datetime  # Basic date and time types
# import operator  # Standard operators as functions
# import collections  # High-performance container datatypes
# import itertools  # Functions creating iterators for efficient looping
# import functools  # Higher-order functions and operations on callable objects
# import argparse  # Parser for command-line options, arguments and subcommands
# import subprocess  # Subprocess management
# import multiprocessing  # Process-based parallelism
# import fractions  # Rational numbers
# import csv  # CSV File Reading and Writing [CSV: Comma-Separated Values]
# import json  # JSON encoder and decoder [JSON: JavaScript Object Notation]
# import inspect  # Inspect live objects
# import unittest  # Unit testing framework
import doctest  # Test interactive Python examples

# :: External Imports
import numpy as np  # NumPy (multidimensional numerical arrays library)
import scipy as sp  # SciPy (signal and image processing library)
# import matplotlib as mpl  # Matplotlib (2D/3D plotting library)
# import sympy as sym  # SymPy (symbolic CAS library)
# import PIL  # Python Image Library (image manipulation toolkit)
# import SimpleITK as sitk  # Image ToolKit Wrapper
import nibabel as nib  # NiBabel (NeuroImaging I/O Library)
# import nipy  # NiPy (NeuroImaging in Python)
# import nipype  # NiPype (NiPy Pipelines and Interfaces)

# :: External Imports Submodules
# import matplotlib.pyplot as plt  # Matplotlib's pyplot: MATLAB-like syntax
# import scipy.optimize  # SciPy: Optimization Algorithms
# import scipy.integrate  # SciPy: Integrations facilities
# import scipy.constants  # SciPy: Mathematal and Physical Constants
import scipy.ndimage  # SciPy: ND-image Manipulation
# :: Local Imports
import pymrt.base as mrb
import pymrt.geometry as mrg
import pymrt.plot as mrp
import pymrt.segmentation as mrs


# from pymrt import INFO
# from pymrt import VERB_LVL
# from pymrt import D_VERB_LVL
# from pymrt import get_first_line

# ======================================================================
# :: Custom defined constants

# ======================================================================
# :: Default values usable in functions


# ======================================================================
def load(
        in_filepath,
        full=False):
    """
    Load a NiBabel-supported image.

    Args:
        in_filepath (str): The input file path.
        full (bool): Return the image data, the affine and the header.

    Returns:
        ndarray|[ndarray,ndarray,header]: Returns the image data, and,
            if `full` is set to True, also the affine transformation matrix
            and the data header.

    See Also:
        nibabel.load, nibabel.get_data, nibabel.get_affine, nibabel.get_header
    """
    obj = nib.load(in_filepath)
    if full:
        return obj.get_data(), obj.get_affine(), obj.get_header()
    else:
        return obj.get_data()


# ======================================================================
def save(
        out_filepath,
        array,
        affine=None,
        header=None):
    """
    Save a NiBabel-supported image

    Args:
        out_filepath (str): Output file path
        array (np.ndarray): Data to be stored
        affine (np.ndarray): 3D affine transformation (4x4 matrix)
        header: Header of the image (refer to NiBabel).

    Returns:
        None
    """
    if affine is None:
        affine = np.eye(4)
    obj = nib.Nifti1Image(array, affine, header)
    obj.to_filename(out_filepath)


# ======================================================================
def masking(
        in_filepath,
        mask_filepath,
        out_filepath,
        mask_val=np.nan):
    """
    Apply a mask to a given file path.

    Args:
        in_filepath (str): Input file path
        mask_filepath (str): Mask file path
        out_filepath (str): Output file path
        mask_val: (int|float|complex): Value of masked out voxels

    Returns:
        None
    """
    obj = nib.load(in_filepath)
    obj_mask = nib.load(mask_filepath)
    img = obj.get_data()
    mask = obj_mask.get_data()
    mask = mask.astype(bool)
    img[~mask] = mask_val
    save(out_filepath, img, obj.get_affine())


# ======================================================================
def filter_1_1(
        in_filepath,
        out_filepath,
        func,
        *args,
        **kwargs):
    """
    Interface to generic 1-1 filter.
    filter(in_filepath) -> out_filepath

    Note that the function must return the affine matrix and the header.
    If the affine matrix is None, it is assumed to be the identity.
    If the returned header is None, it is autocalculated.

    Args:
        in_filepath (str): Input file path
        out_filepath (str): Output file path
        func (callable): Filtering function
            (img: ndarray, aff:ndarray, hdr:header)
            func(img, aff, hdr, *args, *kwargs) -> img, aff, hdr
        *args (tuple): Positional arguments passed to the filtering function
        **kwargs (dict): Keyword arguments passed to the filtering function

    Returns:
        None
    """
    obj = nib.load(in_filepath)
    img, aff, hdr = func(
        obj.get_data(), obj.get_affine(), obj.get_header(), *args,
        **kwargs)
    save(out_filepath, img, aff, hdr)


# ======================================================================
def filter_n_1(
        in_filepaths,
        out_filepath,
        func,
        *args,
        **kwargs):
    """
    Interface to generic n-1 filter.
    filter(in_filepaths) -> out_filepath

    Note that the function must return the affine matrix and the header.
    If the affine matrix is None, it is assumed to be the identity.
    If the returned header is None, it is autocalculated.

    Args:
        in_filepaths (list[str]): List of input file paths.
        out_filepath (str): Output file path
        func (callable): Filtering function
            (img: ndarray, aff:ndarray, hdr:header)
            func(list[img, aff, hdr], *args, *kwargs)) -> img, aff, hdr
        *args (tuple): Positional arguments passed to the filtering function.
        **kwargs (dict): Keyword arguments passed to the filtering function.

    Returns:
        None
    """
    input_list = []
    for in_filepath in in_filepaths:
        obj = nib.load(in_filepath)
        input_list.append((obj.get_data(), obj.get_affine(), obj.get_header()))
    img, aff, hdr = func(input_list, *args, **kwargs)
    save(out_filepath, img, aff, hdr)


# ======================================================================
def filter_n_m(
        in_filepaths,
        out_filepaths,
        func,
        *args,
        **kwargs):
    """
    Interface to generic n-m filter:
    filter(in_filepaths) -> out_filepaths

    Note that the function must return the list of affine matrices and headers.
    If the affine matrix is None, it is assumed to be the identity.
    If the returned header is None, it is autocalculated.

    Args:
        in_filepaths (list[str]): List of input file paths.
            The shape of each array must be identical.
            The affine matrix is taken from the last item
        out_filepaths (list[str]): List of output file paths
        func (callable): Filtering function
            (img: ndarray, aff:ndarray, hdr:header)
            func(list[img, aff, hdr], *args, *kwargs)) -> list[img, aff, hdr]
        *args (tuple): Positional arguments passed to the filtering function
        **kwargs (dict): Keyword arguments passed to the filtering function

    Returns:
        None
    """
    input_list = []
    for in_filepath in in_filepaths:
        obj = nib.load(in_filepath)
        input_list.append((obj.get_data(), obj.get_affine(), obj.get_header()))
    output_list = func(input_list, *args, **kwargs)
    for (img, aff, hdr), out_filepath in zip(output_list, out_filepaths):
        save(out_filepath, img, aff, hdr)


# ======================================================================
def filter_n_x(
        in_filepaths,
        out_filepath_template,
        func,
        *args,
        **kwargs):
    """
    Interface to generic n-x filter:
    calculation(i_filepaths) -> o_filepaths

    Note that the function must return the list of affine matrices and headers.
    If the affine matrix is None, it is assumed to be the identity.
    If the returned header is None, it is autocalculated.
    The number of output image is not known in advance.

    Args:
        in_filepaths (list[str]): List of input file paths.
            The shape of each array must be identical.
            The affine matrix is taken from the last item
        out_filepath_template (str): Output file path template
        func (callable): Filtering function
            (img: ndarray, aff:ndarray, hdr:header)
            func(list[img, aff, hdr], *args, *kwargs)) -> list[img, aff, hdr]
        *args (tuple): Positional arguments passed to the filtering function
        **kwargs (dict): Keyword arguments passed to the filtering function

    Returns:
        None
    """
    pass


# ======================================================================
def simple_filter_1_1(
        in_filepath,
        out_filepath,
        func,
        *args,
        **kwargs):
    """
    Interface to simplified 1-1 filter.
    filter(in_filepath) -> out_filepath

    Args:
        in_filepath (str): Input file path
            The affine matrix is taken from the input.
            The header is calculated automatically.
        out_filepath (str): Output file path
        func (callable): Filtering function (img: ndarray)
            func(img, *args, *kwargs) -> img
        *args (tuple): Positional arguments passed to the filtering function
        **kwargs (dict): Keyword arguments passed to the filtering function

    Returns:
        None
    """
    obj = nib.load(in_filepath)
    img = func(obj.get_data(), *args, **kwargs)
    aff = obj.get_affine()
    save(out_filepath, img, aff)


# ======================================================================
def simple_filter_n_1(
        in_filepaths,
        out_filepath,
        func,
        *args,
        **kwargs):
    """
    Interface to simplified n-1 filter.
    filter(in_filepaths) -> out_filepath

    Args:
        in_filepaths (list[str]): List of input file paths.
            The affine matrix is taken from the last item.
            The header is calculated automatically.
        out_filepath (str): Output file path.
        func (callable): Filtering function (img: ndarray)
            func(list[img], *args, *kwargs)) -> img
        *args (tuple): Positional arguments passed to the filtering function.
        **kwargs (dict): Keyword arguments passed to the filtering function.

    Returns:
        None.
    """
    img_list = []
    aff_list = []
    for in_filepath in in_filepaths:
        obj = nib.load(in_filepath)
        img_list.append(obj.get_data())
        aff_list.append(obj.get_affine())
    img = func(img_list, *args, **kwargs)
    aff = aff_list[-1]  # the affine of the first image
    save(out_filepath, img, aff)


# ======================================================================
def simple_filter_n_m(
        in_filepaths,
        out_filepaths,
        func,
        *args,
        **kwargs):
    """
    Interface to simplified n-m filter.
    filter(in_filepaths) -> out_filepaths

    Args:
        in_filepaths (list[str]): List of input file paths.
            The shape of each array must be identical.
            The affine matrix is taken from the last item.
            The header is calculated automatically.
        out_filepaths (list[str]): List of output file paths.
        func (callable): Filtering function (img: ndarray)
            func(list[img], *args, *kwargs) -> list[ndarray]
        *args (tuple): Positional arguments passed to the filtering function
        **kwargs (dict): Keyword arguments passed to the filtering function

    Returns:
        None.
    """
    i_img_list = []
    aff_list = []
    for in_filepath in in_filepaths:
        obj = nib.load(in_filepath)
        i_img_list.append(obj.get_data())
        aff_list.append(obj.get_affine())
    o_img_list = func(i_img_list, *args, **kwargs)
    aff = aff_list[0]  # the affine of the first image
    for img, out_filepath in zip(o_img_list, out_filepaths):
        save(out_filepath, img, aff)


# ======================================================================
def simple_filter_n_x(
        in_filepaths,
        out_filepath_template,
        func,
        *args,
        **kwargs):
    """
    Interface to simplified n-x filter.
    filter(in_filepaths) -> out_filepaths

    Note that the number of output image is not known in advance.

    Args:
        in_filepaths (list[str]): List of input file paths.
            The shape of each array must be identical.
            The affine matrix is taken from the last item.
        out_filepaths (list[str]): List of output file paths.
        func (callable): Filtering function (img: ndarray).
            func(list[img], *args, *kwargs) -> list[ndarray]
        *args (tuple): Positional arguments passed to the filtering function.
        **kwargs (dict): Keyword arguments passed to the filtering function.

    Returns:
        None.
    """
    pass


# ======================================================================
def stack(
        in_filepaths,
        out_filepath,
        axis=-1):
    """
    Join images together.

    Args:
        in_filepaths (list[str]): List of input file paths.
            The shape of each array must be identical.
            The affine matrix is taken from the last item.
        out_filepath (str): Output file path.
        axis (int): Joining axis of orientation.
            Must be a valid index for the input shape.

    Returns:
        None.
    """
    simple_filter_n_1(in_filepaths, out_filepath, np.stack, axis)


# ======================================================================
def split(
        in_filepath,
        out_dirpath=None,
        out_basename=None,
        axis=-1):
    """
    Split images apart.

    Args:
        in_filepath (str): Input file path.
            The affine matrix is taken from the input.
        axis (int): Joining axis of orientation.
            Must be a valid index for the input shape
        out_dirpath (str): Path to directory where to store results.
        out_filename (str): Output filename (without extension).

    Returns:
        out_filepaths (list[str]): List of output file paths.
    """
    # todo: refactor to use simple_filter_n_x
    if not out_dirpath or not os.path.exists(out_dirpath):
        out_dirpath = os.path.dirname(in_filepath)
    if not out_basename:
        out_basename = mrb.change_ext(
            os.path.basename(in_filepath), '', mrb.EXT['niz'])
    out_filepaths = []
    # load source image
    obj = nib.load(in_filepath)
    img = obj.get_data()
    # split data
    img_list = mrb.ndsplit(img, axis)
    # save data to output
    for i, image in enumerate(img_list):
        i_str = str(i).zfill(len(str(len(img_list))))
        out_filepath = os.path.join(
            out_dirpath,
            mrb.change_ext(out_basename + '-' + i_str, mrb.EXT['niz'], ''))
        save(out_filepath, image, obj.get_affine())
        out_filepaths.append(out_filepath)
    return out_filepaths


# ======================================================================
def zoom(
        in_filepath,
        out_filepath,
        zoom=1.0,
        interp_order=1,
        extra_dim=True,
        fill_dim=True):
    """
    Zoom the image with a specified magnification factor.

    Args:
        in_filepath (str): Input file path
        out_filepath (str): Output file path
        zoom (float|iterable): The zoom factor along the axes
        interp_order (int): Order of the spline interpolation
            0: nearest. Accepted range: [0, 5]
        extra_dim (bool): Force extra dimensions in the zoom parameters
        fill_dim (bool): Dimensions not specified are left untouched

    Returns:    
        None

    See Also:
        pymrt.geometry
    """

    def _zoom(array, zoom, interp_order, extra_dim, fill_dim):
        zoom, shape = mrg.zoom_prepare(zoom, array.shape, extra_dim, fill_dim)
        array = sp.ndimage.zoom(
            array.reshape(shape), zoom, order=interp_order)
        aff_transform = np.diag(1.0 / np.array(zoom[:3] + [1.0]))
        return array, aff_transform

    simple_filter_1_1(
        in_filepath, out_filepath, _zoom,
        zoom, interp_order, extra_dim, fill_dim)


# ======================================================================
def resample(
        in_filepath,
        out_filepath,
        new_shape,
        aspect=None,
        interp_order=1,
        extra_dim=True,
        fill_dim=True):
    """
    Resample the image to a new shape (different resolution / voxel size).

    Args:
        in_filepath (str): Input file path
        out_filepath (str): Output file path
        new_shape (tuple[int]): New dimensions of the image
        aspect (callable|list[callable]): Transformation applied to
        interp_order (int): Order of the spline interpolation
            0: nearest. Accepted range: [0, 5]
        extra_dim (bool): Force extra dimensions in the zoom parameters
        fill_dim (bool): Dimensions not specified are left untouched

    Returns:
        None
    """

    # todo: check for correctness

    def _zoom(
            array, new_shape, aspect, interp_order, extra_dim, fill_dim):
        zoom = mrg.shape2zoom(array.shape, new_shape, aspect)
        zoom, shape = mrg.zoom_prepare(zoom, array.shape, extra_dim, fill_dim)
        array = sp.ndimage.zoom(
            array.reshape(shape), zoom, order=interp_order)
        # aff_transform = np.diag(1.0 / np.array(zoom[:3] + [1.0]))
        return array

    simple_filter_1_1(
        in_filepath, out_filepath, _zoom,
        new_shape, aspect, extra_dim, fill_dim,
        interp_order)


# ======================================================================
def frame(
        in_filepath,
        out_filepath,
        border,
        background=0,
        use_longest=True):
    """
    Add a border frame to the image (same resolution / voxel size)

    Args:
        in_filepath (str): Input file path
        out_filepath (str): Output file path
        border (float|tuple[float]): The relative size of the borders
        background (int|float|complex): The value used for the frame
        use_longest (bool): Use longest dimension to calculate the border size

    Returns:
        None
    """
    simple_filter_1_1(
        in_filepath, out_filepath, mrg.frame, border, background,
        use_longest)


# ======================================================================
def reframe(
        in_filepath,
        out_filepath,
        new_shape,
        background=0):
    """
    Reframe an image into a new shape (same resolution / voxel size)

    Args:
        in_filepath (str): Input file path
        out_filepath (str): Output file path
        new_shape (tuple[int]): The new shape of the image
        background (int|float|complex): The value used for the frame
        use_longest (bool): Use longest dimension to calculate the border size

    Returns:
        None
    """
    simple_filter_1_1(
        in_filepath, out_filepath, mrg.reframe, new_shape, background)


# ======================================================================
def common_sampling(
        in_filepaths,
        out_filepaths=None,
        new_shape=None,
        lossless=False,
        extra_dim=True,
        fill_dim=True):
    """
    Resample images sizes and affine transformations to match the same shape.

    Note that:
        - uses 'resample' under the hood
        - the sampling / resolution / voxel size will change
        - the support space / field-of-view will NOT change

    Args:
        in_filepaths (list[str]): List of input file paths.
        out_filepaths (list[str]): List of output file paths.
        new_shape (tuple[int]): The new shape of the images
        lossless (bool): allow for lossy resampling
        extra_dim (bool): Force extra dimensions in the zoom parameters
        fill_dim (bool): Dimensions not specified are left untouched

    Returns:
        None
    """

    def combine_shape(shape_list, lossless=lossless):
        new_shape = [1] * max([len(shape) for shape in shape_list])
        shape_arr = np.ones((len(shape_list), len(new_shape))).astype(np.int)
        for i, shape in enumerate(shape_list):
            shape_arr[i, :len(shape)] = np.array(shape)
        combiner = mrb.lcm if lossless else max
        new_shape = [
            combiner(*list(shape_arr[:, i]))
            for i in range(len(new_shape))]
        return tuple(new_shape)

    # calculate new shape
    if new_shape is None:
        shape_list = []
        for in_filepath in in_filepaths:
            obj = nib.load(in_filepath)
            shape_list.append(obj.get_data().shape)
        new_shape = combine_shape(shape_list)

    # resample images
    interpolation_order = 0 if lossless else 1

    # when output files are not specified, modify inputs
    if out_filepaths is None:
        out_filepaths = in_filepaths

    for in_filepath, out_filepath in zip(in_filepaths, out_filepaths):
        # ratio should not be kept: keep_ratio_method=None
        resample(
            in_filepath, out_filepath, new_shape, None,
            interpolation_order,
            extra_dim, fill_dim)
    return out_filepaths


# ======================================================================
def common_support(
        in_filepaths,
        out_filepaths=None,
        new_shape=None,
        background=0):
    """
    Reframe images sizes (by adding border) to match the same shape.

    Note that:
        - uses 'reframe' under the hood
        - the sampling / resolution / voxel size will NOT change
        - the support space / field-of-view will change

    Args:
        in_filepaths (list[str]): List of input file paths.
        out_filepaths (list[str]): List of output file paths.
        new_shape (tuple[int]): The new shape of the images
        background (int|float|complex): The value used for the frame

    Returns:
        None
    """

    def combine_shape(shape_list):
        new_shape = [1] * max([len(shape) for shape in shape_list])
        if any([len(shape) != len(new_shape) for shape in shape_list]):
            raise IndexError('shape length must match')
        shape_arr = np.ones((len(shape_list), len(new_shape))).astype(np.int)
        for i, shape in enumerate(shape_list):
            shape_arr[i, :len(shape)] = np.array(shape)
        new_shape = [
            max(*list(shape_arr[:, i]))
            for i in range(len(new_shape))]
        return tuple(new_shape)

    # calculate new shape
    if new_shape is None:
        shape_list = []
        for in_filepath in in_filepaths:
            obj = nib.load(in_filepath)
            shape_list.append(obj.get_data().shape)
        new_shape = combine_shape(shape_list)

    if out_filepaths is None:
        out_filepaths = in_filepaths

    for in_filepath, out_filepath in zip(in_filepaths, out_filepaths):
        reframe(in_filepath, out_filepath, new_shape, background)

    return out_filepaths


# ======================================================================
def mask_threshold(
        in_filepath,
        out_filepath,
        threshold=None,
        comparison=None,
        mode=None):
    """
    Extract a mask from an array using a threshold.

    Args:
        in_filepath (str): The input file path
        out_filepath (str): The output file path
        threshold : int, float or tuple or None (optional)
            Value(s) to be used for determining the threshold.
        comparison : str or None(optional)
            A string representing the numeric relationship: [=, !=, >, <,
            >=, <=]
        mode : str or None (optional)
            Determines how to interpret / process the threshold value.
            Available values are:
            | 'absolute': use the absolute value
            | 'relative': use a value relative to dynamic range
            | 'percentile': use the value obtained from the percentiles

    Returns:
        None

    See Also:
        pymrt.segmentation.mask_threshold
    """

    def _img_mask_threshold(array, *args, **kwargs):
        return mrs.mask_threshold(array, *args, **kwargs).astype(float)

    kw_params = mrb.set_keyword_parameters(mrs.mask_threshold, locals())
    simple_filter_1_1(in_filepath, out_filepath, _img_mask_threshold,
                      **kw_params)


# ======================================================================
def find_objects(
        in_filepath,
        out_filepath,
        structure=None,
        max_label=0):
    """
    Extract labels using: pymrt.geometry.find_objects

    Args:
        in_filepath (str): The input file path
        out_filepath (str): The output file path
        structure (ndarray|None): The definition of the feature connections.
            If None, use default.
        max_label (int): Limit the number of labels to search through.

    See Also:
        pymrt.geometry.find_objects
    """

    def _find_objects(array, structure, max_label):
        labels, masks = mrs.find_objects(array, structure, max_label, False)
        return labels

    simple_filter_1_1(
        in_filepath, out_filepath,
        _find_objects, structure, max_label)


# ======================================================================
def calc_stats(
        img_filepath,
        mask_filepath=None,
        *args,
        **kwargs):
    # save_filepath=None,
    # mask_nan=True,
    # mask_inf=True,
    # mask_vals=(0.0,),
    # printing=None,
    # title=None,
    # compact=False):
    """
    Calculate statistical information (min, max, avg, std, sum).

    Args:
        img_filepath (str): The image file path
        mask_filepath (str): The mask file path
        save_filepath (str): The path where results are saved. If None,
        no output.
        mask_nan (bool): Mask NaN values.
        mask_inf (bool): Mask Inf values.
        mask_vals (iterable): List of values to mask out
        printing : bool or None (optional)
            Force printing of results, even when result is saved to file.
        title : str or None (optional)
            Title to be printed before results.
        compact : bool (optional)
            Use a compact format string for displaying results.

    Returns
        stats_dict (dict):
            - 'min': minimum value
            - 'max': maximum value
            - 'avg': average or mean
            - 'std': standard deviation
            - 'sum': summation
    """
    obj = nib.load(img_filepath)
    img = obj.get_data()
    if mask_filepath:
        obj_mask = nib.load(mask_filepath)
        mask = obj_mask.get_data().astype(bool)
    else:
        mask = slice(None)
    # if not save_filepath and printing is not False:
    #     printing = True
    # if printing:
    #     if not title:
    #         if save_filepath:
    #             title = os.path.basename(save_filepath)
    #         else:
    #             title = os.path.basename(img_filepath)
    #     print(save_filepath)
    #     stats_dict = mrb.calc_stats(
    #         img[mask], mask_nan, mask_inf, mask_vals, save_filepath, title)
    # else:
    #     stats_dict = mrb.calc_stats(
    #         img[mask], mask_nan, mask_inf, mask_vals, save_filepath, title,
    #         compact)
    return mrb.calc_stats(img[mask], *args, **kwargs)


# ======================================================================
def change_data_type(
        in_filepath,
        out_filepath,
        data_type=complex):
    """
    Change image data type.

    Args:
        in_filepath (str): The input file path
        out_filepath (str): The output file path
        data_type (dtype): The data type

    Returns:
        None
    """
    obj = nib.load(in_filepath)
    img = obj.get_data()
    save(out_filepath, img.astype(data_type), obj.get_affine())


# ======================================================================
def plot_sample2d(
        in_filepath,
        *args,
        **kwargs):
    """
    Plot a 2D sample image of a ND image.

    Uses the function: pymrt.plot.sample2d

    Args:
        in_filepath (str): The input file path
        *args (tuple): Positional arguments passed to the plot function
        **kwargs (dict): Keyword arguments passed to the plot function

    Returns:
        The result of `pymrt.plot.sample2d`

    See Also:
        pymrt.plot
    """
    obj = nib.load(in_filepath)
    img = obj.get_data()
    if 'resolution' not in kwargs:
        resolution = np.array(
            [round(x, 3) for x in obj.get_header()['pixdim'][1:img.ndim + 1]])
        kwargs.update({'resolution': resolution})
    sample, plot = mrp.sample2d(img, *args, **kwargs)
    return sample, plot


# ======================================================================
def plot_sample2d_anim(
        in_filepath,
        *args,
        **kwargs):
    """
    Plot a 2D sample image of a ND image.

    Uses the function: pymrt.plot.sample2d

    Args:
        in_filepath (str): The input file path
        *args (tuple): Positional arguments passed to the plot function
        **kwargs (dict): Keyword arguments passed to the plot function

    Returns:
        The result of `pymrt.plot.sample2d`

    See Also:
        pymrt.plot
    """
    obj = nib.load(in_filepath)
    img = obj.get_data()
    if 'resolution' not in kwargs:
        resolution = np.array(
            [round(x, 3) for x in obj.get_header()['pixdim'][1:img.ndim + 1]])
        kwargs.update({'resolution': resolution})
    mov = mrp.sample2d_anim(img, *args, **kwargs)
    return mov


# ======================================================================
def plot_histogram1d(
        in_filepath,
        mask_filepath=None,
        *args,
        **kwargs):
    """
    Plot the 1D histogram of the image using MatPlotLib.

    Uses the function: pymrt.plot.histogram1d

    Args:
        in_filepath (str): The input file path
        mask_filepath (str): The mask file path
        *args (tuple): Positional arguments passed to the plot function
        **kwargs (dict): Keyword arguments passed to the plot function

    Returns:
        The result of `pymrt.plot.histogram1d`

    See Also:
        pymrt.plot
    """
    obj = nib.load(in_filepath)
    img = obj.get_data().astype(np.double)
    if mask_filepath:
        obj_mask = nib.load(mask_filepath)
        mask = obj_mask.get_data().astype(bool)
    else:
        mask = slice(None)
    result = mrp.histogram1d(img[mask], *args, **kwargs)
    return result


# ======================================================================
def plot_histogram1d_list(
        in_filepaths,
        mask_filepath=None,
        *args,
        **kwargs):
    """
    Plot 1D overlapping histograms of images using MatPlotLib.

    Uses the function: pymrt.plot.histogram1d_list

    Args:
        in_filepaths (list[str]): The list of input file paths
        mask_filepath (str): The mask file path
        *args (tuple): Positional arguments passed to the plot function
        **kwargs (dict): Keyword arguments passed to the plot function

    Returns:
        The result of `pymrt.plot.histogram1d_list`

    See Also:
        pymrt.plot
    """
    if mask_filepath:
        obj_mask = nib.load(mask_filepath)
        mask = obj_mask.get_data().astype(bool)
    else:
        mask = slice(None)
    img_list = []
    for in_filepath in in_filepaths:
        obj = nib.load(in_filepath)
        img = obj.get_data()
        img_list.append(img[mask])
    hist, bin_edges, plot = mrp.histogram1d_list(img_list, *args, **kwargs)
    return hist, bin_edges, plot


# ======================================================================
def plot_histogram2d(
        in1_filepath,
        in2_filepath,
        mask1_filepath=None,
        mask2_filepath=None,
        *args,
        **kwargs):
    """
    Plot 2D histogram of two arrays with MatPlotLib.

    Uses the function: pymrt.plot.histogram2d

    Args:
        in1_filepath (str): The first input file path.
        in2_filepath (str): The second input file path.
        mask1_filepath (str): The first mask file path.
        mask2_filepath (str): The second mask file path.
        *args (tuple): Positional arguments passed to the plot function
        **kwargs (dict): Keyword arguments passed to the plot function

    Returns:
        The result of `pymrt.plot.histogram2d`

    See Also:
        pymrt.plot.histogram2d
    """
    obj1 = nib.load(in1_filepath)
    obj2 = nib.load(in2_filepath)
    img1 = obj1.get_data().astype(np.double)
    img2 = obj2.get_data().astype(np.double)
    if mask1_filepath:
        obj1_mask = nib.load(mask1_filepath)
        mask1 = obj1_mask.get_data().astype(bool)
    else:
        mask1 = slice(None)
    if mask2_filepath:
        obj2_mask = nib.load(mask2_filepath)
        mask2 = obj2_mask.get_data().astype(bool)
    else:
        mask2 = slice(None)
    hist2d, x_edges, y_edges, plot = \
        mrp.histogram2d(img1[mask1], img2[mask2], *args, **kwargs)

    return hist2d, x_edges, y_edges, plot


# ======================================================================
if __name__ == '__main__':
    print(__doc__)
    doctest.testmod()
