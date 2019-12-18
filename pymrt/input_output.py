#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PyMRT: useful I/O utilities.

TODO:
- improve affine support
- improve header support
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
# import scipy as sp  # SciPy (signal and image processing library)
# import matplotlib as mpl  # Matplotlib (2D/3D plotting library)
# import sympy as sym  # SymPy (symbolic CAS library)
# import PIL  # Python Image Library (image manipulation toolkit)
# import SimpleITK as sitk  # Image ToolKit Wrapper
import nibabel as nib  # NiBabel (NeuroImaging I/O Library)
# import nipy  # NiPy (NeuroImaging in Python)
# import nipype  # NiPype (NiPy Pipelines and Interfaces)
import flyingcircus as fc  # Everything you always wanted to have in Python*

# :: External Imports Submodules
# import matplotlib.pyplot as plt  # Matplotlib's pyplot: MATLAB-like syntax
# import scipy.optimize  # SciPy: Optimization Algorithms
# import scipy.integrate  # SciPy: Integrations facilities
# import scipy.constants  # SciPy: Mathematal and Physical Constants
# import scipy.ndimage  # SciPy: ND-image Manipulation
import raster_geometry  # Create/manipulate N-dim raster geometric shapes.

from nibabel.spatialimages import (
    HeaderDataError, HeaderTypeError, ImageDataError, )

# :: Local Imports
import pymrt as mrt  # Python Magnetic Resonance Tools: the multi-tool of MRI
import pymrt.util
import pymrt.plot
import pymrt.segmentation
import pymrt.animate

from pymrt import INFO, PATH
from pymrt import VERB_LVL, D_VERB_LVL, VERB_LVL_NAMES
from pymrt import elapsed, report, run_doctests
from pymrt import msg, dbg, fmt, fmtm


# ======================================================================
# :: Custom defined constants

# ======================================================================
# :: Default values usable in functions


# ======================================================================
def load(
        in_filepath,
        meta=False,
        *_args,
        **_kws):
    """
    Load a NiBabel-supported image.

    Args:
        in_filepath (str): The input file path.
        meta (bool): Include metadata.

    Returns:
        arr (np.ndarray): The array data.
        meta (dict): The metadata information.
            This is only produced if `meta` is True.
        *_args: Positional arguments for `nibabel.load()`.
        **_kws: Keyword arguments for `nibabel.load()`.

    See Also:
        nibabel.load, nibabel.get_data, nibabel.get_affine, nibabel.get_header
    """
    obj = nib.load(in_filepath)
    arr = obj.get_data()
    if meta:
        # todo: generalize to multiple image types
        meta = dict(
            affine=obj.get_affine(),
            header=obj.get_header())
        return arr, meta
    else:
        return arr


# ======================================================================
def save(
        out_filepath,
        arr,
        img_type=nib.Nifti1Image,
        *_args,
        **_kws):
    """
    Save a NiBabel-supported image.

    If the output directory does not exists, saving fails.

    Args:
        out_filepath (str): Output file path.
        arr (np.ndarray): Data to be stored.
        img_type (callable): The NiBabel image object to use for saving.
        *_args: Positional arguments for NiBabel image object.
        **_kws: Keyword arguments for NiBabel image object.

    Returns:
        None.
    """
    _args = tuple(_args) if _args else ()
    _kws = dict(_kws)
    if img_type == nib.Nifti1Image:
        if arr.dtype == bool:
            arr = arr.astype(int)
        if arr.dtype == float:
            mask = np.isnan(arr)
            arr[mask] = 0.0
        if 'affine' not in _kws:
            _kws['affine'] = np.eye(4)
        if 'header' in _kws:
            _kws['header'] = None
        if '_header' in _kws:
            _kws['header'] = _kws['_header']
            _kws.pop('_header')

    try:
        obj = img_type(arr, *_args, **_kws)
    except (HeaderDataError, HeaderTypeError, ImageDataError):
        raise ValueError('Could not create a NiBabel object.')
    else:
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
        in_filepath (str): Input file path.
        mask_filepath (str): Mask file path.
        out_filepath (str): Output file path.
        mask_val: (Number): Value of masked out voxels.

    Returns:
        None.
    """
    arr, meta = load(in_filepath, meta=True)
    mask = load(mask_filepath).astype(bool)
    arr[~mask] = mask_val
    save(out_filepath, arr, **meta)


# ======================================================================
def filter_1_1(
        in_filepath,
        out_filepath,
        func,
        *_args,
        **_kws):
    """
    Interface to generic 1-1 filter.
    filter(in_filepath) -> out_filepath

    Note that the function must return a tuple with the output
    array and the its corresponding metadata.
    If the metadata is None, it is automatically calculated.

    Args:
        in_filepath (str): Input file path.
        out_filepath (str): Output file path.
        func (callable): Filtering function.
            (arr: ndarray, aff:ndarray, hdr:header).
            func(arr, aff, hdr, *_args, *kws) -> arr, aff, hdr.
        *_args: Positional arguments for the filtering function.
        **_kws: Keyword arguments for the filtering function.

    Returns:
        None
    """
    arr, meta = load(in_filepath, meta=True)
    arr, meta = func(arr, meta, *_args, **_kws)
    save(out_filepath, arr, **meta)


# ======================================================================
def filter_n_1(
        in_filepaths,
        out_filepath,
        func,
        *_args,
        **_kws):
    """
    Interface to generic n-1 filter.
    filter(in_filepaths) -> out_filepath

    Note that the function must return a tuple with the output
    array and the its corresponding metadata.
    If the metadata is None, it is automatically calculated.

    Args:
        in_filepaths (Iterable[str]): List of input file paths.
            The metadata is taken from the last item.
        out_filepath (str): Output file path
        func (callable): Filtering function
            (arr: ndarray, aff:ndarray, hdr:header)
            func(list[arr, aff, hdr], *_args, *kws)) -> arr, aff, hdr
        *_args: Positional arguments for the filtering function.
        **_kws: Keyword arguments for the filtering function.

    Returns:
        None
    """
    arrs, metas = [], []
    for in_filepath in in_filepaths:
        arr, meta = load(in_filepath, meta=True)
        arrs.append(arr)
        metas.append(meta)
    arr, meta = func(arrs, metas, *_args, **_kws)
    save(out_filepath, arr, **meta)


# ======================================================================
def filter_n_m(
        in_filepaths,
        out_filepaths,
        func,
        *_args,
        **_kws):
    """
    Interface to generic n-m filter:
    filter(in_filepaths) -> out_filepaths

    Note that the function must return an Iterable of tuples with the output
    array and the its corresponding metadata.
    If the metadata is None, it is automatically calculated.

    Args:
        in_filepaths (Iterable[str]): List of input file paths.
            The metadata is taken from the last item.
        out_filepaths (Iterable[str]): List of output file paths
        func (callable): Filtering function
            (arr: ndarray, aff:ndarray, hdr:header)
            func(list[arr, aff, hdr], *_args, *kws)) -> list[arr, aff, hdr]
        *_args: Positional arguments for the filtering function
        **_kws: Keyword arguments for the filtering function

    Returns:
        None.
    """
    arrs, metas = [], []
    for in_filepath in in_filepaths:
        arr, meta = load(in_filepath, meta=True)
        arrs.append(arr)
        metas.append(meta)
    output_list = func(arrs, metas, *_args, **_kws)
    for (arr, meta), out_filepath in zip(output_list, out_filepaths):
        save(out_filepath, arr, **meta)


# ======================================================================
def filter_n_x(
        in_filepaths,
        out_dirpath,
        func,
        *_args,
        **_kws):
    """
    Interface to generic n-x filter:
    calculation(i_filepaths) -> o_filepaths

    Note that the function must return an Iterable of tuples with the output
    array and the its corresponding metadata.
    If the metadata is None, it is automatically calculated.
    The number of output image is not known in advance.

    Args:
        in_filepaths (Iterable[str]): List of input file paths.
            The metadata is taken from the last item.
        out_dirpath (str): Output file path template.
        func (callable): Filtering function
            (arr: ndarray, aff:ndarray, hdr:header)
            func(list[arr, aff, hdr], *_args, **_kws)) -> list[arr, aff, hdr]
        *_args: Positional arguments of the filtering function.
        **_kws: Keyword arguments of the filtering function.

    Returns:
        None.
    """
    raise NotImplementedError


# ======================================================================
def simple_filter_1_1(
        in_filepath,
        out_filepath,
        func,
        *_args,
        **_kws):
    """
    Interface to simplified 1-1 filter.
    filter(in_filepath) -> out_filepath

    Args:
        in_filepath (str): Input file path
            The affine matrix is taken from the input.
            The header is calculated automatically.
        out_filepath (str): Output file path
        func (callable): Filtering function (arr: np.ndarray)
            func(arr, *_args, *kws) -> arr
        *_args: Positional arguments of the filtering function.
        **_kws: Keyword arguments of the filtering function.

    Returns:
        None
    """
    arr, meta = load(in_filepath, meta=True)
    arr = func(arr, *_args, **_kws)
    save(out_filepath, arr, **meta)


# ======================================================================
def simple_filter_n_1(
        in_filepaths,
        out_filepath,
        func,
        *_args,
        **_kws):
    """
    Interface to simplified n-1 filter.
    filter(in_filepaths) -> out_filepath

    Args:
        in_filepaths (Iterable[str]): List of input file paths.
            The shape of each array must be identical.
            The metadata is taken from the last item.
        out_filepath (str): Output file path.
        func (callable): Filtering function (arr: ndarray)
            func(list[arr], *_args, **_kws)) -> arr
        *_args: Positional arguments of the filtering function.
        **_kws: Keyword arguments of the filtering function.

    Returns:
        None.
    """
    arrs, metas = [], []
    for in_filepath in in_filepaths:
        arr, meta = load(in_filepath, meta=True)
        arrs.append(arr)
        metas.append(meta)
    arr = func(arrs, *_args, **_kws)
    meta = metas[-1]  # the metadata of the first image
    save(out_filepath, arr, **meta)


# ======================================================================
def simple_filter_nn_1(
        in_filepaths,
        out_filepath,
        func,
        *_args,
        **_kws):
    """
    Interface to simplified n-1 filter.

    Supports a different signature for `func` compared to `_n_1`.
    Specifically, the input arrays are listed as arguments.
    This is useful when the number of input arrays must be forced.

    filter(in_filepaths) -> out_filepath

    Args:
        in_filepaths (Iterable[str]): List of input file paths.
            The metadata is taken from the last item.
        out_filepath (str): Output file path.
        func (callable): Filtering function (arr: ndarray)
            func(*_args, **_kws)) -> arr
        *_args: Positional arguments of the filtering function.
        **_kws: Keyword arguments of the filtering function.

    Returns:
        None.
    """
    arrs, metas = [], []
    for in_filepath in in_filepaths:
        arr, meta = load(in_filepath, meta=True)
        arrs.append(arr)
        metas.append(meta)
    arr = func(*(arrs + list(_args)), **_kws)
    meta = metas[-1]  # the affine of the last image
    save(out_filepath, arr, **meta)


# ======================================================================
def simple_filter_inn_1(
        in_filepaths,
        out_filepath,
        func,
        *_args,
        **_kws):
    """
    Interface to simplified n-1 filter.

    Supports a different signature for `func` compared to `_n_1`.
    Specifically, two arrays are expected where the first one is iteratively
    updated with the second one looping through the list
    This is useful when loading the inputs must be loaded one by one to avoid
    memory constraints.

    filter(in_filepaths) -> out_filepath

    Args:
        in_filepaths (Iterable[str]): List of input file paths.
            The metadata is taken from the first item.
        out_filepath (str): Output file path.
        func (callable): Filtering function (arr: ndarray)
            func(result, arr_i, *_args, *kws)) -> result
        *_args: Positional arguments of the filtering function.
        **_kws: Keyword arguments of the filtering function.

    Returns:
        None.
    """
    res, meta = load(in_filepaths[0], meta=True)
    for i, in_filepath in enumerate(in_filepaths[1:]):
        res = func(res, load(in_filepath), i, *_args, **_kws)
    save(out_filepath, res, **meta)


# ======================================================================
def simple_filter_1_m(
        in_filepath,
        out_filepaths,
        func,
        *_args,
        **_kws):
    """
    Interface to simplified n-m filter.
    filter(in_filepaths) -> out_filepaths

    Args:
        in_filepath (str): Input file path
            The metadata information is taken from the input.
        out_filepaths (Iterable[str]): List of output file paths.
        func (callable): Filtering function (arr: ndarray)
            func(list[arr], *_args, **_kws) -> list[ndarray]
        *_args: Positional arguments of the filtering function
        **_kws: Keyword arguments of the filtering function

    Returns:
        None.
    """
    arr, meta = load(in_filepath, meta=True)
    o_arrs = func(arr, *_args, **_kws)
    for arr, out_filepath in zip(o_arrs, out_filepaths):
        save(out_filepath, arr, **meta)


# ======================================================================
def simple_filter_n_m(
        in_filepaths,
        out_filepaths,
        func,
        *_args,
        **_kws):
    """
    Interface to simplified n-m filter.
    filter(in_filepaths) -> out_filepaths

    Args:
        in_filepaths (Iterable[str]): List of input file paths.
            The metadata information is taken from the last item.
        out_filepaths (Iterable[str]): List of output file paths.
        func (callable): Filtering function (arr: ndarray)
            func(list[arr], *_args, *kws) -> list[ndarray]
        *_args: Positional arguments of the filtering function
        **_kws: Keyword arguments of the filtering function

    Returns:
        None.
    """
    i_arrs, metas = [], []
    for in_filepath in in_filepaths:
        arr, meta = load(in_filepath, meta=True)
        i_arrs.append(arr)
        metas.append(meta)
    o_arrs = func(i_arrs, *_args, **_kws)
    meta = metas[-1]  # the affine of the last image
    for arr, out_filepath in zip(o_arrs, out_filepaths):
        save(out_filepath, arr, **meta)


# ======================================================================
def simple_filter_nn_m(
        in_filepaths,
        out_filepaths,
        func,
        *_args,
        **_kws):
    """
    Interface to simplified n-m filter.
    filter(in_filepaths) -> out_filepaths

    Supports a different signature for `func` compared to `_n_1`.
    Specifically, the input arrays are listed as arguments.
    This is useful when the number of input arrays must be forced.

    Args:
        in_filepaths (Iterable[str]): List of input file paths.
            The metadata information is taken from the last item.
        out_filepaths (Iterable[str]): List of output file paths.
        func (callable): Filtering function (arr: ndarray)
            func(list[arr], *_args, *kws) -> list[ndarray]
        *_args: Positional arguments of the filtering function
        **_kws: Keyword arguments of the filtering function

    Returns:
        None.
    """
    i_arrs, metas = [], []
    for in_filepath in in_filepaths:
        arr, meta = load(in_filepath, meta=True)
        i_arrs.append(arr)
        metas.append(meta)
    o_arrs = func(*(i_arrs + list(_args)), **_kws)
    meta = metas[-1]  # the affine of the last image
    for arr, out_filepath in zip(o_arrs, out_filepaths):
        save(out_filepath, arr, **meta)


# ======================================================================
def simple_filter_1_x(
        in_filepath,
        out_basepath,
        func,
        out_filename_template='{base}_{name}{ext}',
        *_args,
        **_kws):
    """
    Interface to simplified n-m filter.
    filter(in_filepaths) -> out_filepaths

    Args:
        in_filepath (str): Input file path
            The metadata information is taken from the input.
        out_basepath (str): Base output directory path.
            Additional subdirectories can be specified through the
            `out_filename_template` parameter.
        out_filename_template (str):
            If subdirectories do not exists, they are created.
            The following substitution are valid:
             - `{path}`: the input directory path.
             - `{base}`: the input file name without extension.
             - `{ext}`: the input file extension.
             - `{name}`: the result identifier from the computing function.
        func (callable): Filtering function (arr: ndarray)
            func(list[arr], *_args, *kws) -> list[ndarray]
        *_args: Positional arguments of the filtering function
        **_kws: Keyword arguments of the filtering function

    Returns:
        None.
    """
    arr, meta = load(in_filepath, meta=True)
    results = func(arr, *_args, **_kws)
    path, base, ext = fc.base.split_path(in_filepath)
    for i, (name, arr) in enumerate(results.items()):
        out_filepath = os.path.join(
            out_basepath, fmtm(out_filename_template))
        out_dirpath = os.path.dirname(out_filepath)
        if not os.path.isdir(out_dirpath):
            os.makedirs(out_dirpath)
        save(out_filepath, arr, **meta)


# ======================================================================
def simple_filter_n_x(
        in_filepaths,
        out_dirpath,
        func,
        *_args,
        **_kws):
    """
    Interface to simplified n-x filter.
    filter(in_filepaths) -> out_filepaths

    Note that the number of output image is not known in advance.

    Args:
        in_filepaths (Iterable[str]): List of input file paths.
            The metadata information is taken from the last item.
        out_filepaths (Iterable[str]): List of output file paths.
        func (callable): Filtering function (arr: ndarray).
            func(list[arr], *_args, *kws) -> list[ndarray]
        *_args: Positional arguments of the filtering function.
        **_kws: Keyword arguments of the filtering function.

    Returns:
        None.
    """
    raise NotImplementedError


# ======================================================================
def simple_filter_nn_x(
        in_filepaths,
        out_dirpath,
        func,
        *_args,
        **_kws):
    """
    Interface to simplified n-x filter.
    filter(in_filepaths) -> out_filepaths

    Note that the number of output image is not known in advance.

    Args:
        in_filepaths (Iterable[str]): List of input file paths.
            The metadata information is taken from the last item.
        out_filepaths (Iterable[str]): List of output file paths.
        func (callable): Filtering function (arr: ndarray).
            func(*_args, *kws) -> list[ndarray]
        *_args: Positional arguments of the filtering function.
        **_kws: Keyword arguments of the filtering function.

    Returns:
        None.
    """
    raise NotImplementedError


# ======================================================================
def stack(
        in_filepaths,
        out_filepath,
        axis=-1):
    """
    Join images together.

    Args:
        in_filepaths (Iterable[str]): List of input file paths.
            The metadata information is taken from the last item.
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
            The metadata information is taken from the input.
        out_dirpath (str): Path to directory where to store results.
        out_basename (str): Output filename (without extension).
        axis (int): Joining axis of orientation.
            Must be a valid index for the input shape

    Returns:
        out_filepaths (Iterable[str]): List of output file paths.
    """
    # todo: refactor to use simple_filter_n_y
    if not out_dirpath or not os.path.exists(out_dirpath):
        out_dirpath = os.path.dirname(in_filepath)
    if not out_basename:
        out_basename = fc.base.change_ext(
            os.path.basename(in_filepath), '', mrt.util.EXT['niz'])
    out_filepaths = []

    arr, meta = load(in_filepath, meta=True)
    # split data
    arrs = np.split(arr, arr.shape[axis], axis)
    # save data to output
    for i, image in enumerate(arrs):
        i_str = str(i).zfill(len(str(len(arrs))))
        out_filepath = os.path.join(
            out_dirpath,
            fc.base.change_ext(out_basename + '-' + i_str,
                               mrt.util.EXT['niz'], ''))
        save(out_filepath, image, **meta)
        out_filepaths.append(out_filepath)
    return out_filepaths


# ======================================================================
def frame(
        in_filepath,
        out_filepath,
        borders,
        background=0,
        use_longest=True):
    """
    Add a border frame to the image (same resolution / voxel size)

    Args:
        in_filepath (str): Input file path.
        out_filepath (str): Output file path.
        borders (int|float|Iterable[int|float]): The border size(s).
            If int, this is in units of pixels.
            If float, this is proportional to the initial array shape.
            If int or float, uses the same value for all dimensions.
            If Iterable, the size must match `arr` dimensions.
            If 'use_longest' is True, use the longest dimension for the
            calculations.
        background (int|float): The background value to be used for the frame.
        use_longest (bool): Use longest dimension to get the border size.

    Returns:
        None
    """
    simple_filter_1_1(
        in_filepath, out_filepath, fc.extra.frame,
        borders, background, use_longest)


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
        new_shape (int|Iterable[int]): The shape of the output array.
            If int, uses the same value for all dimensions.
            If Iterable, the size must match `arr` dimensions.
            Additionally, each value of `new_shape` must be greater than or
            equal to the corresponding dimensions of `arr`.
        background (int|float): The background value to be used for the frame.

    Returns:
        None
    """
    simple_filter_1_1(
        in_filepath, out_filepath, fc.extra.reframe,
        new_shape, background)


# ======================================================================
def multi_reframe(
        in_filepaths,
        out_filepath,
        new_shape=None,
        background=0.0,
        dtype=None):
    """
    Reframe arrays (by adding border) to match the same shape.

    Note that:
        - uses 'reframe' under the hood
        - the sampling / resolution / voxel size will NOT change
        - the support space / field-of-view will change

    Args:
        in_filepaths (Iterable[str]): List of input file paths.
        out_filepath (Iterable[str]): The output file path.
        new_shape (Iterable[int]): The new base shape of the arrays.
        background (Number): The background value for the frame.
        dtype (data-type): Desired output data-type.
            If None, its guessed from dtype of arrs.
            See `np.ndarray()` for more.

    Returns:
        None
    """
    simple_filter_n_1(
        in_filepaths, out_filepath,
        fc.extra.multi_reframe,
        new_shape=new_shape, background=background, dtype=dtype)


# ======================================================================
def zoom(
        in_filepath,
        out_filepath,
        factors,
        window=None,
        interp_order=0,
        extra_dim=True,
        fill_dim=True):
    """
    Zoom the image with a specified magnification factor.

    Args:
        in_filepath (str): Input file path.
        out_filepath (str): Output file path.
        factors (int|float|Iterable[int|float]): The zoom factor(s).
            If int or float, uses isotropic factor along all axes.
            If Iterable, its size must match the number of dims of `arr`.
            Values larger than 1 increase `arr` size along the axis.
            Values smaller than 1 decrease `arr` size along the axis.
        window (int|Iterable[int]|None): Uniform pre-filter window size.
            This is the size of the window for the uniform filter using
            `sp.ndimage.uniform_filter()`.
            If Iterable, its size must match the number of dims of `arr`.
            If int, uses an isotropic window with the specified size.
            If None, the window is calculated automatically from the `zoom`
            parameter.
        interp_order (int): Order of the spline interpolation.
            0: nearest. Accepted range: [0, 5].
        extra_dim (bool): Force extra dimensions in the zoom parameters.
        fill_dim (bool): Dimensions not specified are left untouched.

    Returns:
        None.
    """
    simple_filter_1_1(
        in_filepath, out_filepath, fc.extra.zoom,
        factors, window, interp_order, extra_dim, fill_dim)


# ======================================================================
def resample(
        in_filepath,
        out_filepath,
        new_shape,
        aspect=None,
        window=None,
        interp_order=0,
        extra_dim=True,
        fill_dim=True):
    """
    Reshape the image to a new shape (different resolution / voxel size).

    Warning: use `scipy.ndimage.zoom` internally!
    For downsampling applications, this might not be appropriate.

    Args:
        in_filepath (str): Input file path.
        out_filepath (str): Output file path.
        new_shape (tuple[int|None]): New dimensions of the array.
        aspect (callable|Iterable[callable]|None): Zoom shape manipulation.
            Useful for obtaining specific aspect ratio effects.
            This is passed to `fc.extra.shape2zoom()`.
        window (int|Iterable[int]|None): Uniform pre-filter window size.
            This is the size of the window for the uniform filter using
            `sp.ndimage.uniform_filter()`.
            If Iterable, its size must match the number of dims of `arr`.
            If int, uses an isotropic window with the specified size.
            If None, the window is calculated automatically from `new_shape`.
        interp_order (int|None): Order of the spline interpolation.
            0: nearest. Accepted range: [0, 5].
        extra_dim (bool): Force extra dimensions in the zoom parameters.
        fill_dim (bool): Dimensions not specified are left untouched.

    Returns:
        None.
    """
    simple_filter_1_1(
        in_filepath, out_filepath, fc.extra.resample,
        new_shape, aspect, window, extra_dim, fill_dim, interp_order)


# ======================================================================
def multi_resample(
        in_filepaths,
        out_filepath,
        new_shape=None,
        lossless=False,
        window=None,
        interp_order=0,
        extra_dim=True,
        fill_dim=True,
        dtype=None):
    """
    Resample arrays to match the same shape.

    Note that:
        - uses 'flyingcircus.extra.multi_resample()' internally;
        - the sampling / resolution / voxel size will change;
        - the support space / field-of-view will NOT change.

    Args:
        in_filepaths (Iterable[str]): List of input file paths.
        out_filepath (Iterable[str]): The output file path.
        new_shape (Iterable[int]): The new shape of the arrays.
        lossless (bool): allow for lossy resampling.
        window (int|Iterable[int]|None): Uniform pre-filter window size.
            This is the size of the window for the uniform filter using
            `sp.ndimage.uniform_filter()`.
            If Iterable, its size must match the number of dims of `arr`.
            If int, uses an isotropic window with the specified size.
            If None, the window is calculated automatically from `new_shape`.
        interp_order (int|None): Order of the spline interpolation.
            0: nearest. Accepted range: [0, 5].
        extra_dim (bool): Force extra dimensions in the zoom parameters.
        fill_dim (bool): Dimensions not specified are left untouched.
        dtype (data-type): Desired output data-type.
            If None, its guessed from dtype of arrs.
            See `np.ndarray()` for more.

    Returns:
        None
    """
    simple_filter_n_1(
        in_filepaths, out_filepath,
        fc.extra.multi_resample,
        new_shape=new_shape, lossless=lossless, window=window,
        interp_order=interp_order, extra_dim=extra_dim, fill_dim=fill_dim,
        dtype=dtype)


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
            | 'relative': use a value relative to values interval
            | 'percentile': use the value obtained from the percentiles

    Returns:
        None

    See Also:
        segmentation.mask_threshold
    """
    kw_params = fc.base.set_func_kws(
        mrt.segmentation.mask_threshold, locals())
    simple_filter_1_1(
        in_filepath, out_filepath, mrt.segmentation.mask_threshold,
        **kw_params)


# ======================================================================
def find_objects(
        in_filepath,
        out_filepath,
        structure=None,
        max_label=0):
    """
    Extract labels using: `pymrt.segmentation.find_objects()`

    Args:
        in_filepath (str): The input file path
        out_filepath (str): The output file path
        structure (ndarray|None): The definition of the feature connections.
            If None, use default.
        max_label (int): Limit the number of labels to search through.

    See Also:
        - pymrt.segmentation.find_objects()
    """

    def _find_objects(array, structure_, max_label_):
        labels, masks = mrt.segmentation.find_objects(
            array, structure_, max_label_, False)
        return labels

    simple_filter_1_1(
        in_filepath, out_filepath,
        _find_objects, structure, max_label)


# ======================================================================
def calc_stats(
        arr_filepath,
        mask_filepath=None,
        *_args,
        **_kws):
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
        arr_filepath (str): The image file path.
        mask_filepath (str): The mask file path.
        *_args: Positional arguments for the `calc_stats()`.
        **_kws: Keyword arguments for the `calc_stats()`.

    Returns
        stats_dict (dict):
            - 'min': minimum value
            - 'max': maximum value
            - 'avg': average or mean
            - 'std': standard deviation
            - 'sum': summation

    See Also:
        utils.calc_stats
    """
    arr = load(arr_filepath)
    mask = load(mask_filepath).astype(bool) if mask_filepath else None
    return fc.extra.calc_stats(
        arr[mask] if mask is None else arr, *_args, **_kws)


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
    arr, meta = load(in_filepath, meta=True)
    save(
        out_filepath, arr.astype(data_type), **meta)


# ======================================================================
def isum(
        in_filepaths,
        out_filepath):
    """
    Compute the sum of multiple arrays.

    This is implemented using a memory-efficienty iterative approach.
    Inputs must have the same shape.

    Args:
        in_filepaths (str): The input file path.
        out_filepath (str): The output file path.

    Returns:
        None.
    """
    simple_filter_inn_1(in_filepaths, out_filepath, lambda r, x, i: r + x)


# ======================================================================
def imean(
        in_filepaths,
        out_filepath):
    """
    Compute the mean of multiple arrays.

    This is implemented using a memory-efficienty iterative approach.
    Inputs must have the same shape.

    Args:
        in_filepaths (str): The input file path.
        out_filepath (str): The output file path.

    Returns:
        None.
    """
    simple_filter_inn_1(
        in_filepaths, out_filepath,
        lambda r, x, i: (r * (i + 1) + x) / (i + 2))


# ======================================================================
def plot_sample2d(
        in_filepath,
        *_args,
        **_kws):
    """
    Plot a 2D sample image of a ND image.

    Uses the function: `plot.sample2d`.

    Args:
        in_filepath (str): The input file path
        *_args: Positional arguments for the plot function.
        **_kws: Keyword arguments for the plot function.

    Returns:
        The result of `plot.sample2d`

    See Also:
        pymrt.plot
    """
    obj = nib.load(in_filepath)
    arr = obj.get_data()
    if 'resolution' not in _kws:
        resolution = np.array(
            [round(x, 3) for x in obj.get_header()['pixdim'][1:arr.ndim + 1]])
        _kws.update({'resolution': resolution})
    result = mrt.plot.sample2d(arr, *_args, **_kws)
    return result


# ======================================================================
def plot_sample2d_anim(
        in_filepath,
        *_args,
        **_kws):
    """
    Plot a 2D sample image of a ND image.

    Uses the function: `plot.sample2d()`

    Args:
        in_filepath (str): The input file path
        *_args: Positional arguments for the plot function
        **_kws: Keyword arguments for the plot function

    Returns:
        The result of `plot.sample2d()`

    See Also:
        plot
    """
    obj = nib.load(in_filepath)
    arr = obj.get_data()
    if 'resolution' not in _kws:
        resolution = np.array(
            [round(x, 3) for x in obj.get_header()['pixdim'][1:arr.ndim + 1]])
        _kws.update({'resolution': resolution})
    mov = mrt.animate.sample2d(arr, *_args, **_kws)
    return mov


# ======================================================================
def plot_histogram1d(
        in_filepath,
        mask_filepath=None,
        *_args,
        **_kws):
    """
    Plot the 1D histogram of the image using MatPlotLib.

    Uses the function: `plot.histogram1d()`

    Args:
        in_filepath (str): The input file path
        mask_filepath (str): The mask file path
        *_args: Positional arguments for the plot function
        **_kws: Keyword arguments for the plot function

    Returns:
        The result of `plot.histogram1d()`

    See Also:
        plot
    """
    obj = nib.load(in_filepath)
    arr = obj.get_data().astype(np.double)
    mask = load(mask_filepath).astype(bool) if mask_filepath else None
    result = mrt.plot.histogram1d(
        arr[mask] if mask is not None else arr, *_args, **_kws)
    return result


# ======================================================================
def plot_histogram1d_list(
        in_filepaths,
        mask_filepath=None,
        *_args,
        **_kws):
    """
    Plot 1D overlapping histograms of images using MatPlotLib.

    Uses the function: `plot.histogram1d_list()`

    Args:
        in_filepaths (Iterable[str]): The list of input file paths
        mask_filepath (str): The mask file path
        *_args: Positional arguments for the plot function
        **_kws: Keyword arguments for the plot function

    Returns:
        The result of `plot.histogram1d_list()`

    See Also:
        plot
    """
    mask = load(mask_filepath).astype(bool) if mask_filepath else None
    arr_list = []
    for in_filepath in in_filepaths:
        obj = nib.load(in_filepath)
        arr = obj.get_data()
        arr_list.append(arr[mask] if mask is not None else arr)
    result = mrt.plot.histogram1d_list(arr_list, *_args, **_kws)
    return result


# ======================================================================
def plot_histogram2d(
        in1_filepath,
        in2_filepath,
        mask1_filepath=None,
        mask2_filepath=None,
        *_args,
        **_kws):
    """
    Plot 2D histogram of two arrays with MatPlotLib.

    Uses the function: `plot.histogram2d()`

    Args:
        in1_filepath (str): The first input file path.
        in2_filepath (str): The second input file path.
        mask1_filepath (str): The first mask file path.
        mask2_filepath (str): The second mask file path.
        *_args: Positional arguments for the plot function
        **_kws: Keyword arguments for the plot function

    Returns:
        The result of `plot.histogram2d()`

    See Also:
        plot.histogram2d
    """
    obj1 = nib.load(in1_filepath)
    obj2 = nib.load(in2_filepath)
    arr1 = obj1.get_data().astype(np.double)
    arr2 = obj2.get_data().astype(np.double)
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
    result = \
        mrt.plot.histogram2d(arr1[mask1], arr2[mask2], *_args, **_kws)

    return result


# ======================================================================
elapsed(__file__[len(PATH['base']) + 1:])

# ======================================================================
if __name__ == '__main__':
    run_doctests(__doc__)
