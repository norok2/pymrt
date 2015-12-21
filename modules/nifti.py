#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
mr_lib: useful NIfTI-1 utilities.

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
import matplotlib.pyplot as plt  # Matplotlib's pyplot: MATLAB-like syntax
# import mayavi.mlab as mlab  # Mayavi's mlab: MATLAB-like syntax
# import scipy.optimize  # SciPy: Optimization Algorithms
# import scipy.integrate  # SciPy: Integrations facilities
# import scipy.constants  # SciPy: Mathematal and Physical Constants
import scipy.ndimage  # SciPy: ND-image Manipulation
# :: Local Imports
import mri_tools.modules.base as mrb
import mri_tools.modules.geometry as mrg
import mri_tools.modules.plot as mrp
import mri_tools.modules.segmentation as mrs

# from mri_tools import INFO
# from mri_tools import VERB_LVL
# from mri_tools import D_VERB_LVL
# from mri_tools import get_first_line

# ======================================================================
# :: Custom defined constants

# ======================================================================
# :: Default values usable in functions.
EXT_UNCOMPRESSED = 'nii'
EXT_COMPRESSED = 'nii.gz'
D_EXT = EXT_COMPRESSED
D_DICOM_RANGE = (0.0, 4096.0)


# ======================================================================
def filename_noext(filename):
    """
    Remove NIfTI-1 extension from filename.

    Parameters
    ==========
    filename : str
        The filename from which the extension is to be removed.

    Returns
    =======
    filename_noext : str
        The filename without the NIfTI-1 extension.

    """
    ext = None
    for test_ext in [EXT_UNCOMPRESSED, EXT_COMPRESSED]:
        if filename.endswith(test_ext):
            ext = test_ext
    if ext:
        filename_noext = filename[:-(len(ext) + 1)]
    else:
        filename_noext = filename
    return filename_noext


# ======================================================================
def filename_addext(
        filename,
        compressed=True):
    """
    Add NIfTI-1 extension to filename.

    Parameters
    ==========
    filename : str
        The filename to which the extension is to be added.

    Returns
    =======
    filename_noext : str
        The filename with the NIfTI-1 extension.

    """
    if compressed:
        filename += mrb.add_extsep(EXT_COMPRESSED)
    else:
        filename += mrb.add_extsep(EXT_UNCOMPRESSED)
    return filename


def load(
        in_filepath,
        full=False):
    """

    Args:
        in_filepath:
        full:

    Returns:

    """
    nii = nib.load(in_filepath)
    if full:
        return nii.get_data(), nii.get_affine(), nii.get_header()
    else:
        return nii.get_data()


# ======================================================================
def save(
        out_filepath,
        array,
        affine=None,
        header=None):
    """
    Interface to NIfTI-1 generic image creation.

    Parameters
    ==========
    out_filepath : str
        Output file path
    array : ndarray
        Data to be stored.
    affine : ndarray (optional)
        Affine transformation.
    header : NIfTI-1-header (optional)
        Header of the image (refer to NiBabel).

    Returns
    =======
    None.

    """
    if affine is None:
        affine = np.eye(4)
    nii = nib.Nifti1Image(array, affine, header)
    nii.to_filename(out_filepath)


# ======================================================================
def masking(
        in_filepath,
        mask_filepath,
        out_filepath,
        mask_val=np.nan):
    """
    Interface to NIfTI-1 generic image creation.

    Parameters
    ==========
    in_filepath : str
        Input file path.
    mask_filepath : str
        Mask file path.
    out_filepath : str
        Output file path.
    mask_val : int or float
        Value of masked out voxels.

    Returns
    =======
    None.

    """
    img_nii = nib.load(in_filepath)
    mask_nii = nib.load(mask_filepath)
    img = img_nii.get_data()
    mask = mask_nii.get_data()
    mask = mask.astype(bool)
    img[~mask] = mask_val
    save(out_filepath, img, img_nii.get_affine())


# ======================================================================
def filter(
        in_filepath,
        out_filepath,
        func,
        *args,
        **kwargs):
    """
    Interface to NIfTI-1 generic filter:
    calculation(i_filepath) -> o_filepath

    Parameters
    ==========
    in_filepath : str
        Input file path.
    out_filepath : str
        Output file path.
    func : function
        | Filtering function (img: ndarray, aff: ndarray, hdr: NIfTI-1 header):
        | func(img, aff, hdr, *args, *kwargs) -> img, aff, hdr
    args : tuple (optional)
    kwargs : dict (optional)
        Additional arguments to be passed to the filtering function.

    Returns
    =======
    None.

    """
    nii = nib.load(in_filepath)
    img, aff, hdr = func(
        nii.get_data(), nii.get_affine(), nii.get_header(), *args, **kwargs)
    save(out_filepath, img, aff, hdr)


# ======================================================================
def filter_n(
        in_filepath_list,
        out_filepath,
        func,
        *args,
        **kwargs):
    """
    Interface to NIfTI-1 generic filter:
    calculation(i_filepath_list) -> o_filepath

    Note that only

    Parameters
    ==========
    in_filepath_list : str list
        List of input file paths (affine is taken from last item).
    out_filepath : str
        Output file path.
    func : function
        | Filtering function (img: ndarray, aff: ndarray, hdr: NIfTI-1 header):
        | func((img, aff, hdr) list, *args, *kwargs) -> img, aff, hdr
    args : tuple (optional)
    kwargs : dict (optional)
        Additional arguments to be passed to the filtering function.

    Returns
    =======
    None.

    """
    input_list = []
    for in_filepath in in_filepath_list:
        nii = nib.load(in_filepath)
        input_list.append(nii.get_data(), nii.get_affine(), nii.get_header())
    img, aff, hdr = func(input_list, *args, **kwargs)
    save(out_filepath, img, aff, hdr)


# ======================================================================
def filter_n_m(
        in_filepath_list,
        out_filepath_list,
        func,
        *args,
        **kwargs):
    """
    Interface to NIfTI-1 generic filter:
    calculation(i_filepath_list) -> o_filepath_list

    Parameters
    ==========
    in_filepath_list : str list
        List of input file paths (affine is taken from last item).
    out_filepath_list : str
        List of output file paths.
    func : function
        | Filtering function (img: ndarray, aff: ndarray, hdr: NIfTI-1 header):
        | func((img, aff, hdr) list, *args, *kwargs) -> (img, aff, hdr) list
    args : tuple (optional)
    kwargs : dict (optional)
        Additional arguments to be passed to the filtering function.

    Returns
    =======
    None.

    """
    input_list = []
    for in_filepath in in_filepath_list:
        nii = nib.load(in_filepath)
        input_list.append(nii.get_data(), nii.get_affine(), nii.get_header())
    output_list = func(input_list, *args, **kwargs)
    for (img, aff, hdr), out_filepath in zip(output_list, out_filepath_list):
        save(out_filepath, img, aff, hdr)


# ======================================================================
def filter_n_x(
        in_filepath_list,
        out_filepath_template,
        func,
        *args,
        **kwargs):
    # todo: implement
    pass


# ======================================================================
def simple_filter(
        in_filepath,
        out_filepath,
        func,
        *args,
        **kwargs):
    """
    Interface to simplified NIfTI-1 generic filter:
    calculation(i_filepath) -> o_filepath

    Affine is obtained from the input image.
    The header is calculated automatically.

    Parameters
    ==========
    in_filepath : str
        Input file path.
    out_filepath : str
        Output file path.
    func : function
        | Filtering function (img: ndarray, aff: ndarray, hdr: NIfTI-1 header):
        | func(img, aff, hdr, *args, *kwargs) -> img, aff, hdr
    args : tuple (optional)
    kwargs : dict (optional)
        Additional arguments to be passed to the filtering function.

    Returns
    =======
    None.

    """
    nii = nib.load(in_filepath)
    img = func(nii.get_data(), *args, **kwargs)
    aff = nii.get_affine()
    save(out_filepath, img, aff)


# ======================================================================
def simple_filter_n(
        in_filepath_list,
        out_filepath,
        func,
        *args,
        **kwargs):
    """
    Interface to simplified NIfTI-1 generic filter:
    calculation(i_filepath_list) -> o_filepath

    Affine is obtained from the first input image.
    The header is calculated automatically.

    Parameters
    ==========
    in_filepath_list : str list
        List of input file paths (affine is taken from last item).
    out_filepath : str
        Output file path.
    func : function
        | Filtering function (img: ndarray, aff: ndarray, hdr: NIfTI-1 header):
        | func((img, aff, hdr) list, *args, *kwargs) -> img, aff, hdr
    args : tuple (optional)
    kwargs : dict (optional)
        Additional arguments to be passed to the filtering function.

    Returns
    =======
    None.

    """
    img_list = []
    aff_list = []
    for in_filepath in in_filepath_list:
        nii = nib.load(in_filepath)
        img_list.append(nii.get_data())
        aff_list.append(nii.get_affine())
    img = func(img_list, *args, **kwargs)
    aff = aff_list[0]  # the affine of the first image
    save(out_filepath, img, aff)


# ======================================================================
def simple_filter_n_m(
        in_filepath_list,
        out_filepath_list,
        func,
        *args,
        **kwargs):
    """
    Interface to simplified NIfTI-1 generic filter:
    calculation(i_filepath_list) -> o_filepath_list

    Parameters
    ==========
    in_filepath_list : str list
        List of input file paths (affine is taken from last item).
    out_filepath_list : str
        List of output file paths.
    func : function
        | Filtering function (img: ndarray, aff: ndarray, hdr: NIfTI-1 header):
        | func((list[ndarray], *args, *kwargs) -> list[ndarray]
    args : tuple (optional)
    kwargs : dict (optional)
        Additional arguments to be passed to the filtering function.

    Returns
    =======
    None.

    """
    i_img_list = []
    aff_list = []
    for in_filepath in in_filepath_list:
        nii = nib.load(in_filepath)
        i_img_list.append(nii.get_data())
        aff_list.append(nii.get_affine())
    o_img_list = func(i_img_list, *args, **kwargs)
    aff = aff_list[0]  # the affine of the first image
    for img, out_filepath in zip(o_img_list, out_filepath_list):
        save(out_filepath, img, aff)


# ======================================================================
def simple_filter_n_x(
        in_filepath_list,
        out_filepath_template,
        func,
        *args,
        **kwargs):
    # todo: implement
    pass


# ======================================================================
def img_join(
        in_filepath_list,
        out_filepath,
        axis=-1):
    """
    Join NIfTI-1 images together.

    Parameters
    ==========
    in_filepath_list : str list
        List of input file paths (affine is taken from last item).
    out_filepath : str
        Output file path.
    axis : int [0,N] (optional)
        Orientation along which array is join in the N-dim space.

    Returns
    =======
    None.

    """
    simple_filter_n(in_filepath_list, out_filepath, mrb.ndstack, axis)


# ======================================================================
def img_split(
        in_filepath,
        out_dirpath=None,
        out_basename=None,
        axis=-1):
    """
    Join NIfTI-1 images together.

    Parameters
    ==========
    in_filepath : str
        Input file path (affine is copied from input).
    axis : int [0,N] (optional)
        Orientation along which array is split in the N-dim space.
    out_dirpath : str
        Path to directory where to store results.
    out_filename : str (optional)
        Filename (without extension) where to store results.

    Returns
    =======
    out_filepath_list : str list
        Output file path list.

    """
    # todo: reimplement using simple_filter_n_x
    if not out_dirpath or not os.path.exists(out_dirpath):
        out_dirpath = os.path.dirname(in_filepath)
    if not out_basename:
        out_basename = filename_noext(os.path.basename(in_filepath))
    out_filepath_list = []
    # load source image
    img_nii = nib.load(in_filepath)
    img = img_nii.get_data()
    # split data
    img_list = mrb.ndsplit(img, axis)
    # save data to output
    for idx, image in enumerate(img_list):
        out_filepath = os.path.join(
            out_dirpath,
            filename_addext(out_basename + '-' +
                            str(idx).zfill(len(str(len(img_list))))))
        save(out_filepath, image, img_nii.get_affine())
        out_filepath_list.append(out_filepath)
    return out_filepath_list


# ======================================================================
def img_zoom(
        in_filepath,
        out_filepath,
        zoom=1.0,
        interpolation_order=1,
        extra_dim=True,
        fill_dim=True):
    """
    Zoom the image with a specified magnification factor.

    Parameters
    ==========
    in_filepath : str
        Input file path.
    out_filepath : str
        Output file path.
    zoom : float or sequence
        The zoom factor along the axes.
    interpolation_order : int, optional
        Order of the spline interpolation. 0: nearest. Accepted range: [0, 5].
    force_extra_dim : boolean, optional
        Force extra dimensions in the zoom parameters.
    autofill_dim : boolean, optional
        Dimensions not specified are left untouched.

    Returns
    =======
    None.

    See Also
    ========
    mri_tools.modules.geometry.
    """

    def _zoom(array, zoom, interpolation_order, extra_dim, fill_dim):
        zoom, shape = mrg.zoom_prepare(zoom, array.shape, extra_dim, fill_dim)
        array = sp.ndimage.zoom(
            array.reshape(shape), zoom, order=interpolation_order)
        aff_transform = np.diag(1.0 / np.array(zoom[:3] + [1.0]))
        return array, aff_transform

    simple_filter(
        in_filepath, out_filepath, _zoom,
        zoom, interpolation_order, extra_dim, fill_dim)


# ======================================================================
def img_resample(
        in_filepath,
        out_filepath,
        new_shape,
        keep_ratio_method=None,
        interpolation_order=1,
        extra_dim=True,
        fill_dim=True):
    """
    Resample the image to a new shape (different resolution / voxel size).

    Parameters
    ==========
    in_filepath : str
        Input file path.
    out_filepath : str
        Output file path.
    new_shape : int tuple
        New dimensions of the image.
    interpolation_order : int, optional
        Order of the spline interpolation. 0: nearest. Accepted range: [0, 5].
    extra_dim : bool, optional
        Force extra dimensions in the zoom parameters.
    fill_dim : bool, optional
        Dimensions not specified are left untouched.

    Returns
    =======
    None

    """

    def _zoom(
            array, new_shape, keep_ratio_method, interpolation_order,
            extra_dim, fill_dim):
        zoom = mrg.shape2zoom(array.shape, new_shape, keep_ratio_method)
        zoom, shape = mrg.zoom_prepare(zoom, array.shape, extra_dim, fill_dim)
        array = sp.ndimage.zoom(
            array.reshape(shape), zoom, order=interpolation_order)
        # aff_transform = np.diag(1.0 / np.array(zoom[:3] + [1.0]))
        return array

    simple_filter(
        in_filepath, out_filepath, _zoom,
        new_shape, keep_ratio_method, extra_dim, fill_dim, interpolation_order)


# ======================================================================
def img_frame(
        in_filepath,
        out_filepath,
        border,
        background=0,
        use_longest=True):
    """
    Add a border frame to the image (same resolution / voxel size).
    TODO: check with 'img_reframe'

    Parameters
    ==========
    in_filepath : str
        Input file path.
    out_filepath : str
        Output file path.
    border : int or int tuple (optional)
        The size of the border relative to the initial array shape.
    background : int or float (optional)
        The background value to be used for the frame.
    use_longest : bool (optional)
        Use longest dimension to get the border size.
    """
    simple_filter(
        in_filepath, out_filepath, mrg.frame, border, background, use_longest)


# ======================================================================
def img_reframe(
        in_filepath,
        out_filepath,
        new_shape,
        background=0):
    """
    Add a border frame to the image (same resolution / voxel size).
    TODO: check with 'img_frame'

    Parameters
    ==========
    in_filepath : str
        Input file path.
    out_filepath : str
        Output file path.
    new_shape : int tuple
        New dimensions of the image.
    background : int or float (optional)
        The background value to be used for the frame.

    """
    simple_filter(
        in_filepath, out_filepath, mrg.reframe, new_shape, background)


# ======================================================================
def img_common_sampling(
        in_filepath_list,
        out_filepath_list=None,
        new_shape=None,
        lossless=False,
        extra_dim=True,
        fill_dim=True):
    """
    Resample images sizes and affine transformations to match the same shape.

    Note that:
    | - the sampling / resolution / voxel size will change
    | - the support space / field-of-view will NOT change

    Parameters
    ==========
    filepath_list : str list
        List of input file paths (affine is taken from last item).
    suffix : str
        Suffix to append to the output filenames. Empty string to overwrite.

    Returns
    =======
    None.

    """

    def combine_shape(shape_list, lossless=lossless):
        new_shape = [1] * max([len(shape) for shape in shape_list])
        shape_arr = np.ones((len(shape_list), len(new_shape))).astype(np.int)
        for idx, shape in enumerate(shape_list):
            shape_arr[idx, :len(shape)] = np.array(shape)
        combiner = mrb.lcm if lossless else max
        new_shape = [
            combiner(*list(shape_arr[:, idx]))
            for idx in range(len(new_shape))]
        return tuple(new_shape)

    # calculate new shape
    if new_shape is None:
        shape_list = []
        for in_filepath in in_filepath_list:
            img_nii = nib.load(in_filepath)
            shape_list.append(img_nii.get_data().shape)
        new_shape = combine_shape(shape_list)

    # resample images
    interpolation_order = 0 if lossless else 1

    # when output files are not specified, modify inputs
    if out_filepath_list is None:
        out_filepath_list = in_filepath_list

    for in_filepath, out_filepath in zip(in_filepath_list, out_filepath_list):
        # ratio should not be kept: keep_ratio_method=None
        img_resample(
            in_filepath, out_filepath, new_shape, None, interpolation_order,
            extra_dim, fill_dim)
    return out_filepath_list


# ======================================================================
def img_common_support(
        in_filepath_list,
        out_filepath_list=None,
        new_shape=None,
        background=0):
    """
    Reframe images sizes (by adding border) to match the same shape.

    Note that:
    | - the sampling / resolution / voxel size will NOT change
    | - the support space / field-of-view will change

    Parameters
    ==========
    filepath_list : str list
        List of input file paths (affine is taken from last item).
    suffix : str
        Suffix to append to the output filenames. Empty string to overwrite.

    Returns
    =======
    None.

    """

    def combine_shape(shape_list):
        new_shape = [1] * max([len(shape) for shape in shape_list])
        if any([len(shape) != len(new_shape) for shape in shape_list]):
            raise IndexError('shape length must match')
        shape_arr = np.ones((len(shape_list), len(new_shape))).astype(np.int)
        for idx, shape in enumerate(shape_list):
            shape_arr[idx, :len(shape)] = np.array(shape)
        new_shape = [
            max(*list(shape_arr[:, idx]))
            for idx in range(len(new_shape))]
        return tuple(new_shape)

    # calculate new shape
    if new_shape is None:
        shape_list = []
        for in_filepath in in_filepath_list:
            img_nii = nib.load(in_filepath)
            shape_list.append(img_nii.get_data().shape)
        new_shape = combine_shape(shape_list)

    if out_filepath_list is None:
        out_filepath_list = in_filepath_list

    for in_filepath, out_filepath in zip(in_filepath_list, out_filepath_list):
        img_reframe(in_filepath, out_filepath, new_shape, background)

    return out_filepath_list


# ======================================================================
def img_mask_threshold(
        in_filepath,
        out_filepath,
        threshold=None,
        comparison=None,
        mode=None):
    """
    Extract a mask from an array using a threshold.

    Parameters
    ==========
    in_filepath : str
        Input file path.
    out_filepath : str
        Output file path.
    threshold : int, float or tuple or None (optional)
        Value(s) to be used for determining the threshold.
    comparison : str or None(optional)
        A string representing the numeric relationship: [=, !=, >, <, >=, <=]
    mode : str or None (optional)
        Determines how to interpret / process the threshold value.
        Available values are:
        | 'absolute': use the absolute value
        | 'relative': use a value relative to dynamic range
        | 'percentile': use the value obtained from the percentiles

    Returns
    =======
    None.

    See Also
    ========
    mri_tools.modules.segmentation.mask_threshold

    """

    def _img_mask_threshold(array, *args, **kwargs):
        return mrs.mask_threshold(array, *args, **kwargs).astype(float)

    kw_params = mrb.set_keyword_parameters(mrs.mask_threshold, locals())
    simple_filter(in_filepath, out_filepath, _img_mask_threshold, **kw_params)


# ======================================================================
def calc_labels(
        in_filepath,
        out_filepath,
        *args,
        **kwargs):
    """
    Extract labels using: mri_tools.modules.geometry.calc_labels

    Parameters
    ==========
    in_filepath : str
        Input file path.
    out_filepath : str
        Output file path.
    args : tuple
        arguments to be passed through
    kwargs : dict
        keyword arguments to be passed through

    Returns
    =======
    None.

    See Also
    ========
    mri_tools.modules.geometry.calc_labels

    """

    # todo: fixme

    def _calc_labels(array, *params):
        labels, masks = mrs.find_objects()
        return

    simple_filter(
        in_filepath, out_filepath,
        lambda x, p: (mrs.find_objects(x.astype(int), *p).astype(int)), *pars)


# ======================================================================
def calc_stats(
        img_filepath,
        mask_filepath=None,
        save_path=None,
        mask_nan=True,
        mask_inf=True,
        mask_vals=[0.0],
        printing=None,
        title=None,
        compact=False):
    """
    Calculate statistical information (min, max, avg, std, sum).

    Parameters
    ==========
    img_filepath : str
        Input file path.
    mask_filepath : str (optional)
        Mask file path.
    save_path : str (optional)
        The path to which the plot is to be saved. If unset, no output.
    mask_nan : bool (optional)
        Mask NaN values.
    mask_inf : bool (optional)
        Mask Inf values.
    mask_vals : list (optional)
        List of values to mask out.
    no_labels : bool (optional)
        Strip labels from return.
    printing : bool or None (optional)
        Force printing of results, even when result is saved to file.
    title : str or None (optional)
        Title to be printed before results.
    compact : bool (optional)
        Use a compact format string for displaying results.

    Returns
    =======
    stats_dict : dictionary
        | 'min': minimum value
        | 'max': maximum value
        | 'avg': average or mean
        | 'std': standard deviation
        | 'sum': summation

    """
    img_nii = nib.load(img_filepath)
    img = img_nii.get_data()
    if mask_filepath:
        mask_nii = nib.load(mask_filepath)
        mask = mask_nii.get_data().astype(bool)
    else:
        mask = slice(None)
    if not save_path and printing is not False:
        printing = True
    if printing:
        if not title:
            if save_path:
                title = os.path.basename(save_path)
            else:
                title = os.path.basename(img_filepath)
        stats_dict = mrb.calc_stats(
            img[mask], mask_nan, mask_inf, mask_vals, save_path, title)
    else:
        stats_dict = mrb.calc_stats(
            img[mask], mask_nan, mask_inf, mask_vals, save_path)
    return stats_dict


# ======================================================================
def change_data_type(
        in_filepath,
        out_filepath,
        data_type=complex):
    """
    Change NIfTI-1 image data type.

    Parameters
    ==========
    in_filepath : str list
        Input file path.
    out_filepath : str
        Output file path.
    data_type : datatype (optional)
        Orientation along which array is join in the N-dim space.

    Returns
    =======
    None.

    """
    img_nii = nib.load(in_filepath)
    img = img_nii.get_data()
    img = img.astype(data_type)
    save(out_filepath, img, img_nii.get_affine())


# ======================================================================
def plot_sample2d(
        img_filepath,
        *args,
        **kwargs):
    """
    Plot a 2D sample image of a 3D NIfTI-1 image.

    Parameters
    ==========
    img_filepath : str
        Input file path.
    args : tuple (optional)
    kwargs : dict (optional)
        Additional arguments to be passed to the transformation function:
        `mri_tools.modules.plot.sample2d()`

    Returns
    =======
    sample : ndarray
        The displayed image.
    plot : matplotlib.pyplot.Axes
        The Axes object containing the plot.

    """
    img_nii = nib.load(img_filepath)
    img = img_nii.get_data()
    sample, plot = mrp.sample2d(img, *args, **kwargs)
    return sample, plot


# ======================================================================
def plot_histogram1d(
        in_filepath,
        mask_filepath=None,
        *args,
        **kwargs):
    """
    Plot 1D histogram of NIfTI-1 image using MatPlotLib.

    Parameters
    ==========
    in_filepath : str
        Input file path.
    mask_filepath : str
        Mask file path.
    args : tuple (optional)
    kwargs : dict (optional)
        Additional arguments to be passed to the transformation function:
        `mri_tools.modules.plot.histogram1d()`

    Returns
    =======
    hist : array
        The calculated histogram.

    """
    img_nii = nib.load(in_filepath)
    img = img_nii.get_data().astype(np.double)
    if mask_filepath:
        mask_nii = nib.load(mask_filepath)
        mask = mask_nii.get_data().astype(bool)
    else:
        mask = slice(None)
    hist, bin_edges, plot = mrp.histogram1d(img[mask], *args, **kwargs)
    return hist, bin_edges, plot


# ======================================================================
def plot_histogram1d_list(
        in_filepath_list,
        mask_filepath=None,
        *args,
        **kwargs):
    """
    Plot 1D overlapping histograms of NIfTI-1 images using MatPlotLib.

    Parameters
    ==========
    in_filepath_list : str list
        List of input file paths (affine is taken from last item).
    mask_filepath : str
        Mask file path.
    args : tuple (optional)
    kwargs : dict (optional)
        Additional arguments to be passed to the transformation function:
        `mri_tools.modules.plot.histogram1d_list()`

    Returns
    =======
    hist : array
        The calculated histogram.
    bin_edges : array
        The bin edges of the calculated histogram.
    plot : array
        The plot for further manipulation of the figure.

    """
    if mask_filepath:
        mask_nii = nib.load(mask_filepath)
        mask = mask_nii.get_data().astype(bool)
    else:
        mask = slice(None)
    img_list = []
    for in_filepath in in_filepath_list:
        img_nii = nib.load(in_filepath)
        img = img_nii.get_data()
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

    See Also:
        mri_tools.modules.plot.histogram2d()

    Args:
        in1_filepath (str): First input file path.
        in2_filepath (str): Second input file path.
        mask1_filepath (str): First mask file path.
        mask2_filepath (str): Second mask file path.
        *args (tuple): Additional arguments for:
            mri_tools.modules.plot.histogram2d()`
        **kwargs (dict): Additional arguments for:
            mri_tools.modules.plot.histogram2d()`

    Returns:
        (ndarray, ndarray, ndarray, matplotlib.Figure):
            - hist2d: The calculated 2D histogram.
            - x_edges: The bin edges on the x-axis.
            - y_edges: The bin edges on the y-axis.
            - plot: The figure object containing the plot.
    """
    img1_nii = nib.load(in1_filepath)
    img2_nii = nib.load(in2_filepath)
    img1 = img1_nii.get_data().astype(np.double)
    img2 = img2_nii.get_data().astype(np.double)
    if mask1_filepath:
        mask1_nii = nib.load(mask1_filepath)
        mask1 = mask1_nii.get_data().astype(bool)
    else:
        mask1 = slice(None)
    if mask2_filepath:
        mask2_nii = nib.load(mask2_filepath)
        mask2 = mask2_nii.get_data().astype(bool)
    else:
        mask2 = slice(None)
    hist2d, x_edges, y_edges, plot = \
        mrp.histogram2d(img1[mask1], img2[mask2], *args, **kwargs)

    return hist2d, x_edges, y_edges, plot


# ======================================================================
if __name__ == '__main__':
    print(__doc__)
