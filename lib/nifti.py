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
#from __future__ import unicode_literals


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
import mri_tools.lib.base as mrb
# from mri_tools import INFO
# from mri_tools import VERB_LVL
# from mri_tools import D_VERB_LVL
# from mri_tools import _firstline

# ======================================================================
# :: Custom defined constants

# ======================================================================
# :: Default values usable in functions.
EXT_UNCOMPRESSED = 'nii'
EXT_COMPRESSED = 'nii.gz'
D_EXT = EXT_COMPRESSED
D_RANGE = (0.0, 4096.0)


# ======================================================================
def filename_noext(filename):
    """
    Remove NIfTI-1 extension from filename.

    Parameters
    ==========
    filename : string
        The filename from which the extension is to be removed.

    Returns
    =======
    filename_noext : string
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
    filename : string
        The filename to which the extension is to be added.

    Returns
    =======
    filename_noext : string
        The filename with the NIfTI-1 extension.

    """
    if compressed:
        filename += mrb.add_extsep(EXT_COMPRESSED)
    else:
        filename += mrb.add_extsep(EXT_UNCOMPRESSED)
    return filename


# ======================================================================
def img_maker(
        out_filepath,
        img,
        affine=None,
        header=None):
    """
    Interface to NIfTI-1 generic image creation.

    Parameters
    ==========
    out_filepath : string
        Output file path
    img : ndarray
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
    img_nii = nib.Nifti1Image(img, affine, header)
    img_nii.to_filename(out_filepath)


# ======================================================================
def img_mask(
        in_filepath,
        mask_filepath,
        out_filepath,
        mask_val=np.nan):
    """
    Interface to NIfTI-1 generic image creation.

    Parameters
    ==========
    in_filepath : string
        Input file path.
    mask_filepath : string
        Mask file path.
    out_filepath : string
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
    img_maker(out_filepath, img, img_nii.get_affine())


# ======================================================================
def img_filter(
        in_filepath,
        out_filepath,
        calc_func,
        calc_params):
    """
    Interface to NIfTI-1 generic filter:
    calculation(i_filepath) -> o_filepath

    Parameters
    ==========
    in_filepath : string
        Input file path.
    out_filepath : string
        Output file path.
    calc_func : func
        | Filtering function accepting and returning ndarrays:
        | calc_func(i_img, calc_params...) -> o_img
    calc_params : list (optional)
        List of function parameters to be passed to calc_func.

    Returns
    =======
    None.

    """
    img_nii = nib.load(in_filepath)
    img = img_nii.get_data()
    img = calc_func(img, *calc_params)
    img_maker(out_filepath, img, img_nii.get_affine())


# ======================================================================
def img_filter2(
        in1_filepath,
        in2_filepath,
        out_filepath,
        calc_func,
        calc_params=None):
    """
    Interface to NIfTI-1 generic binary filter:
    calculation(i_filepath1, i_filepath2) -> o_filepath

    Parameters
    ==========
    in1_filepath : string
        First input file path (affine is taken from this).
    in2_filepath : string
        Second input file path (affine is ignored).
    out_filepath : string
        Output file path.
    calc_func : func
        | Filering function accepting and returning ndarrays:
        | calc_func(i1_img, i2_img, calc_params...) -> o_img
    calc_params : list (optional)
        List of function parameters to be passed to calc_func.

    Returns
    =======
    None.

    """
    if calc_params is None:
        calc_params = []
    img1_nii = nib.load(in1_filepath)
    img2_nii = nib.load(in2_filepath)
    img1 = img1_nii.get_data()
    img2 = img2_nii.get_data()
    img1 = calc_func(img1, img2, *calc_params)
    img_maker(out_filepath, img1, img1_nii.get_affine())


# ======================================================================
def img_filter_n(
        in_filepath_list,
        out_filepath,
        calc_func,
        calc_params=None):
    """
    Interface to NIfTI-1 generic filter:
    calculation(i_filepath_list) -> o_filepath

    Parameters
    ==========
    in_filepath_list : string list
        List of input file paths (affine is taken from last item).
    out_filepath : string
        Output file path.
    calc_func : func
        | Filering function accepting and returning ndarrays:
        | calc_func(img_list, calc_params...) -> o_img
    calc_params : list (optional)
        List of function parameters to be passed to calc_func.

    Returns
    =======
    None.

    """
    if calc_params is None:
        calc_params = []
    img_list = []
    for in_filepath in in_filepath_list:
        img_nii = nib.load(in_filepath)
        img = img_nii.get_data()
        img_list.append(img)
    img = calc_func(img_list, *calc_params)
    img_maker(out_filepath, img, img_nii.get_affine())


# ======================================================================
def img_filter_n_n(
        in_filepath_list,
        out_filepath_list,
        calc_func,
        calc_params=None):
    """
    Interface to NIfTI-1 generic filter:
    calculation(i_filepath_list) -> o_filepath_list

    Parameters
    ==========
    in_filepath_list : string list
        List of input file paths (affine is taken from last item).
    out_filepath_list : string
        List of output file paths.
    calc_func : func
        | Filering function accepting and returning ndarrays:
        | calc_func(img_list, calc_params...) -> o_img_list
    calc_params : list (optional)
        List of function parameters to be passed to calc_func.

    Returns
    =======
    None.

    """
    if calc_params is None:
        calc_params = []
    in_img_list = []
    for in_filepath in in_filepath_list:
        img_nii = nib.load(in_filepath)
        img = img_nii.get_data()
        in_img_list.append(img)
    out_img_list = calc_func(in_img_list, *calc_params)
    for img, out_filepath in zip(out_img_list, out_filepath_list):
        img_maker(out_filepath, img, img_nii.get_affine())


# ======================================================================
def img_join(
        in_filepath_list,
        out_filepath,
        axis=-1):
    """
    Join NIfTI-1 images together.

    Parameters
    ==========
    in_filepath_list : string list
        List of input file paths (affine is taken from last item).
    out_filepath : string
        Output file path.
    axis : int [0,N] (optional)
        Orientation along which array is join in the N-dim space.

    Returns
    =======
    None.

    """
    img_filter_n(in_filepath_list, out_filepath, mrb.ndstack, [axis])


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
    in_filepath : string
        Input file path (affine is copied from input).
    axis : int [0,N] (optional)
        Orientation along which array is split in the N-dim space.
    out_dirpath : string
        Path to directory where to store results.
    out_filename : string (optional)
        Filename (without extension) where to store results.

    Returns
    =======
    out_filepath_list : string list
        Output file path list.

    """
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
        img_maker(out_filepath, image, img_nii.get_affine())
        out_filepath_list.append(out_filepath)
    return out_filepath_list


# ======================================================================
def img_zoom(
        in_filepath,
        out_filepath,
        zoom=1.0,
        interpolation_order=1,
        force_extra_dim=True,
        autofill_dim=True):
    """
    Zoom the image with a specified magnification factor.

    Parameters
    ==========
    in_filepath : string
        Input file path.
    out_filepath : string
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

    """
    img_nii = nib.load(in_filepath)
    img = img_nii.get_data()
    try:
        iter(zoom)
    except (IndexError):
        zoom = [zoom] * len(img.shape)
    else:
        zoom = list(zoom)
    if force_extra_dim:
        expanded_shape = list(img.shape) + \
            [1.0] * (len(zoom) - len(img.shape))
        img = np.reshape(img, expanded_shape)
    else:
        zoom = zoom[:len(img.shape)]
    if autofill_dim and len(zoom) < len(img.shape):
        zoom[len(zoom):] = [1.0] * (len(img.shape) - len(zoom))
    affine = img_nii.get_affine().dot(
        np.diag(1.0 / np.array(zoom[:3] + [1.0])))
    img = sp.ndimage.zoom(img, zoom=zoom, order=interpolation_order)
    img_maker(out_filepath, img, affine)


# ======================================================================
def img_resample(
        in_filepath,
        out_filepath,
        new_shape,
        interpolation_order=1,
        force_extra_dim=True,
        autofill_dim=True):
    """
    Resample the image with a new shape (constant FOV, different voxel-size).

    Parameters
    ==========
    in_filepath : string
        Input file path.
    out_filepath : string
        Output file path.
    new_shape : tuple of int
        New dimensions of the image.
    interpolation_order : int, optional
        Order of the spline interpolation. 0: nearest. Accepted range: [0, 5].
    force_extra_dim : boolean, optional
        Force extra dimensions in the zoom parameters.
    autofill_dim : boolean, optional
        Dimensions not specified are left untouched.

    Returns
    =======
    None

    """
    img_nii = nib.load(in_filepath)
    img = img_nii.get_data()
    if force_extra_dim:
        expanded_shape = list(img.shape) + \
            [1.0] * (len(new_shape) - len(img.shape))
        img = np.reshape(img, expanded_shape)
    zoom = [old / new for new, old in zip(img.shape, new_shape)]
    if autofill_dim and len(zoom) < len(img.shape):
        zoom[len(new_shape):] = [1.0] * (len(img.shape) - len(zoom))
    affine = img_nii.get_affine().dot(
        np.diag(1.0 / np.array(zoom[:3] + [1.0])))
    img = sp.ndimage.zoom(img, zoom=zoom, order=interpolation_order)
    img_maker(out_filepath, img, affine)


# ======================================================================
def img_resize(
        in_filepath,
        out_filepath,
        new_shape,
        interpolation_order=1,
        force_extra_dim=True,
        autofill_dim=True):
    """
    Resize the image into a new shape (constant voxel-size, different FOV).
    TODO: write this function, right now it is a copy of resample

    Parameters
    ==========
    in_filepath : string
        Input file path.
    out_filepath : string
        Output file path.
    new_shape : tuple of int
        New dimensions of the image.
    interpolation_order : int, optional
        Order of the spline interpolation. 0: nearest. Accepted range: [0, 5].
    force_extra_dim : boolean, optional
        Force extra dimensions in the zoom parameters.
    autofill_dim : boolean, optional
        Dimensions not specified are left untouched.

    Returns
    =======
    None.

    """
    img_nii = nib.load(in_filepath)
    img = img_nii.get_data()
    if force_extra_dim:
        expanded_shape = list(img.shape) + \
            [1.0] * (len(new_shape) - len(img.shape))
        img = np.reshape(img, expanded_shape)
    zoom = [old / new for new, old in zip(img.shape, new_shape)]
    if autofill_dim and len(zoom) < len(img.shape):
        zoom[len(new_shape):] = [1.0] * (len(img.shape) - len(zoom))
    affine = img_nii.get_affine().dot(
        np.diag(1.0 / np.array(zoom[:3] + [1.0])))
    img = sp.ndimage.zoom(img, zoom=zoom, order=interpolation_order)
    img_maker(out_filepath, img, affine)


# ======================================================================
def img_common_shape(
        in_filepath_list,
        out_filepath_list=None,
        new_shape=None,
        lossless=False,
        ignore_extra_dim=False):
    """
    Resample NIfTI-1 sizes and affines to a common shape
    (without approximations and assuming identical field-of-view).

    Parameters
    ==========
    filepath_list : string list
        List of input file paths (affine is taken from last item).
    suffix : string
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
        if lossless:
            combiner = mrb.lcm
        else:
            combiner = max
        new_shape = [combiner(*list(shape_arr[:, idx]))
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
    if lossless:
        interpolation_order = 0
    else:
        interpolation_order = 1
    for in_filepath, out_filepath in zip(in_filepath_list, out_filepath_list):
        img_zoom(in_filepath, out_filepath)
        img_resample(
            in_filepath, out_filepath, new_shape, interpolation_order,
            ignore_extra_dim)
    return out_filepath_list


# ======================================================================
def calc_mask(
        in_filepath,
        out_filepath,
        smoothing=1.0,
        hist_dev_factor=4.0,
        rel_threshold=0.01,
        comparison='>',
        erosion_iter=0):
    """
    Extract a mask from an array.
    | Workflow is:
    * Gaussian filter (smoothing) with specified sigma
    * histogram deviation reduction by a specified factor
    * masking values using a relative threshold (and thresholding method)
    * binary erosion(s) witha specified number of iterations.

    Parameters
    ==========
    arr : nd-array
        Array from which mask is created.
    smoothing : float
        Sigma to be used for Gaussian filtering. If zero, no filtering done.
    hist_dev_factor : float
        Histogram deviation reduction factor (in std-dev units):
        values that are the specified number of standard deviations away from
        the average are not used for the absolute thresholding calculation.
    rel_threshold : (0,1)-float
        Relative threshold for masking out values.
    comparison : string
        Comparison mode: [=|>|<|>=|<=|~]
    erosion_iter : integer
        Number of binary erosion iteration in mask post-processing.

    Returns
    =======
    mask : nd-array
        The extracted mask.

    """
    par_list = \
        [smoothing, hist_dev_factor, rel_threshold, comparison, erosion_iter]
    float_calc_mask = lambda array, par_list: \
        mrb.calc_mask(array, *par_list).astype(np.double)
    img_filter(in_filepath, out_filepath, float_calc_mask, [par_list])


# ======================================================================
def change_data_type(
        in_filepath,
        out_filepath,
        data_type=complex):
    """
    Change NIfTI-1 image data type.

    Parameters
    ==========
    in_filepath : string list
        Input file path.
    out_filepath : string
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
    img_maker(out_filepath, img, img_nii.get_affine())


# ======================================================================
def plot_histogram1d(
        in_filepath,
        mask_filepath=None,
        bin_size=1,
        hist_range=(0.01, 0.99),
        bins=None,
        array_range=None,
        scale='linear',
        title='Histogram',
        labels=('Value', 'Value Frequency'),
        style='-k',
        use_new_figure=True,
        close_figure=False,
        save_path=None):
    """
    Plot 1D histogram of NIfTI-1 image using MatPlotLib.

    Parameters
    ==========
    in_filepath : string
        Input file path.
    mask_filepath : string
        Mask file path.
    bin_size : int or float (optional)
        The size of the bins.
    hist_range : float 2-tuple (optional)
        The range of the histogram to display in percentage.
    bins : int (optional)
        The number of bins to use. If set, overrides bin_size parameter.
    array_range : float 2-tuple (optional)
        Theoretical range of values for the array. If unset, uses min and max.
    scale : ['linear'|'log'|'log10'|'normed'] string (optional)
        The frequency value scaling.
    title : string (optional)
        The title of the plot.
    labels : string 2-tuple (optional)
        A 2-tuple of strings containing x-labels and y-labels.
    style : string (optional)
        Plotting style string (as accepted by MatPlotLib).
    use_new_figure : boolean (optional)
        Plot the histogram in a new figure.
    close_figure : boolean (optional)
        Close the figure after saving (useful for batch processing).
    save_path : string (optional)
        The path to which the plot is to be saved. If unset, no output.

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
    hist, bin_edges = mrb.plot_histogram1d(
        img[mask], bin_size, hist_range, bins, array_range, scale, title,
        labels, style, use_new_figure, close_figure, save_path)
    return hist, bin_edges


# ======================================================================
def plot_histogram1d_list(
        in_filepath_list,
        mask_filepath=None,
        bin_size=1,
        hist_range=(0.01, 0.99),
        bins=None,
        array_range=None,
        scale='linear',
        title='Histogram',
        labels=('Value', 'Value Frequency'),
        legends=None,
        styles=None,
        use_new_figure=True,
        close_figure=False,
        save_path=None):
    """
    Plot 1D overlapping histograms of NIfTI-1 images using MatPlotLib.

    Parameters
    ==========
    in_filepath_list : string list
        List of input file paths (affine is taken from last item).
    mask_filepath : string
        Mask file path.
    bin_size : int or float (optional)
        The size of the bins.
    hist_range : float 2-tuple (optional)
        The range of the histogram to display in percentage.
    bins : int (optional)
        The number of bins to use. If set, overrides bin_size parameter.
    array_range : float 2-tuple (optional)
        Theoretical range of values for the array. If unset, uses min and max.
    scale : ['linear'|'log'|'log10'|'normed'] string (optional)
        The frequency value scaling.
    title : string (optional)
        The title of the plot.
    labels : string 2-tuple (optional)
        A 2-tuple of strings containing x-labels and y-labels.
    legends : string list (optional)
        Legend for each histogram. If None, no legend will be displayed.
    styles : string list (optional)
        MatPlotLib's plotting style strings. If None, uses color cyclying.
    use_new_figure : boolean (optional)
        Plot the histogram in a new figure.
    close_figure : boolean (optional)
        Close the figure after saving (useful for batch processing).
    save_path : string (optional)
        The path to which the plot is to be saved. If unset, no output.

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
    hist, bin_edges, plot = mrb.plot_histogram1d_list(
        img_list, bin_size, hist_range, bins, array_range, scale, title,
        labels, legends, styles, use_new_figure, close_figure, save_path)
    return hist, bin_edges, plot


# ======================================================================
def plot_histogram2d(
        in1_filepath,
        in2_filepath,
        mask1_filepath=None,
        mask2_filepath=None,
        bin_size=1,
        hist_range=(0.0, 1.0),
        bins=None,
        array_range=None,
        use_separate_range=False,
        scale='linear',
        interpolation='bicubic',
        title='2D Histogram',
        labels=('Array 1 Values', 'Array 2 Values'),
        cmap=plt.cm.jet,
        show_contour=False,
        bisector=None,
        use_new_figure=True,
        close_figure=False,
        save_path=None):
    """
    Plot 2D histogram of two arrays with MatPlotLib.

    Parameters
    ==========
    in1_filepath : string
        First input file path.
    in2_filepath : string
        Second input file path.
    mask1_filepath : string
        First mask file path.
    mask2_filepath : string
        Second mask file path.
    bin_size : int or float | int 2-tuple (optional)
        The size of the bins.
    hist_range : float 2-tuple | 2-tuple of float 2-tuple (optional)
        The range of the histogram to display in percentage.
    bins : int | int 2-tuple (optional)
        The number of bins to use. If set, overrides bin_size parameter.
    array_range : float 2-tuple | 2-tuple of float 2-tuple (optional)
        Theoretical range of values for the array. If unset, uses min and max.
    use_separate_range : boolean (optional)
        Select if display ranges in each dimension are determined separately.
    scale : ['linear'|'log'|'log10'|'normed'] string (optional)
        The frequency value scaling.
    interpolation : string (optional)
        Interpolation method (see imshow()).
    title : string (optional)
        The title of the plot.
    labels : string 2-tuple (optional)
        A 2-tuple of strings containing x-labels and y-labels.
    cmap : MatPlotLib ColorMap (optional)
        The colormap to be used for displaying the histogram.
    bisector : string or None (optional)
        If not None, show the first bisector using specified line style.
    use_new_figure : boolean (optional)
        Plot the histogram in a new figure.
    close_figure : boolean (optional)
        Close the figure after saving (useful for batch processing).
    save_path : string (optional)
        The path to which the plot is to be saved. If unset, no output.

    Returns
    =======
    hist : array
        The calculated 2D histogram.

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
    hist = mrb.plot_histogram2d(
        img1[mask1], img2[mask2], bin_size, hist_range, bins, array_range,
        use_separate_range, scale, interpolation, title, labels, cmap,
        bisector, use_new_figure, close_figure, save_path)
    return hist


# ======================================================================
def plot_sample(
        img_filepath,
        axis=0,
        index=None,
        title=None,
        val_range=None,
        cmap=None,
        use_new_figure=True,
        close_figure=False,
        save_path=None):
    """
    Plot a 2D sample image of a 3D NIfTI-1 image.

    Parameters
    ==========
    img_filepath : string
        Input file path.
    axis : int (optional)
        The slicing axis.
    index : int (optional)
        The slicing index. If None, mid-value is taken.
    title : string (optional)
        The title of the plot.
    val_range : 2-tuple (optional)
        The (min, max) values range.
    cmap : MatPlotLib ColorMap (optional)
        The colormap to be used for displaying the histogram.
    use_new_figure : boolean (optional)
        Plot the histogram in a new figure.
    close_figure : boolean (optional)
        Close the figure after saving (useful for batch processing).
    save_path : string (optional)
        The path to which the plot is to be saved. If unset, no output.

    Returns
    =======
    sample : ndarray
        The displayed image.

    """
    img_nii = nib.load(img_filepath)
    img = img_nii.get_data()
    if not cmap:
        if val_range is None:
            min_val, max_val = np.min(img), np.max(img)
        else:
            min_val, max_val = val_range
        if min_val * max_val < 0:
            cmap = plt.cm.bwr
        else:
            cmap = plt.cm.binary
    sample = mrb.plot_sample2d(
        img, axis, index, title, val_range, cmap, use_new_figure, close_figure,
        save_path)
    return sample


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
    img_filepath : string
        Input file path.
    mask_filepath : string (optional)
        Mask file path.
    save_path : string (optional)
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
    title : string or None (optional)
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
if __name__ == '__main__':
    print(__doc__)
