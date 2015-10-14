#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
mri_tools.registration: generic registration using numpy/scipy
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
import itertools  # Functions creating iterators for efficient looping
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
# import matplotlib.pyplot as plt  # Matplotlib's pyplot: MATLAB-like syntax
# import mayavi.mlab as mlab  # Mayavi's mlab: MATLAB-like syntax
# import scipy.optimize  # SciPy: Optimization Algorithms
# import scipy.integrate  # SciPy: Integrations facilities
# import scipy.constants  # SciPy: Mathematal and Physical Constants
import scipy.ndimage  # SciPy: ND-image Manipulation

# :: Local Imports
import mri_tools.modules.base as mrb
# import mri_tools.modules.nifti as mrn
import mri_tools.modules.geometry as mrg
# from mri_tools import INFO
# from mri_tools import VERB_LVL
# from mri_tools import D_VERB_LVL
# from mri_tools import get_first_line
from mri_tools.config import EXT_CMD


# ======================================================================
def affine_registration(
        array,
        fixed,
        transform='affine',
        interp_order=1,
        metric=None):
    """
    Register the 'moving' image to the 'fixed' image, using only the specified
    transformation and the given metric.

    Parameters
    ==========
    array : ndarray
        The image to be registered.
    fixed : ndarray
        The reference (or template) image.
    transform : str (optional)
        The allowed transformations:
        | affine : general linear transformation and translation
        | similarity : scaling, rotation and translation
        | rigid : rotation and translation
        | scaling : only scaling (TODO: include translation)
        | translation : only translation

    Returns
    =======
    affine :
        The n+1 square matrix describing the affine transformation that
        minimizes the specified metric and can be used for registration.

    """

    def min_func_translation(shift, num_dim, moving, fixed, interp_order):
        """
        Function to minimize for translation transformation.
        """
        if all(np.abs(shift)) < np.max(moving.shape):
            moved = scipy.ndimage.shift(moving, shift, order=interp_order)
            diff = moved - fixed
        else:
            diff = np.tile(np.inf, len(moving))
        return np.abs(diff.ravel())

    def min_func_scaling(scaling, num_dim, moving, fixed, interp_order):
        """
        Function to minimize for scaling transformation.
        """
        if all(scaling) > 0.0:
            moved = scipy.ndimage.zoom(moving, scaling, order=interp_order)
            diff = moved - fixed
        else:
            diff = np.tile(np.inf, len(moving))
        return np.abs(diff.ravel())

    def min_func_rigid(par, num_dim, moving, fixed, axes_list, interp_order):
        """
        Function to minimize for rigid transformation.
        """
        shift = par[:num_dim]
        angle_list = par[num_dim:]
        if all(np.abs(shift)) < np.max(moving.shape) and \
                                        -2.0 * np.pi <= all(
                            angle_list) <= 2.0 * np.pi:
            axes_list = list(itertools.combinations(range(num_dim), 2))
            moved = scipy.ndimage.shift(moving, shift, order=interp_order)
            for angle, axes in zip(angle_list, axes_list):
                moved = scipy.ndimage.rotate(moved, angle, axes, reshape=False)
            diff = moved - fixed
        else:
            diff = np.tile(np.inf, len(moving))
        return np.abs(diff.ravel())

    def min_func_affine(par, num_dim, moving, fixed, interp_order):
        """
        Function to minimize for affine transformation.
        """
        shift = par[:num_dim]
        linear = par[num_dim:].reshape((num_dim, num_dim))
        moved = sp.ndimage.affine_transform(moving, linear, shift,
                                               order=interp_order)
        diff = moved - fixed
        return np.abs(diff.ravel())

    # determine number of dimensions
    num_dim = len(array.shape)

    # calculate starting points
    shift = np.zeros(num_dim)  # TODO: use center-of-mass
    linear = np.eye(num_dim)  # TODO: use rotational tensor (of inerita)
    if transform == 'translation':
        par0 = shift
        res = sp.optimize.leastsq(
            min_func_translation, par0,
            args=(num_dim, array, fixed, interp_order))
        opt_par = res[0]
        shift = -opt_par
        linear = np.eye(num_dim)
    elif transform == 'scaling':  # TODO: improve scaling
        scaling = np.ones(num_dim)
        par0 = scaling
        res = scipy.optimize.leastsq(
            min_func_scaling, par0,
            args=(num_dim, array, fixed, interp_order))
        opt_par = res[0]
        shift = np.zeros(num_dim)
        linear = np.diag(opt_par)
    elif transform == 'rigid':
        shift = np.zeros(num_dim)
        axes_list = list(itertools.combinations(range(num_dim), 2))
        angles = np.zeros(len(axes_list))
        par0 = np.concatenate((shift, angles))
        res = scipy.optimize.leastsq(
            min_func_rigid, par0,
            args=(num_dim, array, fixed, axes_list, interp_order))
        opt_par = res[0]
        shift = opt_par[:num_dim]
        angles = opt_par[num_dim:]
        linear = mrg.angles2linear(angles, axes_list)
    elif transform == 'affine':
        par0 = np.concatenate((shift, linear.ravel()))
        res = scipy.optimize.leastsq(
            min_func_affine, par0,
            args=(num_dim, array, fixed, interp_order))
        opt_par = res[0]
        shift = opt_par[:num_dim]
    affine = mrg.compose_affine(linear, shift)
    return affine


# =============================================================================
def external_registration(
        array,
        fixed,
        weight_array=None,
        weight_fixed=None,
        tool='FSL',
        bridge=nib.Nifti1Image,
        tmp_output='tmp_{}~',
        clean_after=True,
        **arguments):
    """
    Register the array to the reference using an external tool.

    Some tools may interpret data differently and should NOT be intermixed.

    Parameters
    ==========
    array :
    """
    # todo: generalize to support any tool
    # temporary filepaths
    img_filepath = tmp_output.format('img')
    fix_filepath = tmp_output.format('ref')
    # create a dummy affine
    dummy = np.eye(array.ndim + 1)  # affine matrix has an extra dimension
    # output image
    img_nii = bridge(array, dummy)
    img_nii.to_filename(img_filepath)
    # output reference
    fix_nii = bridge(fixed, dummy)
    fix_nii.to_filename(fix_filepath)
    # generate
    if tool.startswith('FSL'):
        cmd = EXT_CMD['fsl/4.1/flirt']
        mrb.execute(cmd)
    else:
        affine = np.eye(array.ndim + 1)  # affine matrix has an extra dimension

    return affine
