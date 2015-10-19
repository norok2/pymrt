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
# import os  # Miscellaneous operating system interfaces
# import shutil  # High-level file operations
# import math  # Mathematical functions
import time  # Time access and conversions
import datetime  # Basic date and time types
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
import scipy.optimize  # SciPy: Optimization Algorithms
# import scipy.integrate  # SciPy: Integrations facilities
# import scipy.constants  # SciPy: Mathematal and Physical Constants
# import scipy.ndimage  # SciPy: ND-image Manipulation



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
def affine2parameters(linear, shift, num_dim, transform):
    if transform == 'affine':
        parameters = np.concatenate((shift, linear.ravel()))
    else:
        parameters = None
    return parameters


# ======================================================================
def parameters2affine(params, num_dim, transform):
    if 'translation' in transform or transform in ['rigid', 'affine']:
        shift = params[:num_dim]
        params = params[num_dim:]

    if 'rotation' in transform or transform in ['rigid']:
        linear = mrg.angles2linear(params)
    elif 'scaling' in transform:
        linear = np.diag(params)
    elif transform == 'affine':
        linear = params.reshape((num_dim, num_dim))
    else:
        linear = np.eye(num_dim)
        shift = np.zeros(num_dim)
    return linear, shift


# ======================================================================
def set_init_parameters(
        moving,
        fixed,
        transform,
        init_guess=('none', 'none')):
    """
    Set initial parameters according
    """
    init_guess_shift, init_guess_other = init_guess
    params = np.array([])

    # :: set up shift
    if 'translation' in transform or transform in ['rigid', 'affine']:
        if init_guess_shift == 'weights':
            shift = mrg.weighted_center(moving) - mrg.weighted_center(fixed)
        elif init_guess_shift == 'random':
            shift = np.random.rand(moving.ndim) * moving.shape / 2.0
        else:  # 'none' or not known
            shift = np.zeros(moving.ndim)
        params = np.concatenate((params, shift))

    # :: set up other parameters, according to transform
    if 'rotation' in transform or transform in ['rigid']:
        # todo: use inertia for rotation angles?
        num_angles = mrg.num_angles_from_dim(moving.ndim)
        if init_guess_other == 'random':
            angles = np.random.rand(num_angles) * np.pi / 2.0
        else:  # 'none' or not known
            angles = np.zeros((num_angles,))
        params = np.concatenate((params, angles))

    elif 'scaling' in transform:
        max_scaling = np.max(moving.shape)
        # todo: use inertia for scaling?
        if init_guess_other == 'random':
            factors = np.random.rand(moving.dim) * max_scaling
        else:  # 'none' or not known
            factors = np.ones((moving.dim,))
        params = np.concatenate((params, factors))

    elif transform == 'affine':
        if init_guess_other == 'weights':
            # todo: improve to find real rotation
            rot_moving = mrg.rotation_axes(moving)
            rot_fixed = mrg.rotation_axes(fixed)
            linear = np.dot(rot_fixed.transpose(), rot_moving)
        elif init_guess_other == 'random':
            linear = np.random.rand(moving.ndim, moving.ndim)
        else:  # 'none' or not known
            linear = np.eye(moving.ndim)
        params = np.concatenate((params, linear.ravel()))

    return params


# ======================================================================
def _discrete_generator(transform, num_dim):
    """

    """
    if transform == 'reflection':
        shift = np.zeros((num_dim,))
        for item in itertools.product([-1, 0, 1], repeat=num_dim * num_dim):
            linear = np.array(item).reshape((num_dim, num_dim))
            if np.abs(np.linalg.det(linear)) == 1:
                yield linear, shift
    elif transform == 'reflection_simple':
        shift = np.zeros((num_dim,))
        for item in itertools.product([-1, 1], repeat=num_dim):
            linear = np.diag(item)
            if np.abs(np.linalg.det(linear)) == 1:
                yield linear, shift
    else:
        shift = np.zeros(num_dim)
        linear = np.eye(num_dim)
        yield linear, shift


# ======================================================================
def minimize_discrete(
        moving,
        fixed,
        transform,
        cost_threshold=None,
        cost_func=lambda x: np.sqrt(np.sum(x ** 2)),
        interp_order=0):
    """
    Function to minimize for discrete transformations.
    """
    best_cost = np.inf
    best_params = set_init_parameters(moving, fixed, ('none', 'none'))
    if cost_threshold == 'auto':
        cost_threshold = cost_func(
            _min_func_affine(
                best_params, moving, fixed, 'affine',
                interp_order=interp_order))
    for linear, shift in _discrete_generator(transform, moving.ndim):
        params = affine2parameters(linear, shift, moving.ndim, 'affine')
        cost = cost_func(
            _min_func_affine(
                params, moving, fixed, 'affine', interp_order=interp_order))
        if cost < best_cost:
            best_cost = cost
            best_params = params
        if cost_threshold and best_cost < cost_threshold:
            break
    return parameters2affine(best_params, moving.ndim, 'affine')


# ======================================================================
def _min_func_affine(
        params,
        moving,
        fixed,
        transform='affine',
        distance=lambda x, y: (x - y),
        interp_order=1):
    """
    Function to minimize for affine transformation.
    """
    linear, shift = parameters2affine(params, moving.ndim, transform)
    check_shift = all(shift < moving.shape)
    check_linear = np.sum(np.abs(linear)) < np.sum(moving.shape)
    if check_shift and check_linear:
        # the other valid parameters of the `affine_transform` function are:
        #     output=None, order=3, mode='constant', cval=0.0, prefilter=True
        moved = mrg.affine_transform(moving, linear, shift, order=interp_order)
        cost = distance(moved, fixed).ravel()
    else:
        cost = np.tile(np.inf, len(moving))
    return cost


# ======================================================================
def affine_registration(
        moving,
        fixed,
        moving_weights=None,
        fixed_weights=None,
        transform=None,
        init_guess=('weights', 'random'),
        interp_order=1):
    """
    Register the 'moving' image to the 'fixed' image, using only the specified
    transformation and the given metric.

    Parameters
    ==========
    moving : ndarray
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
    # :: correct for weighting
    if moving_weights is not None:
        moving = moving * moving_weights
    if fixed_weights is not None:
        fixed = fixed * fixed_weights

    # check that transform is supported
    continuous_transforms = (
        'affine', 'rigid', 'translation', 'rotation', 'scaling',
        'translation_rotation', 'translation_scaling')
    discrete_transforms = ('reflection', 'reflection_simple')
    if transform in continuous_transforms:
        params = set_init_parameters(moving, fixed, transform, init_guess)
        kwargs = mrb.set_keyword_parameters(_min_func_affine, {})
        distance = kwargs['distance']
        results = sp.optimize.leastsq(
            _min_func_affine, params,
            args=(moving, fixed, transform, distance, interp_order))
        opt_par = results[0]
        linear, shift = parameters2affine(opt_par, moving.ndim, transform)
    elif transform in discrete_transforms:
        linear, shift = minimize_discrete(moving, fixed, transform)
    else:
        linear, shift = parameters2affine(None, moving.ndim, '')

    return linear, shift


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


# # :: test
# s1 = '/nobackup/isar2/cache/ecm-mri/sandbox/test/T1_sample2/s018__MP2RAGE_e' \
#      '=post0,l=2__T1.nii.gz'
# s2 = '/nobackup/isar2/cache/ecm-mri/sandbox/test/T1_sample2/s020__MP2RAGE_e' \
#      '=post2,l=2__T1.nii.gz'
# s3 = '/nobackup/isar2/cache/ecm-mri/sandbox/test/T1_sample2/s031__MP2RAGE_e' \
#      '=pre0,l=2__T1.nii.gz'
# t12 = '/nobackup/isar2/cache/ecm-mri/sandbox/test/T1_sample2/s018__MP2RAGE_e' \
#       '=post0,l=2,reg=s2__T1.nii.gz'
# t13 = '/nobackup/isar2/cache/ecm-mri/sandbox/test/T1_sample2/s018__MP2RAGE_e' \
#       '=post0,l=2,reg=s3__T1.nii.gz'
#
# import mri_tools.modules.nifti as mrn
#
#
# def my_reg(array_list, *args, **kwargs):
#     img = array_list[0]
#     ref = array_list[1]
#     # at first translate...
#     linear, shift = affine_registration(
#         img, ref, transform='translation', init_guess=('weights', 'weights'))
#     # print(shift)
#     img = mrg.affine_transform(img, linear, shift)
#     # ... then reorient
#     linear, shift = affine_registration(
#         img, ref, transform='reflection_simple')
#     # print(linear)
#     img = mrg.affine_transform(img, linear, shift)
#     # ... and finally perform finer registration
#     linear, shift = affine_registration(img, ref, *args, **kwargs)
#     array = mrg.affine_transform(img, linear, shift)
#     # print(mrg.compose_affine(linear, shift))
#     return array
#
#
# # ======================================================================
# if __name__ == '__main__':
#     print(__doc__)
#     begin_time = time.time()
#
#     mrn.simple_filter_n(
#         [s1, s2], t12, my_reg,
#         transform='rigid', interp_order=1, init_guess=('none', 'none'))
#
#     mrn.simple_filter_n(
#         [s1, s3], t13, my_reg,
#         transform='rigid', interp_order=1, init_guess=('none', 'none'))
#
#     end_time = time.time()
#     print('ExecTime: ', datetime.timedelta(0, end_time - begin_time))
