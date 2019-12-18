#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pymrt.registration: generic simple registration.
"""

# ======================================================================
# :: Future Imports
from __future__ import (
    absolute_import, division, print_function, unicode_literals)

# ======================================================================
# :: Python Standard Library Imports
# import os  # Miscellaneous operating system interfaces
# import shutil  # High-level file operations
# import math  # Mathematical functions
import time  # Time access and conversions
import datetime  # Basic date and time types
# import operator  # Standard operators as functions
# import collections  # Container datatypes
import itertools  # Functions creating iterators for efficient looping
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
import flyingcircus as fc  # Everything you always wanted to have in Python*

# :: External Imports Submodules
# import matplotlib.pyplot as plt  # Matplotlib's pyplot: MATLAB-like syntax
import scipy.optimize  # SciPy: Optimization Algorithms
# import scipy.integrate  # SciPy: Integrations facilities
# import scipy.constants  # SciPy: Mathematal and Physical Constants
import scipy.ndimage  # SciPy: ND-image Manipulation

# :: Internal Imports
import pymrt as mrt  # Python Magnetic Resonance Tools: the multi-tool of MRI
import pymrt.config

# :: Local Imports
from pymrt import INFO, PATH
from pymrt import VERB_LVL, D_VERB_LVL, VERB_LVL_NAMES
from pymrt import elapsed, report, run_doctests
from pymrt import msg, dbg, fmt, fmtm


# ======================================================================
def affine_to_params(
        linear,
        shift,
        n_dim,
        transform):
    """

    Args:
        linear ():
        shift ():
        n_dim ():
        transform ():

    Returns:

    """
    if transform == 'affine':
        parameters = np.concatenate((shift, linear.ravel()))
    else:
        parameters = None
    return parameters


# ======================================================================
def params_to_affine(
        params,
        n_dim,
        transform):
    """

    Args:
        params ():
        n_dim ():
        transform ():

    Returns:

    """
    linear = np.eye(n_dim)
    shift = np.zeros(n_dim)
    if 'translation' in transform or transform in ['rigid', 'affine']:
        shift = params[:n_dim]
        params = params[n_dim:]
    if 'rotation' in transform or transform in ['rigid']:
        linear = fc.extra.angles2rotation(params)
    elif 'scaling' in transform:
        linear = np.diag(params)
    elif transform == 'affine':
        linear = params.reshape((n_dim, n_dim))
    return linear, shift


def set_bounds(
        shape,
        transform):
    """
    Set bounds for registration parameters.
    """
    n_dim = len(shape)
    # todo: implement sensible bounds for different transformations
    return None


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
            shift = \
                fc.extra.weighted_center(moving) - fc.extra.weighted_center(
                    fixed)
        elif init_guess_shift == 'random':
            shift = np.random.rand(moving.ndim) * moving.shape / 2.0
        else:  # 'none' or not known
            shift = np.zeros(moving.ndim)
        params = np.concatenate((params, shift))

    # :: set up other parameters, according to transform
    if 'rotation' in transform or transform in ['rigid']:
        # todo: use inertia for rotation angles?
        num_angles = fc.extra.num_angles_from_dim(moving.ndim)
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
            rot_moving = fc.extra.rotation_axes(moving)
            rot_fixed = fc.extra.rotation_axes(fixed)
            linear = np.dot(rot_fixed.transpose(), rot_moving)
        elif init_guess_other == 'random':
            linear = np.random.rand(moving.ndim, moving.ndim)
        else:  # 'none' or not known
            linear = np.eye(moving.ndim)
        params = np.concatenate((params, linear.ravel()))

    return params


# ======================================================================
def _discrete_generator(transform, n_dim):
    """
    Generator of discrete transformations.

    Parameters
    ==========
    transform : str
        The name of the accepted transformation.
    """
    if transform == 'reflection':
        shift = np.zeros((n_dim,))
        for elements in itertools.product([-1, 0, 1],
                                          repeat=n_dim * n_dim):
            linear = np.array(elements).reshape((n_dim, n_dim))
            if np.abs(np.linalg.det(linear)) == 1:
                linear = linear.astype(np.float)
                yield linear, shift
    elif transform == 'reflection_simple':
        shift = np.zeros((n_dim,))
        for diagonal in itertools.product([-1, 1], repeat=n_dim):
            linear = np.diag(diagonal).astype(np.float)
            yield linear, shift
    elif transform == 'pi_rotation':
        shift = np.zeros((n_dim,))
        for diagonal in itertools.product([-1, 1], repeat=n_dim):
            if np.prod(np.array(diagonal)) == 1:
                linear = np.diag(diagonal).astype(np.float)
                yield linear, shift
    elif transform == 'pi/2_rotation':
        shift = np.zeros((n_dim,))
        num_angles = fc.extra.num_angles_from_dim(n_dim)
        for angles in itertools.product([0, 90, 180, 270], repeat=num_angles):
            linear = fc.extra.angles2rotation(angles)
            yield linear, shift
    elif transform == 'pi/2_rotation+':
        shift = np.zeros((n_dim,))
        num_angles = fc.extra.num_angles_from_dim(n_dim)
        for angles in itertools.product([0, 90, 180, 270], repeat=num_angles):
            for diagonal in itertools.product([-1, 1], repeat=n_dim):
                linear = np.dot(
                    np.diag(diagonal).astype(np.float),
                    fc.extra.angles2rotation(angles))
                yield linear, shift
    else:
        shift = np.zeros(n_dim)
        linear = np.eye(n_dim)
        yield linear, shift


# ======================================================================
def minimize_discrete(
        moving,
        fixed,
        transform,
        cost_threshold=None,
        cost_func=lambda x, y: np.sqrt(np.sum((x - y) ** 2)),
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
    if cost_func is None:
        cost_func = \
            fc.base.set_func_kws(_min_func_affine, {})['cost_func']
    for linear, shift in _discrete_generator(transform, moving.ndim):
        params = affine_to_params(linear, shift, moving.ndim, 'affine')
        cost = _min_func_affine(
            params, moving.ravel(), fixed.ravel(), moving.shape, 'affine',
            cost_func, interp_order=interp_order)
        if cost < best_cost:
            best_cost = cost
            best_params = params
        if cost_threshold and best_cost < cost_threshold:
            break
    return params_to_affine(best_params, moving.ndim, 'affine')


# ======================================================================
def _min_func_affine(
        params,
        moving_ravel,
        fixed_ravel,
        shape,
        transform='affine',
        cost_func=lambda x, y: \
                np.sqrt(np.sum((x - y) ** 2)) / (np.sum(np.abs(x)) + 1),
        interp_order=1):
    """
    Function to minimize for affine transformation.
    """
    n_dim = len(shape)
    linear, shift = params_to_affine(params, n_dim, transform)
    # the other valid parameters of the `affine_transform` function are:
    #     output=None, order=3, mode='constant', cval=0.0, prefilter=True
    moved_ravel = sp.ndimage.affine_transform(
        moving_ravel.reshape(shape), linear, shift, order=interp_order).ravel()
    return cost_func(moved_ravel, fixed_ravel)


# ======================================================================
def affine_registration(
        moving,
        fixed,
        moving_weights=None,
        fixed_weights=None,
        transform=None,
        init_guess=('weights', 'random'),
        cost_func=None,
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
    linear, shift :
        The n+1 square matrix describing the affine transformation that
        minimizes the specified metric and can be used for registration.

    """
    # :: correct for weighting
    if moving_weights is not None:
        moving = moving * moving_weights
    if fixed_weights is not None:
        fixed = fixed * fixed_weights

    # : check that transform is supported
    continuous_transforms = (
        'affine', 'rigid', 'translation', 'rotation', 'scaling',
        'translation_rotation', 'translation_scaling')
    discrete_transforms = (
        'reflection', 'reflection_simple', 'pi_rotation', 'pi/2_rotation',
        'pi/2_rotation+')
    if transform in continuous_transforms:
        init_params = set_init_parameters(moving, fixed, transform, init_guess)
        bounds = set_bounds(moving.shape, transform)
        if bounds is not None:
            method = 'L-BFGS-B'
        else:
            method = 'BFGS'
        if cost_func is None:
            kwargs__min_func_affine = \
                fc.base.set_func_kws(_min_func_affine, {})
            cost_func = kwargs__min_func_affine['cost_func']
        args__min_func_affine = (
            moving.ravel(), fixed.ravel(), moving.shape,
            transform, cost_func, interp_order)
        results = sp.optimize.minimize(
            _min_func_affine, init_params, args=args__min_func_affine,
            method=method, bounds=bounds)
        opt_par = results.x
        linear, shift = params_to_affine(opt_par, moving.ndim, transform)
    elif transform in discrete_transforms:
        linear, shift = minimize_discrete(moving, fixed, transform)
    else:
        linear, shift = params_to_affine(None, moving.ndim, '')

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
    # todo: generalize to support any tool?
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
        cmd = mrt.config.EXT_CMD['fsl/5.0/flirt']
        fc.base.execute(cmd)
    else:
        affine = np.eye(array.ndim + 1)  # affine matrix has an extra dimension
    return affine


# ======================================================================
elapsed(__file__[len(PATH['base']) + 1:])

# ======================================================================
if __name__ == '__main__':
    run_doctests(__doc__)
