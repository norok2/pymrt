#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
mri_tools: Matrix algebra formalism for Magnetization Transfer MRI experiments.
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
import cmath  # Mathematical functions for complex numbers
import time  # Time access and conversions
import datetime  # Basic date and time types
# import operator  # Standard operators as functions
# import collections  # High-performance container datatypes
# import argparse  # Parser for command-line options, arguments and subcommands

# import itertools  # Functions creating iterators for efficient looping
# import functools  # Higher-order functions and operations on callable objects
# import subprocess  # Subprocess management
# import multiprocessing  # Process-based parallelism
# import csv  # CSV File Reading and Writing [CSV: Comma-Separated Values]
# import json  # JSON encoder and decoder [JSON: JavaScript Object Notation]

# :: External Imports
import numpy as np  # NumPy (multidimensional numerical arrays library)
import scipy as sp  # SciPy (signal and image processing library)
# import matplotlib as mpl  # Matplotlib (2D/3D plotting library)
import sympy as sym  # SymPy (symbolic CAS library)
# import PIL  # Python Image Library (image manipulation toolkit)
# import SimpleITK as sitk  # Image ToolKit Wrapper
# import nibabel as nib  # NiBabel (NeuroImaging I/O Library)
# import nipy  # NiPy (NeuroImaging in Python)
# import nipype  # NiPype (NiPy Pipelines and Interfaces)
# import numba  # Numba Just-In-Time compiler for Python / NumPy
import cProfile as profile  # Deterministic Profiler

# :: External Imports Submodules
import matplotlib.pyplot as plt  # Matplotlib's pyplot: MATLAB-like syntax
import mpl_toolkits.mplot3d as mpl3  # Matplotlib's 3D support
# import scipy.optimize  # SciPy: Optimization
import scipy.integrate  # SciPy: Integration
import scipy.interpolate  # SciPy: Interpolation
import scipy.constants  # SciPy: Constants
# import scipy.ndimage  # SciPy: Multidimensional image processing
import scipy.linalg  # SciPy: Linear Algebra
import scipy.stats  # SciPy: Statistical functions
import scipy.misc  # SciPy: Miscellaneous routines

from numpy import pi, sin, cos, exp, sqrt, sinc
# from sympy import pi, sin, cos, exp, sqrt, sinc
# from sympy import re, im
# from numba import jit

# :: Local Imports
import mri_tools.modules.base as mrb

# import mri_tools.modules.geometry as mrg
# import mri_tools.modules.plot as mrp
# import mri_tools.modules.segmentation as mrs

# from mri_tools import INFO
# from mri_tools import VERB_LVL
# from mri_tools import D_VERB_LVL
from mri_tools import _EVENTS
from mri_tools.modules.base import _elapsed, _print_elapsed

# from mri_tools import get_first_line


# ======================================================================
# Proton Gyromagnetic Ratio
# Hz / T / rad
GAMMA = \
    sp.constants.physical_constants['proton gyromag. ratio'][0]
# MHz / T
GAMMA_BAR = \
    sp.constants.physical_constants['proton gyromag. ratio over 2 pi'][0]

# Magnetic Field Strength
B0 = 3.0  # T

_SUPERLORENTZ = {'x': None, 'y': None}


# ======================================================================
def superlorentz_integrand(x, t):
    return sqrt(2.0 / pi) * \
           exp(-2.0 * (x / (3 * cos(t) ** 2 - 1)) ** 2.0) * sin(t) / \
           abs(3 * cos(t) ** 2 - 1)


# ======================================================================
@np.vectorize
def superlorentz(x):
    """

    Args:
        x:

    Returns:

    """
    # scipy.integrate.quad returns both the value and the error, here ignored
    return sp.integrate.quad(
            lambda t: superlorentz_integrand(x, t), 0.0, pi / 2.0)[0]


# ======================================================================
# todo: check that the sampling rate is appropriate: 1024 is usually enough
_SUPERLORENTZ['x'] = np.logspace(-10.0, 1.8, 256)
_SUPERLORENTZ['y'] = superlorentz(_SUPERLORENTZ['x'])
_elapsed('Superlorentz Approx.')


#  ======================================================================
def _superlorentz_approx(
        x,
        x_i=_SUPERLORENTZ['x'],
        y_i=_SUPERLORENTZ['y']):
    """
    something

    Args:
        x (ndarray):
        x_i:
        y_i:

    Returns:

    """
    return np.interp(np.abs(x), x_i, y_i)


#  ======================================================================
def _sat_rate_lineshape(
        r2,
        w0,
        w_rf,
        w1,
        lineshape):
    """

    Args:
        r2:
        w0:
        w_rf:
        w1:
        lineshape:

    Returns:

    """
    if lineshape == 'superlorentz':
        lineshape_factor = superlorentz((w0 - w_rf) / r2)
    elif lineshape == 'superlorentz_approx':
        lineshape_factor = _superlorentz_approx((w0 - w_rf) / r2)
    elif lineshape in ('cauchy', 'lorentz'):
        lineshape_factor = \
            1.0 / (pi * (1.0 + ((w0 - w_rf) / r2) ** 2.0))
    elif lineshape in ('gauss', 'normal'):
        lineshape_factor = \
            exp(- ((w0 - w_rf) / r2) ** 2.0 / 2.0) / sqrt(2.0 * pi)
    else:
        lineshape_factor = 1.0
    saturation_rate = pi * np.abs(w1) ** 2.0 * lineshape_factor / r2
    return saturation_rate


#  ======================================================================
def _get_flip_angle(
        w1_arr,
        dt):
    """

    Args:
        w1_arr:
        dt:

    Returns:

    """
    return np.sum(np.abs(w1_arr * dt))


#  ======================================================================
def _set_flip_angle(
        w1_arr,
        dt,
        flip_angle):
    """

    Args:
        w1_arr:
        dt:
        flip_angle:

    Returns:

    """
    return w1_arr * flip_angle / np.sum(np.abs(w1_arr * dt))


#  ======================================================================
def _model_info(approx):
    """

    Args:
        approx:

    Returns:
        model_info (dict): A dictionary containing spin model information:
        | 'pools': the number of spin pools
        | 'exchange': the number of equilibrium exchange constants
        | 'approx': the number of spin pools using lineshape approximation
        | 'exact': the number of spin pools calculated exactly
        | 'operator_dim_base': the base operators' dimension
        | 'operator_dim': the operators' dimension after approximations
    """
    model_info = {
        'num_pools': len(approx),
        'num_approx': np.sum(
                [0 if lineshape is None else 1 for lineshape in approx])}
    model_info['num_exact'] = \
        model_info['num_pools'] - model_info['num_approx']
    # exchange at equilibrium between each two pools
    model_info['num_exchange'] = scipy.misc.comb(model_info['num_pools'], 2)
    # 3: cartesian dimensions
    # +1 for homogeneous operators
    # 2: transverse dimensions (omitted when using lineshape approximation)
    model_info['operator_dim'] = \
        1 + 3 * model_info['num_pools'] - 2 * model_info['num_approx']
    model_info['operator_shape'] = (model_info['operator_dim'],) * 2
    return model_info


#  ======================================================================
def _pulse_info(
        w1_arr,
        dt):
    """

    Args:
        w1_arr:
        dt:

    Returns:
        pulse_info (dict): A dictionary containing spin model information:
    """
    pulse_info = {
        'num_steps': w1_arr.size,
        'duration': w1_arr.size * dt,
        'amplitude': np.max(np.abs(w1_arr)),
        'flip_angle': _get_flip_angle(w1_arr, dt),
        'is_real': np.isreal(w1_arr[0]),
        'is_imag': np.isreal(w1_arr[0] / 1j),
    }
    return pulse_info


#  ======================================================================
def _shape_normal(
        num_steps,
        truncation=(0.001, 0.001)):
    """

    Args:
        num_steps:
        truncation:

    Returns:

    """
    x = np.linspace(
            scipy.stats.norm.ppf(0.0 + truncation[0]),
            scipy.stats.norm.ppf(1.0 - truncation[1]),
            num_steps)
    y = scipy.stats.norm.pdf(x)
    return y


#  ======================================================================
def _shape_cauchy(
        num_steps,
        truncation=(0.001, 0.001)):
    """

    Args:
        num_steps:
        truncation:

    Returns:

    """
    x = np.linspace(
            scipy.stats.cauchy.ppf(0.0 + truncation[0]),
            scipy.stats.cauchy.ppf(1.0 - truncation[1]),
            num_steps)
    y = scipy.stats.cauchy.pdf(x)
    return y


#  ======================================================================
def _shape_sinc(
        num_steps,
        roots=(3.0, 3.0)):
    """

    Args:
        num_steps:
        truncation:

    Returns:

    """
    x = np.linspace(-roots[0] * pi, roots[1] * pi, num_steps)
    y = sinc(x)
    return y


#  ======================================================================
def _shape_cos_sin(
        num_steps,
        roots=(1.0, 1.0)):
    """

    Args:
        num_steps:
        truncation:

    Returns:

    """
    x = np.linspace(-roots[0] * pi, roots[1] * pi, num_steps)
    y_re = cos(x)
    y_im = sin(x)
    return y_re + 1j * y_im


# ======================================================================
def make_pulse(
        duration,
        flip_angle=0.0,
        num_steps=1,
        shape=None):
    """

    Args:
        duration (float): Pulse duration in s
        flip_angle (float): Flip angle (pulse normalization) in rad
        num_steps (int): Number of sampling steps
        shape (str): The shape identifier. Accept

    Returns:
        w1_arr (ndarray[complex]

    """
    if shape == 'gauss':
        w1_arr = _shape_normal(num_steps)
    elif shape == 'lorentz':
        w1_arr = _shape_cauchy(num_steps)
    elif shape == 'sinc':
        w1_arr = _shape_sinc(num_steps)
    elif shape == 'rect' or shape is None:
        w1_arr = np.array((flip_angle / duration,))
    else:
        w1_arr = eval('_shape_' + shape)(num_steps)
    dt = duration / num_steps
    norm = np.sum(np.abs(w1_arr * dt))
    if norm:
        w1_arr = w1_arr * flip_angle / norm
    return w1_arr, dt


#  ======================================================================
def equilibrium_magnetization(
        m0,
        approx):
    """
    Generate the equilibrium magnetization vector.

    Note that B0 is assumed parallel to z-axis, therefore:
    - the transverse magnetization is zero
    - the longitudinal magnetization is only in the z-axis

    Args:
        m0:
        approx (bool): approximation to neglect

    Returns:
        m_eq (ndarray[complex])
    """
    model_info = _model_info(approx)
    m_eq = np.zeros(model_info['operator_dim']).astype(type(m0[0]))
    m_eq[0] = 0.5
    num_exact, num_approx = 0, 0
    for m0z, lineshape in zip(m0, approx):
        # 3: cartesian dims; +1: hom. operator
        if lineshape:
            pos = 1 + num_exact * 3 + num_approx
            num_approx += 1
        else:
            pos = 1 + num_exact * 3 + num_approx + 2
            num_exact += 1
        m_eq[pos] = m0z
    return m_eq


#  ======================================================================
def dynamics_operator(
        m0,
        w0,
        r1,
        r2,
        k,
        approx,
        w_rf,
        w1):
    """
    Calculate the Bloch-McConnell dynamics operator, L.

    Args:
        s0 (float):
        m0 (ndarray[float]):
        r1 (ndarray[float]):
        r2 (ndarray[float]):
        w0 (ndarray[float]):
        k (ndarray[float]):
        w_rf (float): Modulation frequency in Hz
        w1 (complex): Excitation (carrier) frequency in Hz
        approx:

    Returns:
        l_op (ndarray[float]): The dynamics operator L
    """
    # todo: fix it!
    # todo: separate spin_model and excitation parts
    # todo: include chemical exchange
    # todo: include cross-relaxation
    # :: no-delete version / without cross-relaxation
    # model_info = _model_info(approx)
    # l_op = np.zeros(model_info['operator_shape']).astype(type(m0[0]))
    # # # ...to make L invertible
    # # L[0, 0] = -2.0
    # num_exact, num_approx = 0, 0
    # for i, lineshape in enumerate(approx):
    #     # 3: cartesian dims; +1: hom. operator
    #     base = 1 + num_exact * 3 + num_approx
    #     w1x, w1y = w1.real, w1.imag
    #     bloch_core = np.array([
    #         [r2[i], w0[i] - w_rf, -w1y],
    #         [w_rf - w0[i], r2[i], w1x],
    #         [w1y, -w1x, r1[i]]])
    #     if lineshape:
    #         # Henkelman additional saturation rate approximation
    #         r_rf = _sat_rate_lineshape(r2[i], w0[i], w_rf, w1, lineshape)
    #         # r_rf = sym.symbols('r_rf')
    #         l_op[base, base] = bloch_core[-1, -1] + r_rf
    #         l_op[base, 0] = -2.0 * r1[i] * s0 * m0[i]
    #         num_approx += 1
    #     else:
    #         l_op[base:base + 3, base:base + 3] = bloch_core
    #         l_op[base + 2, 0] = -2.0 * r1[i] * s0 * m0[i]
    #         num_exact += 1

    num_pools = len(approx)
    base_len = 1 + num_pools * 3
    l_op = np.zeros((base_len,) * 2).astype(type(m0[0]))
    k_op_base = np.zeros((num_pools,) * 2).astype(type(m0[0]))
    # # ...to make L invertible
    # L[0, 0] = -2.0
    to_remove = []
    for i, lineshape in enumerate(approx):
        base = 1 + i * 3
        w1x, w1y = w1.real, w1.imag
        # Bloch operator core...
        l_op[base:base + 3, base:base + 3] = np.array([
            [r2[i], w0[i] - w_rf, -w1y],
            [w_rf - w0[i], r2[i], w1x],
            [w1y, -w1x, r1[i]]])
        # ...additional modification for homogeneous form
        l_op[base + 2, 0] = -2.0 * r1[i] * m0[i]
        # deal with approximations
        if lineshape:
            to_remove.extend([base, base + 1])
            r_rf = _sat_rate_lineshape(r2[i], w0[i], w_rf, w1, lineshape)
            # r_rf = sym.symbols('r_rf')
            l_op[base + 2, base + 2] += r_rf
    # include cross-relaxation
    # k_op_base += np.diag( * num_pools)
    k_op = np.kron(np.eye(3), k_op_base)
    l_op[1:,1:] += k_op
    # remove transverse components of approximated pools
    l_op = np.delete(l_op, to_remove, 0)
    l_op = np.delete(l_op, to_remove, 1)
    return l_op


# ======================================================================
def propagator_pulse(
        m0,
        w0,
        r1,
        r2,
        k,
        approx,
        w_rf,
        w1_arr,
        dt,
        mode='exact'):
    """
    Calculate the Bloch-McConnell propagator: expm(-L * Dt).

    L is the dynamics operator of the rf-excited system.
    Dt is a time interval where spin exchange dynamics is negligible.

    Args:
        m0:
        w0:
        r1:
        r2:
        k:
        w_rf:
        w1_arr:
        dt:
        approx:
        mode (str|None): Approximation to use for faster computation.
            Accepted values are:

            exact
                do not perform additional approximations
            sum_simple
                use expm(sum(M_i)) = prod(expm(M_i))
            sum_order1
                perform pseudo-first-order correction for non-commuting
                matrices:
                expm(sum(M_i) + sum([M_i, M_j]/2) = prod(expm(M_i))
            poly_abs:
                Use a fast polynomial approximation based on |w1| values
            interp_abs:
                Use a fast linear interpolation based on |w1| values
            reduced:
                Calculate the propagator on a coarse pulse, obtained by
                riducing the number of steps - but the same flip angle
                (norm-invariant).

    Returns:
        p_op (ndarray): The propagator operator P.

    """
    model_info = _model_info(approx)
    pulse_info = _pulse_info(w1_arr, dt)

    if mode == 'sum_simple':
        l_op_sum = np.zeros(model_info['operator_shape'])
        for w1 in w1_arr:
            l_op = dynamics_operator(m0, w0, r1, r2, k, approx, w_rf, w1)
            l_op_sum += dt * l_op
        p_op = scipy.linalg.expm(-l_op_sum)

    elif mode == 'sum_order1':
        l_op_list = []
        for w1 in w1_arr:
            l_op = dynamics_operator(m0, w0, r1, r2, k, approx, w_rf, w1)
            l_op_list.append(dt * l_op)
        l_op_sum = sum(l_op_list)
        # pseudo-first-order correction
        comm_list = [
            mrb.commutator(l_op_list[i], l_op_list[i + 1]) / 2.0
            for i in range(len(l_op_list[:-1]))]
        comm_sum = sum(comm_list)
        p_op = scipy.linalg.expm(-(l_op_sum + comm_sum))

    elif mode == 'sum_sep':
        p_op_w1_sum = np.zeros(model_info['operator_shape'])
        l_op_free = dynamics_operator(m0, r1, r2, w0, k, approx, w_rf, 0.0)
        for w1 in w1_arr:
            l_op = dynamics_operator(m0, r1, r2, w0, k, approx, w_rf, w1)
            p_op_w1_sum += (dt * (l_op - l_op_free))
        # calculate propagators
        p_op_free = scipy.linalg.expm(-l_op_free * dt)
        p_op_pow = scipy.linalg.fractional_matrix_power(
                p_op_free, pulse_info['num_steps'])
        p_op_w1 = scipy.linalg.expm(-p_op_w1_sum)
        p_op = np.dot(p_op_pow, p_op_w1_sum)

    elif mode.startswith('poly'):
        if pulse_info['is_real'] or pulse_info['is_imag']:
            _w1_arr = w1_arr.real if pulse_info['is_real'] else w1_arr.imag
            fit_order = int(mode.split('_')[-1])
            # :: calculate samples
            num_samples = fit_order + 1
            p_op_approx = np.zeros(
                    model_info['operator_shape'] + (num_samples,))
            w1_approx = np.linspace(
                    np.min(_w1_arr), np.max(_w1_arr), num_samples)
            for i, w1 in enumerate(w1_approx):
                l_op = dynamics_operator(m0, w0, r1, r2, k, approx, w_rf, w1)
                p_op_approx[:, :, i] = scipy.linalg.expm(-dt * l_op)
            # :: prepare for polynomial fit
            x_arr = w1_approx
            y_arr = p_op_approx
            # :: perform polynomial fit
            support_axis = -1
            shape = y_arr.shape
            support_size = shape[support_axis]
            y_arr = y_arr.reshape((-1, support_size))
            # polyfit requires to change matrix orientation using transpose
            p_arr = np.polyfit(x_arr, y_arr.transpose(), fit_order)
            # transpose the results back
            p_arr = p_arr.transpose()
            # revert to original shape
            p_arr = p_arr.reshape(list(shape[:support_axis]) + [fit_order + 1])
            # :: approximate all propagators and calculate final result
            p_op_arr = np.zeros(
                    (pulse_info['num_steps'],) + model_info['operator_shape'])
            for i in range(model_info['operator_dim']):
                for j in range(model_info['operator_dim']):
                    p_op_arr[:, i, j] = np.polyval(p_arr[i, j, :], _w1_arr)
            p_op_list = [p_op_arr[j, :, :] for j in
                range(pulse_info['num_steps'])]
            p_op = mrb.mdot(*p_op_list[::-1])
        else:
            fit_order = int(mode.split('_')[-1])
            # :: calculate samples
            num_samples = fit_order + 1
            # :: calculate samples
            num_extra_samples = num_samples * num_samples
            re_min, im_min = np.min(w1_arr.real), np.min(w1_arr.imag)
            re_max, im_max = np.max(w1_arr.real), np.max(w1_arr.imag)
            re_ptp, im_ptp = re_max - re_min, im_max - im_min
            _tmp = sqrt(re_ptp ** 2 +
                        (im_ptp * num_extra_samples + 2 * im_ptp) * re_ptp +
                        im_ptp ** 2) - re_ptp - im_ptp
            num_re_samples = np.floor(_tmp / im_ptp) + 2
            num_im_samples = np.floor(_tmp / re_ptp) + 2
            num_samples = num_re_samples * num_im_samples
            w1_re_approx = np.linspace(re_min, re_max, num_re_samples)
            w1_im_approx = np.linspace(im_min, im_max, num_im_samples)
            p_op_approx = np.zeros(
                    model_info['operator_shape'] + (num_samples,))
            w1_approx = np.zeros(
                    num_re_samples * num_im_samples).astype(complex)
            i = 0
            for w1_re in w1_re_approx:
                for w1_im in w1_im_approx:
                    l_op = dynamics_operator(
                            m0, w0, r1, r2, k, approx, w_rf, w1_re + 1j * w1_im)
                    p_op_approx[:, :, i] = scipy.linalg.expm(-dt * l_op)
                    w1_approx[i] = w1_re + 1j * w1_im
                    i += 1
            # :: prepare for polynomial fit
            x_arr = w1_approx
            y_arr = p_op_approx
            # :: perform polynomial fit
            support_axis = -1
            shape = y_arr.shape
            support_size = shape[support_axis]
            y_arr = y_arr.reshape((-1, support_size))
            # polyfit requires to change matrix orientation using transpose
            p_arr = np.polyfit(x_arr, y_arr.transpose(), fit_order)
            # transpose the results back
            p_arr = p_arr.transpose()
            # revert to original shape
            p_arr = p_arr.reshape(list(shape[:support_axis]) + [fit_order + 1])
            # :: approximate all propagators and calculate final result
            p_op_arr = np.zeros(
                    (pulse_info['num_steps'],) + model_info['operator_shape'])
            for i in range(model_info['operator_dim']):
                for j in range(model_info['operator_dim']):
                    p_op_arr[:, i, j] = np.real(
                            np.polyval(p_arr[i, j, :], w1_arr))
            p_op_list = [p_op_arr[j, :, :] for j in
                range(pulse_info['num_steps'])]
            p_op = mrb.mdot(*p_op_list[::-1])

    elif mode.startswith('interp'):
        method, num_samples = mode.split('_')[1:]
        num_samples = int(num_samples)
        if pulse_info['is_real'] or pulse_info['is_imag']:
            _w1_arr = w1_arr.real if pulse_info['is_real'] else w1_arr.imag
            # :: calculate samples
            p_op_approx = np.zeros(
                    model_info['operator_shape'] + (num_samples,))
            w1_approx = np.linspace(
                    np.min(_w1_arr), np.max(_w1_arr), num_samples)
            for i, w1 in enumerate(w1_approx):
                l_op = dynamics_operator(m0, w0, r1, r2, k, approx, w_rf, w1)
                p_op_approx[:, :, i] = scipy.linalg.expm(-dt * l_op)
            # perform interpolation
            p_op_arr = np.zeros(
                    (pulse_info['num_steps'],) + model_info['operator_shape'])
            for i in range(model_info['operator_dim']):
                for j in range(model_info['operator_dim']):
                    p_op_arr[:, i, j] = scipy.interpolate.griddata(
                            w1_approx, p_op_approx[i, j, :], _w1_arr,
                            method=method, fill_value=0.0)
            p_op_list = [p_op_arr[j, :, :] for j in
                range(pulse_info['num_steps'])]
            p_op = mrb.mdot(*p_op_list[::-1])
        else:
            # :: calculate samples
            num_extra_samples = num_samples * num_samples
            re_min, im_min = np.min(w1_arr.real), np.min(w1_arr.imag)
            re_max, im_max = np.max(w1_arr.real), np.max(w1_arr.imag)
            re_ptp, im_ptp = re_max - re_min, im_max - im_min
            _tmp = sqrt(re_ptp ** 2 +
                        (im_ptp * num_extra_samples + 2 * im_ptp) * re_ptp +
                        im_ptp ** 2) - re_ptp - im_ptp
            num_re_samples = np.floor(_tmp / im_ptp) + 2
            num_im_samples = np.floor(_tmp / re_ptp) + 2
            num_samples = num_re_samples * num_im_samples
            w1_re_approx = np.linspace(re_min, re_max, num_re_samples)
            w1_im_approx = np.linspace(im_min, im_max, num_im_samples)
            p_op_approx = np.zeros(
                    model_info['operator_shape'] + (num_samples,))
            w1_approx = np.zeros((num_re_samples * num_im_samples, 2))
            i = 0
            for w1_re in w1_re_approx:
                for w1_im in w1_im_approx:
                    l_op = dynamics_operator(
                            m0, w0, r1, r2, k, approx, w_rf, w1_re + 1j * w1_im)
                    p_op_approx[:, :, i] = scipy.linalg.expm(-dt * l_op)
                    w1_approx[i, :] = (w1_re, w1_im)
                    i += 1
            # perform interpolation
            p_op_arr = np.zeros(
                    (pulse_info['num_steps'],) + model_info['operator_shape'])
            for i in range(model_info['operator_dim']):
                for j in range(model_info['operator_dim']):
                    p_op_arr[:, i, j] = scipy.interpolate.griddata(
                            w1_approx,
                            p_op_approx[i, j, :],
                            (w1_arr.real, w1_arr.imag),
                            method=method, fill_value=0.0)
            p_op_list = [p_op_arr[j, :, :] for j in
                range(pulse_info['num_steps'])]
            p_op = mrb.mdot(*p_op_list[::-1])

    elif mode == 'linear':
        num_samples = 7
        if pulse_info['is_real'] or pulse_info['is_imag']:
            _w1_arr = w1_arr.real if pulse_info['is_real'] else w1_arr.imag
            # :: calculate samples
            p_op_approx = np.zeros(
                    model_info['operator_shape'] + (num_samples,))
            w1_approx = np.linspace(
                    np.min(_w1_arr), np.max(_w1_arr), num_samples)
            for i, w1 in enumerate(w1_approx):
                l_op = dynamics_operator(m0, w0, r1, r2, k, approx, w_rf, w1)
                p_op_approx[:, :, i] = scipy.linalg.expm(-dt * l_op)
            # perform interpolation
            p_op_arr = np.zeros(
                    (pulse_info['num_steps'],) + model_info['operator_shape'])
            for i in range(model_info['operator_dim']):
                for j in range(model_info['operator_dim']):
                    p_op_arr[:, i, j] = np.interp(
                            _w1_arr, w1_approx, p_op_approx[i, j, :])
            p_op_list = [p_op_arr[j, :, :] for j in
                range(pulse_info['num_steps'])]
            p_op = mrb.mdot(*p_op_list[::-1])
        else:
            # :: calculate samples
            num_extra_samples = num_samples * num_samples
            re_min, im_min = np.min(w1_arr.real), np.min(w1_arr.imag)
            re_max, im_max = np.max(w1_arr.real), np.max(w1_arr.imag)
            re_ptp, im_ptp = re_max - re_min, im_max - im_min
            _tmp = sqrt(re_ptp ** 2 +
                        (im_ptp * num_extra_samples + 2 * im_ptp) * re_ptp +
                        im_ptp ** 2) - re_ptp - im_ptp
            num_re_samples = np.floor(_tmp / im_ptp) + 2
            num_im_samples = np.floor(_tmp / re_ptp) + 2
            num_samples = num_re_samples * num_im_samples
            w1_re_approx = np.linspace(re_min, re_max, num_re_samples)
            w1_im_approx = np.linspace(im_min, im_max, num_im_samples)
            p_op_re_approx = np.zeros(
                    model_info['operator_shape'] + (num_re_samples,))
            p_op_im_approx = np.zeros(
                    model_info['operator_shape'] + (num_im_samples,))
            for i, w1_re in enumerate(w1_re_approx):
                l_op = dynamics_operator(
                        m0, w0, r1, r2, k, approx, w_rf, w1_re)
                p_op_re_approx[:, :, i] = scipy.linalg.expm(-dt * l_op)
            for i, w1_im in enumerate(w1_im_approx):
                l_op = dynamics_operator(
                        m0, w0, r1, r2, k, approx, w_rf, w1_im)
                p_op_re_approx[:, :, i] = scipy.linalg.expm(-dt * l_op)
            # perform interpolation
            p_op_arr = np.zeros(
                    model_info['operator_shape'] + (
                        pulse_info['num_steps'],))
            for i in range(model_info['operator_dim']):
                for j in range(model_info['operator_dim']):
                    p_op_arr_re = np.interp(
                            w1_arr.real, w1_re_approx, p_op_re_approx[i, j, :])
                    p_op_arr_im = np.interp(
                            w1_arr.imag, w1_im_approx, p_op_im_approx[i, j, :])
                    weighted = \
                        (p_op_arr_re * np.abs(w1_arr.real) +
                         p_op_arr_im * np.abs(w1_arr.imag)) / \
                        (np.abs(w1_arr.real) + np.abs(w1_arr.imag))
                    p_op_arr[:, i, j] = weighted
            p_op_list = [p_op_arr[:, :, j] for j in
                range(pulse_info['num_steps'])]
            p_op = mrb.mdot(*p_op_list[::-1])

    elif mode == 'reduced':
        reduced_size = 32
        chunk_marks = np.round(
                np.linspace(0, pulse_info['num_steps'] - 1, reduced_size + 1))
        dt_reduced = dt * pulse_info['num_steps'] / reduced_size
        w1_reduced_arr = np.zeros(reduced_size).astype(complex)
        for i in range(reduced_size):
            chunk = slice(chunk_marks[i], chunk_marks[i + 1])
            w1_reduced_arr[i] = np.mean(w1_arr[chunk])
        p_op_list = [
            scipy.linalg.expm(
                    -dt_reduced *
                    dynamics_operator(m0, w0, r1, r2, k, approx, w_rf, w1))
            for w1 in w1_reduced_arr]
        p_op = mrb.mdot(*p_op_list[::-1])

    else:  # exact / no additional approximation
        p_op_list = [
            scipy.linalg.expm(
                    -dt *
                    dynamics_operator(m0, w0, r1, r2, k, approx, w_rf, w1))
            for w1 in w1_arr]
        p_op = mrb.mdot(*p_op_list[::-1])
    return p_op


# ======================================================================
def propagator_spoiler(
        approx,
        efficiency=1.0):
    """

    Args:
        approx:
        efficiency:

    Returns:

    """

    model_info = _model_info(approx)
    p_op_diag = np.ones(model_info['operator_dim'])
    num_exact, num_approx = 0, 0
    for lineshape in approx:
        # 3: cartesian dims; +1: hom. operator
        base = 1 + num_exact * 3 + num_approx
        if lineshape:
            num_approx += 1
        else:
            p_op_diag[base:base + 2] -= efficiency
            num_exact += 1
    return np.diag(p_op_diag)


#  ======================================================================
def detector(
        approx,
        phase=pi / 2):
    """

    Args:
        m0:
        approx:
        phase:

    Returns:

    """
    model_info = _model_info(approx)
    detect = np.zeros(model_info['operator_dim']).astype(complex)
    num_exact, num_approx = 0, 0
    for lineshape in approx:
        # 3: cartesian dims; +1: hom. operator
        base = 1 + num_exact * 3 + num_approx
        if lineshape:
            num_approx += 1
        else:
            detect[base: base + 2] = 1.0, 1.0j
            num_exact += 1
    return detect


# ======================================================================
def signal(
        detect,
        propagator,
        m_eq):
    return np.abs(mrb.mdot(detect, propagator, m_eq))


def z_spectrum(spin_model, pulse_sequence, powers, freqs):
    pass


# ======================================================================
# :: Tests

#     The magnetization transfer signal generated by the following sequence:
#
#     RF  _()_/‾\____()_/\________________
#     Gpe ___________/≣\____________
#     Gsl ___________/≣\____________
#     Gro ______________/‾‾‾‾‾‾‾‾\__
#     ADC ______________/‾‾‾‾‾‾‾‾\__
#
#     δx:     Delay
#     σx:     Spoiler, _()_
#     Pp:     Preparation (MT) pulse
#     Ep:     Exc
#
#     RF:     RadioFrequency signal
#     Gpe:    Gradient for phase encoding
#     Gsl:    Gradient for slice selection
#     Gro:    Gradient for readout
#
#     /\      Gaussian pulse
#     {}      Sinc pulse
#     ||      Rect pulse

# ======================================================================
def test_dynamics_operator_symbolic():
    """
    Notes: import pi, sin and cos from sympy

    Returns:

    """
    # todo: reintroduce classes, but keep computation in functions
    s0, m0a, m0b, m0c, w0, r1a, r1b, r1c, r2a, r2b, r2c, k_ab, k_ac, k_bc = \
        sym.symbols('s0 m0a m0b m0c w0 r1a r1b r1c r2a r2b r2c k_ab k_ac k_bc')
    spin_model = {
        's0': s0,
        'm0': np.array((m0a, m0b, m0c)),
        'w0': np.array((w0, w0, w0)),
        'r1': np.array((r1a, r1b, r1c)),
        'r2': np.array((r2a, r2b, r2c)),
        'k': np.array((k_ab, k_ac, k_bc)),
        'approx': (None, None, 'superlorentz')}
    spin_model['m0'] /= np.sum(spin_model['m0'])
    w_rf, w1 = sym.symbols('w_rf w1')
    excitation = {
        'w_rf': w_rf,
        'w1': w1}
    m_eq = equilibrium_magnetization(spin_model['m0'], spin_model['approx'])
    l_op = dynamics_operator(
            spin_model['s0'], spin_model['m0'], spin_model['w0'],
            spin_model['r1'], spin_model['r2'], spin_model['k'],
            spin_model['approx'],
            excitation['w_rf'], excitation['w1'])
    print(m_eq)
    print(l_op)


# ======================================================================
def test_dynamics_operator_numeric():
    """
    Notes: import pi, sin and cos from numpy

    Returns:

    """
    s0 = 100
    m0a, m0b = 1.0, 0.152,
    w0 = GAMMA * B0
    r1a, r1b = 1.8, 1.0
    r2a, r2b = 32.2581, 8.4746e4
    k_ab = 0.05
    spin_model = {
        's0': s0,
        'm0': np.array((m0a, m0b)),
        'w0': np.array((w0, w0)),
        'r1': np.array((r1a, r1b)),
        'r2': np.array((r2a, r2b)),
        'k': np.array((k_ab,)),
        'approx': (None, 'superlorentz_approx')}
    spin_model['m0'] /= np.sum(spin_model['m0'])
    w_rf, w1 = GAMMA * B0 + 10, 1.0
    excitation = {
        'w_rf': np.array((w_rf,)),
        'w1': np.array((w1,))}

    m_eq = equilibrium_magnetization(spin_model['m0'], spin_model['approx'])
    l_op = dynamics_operator(
            spin_model['s0'], spin_model['m0'], spin_model['w0'],
            spin_model['r1'], spin_model['r2'], spin_model['k'],
            spin_model['approx'],
            excitation['w_rf'], excitation['w1'])
    print(spin_model)
    print(m_eq)
    print(l_op)


# ======================================================================
def test_dynamics_operator():
    """
    Full set of tests for the dynamics operator
    """
    s0 = 100
    m0a, m0b, m0c = 1.0, 0.5, 0.15
    w0 = GAMMA * B0
    r1a, r1b, r1c = 2.0, 1.0, 1.0
    r2a, r2b, r2c = 32.0, 64.0, 8.5e4
    k_ab, k_ac, k_bc = 0.05, 0.06, 0.07
    spin_model = {
        's0': s0,
        'm0': np.array((m0a, m0b, m0c)),
        'w0': np.array((w0, w0, w0)),
        'r1': np.array((r1a, r1b, r1c)),
        'r2': np.array((r2a, r2b, r2c)),
        'k': np.array((k_ab, k_ac, k_bc)),
        'approx': (None, None, 'superlorentz_approx')}
    spin_model['m0'] /= np.sum(spin_model['m0'])
    w_rf, w1 = GAMMA * B0 + 5.0, 10.0
    excitation = {
        'w_rf': np.array((w_rf,)),
        'w1': np.array((w1,))}

    m_eq = equilibrium_magnetization(spin_model['m0'], spin_model['approx'])
    l_op = dynamics_operator(
            spin_model['m0'], spin_model['w0'],
            spin_model['r1'], spin_model['r2'], spin_model['k'],
            spin_model['approx'],
            excitation['w_rf'], excitation['w1'])
    print(spin_model)
    print(m_eq)
    print(l_op)


# ======================================================================
def test_mt_sequence():
    """
    Test for the MT sequence.
    """
    s0 = 100
    m0a, m0b = 1.0, 0.15
    w0 = GAMMA * B0
    r1a, r1b = 1.8, 1.0
    r2a, r2b = 32.0, 8.5e4
    k = 0.0
    spin_model = {
        's0': s0,
        'm0': np.array((m0a, m0b)),
        'w0': np.array((w0, w0)),
        'r1': np.array((r1a, r1b)),
        'r2': np.array((r2a, r2b)),
        'k': np.array((k,)),
        'approx': (None, 'superlorentz_approx')}
    spin_model['m0'] /= np.sum(spin_model['m0'])

    num_repetitions = 300

    # sequence pulse elements
    pulse_elements = {
        'delay0': make_pulse(10.0e-3),
        'delay1': make_pulse(20.0e-3),
        'delay2': make_pulse(30.0e-3),
        'readout': make_pulse(10.0e-6, np.deg2rad(90.0)),
        'mt_prep': make_pulse(40.0e-3, np.deg2rad(220.0), 4000, 'gauss')
    }
    w_rf = GAMMA * B0

    # calculate propagators
    p_op_threshold = 5
    propagators = {}
    for key, val in pulse_elements.items():
        mode = 'interp_cubic_6' if val[0].size > p_op_threshold else 'exact'
        excitation = {
            'w1_arr': val[0],
            'dt': val[1],
            'w_rf': np.array(w_rf)}
        propagators[key] = propagator_pulse(
                spin_model['s0'], spin_model['m0'], spin_model['w0'],
                spin_model['r1'], spin_model['r2'], spin_model['k'],
                spin_model['approx'],
                excitation['w_rf'], excitation['w1_arr'], excitation['dt'],
                mode)
    # spoiler
    propagators['spoiler'] = propagator_spoiler(spin_model['approx'])

    kernel_list = (
        propagators['delay0'],
        propagators['spoiler'],
        propagators['mt_prep'],
        propagators['delay1'],
        propagators['spoiler'],
        propagators['readout'],
        propagators['delay2'])

    propagator_kernel = mrb.mdot(*kernel_list[::-1])
    propagator = scipy.linalg.fractional_matrix_power(
            propagator_kernel, num_repetitions)

    s_val = signal(
            detector(spin_model['approx']),
            propagator,
            equilibrium_magnetization(spin_model['m0'], spin_model['approx']))

    print(spin_model)
    print(propagator)
    print(s_val)


# ======================================================================
def test_approx_propagator(
        powers=(1.0,)):
    """
    Test the approximation of propagators - for speeding up.
    """
    s0 = 100
    m0a, m0b = 1.0, 0.15
    w0 = GAMMA * B0
    r1a, r1b = 1.8, 1.0
    r2a, r2b = 32.0, 8.5e4
    k = 0.0
    spin_model = {
        's0': s0,
        'm0': np.array((m0a, m0b)),
        'w0': np.array((w0, w0)),
        'r1': np.array((r1a, r1b)),
        'r2': np.array((r2a, r2b)),
        'k': np.array((k,)),
        'approx': (None, 'superlorentz_approx')}
    spin_model['m0'] /= np.sum(spin_model['m0'])
    w_rf = GAMMA * B0

    modes = ['exact']
    modes += ['linear', 'reduced']
    # modes += ['sum_simple', 'sum_order1', 'sum_sep', 'reduced']
    modes += ['poly_{}'.format(order) for order in range(4, 5)]
    modes += ['interp_{}_{}'.format(mode, num_samples)
        for mode in ['linear', 'cubic'] for num_samples in range(4, 5)]
    shapes = (
        'gauss',
        'lorentz',
        'sinc',
            # 'fermi',
            # 'random',
        'cos_sin',
    )
    for power in powers:
        for shape in shapes:
            rf_pulse = make_pulse(
                    40.0e-3, np.deg2rad(90.0 * power), 4000, shape)
            for i, mode in enumerate(modes):
                excitation = {
                    'w1_arr': rf_pulse[0],
                    'dt': rf_pulse[1],
                    'w_rf': np.array(w_rf)}
                begin_time = time.time()
                p_op = propagator_pulse(
                        spin_model['s0'], spin_model['m0'], spin_model['w0'],
                        spin_model['r1'], spin_model['r2'], spin_model['k'],
                        spin_model['approx'],
                        excitation['w_rf'], excitation['w1_arr'],
                        excitation['dt'],
                        mode)
                end_time = time.time()
                if i == 0:
                    p_op_exact = p_op
                    p_op_norm = np.sum(np.abs(p_op_exact))
                rel_error = np.sum(np.abs(p_op_exact - p_op)) / \
                            np.sum(np.abs(p_op_exact))
                # print('\n', p_op)
                print('{:>12s}, {:>24s}, err.: {:.3e}, time: {}'.format(
                        shape, mode, rel_error,
                        datetime.timedelta(0, end_time - begin_time)))


# ======================================================================
def test_z_spectrum(
        freqs=np.logspace(0, 5, 16),
        powers=np.linspace(0.1, 10, 16),
        save_file='/tmp/mri_tools/z_spectrum_approx.npz'):
    """
    Test calculation of z-spectra

    Args:
        freqs (ndarray[float]):
        powers (ndarray[float]):
        save_file (string):

    Returns:
        freq

    """
    s0 = 100
    m0a, m0b = 1.0, 0.152
    w0 = GAMMA * B0
    r1a, r1b = 1.8, 1.0
    r2a, r2b = 32.0, 8.5e4
    k = 0.0
    spin_model = {
        'm0': np.array((m0a, m0b)),
        'w0': np.array((w0, w0)),
        'r1': np.array((r1a, r1b)),
        'r2': np.array((r2a, r2b)),
        'k': np.array((k,)),
        'approx': (None, 'gauss')}
    spin_model['m0'] /= np.sum(spin_model['m0'])

    num_repetitions = 300

    # sequence pulse elements
    pulse_elements = {
        'delay0': make_pulse(10.0e-3),
        'delay1': make_pulse(10.0e-3),
        'readout': make_pulse(10.0e-6, np.deg2rad(90.0)),
    }
    w_rf = GAMMA * B0

    data = np.zeros((len(freqs), len(powers)))
    for j, freq in enumerate(freqs):
        # calculate propagators
        p_op_threshold = 5
        propagators = {}
        for key, val in pulse_elements.items():
            mode = 'interp_cubic_6' if val[0].size > p_op_threshold else 'exact'
            excitation = {
                'w1_arr': val[0],
                'dt': val[1],
                'w_rf': np.array((w_rf + freq,))}
            propagators[key] = propagator_pulse(
                    spin_model['m0'], spin_model['w0'],
                    spin_model['r1'], spin_model['r2'], spin_model['k'],
                    spin_model['approx'],
                    excitation['w_rf'], excitation['w1_arr'], excitation['dt'],
                    mode)
        # spoiler
        propagators['spoiler'] = propagator_spoiler(spin_model['approx'])
        for i, power in enumerate(powers):
            pulse_elements['mt_prep'] = make_pulse(
                    40.0e-3, np.deg2rad(90.0 * power), 4000, 'gauss')
            excitation = {
                'w1_arr': pulse_elements['mt_prep'][0],
                'dt': pulse_elements['mt_prep'][1],
                'w_rf': np.array((w_rf + freq,))
            }
            propagators['mt_prep'] = propagator_pulse(
                    spin_model['m0'], spin_model['w0'],
                    spin_model['r1'], spin_model['r2'], spin_model['k'],
                    spin_model['approx'],
                    excitation['w_rf'], excitation['w1_arr'], excitation['dt'],
                    'linear')
            kernel_list = (
                propagators['delay0'],
                propagators['spoiler'],
                propagators['mt_prep'],
                propagators['delay1'],
                propagators['spoiler'],
                propagators['readout'],
            )
            propagator_kernel = mrb.mdot(*kernel_list[::-1])
            propagator = scipy.linalg.fractional_matrix_power(
                    propagator_kernel, num_repetitions)
            data[j, i] = signal(
                    detector(spin_model['approx']),
                    propagator,
                    equilibrium_magnetization(
                            spin_model['m0'], spin_model['approx']))

    # plot results
    X, Y = np.meshgrid(powers * 90.0, np.log10(freqs))
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(
            X, Y, data, cmap=plt.cm.hot,
            rstride=1, cstride=1, linewidth=0.01, antialiased=False)
    # np.savez(save_file, freqs, powers, data)
    return data, powers, freqs


# ======================================================================
if __name__ == '__main__':
    print(__doc__)
    # test_dynamics_operator_symbolic()
    # _elapsed('test_dynamics_operator_symbolic')
    # test_dynamics_operator_numeric()
    # _elapsed('test_dynamics_operator_numeric')
    # test_mt_sequence()
    # _elapsed('test_mt_sequence')
    # test_approx_propagator()
    # _elapsed('test_approx_propagator')
    test_z_spectrum()
    _elapsed('test_z_spectrum')

    _print_elapsed()
    # profile.run('test_z_spectrum()', sort=1)
    plt.show()
