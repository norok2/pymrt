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

from numpy import pi, sin, cos, exp, sqrt
# from sympy import pi, sin, cos, exp, sqrt
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

# from mri_tools import get_first_line


# ======================================================================
# Proton Gyromagnetic Ratio
GAMMA = \
    sp.constants.physical_constants['proton gyromag. ratio'][0]
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
    # scipy.integrate.quad returns both the value and the error, here ignored
    return sp.integrate.quad(
            lambda t: superlorentz_integrand(x, t), 0.0, pi / 2.0)[0]


# ======================================================================
# todo: check that the sampling rate is appropriate: 1024 is usually enough
_SUPERLORENTZ['x'] = np.logspace(-10.0, 1.7, 256)
_SUPERLORENTZ['y'] = superlorentz(_SUPERLORENTZ['x'])
_EVENTS += [('Superlorentz Approx.', time.time())]


def _superlorentz_approx(
        x,
        x_i=_SUPERLORENTZ['x'],
        y_i=_SUPERLORENTZ['y']):
    """

    Args:
        x (ndarray[float]:
        x_i:
        y_i:

    Returns:

    """
    return np.interp(np.abs(x), x_i, y_i)


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
    return model_info


#  ======================================================================
def _bloch_core(
        r1,
        r2,
        w0,
        w_rf,
        w1):
    """

    Args:
        r1:
        r2:
        w0:
        w_rf:
        w1:

    Returns:

    """
    w1x, w1y = w1.real, w1.imag
    # w1x, w1y = re(w1), im(w1)
    return np.array([
        [r2, w0 - w_rf, -w1y],
        [w_rf - w0, r2, w1x],
        [w1y, -w1x, r1]])


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
    saturation_rate = pi * abs(w1 * w1.conjugate()) * lineshape_factor / r2
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
    return np.sum(w1_arr * dt)


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
    return w1_arr * flip_angle / np.sum(w1_arr * dt)


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
        'flip_angle': _get_flip_angle(w1_arr, dt)}
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
    x = np.linspace(roots[0] * pi, roots[1] * pi, num_steps)
    y = np.sinc(x)
    return y


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

    # detect = np.zeros(model_info['operator_base_dim']).astype(complex)
    # # print(self.num_pools, self.num_approx, self.num_exact)  # DEBUG
    # to_remove = []
    # for j, lineshape in enumerate(approx):
    #     base = 1 + j * 3  # 3: cartesian dims; +1: hom. operator
    #     # print(j, m0, approx)  # DEBUG
    #     detect[base:base + 2] = 1.0, 1.0j
    #     if lineshape:
    #         to_remove.extend([base, base + 1])
    # # print(to_approx)  # DEBUG
    # detect = np.delete(detect, to_remove, 0)
    # detect *= exp(1j * phase)

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
        m0:
        r1:
        r2:
        w0:
        k:
        w_rf (float): Modulation frequency in Hz
        w1 (complex): Excitation (carrier) frequency in Hz
        approx:

    Returns:
        l_op (ndarray[float]): The dynamics operator L
    """
    # todo: include exchange
    model_info = _model_info(approx)
    l_op = np.zeros((model_info['operator_dim'],) * 2).astype(type(m0[0]))
    # # ...to make L invertible
    # L[0, 0] = -2.0
    num_exact, num_approx = 0, 0
    for i, lineshape in enumerate(approx):
        # 3: cartesian dims; +1: hom. operator
        base = 1 + num_exact * 3 + num_approx
        bloch_core = _bloch_core(r1[i], r2[i], w0[i], w_rf, w1)
        if lineshape:
            r_rf = _sat_rate_lineshape(r2[i], w0[i], w_rf, w1, lineshape)
            # r_rf = sym.symbols('r_rf')
            l_op[base, base] = bloch_core[-1, -1] + r_rf
            l_op[base, 0] = -2.0 * r1[i] * m0[i]
            # l_op[base, base] +=
            num_approx += 1
        else:
            l_op[base:base + 3, base:base + 3] = bloch_core
            l_op[base + 2, 0] = -2.0 * r1[i] * m0[i]
            num_exact += 1
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
        l_op_sum = np.zeros((model_info['operator_dim'],) * 2)
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
        p_op_w1_sum = np.zeros((model_info['operator_dim'],) * 2)
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
    elif mode == 'poly_abs':
        fit_order = 3
        # :: calculate samples
        num_samples = 5
        w1_abs_arr = np.abs(w1_arr)
        p_op_approx = np.zeros(
                (model_info['operator_dim'],) * 2 + (num_samples,))
        w1_approx = np.linspace(
                np.min(w1_abs_arr), np.max(w1_abs_arr), num_samples)
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
        mask = (np.sum(y_arr, -1) == 0.0)
        # polyfit requires to change matrix orientation using transpose
        p_arr = np.polyfit(x_arr, y_arr.transpose(), fit_order)
        # transpose the results back
        p_arr = p_arr.transpose()
        # revert to original shape
        p_arr = p_arr.reshape(list(shape[:support_axis]) + [fit_order + 1])
        # :: approximate all propagators and calculate final result
        p_op_arr = np.zeros(
                (model_info['operator_dim'],) * 2 + (pulse_info['num_steps'],))
        for i in range(model_info['operator_dim']):
            for j in range(model_info['operator_dim']):
                p_op_arr[i, j, :] = np.polyval(p_arr[i, j, :], np.abs(w1_arr))
        p_op_list = [p_op_arr[:, :, j] for j in range(pulse_info['num_steps'])]
        p_op = mrb.mdot(*p_op_list[::-1])
    elif mode == 'interp_abs':
        # :: calculate samples
        num_samples = 5
        w1_abs_arr = np.abs(w1_arr)
        p_op_approx = np.zeros(
                (model_info['operator_dim'],) * 2 + (num_samples,))
        w1_approx = np.linspace(
                np.min(w1_abs_arr), np.max(w1_abs_arr), num_samples)
        for i, w1 in enumerate(w1_approx):
            l_op = dynamics_operator(m0, w0, r1, r2, k, approx, w_rf, w1)
            p_op_approx[:, :, i] = scipy.linalg.expm(-dt * l_op)
        # perform interpolation
        p_op_arr = np.zeros(
                (model_info['operator_dim'],) * 2 + (pulse_info['num_steps'],))
        for i in range(model_info['operator_dim']):
            for j in range(model_info['operator_dim']):
                # interp = scipy.interpolate.interp1d(
                #         w1_approx, p_op_approx[i, j, :], 'cubic',
                #         bounds_error=False, fill_value=0.0)
                # p_op_arr[i, j, :] = interp(np.abs(w1_arr))
                p_op_arr[i, j, :] = np.interp(
                        np.abs(w1_arr), w1_approx, p_op_approx[i, j, :])
        p_op_list = [p_op_arr[:, :, j] for j in range(pulse_info['num_steps'])]
        p_op = mrb.mdot(*p_op_list[::-1])
    elif mode == 'interp_complex':
        # :: calculate samples
        max_num_samples = 16
        re_min = np.min(w1_arr.real)
        re_max = np.max(w1_arr.real)
        im_min = np.min(w1_arr.imag)
        im_max = np.max(w1_arr.imag)
        re_ptp = re_max - re_min
        im_ptp = im_max - im_min
        num_re_samples = max_num_samples * re_ptp // (re_ptp + im_ptp)
        num_im_samples = max_num_samples * im_ptp // (re_ptp + im_ptp)
        w1_re = np.linspace(re_min, re_max, num_re_samples)
        w1_im = np.linspace(im_min, im_max, num_im_samples)
        w1_approx = np.kron(w1_re, 1j * w1_im)
        p_op_approx = np.zeros(
                (model_info['operator_dim'],) * 2 + (num_samples,))
        for i, w1 in enumerate(w1_approx):
            l_op = dynamics_operator(m0, w0, r1, r2, k, approx, w_rf, w1)
            p_op_approx[:, :, i] = scipy.linalg.expm(-dt * l_op)
        # perform interpolation
        p_op_arr = np.zeros(
                (model_info['operator_dim'],) * 2 + (pulse_info['num_steps'],))
        for i in range(model_info['operator_dim']):
            for j in range(model_info['operator_dim']):
                # interp = scipy.interpolate.interp1d(
                #         w1_approx, p_op_approx[i, j, :], 'cubic',
                #         bounds_error=False, fill_value=0.0)
                # p_op_arr[i, j, :] = interp(np.abs(w1_arr))
                p_op_arr[i, j, :] = np.interp(
                        np.abs(w1_arr), w1_approx, p_op_approx[i, j, :])
        p_op_list = [p_op_arr[:, :, j] for j in range(pulse_info['num_steps'])]
        p_op = mrb.mdot(*p_op_list[::-1])
    elif mode == 'reduced':
        reduced_size = 5
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
def generate_pulse(
        duration,
        norm=0.0,
        num_steps=1,
        generator_func=None,
        *args,
        **kwargs):
    """


    Args:
        num_steps:
        duration:
        norm:
        generator_func:
        args:
        kwargs:

    Returns:

    """
    w1_arr = generator_func(num_steps, *args, **kwargs).astype(complex) \
        if generator_func else np.array((norm / duration,))
    dt = duration / num_steps
    w1_arr = w1_arr * norm / np.sum(w1_arr * dt)
    return w1_arr, dt


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
        w1_arr = _shape_normal(num_steps).astype(complex)
    elif shape == 'lorentz':
        w1_arr = _shape_cauchy(num_steps).astype(complex)
    elif shape == 'sinc':
        w1_arr = _shape_sinc(num_steps).astype(complex)
    elif shape == 'rect' or shape is None:
        w1_arr = np.array((flip_angle / duration,)).astype(complex)
    else:
        w1_arr = eval(shape)(num_steps)
    dt = duration / num_steps
    norm = np.sum(w1_arr * dt)
    if norm:
        w1_arr = w1_arr * flip_angle / norm
    return w1_arr, dt


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


# ======================================================================
def signal(
        detect,
        propagator,
        m_eq):
    return np.abs(mrb.mdot(detect, propagator, m_eq))


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
def test_symbolic():
    """
    Notes: import pi, sin and cos from sympy

    Returns:

    """
    m0a, m0b, m0c, w0, r1a, r1b, r1c, r2a, r2b, r2c, k_ab, k_ac, k_bc = \
        sym.symbols('m0a m0b m0c w0 r1a r1b r1c r2a r2b r2c k_ab k_ac k_bc')
    spin_model = {
        'm0': np.array((m0a, m0b, m0c)),
        'w0': np.array((w0, w0, w0)),
        'r1': np.array((r1a, r1b, r1c)),
        'r2': np.array((r2a, r2b, r2c)),
        'k': np.array((k_ab, k_ac, k_bc)),
        'approx': (None, None, 'superlorentz')}
    w_rf, w1 = sym.symbols('w_rf w1')
    excitation = {
        'w_rf': w_rf,
        'w1': w1}
    m_eq = equilibrium_magnetization(spin_model['m0'], spin_model['approx'])
    l_op = dynamics_operator(
            spin_model['m0'], spin_model['w0'],
            spin_model['r1'], spin_model['r2'], spin_model['k'],
            spin_model['approx'],
            excitation['w_rf'], excitation['w1'])
    print(m_eq)
    print(l_op)


# ======================================================================
def test_simple():
    """
    Notes: import pi, sin and cos from numpy

    Returns:

    """
    m0a, m0b, m0c = 1.0, 0.5, 0.15
    w0 = GAMMA * B0
    r1a, r1b, r1c = 2.0, 1.0, 1.0
    r2a, r2b, r2c = 32.0, 64.0, 8.5e4
    k_ab, k_ac, k_bc = 0.05, 0.06, 0.07
    spin_model = {
        'm0': np.array((m0a, m0b, m0c)),
        'w0': np.array((w0, w0, w0)),
        'r1': np.array((r1a, r1b, r1c)),
        'r2': np.array((r2a, r2b, r2c)),
        'k': np.array((k_ab, k_ac, k_bc)),
        'approx': (None, None, 'superlorentz_approx')}
    w_rf, w1 = GAMMA * B0, 10.0
    excitation = {
        'w_rf': np.array((w_rf,)),
        'w1': np.array((w1,))}

    m_eq = equilibrium_magnetization(spin_model['m0'], spin_model['approx'])
    l_op = dynamics_operator(
            spin_model['m0'], spin_model['w0'],
            spin_model['r1'], spin_model['r2'], spin_model['k'],
            spin_model['approx'],
            excitation['w_rf'], excitation['w1'])
    print(m_eq)
    print(l_op)


# ======================================================================
def test_mt_sequence():
    """
    Notes: import pi, sin and cos from numpy

    Returns:

    """
    m0a, m0b = 1.0, 0.15
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
        'approx': (None, 'superlorentz_approx')}

    num_repetitions = 300

    # sequence pulse elements
    pulse_elements = {
        'delay0': make_pulse(10.0e-3),
        'delay1': make_pulse(20.0e-3),
        'delay2': make_pulse(30.0e-3),
        'readout': make_pulse(10.e-6, np.deg2rad(90.0)),
        'mt_prep': make_pulse(40.0e-3, np.deg2rad(220.0), 4000, 'gauss')
    }
    w_rf = GAMMA * B0

    # calculate propagators
    p_op_threshold = 5
    propagators = {}
    for key, val in pulse_elements.items():
        mode = 'poly_abs' if val[0].size > p_op_threshold else None
        excitation = {
            'w1_arr': val[0],
            'dt': val[1],
            'w_rf': np.array(w_rf)}
        propagators[key] = propagator_pulse(
                spin_model['m0'], spin_model['w0'],
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

    Args:
        powers:

    Returns:

    """
    m0a, m0b = 1.0, 0.15
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
        'approx': (None, 'superlorentz_approx')}
    w_rf = GAMMA * B0

    modes = (
        'exact',
        'sum_simple', 'sum_order1', 'sum_sep',
        'poly_abs', 'interp_abs',
        'reduced')
    shapes = (
        'gauss',
        'lorentz',
        'sinc',
        # 'fermi': _shape_fermi,
        # 'random': _shape_random,
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
                        spin_model['m0'], spin_model['w0'],
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
                print(p_op)
                print('{:>8s}, {:>12s}, err.: {:.3f}, time: {}\n'.format(
                        shape, mode, rel_error,
                        datetime.timedelta(0, end_time - begin_time)))


# ======================================================================
def test_z_spectrum(
        freqs=np.logspace(0, 5, 32),
        powers=np.linspace(0.1, 10, 32),
        save_file='z_spectrum_approx.npz'):
    """

    Args:
        freqs:
        powers:
        save_file:

    Returns:

    """
    m0a, m0b = 1.0e2, 0.15e2
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

    num_repetitions = 300

    # sequence pulse elements
    pulse_elements = {
        'delay0': make_pulse(10.0e-3),
        'delay1': make_pulse(10.0e-3),
        'readout': make_pulse(10.e-6, np.deg2rad(90.0)),
    }
    w_rf = GAMMA * B0

    data = np.zeros((len(freqs), len(powers)))
    for j, freq in enumerate(freqs):
        # calculate propagators
        p_op_threshold = 5
        propagators = {}
        for key, val in pulse_elements.items():
            mode = 'reduced' if val[0].size > p_op_threshold else None
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
                'w_rf': np.array((w_rf + freq,)),
                'mode': 'interp_abs'}
            propagators['mt_prep'] = propagator_pulse(
                    spin_model['m0'], spin_model['w0'],
                    spin_model['r1'], spin_model['r2'], spin_model['k'],
                    spin_model['approx'],
                    excitation['w_rf'], excitation['w1_arr'], excitation['dt'],
                    'interp_abs')

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
    # X, Y = np.meshgrid(powers * 90.0, np.log10(freqs))
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.plot_surface(
    #         X, Y, data, cmap=plt.cm.coolwarm,
    #         rstride=1, cstride=1, linewidth=0.01, antialiased=False)
    # np.savez(save_file, freqs, powers, data)
    return data, powers, freqs


# ======================================================================
if __name__ == '__main__':
    print(__doc__)
    # test_symbolic()
    # _EVENTS += [('test_symbolic', time.time())]
    # test_simple()
    # _EVENTS += [('test_simple', time.time())]
    # test_mt_sequence()
    # _EVENTS += [('test_mt_sequence', time.time())]
    # test_approx_propagator()
    # _EVENTS += [('test_approx_propagator', time.time())]
    data, powers, freqs = test_z_spectrum()
    _EVENTS += [('test_z_spectrum', time.time())]

    mrb._print_elapsed(_EVENTS)
    # profile.run('test_z_spectrum()', sort=1)
    plt.show()
