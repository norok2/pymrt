#!python
# -*- coding: utf-8 -*-
"""
mri_tools/sequences/matrix_algebra: solver of the Bloch-McConnell equations.
"""

# todo: add references

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
import cmath  # Mathematical functions for complex numbers
import time  # Time access and conversions
import datetime  # Basic date and time types
# import operator  # Standard operators as functions
# import collections  # High-performance container datatypes
# import argparse  # Parser for command-line options, arguments and subcommands
import itertools  # Functions creating iterators for efficient looping
# import functools  # Higher-order functions and operations on callable objects
# import subprocess  # Subprocess management
# import multiprocessing  # Process-based parallelism
# import csv  # CSV File Reading and Writing [CSV: Comma-Separated Values]
# import json  # JSON encoder and decoder [JSON: JavaScript Object Notation]
import warnings  # Warning control
# import profile  # Deterministic Profiler


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


# :: External Imports Submodules
import matplotlib.pyplot as plt  # Matplotlib's pyplot: MATLAB-like syntax
import mpl_toolkits.mplot3d as mpl3  # Matplotlib's 3D support
import scipy.optimize  # SciPy: Optimization
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
import mri_tools.base as mrb

# import mri_tools.geometry as mrg
# import mri_tools.plot as mrp
# import mri_tools.segmentation as mrs

# from mri_tools import INFO
# from mri_tools import VERB_LVL
# from mri_tools import D_VERB_LVL
from mri_tools.base import elapsed, print_elapsed

# from mri_tools import get_first_line


# ======================================================================
# Proton Gyromagnetic Ratio
# rad/s/T
GAMMA = \
    sp.constants.physical_constants['proton gyromag. ratio'][0]
# Hz/T
GAMMA_BAR = 1.0e-6 * \
            sp.constants.physical_constants['proton gyromag. ratio over 2 pi'][
                0]

# Magnetic Field Strength
B0 = 7.0  # T

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
_SUPERLORENTZ['x'] = np.logspace(-10.0, 1.8, 256)
_SUPERLORENTZ['y'] = superlorentz(_SUPERLORENTZ['x'])
elapsed('Superlorentz Approx.')


# ======================================================================
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


# ======================================================================
def _sat_rate_lineshape(
        r2,
        w0,
        w_c,
        w1,
        lineshape):
    """

    Args:
        r2 (float): transverse relaxation rates in Hz
        w0 (float): resonance angular frequency in rad/s
        w_c (float): carrier angular frequency of the pulse excitation in rad/s
        w1 (complex): pulse excitation angular frequency in rad/s
        lineshape (str): the lineshape of the Henkelman absorption term

    Returns:
        saturation_rate (float): the Henkelman absorption saturation rate
    """
    if lineshape == 'superlorentz':
        lineshape_factor = superlorentz((w0 - w_c) / r2)
    elif lineshape == 'superlorentz_approx':
        lineshape_factor = _superlorentz_approx((w0 - w_c) / r2)
    elif lineshape in ('cauchy', 'lorentz'):
        lineshape_factor = \
            1.0 / (pi * (1.0 + ((w0 - w_c) / r2) ** 2.0))
    elif lineshape in ('gauss', 'normal'):
        lineshape_factor = \
            exp(- ((w0 - w_c) / r2) ** 2.0 / 2.0) / sqrt(2.0 * pi)
    else:
        lineshape_factor = 1.0
    saturation_rate = pi * abs(w1) ** 2.0 * lineshape_factor / r2
    return saturation_rate


# ======================================================================
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


# ======================================================================
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


# ======================================================================
def _shape_sinc(
        num_steps,
        roots=(3.0, 3.0)):
    """

    Args:
        num_steps:
        roots:

    Returns:

    """
    x = np.linspace(-roots[0] * pi, roots[1] * pi, num_steps)
    y = sinc(x)
    return y


# ======================================================================
def _shape_cos_sin(
        num_steps,
        roots=(1.0, 1.0)):
    """

    Args:
        num_steps:
        roots:

    Returns:

    """
    x = np.linspace(-roots[0] * pi, roots[1] * pi, num_steps)
    y_re = cos(x)
    y_im = sin(x)
    return y_re + 1j * y_im


# ======================================================================
def _shape_from_file(
        filename,
        dirpath='pulses'):
    """

    Args:
        num_steps:
        roots:

    Returns:

    """
    tmp_dirpaths = [
        os.path.realpath(dirpath),
        os.path.join(os.path.dirname(__file__), dirpath),
    ]
    for tmp_dirpath in tmp_dirpaths:
        if os.path.isdir(tmp_dirpath):
            dirpath = tmp_dirpath
            break
    filepath = os.path.join(
        dirpath, filename + mrb.add_extsep(mrb.EXT['tab']))
    arr = np.loadtxt(filepath)
    if arr.ndim == 1:
        y_re = arr
        y_im = 0.0
    elif arr.ndim > 1:
        y_re = arr[:, 0]
        y_im = arr[:, 1]
    if arr.ndim > 2:
        warnings.warn('unknown pulse format in: ' + "'{}'")
    return y_re + 1j * y_im


# ======================================================================
def dynamics_operator(
        spin_model,
        w_c,
        w1):
    """
    Calculate the Bloch-McConnell dynamics operator, L.

    Args:
        spin_model (SpinModel): spin model term of the dynamics operator
        w_c (float): carrier angular frequency of the pulse excitation in rad/s
        w1 (complex): pulse excitation angular frequency in rad/s

    Returns:
        l_op (ndarray[float]): The dynamics operator L
    """
    l_op = np.zeros(spin_model.operator_shape).astype(spin_model.dtype)
    num_exact, num_approx = 0, 0
    for i, lineshape in enumerate(spin_model.approx):
        # 3: cartesian dims; +1: hom. operator
        base = 1 + num_exact * 3 + num_approx
        # w1x, w1y = re(w1), im(w1)  # for symbolic
        w1x, w1y = w1.real, w1.imag
        bloch_core = np.array([
            [0.0, - w_c, -w1y],
            [w_c, 0.0, w1x],
            [w1y, -w1x, 0.0]])
        if lineshape:
            r_rf = _sat_rate_lineshape(
                spin_model.r2[i], spin_model.w0[i], w_c, w1, lineshape)
            # r_rf = sym.symbols('r_rf')  # for symbolic
            l_op[base, base] += bloch_core[-1, -1] + r_rf
            num_approx += 1
        else:
            l_op[base:base + 3, base:base + 3] += bloch_core
            num_exact += 1
    return spin_model.l_op + l_op


# ======================================================================
def _propagator_sum(
        pulse_exc,
        spin_model):
    """
    Calculate the time-evolution propagator assuming commutativity.

    The time evolution propagator expm(-L * Dt) can be estimated using:
        expm(sum(M_i)) = prod(expm(M_i))
    which is only valid for commuting operators.

    Args:
        pulse_exc (PulseExc): the e.m. pulse excitation to manipulate the spins
        spin_model (SpinModel): the model for the spin system
        dt (float):

    Returns:
        p_op (ndarray): The (approximated) propagator operator P.
    """
    l_op_sum = np.zeros(spin_model.operator_shape)
    for w1 in pulse_exc.w1_arr:
        l_op = dynamics_operator(spin_model, pulse_exc.w_c, w1)
        l_op_sum += pulse_exc.dt * l_op
    return scipy.linalg.expm(-l_op_sum)


# ======================================================================
def _propagator_sum_order1(
        pulse_exc,
        spin_model):
    """
    Calculate the time-evolution propagator assuming quasi-commutativity.

    The time evolution propagator expm(-L * Dt) can be estimated using:
        expm(sum(M_i)) = prod(expm(M_i))
    including a pseudo-first-order correction for quasi-commuting operators:
        expm(sum(M_i) + sum([M_i, M_j]/2) = prod(expm(M_i))

    Args:
        pulse_exc (PulseExc): the e.m. pulse excitation to manipulate the spins
        spin_model (SpinModel): the model for the spin system

    Returns:
        p_op (ndarray): The (approximated) propagator operator P.
    """
    l_op_list = []
    for w1 in pulse_exc.w1_arr:
        l_op = dynamics_operator(spin_model, pulse_exc.w_c, w1)
        l_op_list.append(pulse_exc.dt * l_op)
    l_op_sum = sum(l_op_list)
    # pseudo-first-order correction
    comm_list = [
        mrb.commutator(l_op_list[i], l_op_list[i + 1]) / 2.0
        for i in range(len(l_op_list[:-1]))]
    comm_sum = sum(comm_list)
    return scipy.linalg.expm(-(l_op_sum + comm_sum))


# ======================================================================
def _propagator_sum_sep(
        pulse_exc,
        spin_model):
    """
    Calculate the time-evolution propagator using a separated sum
    approximation.

    The time evolution propagator expm(-L * Dt) can be estimated assuming:
        prod(expm(-L_i * Dt)) = powm(expm(L_free)) + expm(sum(L_w1_i * Dt))
        (L_i = L_free + L_w1_i)

    Args:
        pulse_exc (PulseExc): the e.m. pulse excitation to manipulate the spins
        spin_model (SpinModel): the model for the spin system

    Returns:
        p_op (ndarray): The (approximated) propagator operator P.
    """
    p_op_w1_sum = np.zeros(spin_model.operator_shape)
    l_op_free = dynamics_operator(spin_model, pulse_exc.w_c, 0.0)
    for w1 in pulse_exc.w1_arr:
        l_op = dynamics_operator(spin_model, pulse_exc.w_c, w1)
        p_op_w1_sum += (pulse_exc.dt * (l_op - l_op_free))
    # calculate propagators
    p_op_free = scipy.linalg.expm(-l_op_free * pulse_exc.dt)
    p_op_pow = scipy.linalg.fractional_matrix_power(
        p_op_free, pulse_exc.num_steps)
    p_op_w1 = scipy.linalg.expm(-p_op_w1_sum)
    return np.dot(p_op_pow, p_op_w1_sum)


# ======================================================================
def _propagator_poly(
        pulse_exc,
        spin_model,
        fit_order=3,
        num_samples=None):
    """
    Calculate the time-evolution propagator using polynomial approximation.

    The time evolution propagator expm(-L * Dt) is estimated with a (fast)
    element-wise polynomial fit, on exact values calculated over the w1 domain.

    Args:
        pulse_exc (PulseExc): the e.m. pulse pulse_exc to manipulate the spins
        spin_model (SpinModel): the model for the spin system
        fit_order (int): the order to be used for the polynomial fit
        num_samples (int): number of samples used for the polynomial fit

    Returns:
        p_op (ndarray): The (approximated) propagator operator P.
    """
    if not num_samples or num_samples < fit_order + 1:
        num_samples = fit_order + 1
    if pulse_exc.is_real or pulse_exc.is_imag:
        _w1_arr = pulse_exc.w1_arr.real if pulse_exc.is_real else \
            pulse_exc.w1_arr.imag
        # :: calculate samples
        p_op_approx = np.zeros(
            spin_model.operator_shape + (num_samples,))
        w1_approx = np.linspace(
            np.min(_w1_arr), np.max(_w1_arr), num_samples)
        for i, w1 in enumerate(w1_approx):
            l_op = dynamics_operator(spin_model, pulse_exc.w_c, w1)
            p_op_approx[:, :, i] = scipy.linalg.expm(-pulse_exc.dt * l_op)
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
            (pulse_exc.num_steps,) + spin_model.operator_shape)
        for i in range(spin_model.operator_dim):
            for j in range(spin_model.operator_dim):
                p_op_arr[:, i, j] = np.polyval(p_arr[i, j, :], _w1_arr)
        p_op_list = [p_op_arr[j, :, :] for j in range(pulse_exc.num_steps)]
        p_op = mrb.mdot(*p_op_list[::-1])
    else:
        # :: calculate samples
        num_extra_samples = num_samples * num_samples
        re_min = np.min(pulse_exc.w1_arr.real)
        im_min = np.min(pulse_exc.w1_arr.imag)
        re_max = np.max(pulse_exc.w1_arr.real)
        im_max = np.max(pulse_exc.w1_arr.imag)
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
            spin_model.operator_shape + (num_samples,))
        w1_approx = np.zeros(
            num_re_samples * num_im_samples).astype(complex)
        i = 0
        for w1_re in w1_re_approx:
            for w1_im in w1_im_approx:
                l_op = dynamics_operator(
                    spin_model, pulse_exc.w_c, w1_re + 1j * w1_im)
                p_op_approx[:, :, i] = scipy.linalg.expm(-pulse_exc.dt * l_op)
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
            (pulse_exc.num_steps,) + spin_model.operator_shape)
        for i in range(spin_model.operator_dim):
            for j in range(spin_model.operator_dim):
                p_op_arr[:, i, j] = np.real(
                    np.polyval(p_arr[i, j, :], pulse_exc.w1_arr))
        p_op_list = [p_op_arr[j, :, :] for j in range(pulse_exc.num_steps)]
        p_op = mrb.mdot(*p_op_list[::-1])
    return p_op


# ======================================================================
def _propagator_interp(
        pulse_exc,
        spin_model,
        method='cubic',
        num_samples=5):
    """
    Calculate the time-evolution propagator using interpolation.

    The time evolution propagator expm(-L * Dt) is estimated with a (fast)
    element-wise interpolation approximation, on exact values calculated
    over the w1 domain.

    Args:
        pulse_exc (PulseExc): the e.m. pulse excitation to manipulate the spins
        spin_model (SpinModel): the model for the spin system
        method (str): the method used by the interpolating function
            'scipy.interpolate.griddata'
        num_samples (int): number of samples for the interpolation

    Returns:
        p_op (ndarray): The (approximated) propagator operator P.
    """
    if pulse_exc.is_real or pulse_exc.is_imag:
        _w1_arr = pulse_exc.w1_arr.real \
            if pulse_exc.is_real else pulse_exc.w1_arr.imag
        # :: calculate samples
        p_op_approx = np.zeros(spin_model.operator_shape + (num_samples,))
        w1_approx = np.linspace(
            np.min(_w1_arr), np.max(_w1_arr), num_samples)
        for i, w1 in enumerate(w1_approx):
            l_op = dynamics_operator(spin_model, pulse_exc.w_c, w1)
            p_op_approx[:, :, i] = scipy.linalg.expm(-pulse_exc.dt * l_op)
        # perform interpolation
        p_op_arr = np.zeros(
            (pulse_exc.num_steps,) + spin_model.operator_shape)
        for i in range(spin_model.operator_dim):
            for j in range(spin_model.operator_dim):
                p_op_arr[:, i, j] = scipy.interpolate.griddata(
                    w1_approx, p_op_approx[i, j, :], _w1_arr,
                    method=method, fill_value=0.0)
        p_op_list = [p_op_arr[j, :, :] for j in
                     range(pulse_exc.num_steps)]
        p_op = mrb.mdot(*p_op_list[::-1])
    else:
        # :: calculate samples
        num_extra_samples = num_samples * num_samples
        re_min = np.min(pulse_exc.w1_arr.real)
        im_min = np.min(pulse_exc.w1_arr.imag)
        re_max = np.max(pulse_exc.w1_arr.real)
        im_max = np.max(pulse_exc.w1_arr.imag)
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
            spin_model.operator_shape + (num_samples,))
        w1_approx = np.zeros((num_re_samples * num_im_samples, 2))
        i = 0
        for w1_re in w1_re_approx:
            for w1_im in w1_im_approx:
                l_op = dynamics_operator(
                    spin_model, pulse_exc.w_c, w1_re + 1j * w1_im)
                p_op_approx[:, :, i] = scipy.linalg.expm(-pulse_exc.dt * l_op)
                w1_approx[i, :] = (w1_re, w1_im)
                i += 1
        # perform interpolation
        p_op_arr = np.zeros(
            (pulse_exc.num_steps,) + spin_model.operator_shape)
        for i in range(spin_model.operator_dim):
            for j in range(spin_model.operator_dim):
                p_op_arr[:, i, j] = scipy.interpolate.griddata(
                    w1_approx, p_op_approx[i, j, :],
                    (pulse_exc.w1_arr.real, pulse_exc.w1_arr.imag),
                    method=method, fill_value=0.0)
        p_op_list = [p_op_arr[j, :, :] for j in range(pulse_exc.num_steps)]
        p_op = mrb.mdot(*p_op_list[::-1])
    return p_op


# ======================================================================
def _propagator_linear(
        pulse_exc,
        spin_model,
        num_samples=32):
    """
    Calculate the time-evolution propagator using linear interpolation.

    The time evolution propagator expm(-L * Dt) is estimated with a (fast)
    element-wise linear interpolation approximation, on exact values calculated
    over the w1 domain. Assumes real and imaginary part to be independent.

    Args:
        pulse_exc (PulseExc): the e.m. pulse excitation to manipulate the spins
        spin_model (SpinModel): the model for the spin system
        num_samples (int): number of samples for the interpolation

    Returns:
        p_op (ndarray): The (approximated) propagator operator P.
    """
    if pulse_exc.is_real or pulse_exc.is_imag:
        _w1_arr = pulse_exc.w1_arr.real \
            if pulse_exc.is_real else pulse_exc.w1_arr.imag
        # :: calculate samples
        p_op_approx = np.zeros(
            spin_model.operator_shape + (num_samples,))
        w1_approx = np.linspace(
            np.min(_w1_arr), np.max(_w1_arr), num_samples)
        for i, w1 in enumerate(w1_approx):
            l_op = dynamics_operator(spin_model, pulse_exc.w_c, w1)
            p_op_approx[:, :, i] = scipy.linalg.expm(-pulse_exc.dt * l_op)
        # perform interpolation
        p_op_arr = np.zeros(
            (pulse_exc.num_steps,) + spin_model.operator_shape)
        for i in range(spin_model.operator_dim):
            for j in range(spin_model.operator_dim):
                p_op_arr[:, i, j] = np.interp(
                    _w1_arr, w1_approx, p_op_approx[i, j, :])
        p_op_list = [p_op_arr[j, :, :] for j in
                     range(pulse_exc.num_steps)]
        p_op = mrb.mdot(*p_op_list[::-1])
    else:
        # :: calculate samples
        num_extra_samples = num_samples * num_samples
        re_min = np.min(pulse_exc.w1_arr.real)
        im_min = np.min(pulse_exc.w1_arr.imag)
        re_max = np.max(pulse_exc.w1_arr.real)
        im_max = np.max(pulse_exc.w1_arr.imag)
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
            spin_model.operator_shape + (num_re_samples,))
        p_op_im_approx = np.zeros(
            spin_model.operator_shape + (num_im_samples,))
        for i, w1_re in enumerate(w1_re_approx):
            l_op = dynamics_operator(spin_model, pulse_exc.w_c, w1_re)
            p_op_re_approx[:, :, i] = scipy.linalg.expm(-pulse_exc.dt * l_op)
        for i, w1_im in enumerate(w1_im_approx):
            l_op = dynamics_operator(spin_model, pulse_exc.w_c, w1_im)
            p_op_re_approx[:, :, i] = scipy.linalg.expm(-pulse_exc.dt * l_op)
        # perform interpolation
        p_op_arr = np.zeros((pulse_exc.num_steps,) + spin_model.operator_shape)
        for i in range(spin_model.operator_dim):
            for j in range(spin_model.operator_dim):
                p_op_arr_re = np.interp(
                    pulse_exc.w1_arr.real,
                    w1_re_approx, p_op_re_approx[i, j, :])
                p_op_arr_im = np.interp(
                    pulse_exc.w1_arr.imag,
                    w1_im_approx, p_op_im_approx[i, j, :])
                weighted = \
                    (p_op_arr_re * np.abs(pulse_exc.w1_arr.real) +
                     p_op_arr_im * np.abs(pulse_exc.w1_arr.imag)) / \
                    (np.abs(pulse_exc.w1_arr.real) +
                     np.abs(pulse_exc.w1_arr.imag))
                p_op_arr[:, i, j] = weighted
        p_op_list = [p_op_arr[j, :, :] for j in range(pulse_exc.num_steps)]
        p_op = mrb.mdot(*p_op_list[::-1])
    return p_op


# ======================================================================
def _propagator_reduced(
        pulse_exc,
        spin_model,
        num_resamples=32):
    """
    Calculate the time-evolution propagator of a coarser pulse excitation.

    The time evolution propagator expm(-L * Dt) is estimated on an approximated
    pulse excitation, obtained by reducing the number of samples of the
    original pulse excitation.

    Args:
        pulse_exc (PulseExc): the e.m. pulse excitation to manipulate the spins
        spin_model (SpinModel): the model for the spin system
        num_resamples (int): number of samples for the reduced resampling

    Returns:
        p_op (ndarray): The (approximated) propagator operator P.
    """
    chunk_marks = np.round(
        np.linspace(0, pulse_exc.num_steps - 1, num_resamples + 1))
    dt_reduced = pulse_exc.dt * pulse_exc.num_steps / num_resamples
    w1_reduced_arr = np.zeros(num_resamples).astype(pulse_exc.w1_arr[0])
    for i in range(num_resamples):
        chunk = slice(chunk_marks[i], chunk_marks[i + 1])
        w1_reduced_arr[i] = np.mean(pulse_exc.w1_arr[chunk])
    p_op_list = [
        sp.linalg.expm(
            -dt_reduced *
            dynamics_operator(spin_model, pulse_exc.w_c, w1))
        for w1 in w1_reduced_arr]
    return mrb.mdot(*p_op_list[::-1])


# ======================================================================
def z_spectrum(spin_model, pulse_sequence, amplitudes, freqs):
    pass


# ======================================================================
class SpinModel:
    """
    Model of the spin system
    """

    def __init__(
            self,
            s0,
            mc,
            w0,
            r1,
            r2,
            k,
            approx=None):
        """
        Base constructor of the spin model class.

        Args:
            s0 (float): magnetization vector magnitude scaling in arb.units
            mc (ndarray[float]): magnetization concentration ratios in #
            w0 (ndarray[float]): resonance angular frequencies in rad/s
            r1 (ndarray[float]): longitudinal relaxation rates in Hz
            r2 (ndarray[float]): transverse relaxation rates in Hz
            k (ndarray[float]): pool-pool exchange rate constants in Hz

        Returns:
            None
        """
        self.num_pools = len(mc)
        # exchange at equilibrium between each two pools
        self.num_exchange = sp.misc.comb(self.num_pools, 2)
        self.approx = approx \
            if approx is not None else [None] * self.num_pools
        self.num_approx = sum(
            [0 if item is None else 1 for item in self.approx])
        self.num_exact = self.num_pools - self.num_approx
        self.operator_dim = 1 + 3 * self.num_exact + self.num_approx
        self.operator_shape = (self.operator_dim,) * 2
        self.operator_base_dim = 1 + 3 * self.num_pools
        self.operator_base_shape = (self.operator_base_dim,) * 2
        self.dtype = type(mc[0])
        # simple check on the number of parameters
        if self.num_pools != len(r1) != len(r2) != len(approx) \
                or len(k) != self.num_exchange:
            raise IndexError('inconsistent spin model')
        sum_mc = sum(mc)
        self.s0 = s0
        self.mc = np.array(mc) / sum_mc
        self.m0 = s0 * self.mc
        self.w0 = np.array(w0)
        self.r1 = np.array(r1)
        self.r2 = np.array(r2)
        self.k = np.array(k)
        self.k_op = self.kinetics_operator()
        self.ignore_k_transverse = True
        self.l_op = self.dynamics_operator()
        self.m_eq = self.equilibrium_magnetization()
        self.det = self.detector()

    # @classmethod
    # def serialized(
    #         cls,
    #         duration,
    #         flip_angle=90.0,
    #         num_steps=1,
    #         shape='rect',
    #         shape_kwargs=None,
    #         w_c=None,
    #         propagator_mode=None,
    #         propagator_kwargs=None):
    #     """
    #     Generate a pulse excitation with a specific shape.
    #
    #     Args:
    #         duration (float): duration of the pulse in s
    #         flip_angle (float): flip angle of the excitation in deg
    #         num_steps (int): number of sampling point for the pulse
    #         shape (str|unicode): name of the desired pulse shape.
    #             Note: a function named '_shape_[SHAPE]' must exist.
    #                 [normal|cauchy|sinc|cos_sin]
    #         shape_kwargs (dict|None): keyword arguments for the shape
    # function
    #         w_c (float): carrier angular frequency of the pulse excitation in
    #             rad/s
    #         propagator_mode (str|unicode): calculation mode for the
    # propagator.
    #             Note: a function named '_propagator_[MODE]' must exist.
    #                 [sum|sum_order1|sum_sep|poly|interp|linear|reduced]
    #         propagator_kwargs (dict|None): keyword arguments for the
    #             propagator function
    #
    #     Returns:
    #         pulse_exc (PulseExc): the generated pulse excitation
    #     """
    #     if shape_kwargs is None:
    #         shape_kwargs = {}
    #     if shape == 'gauss':
    #         w1_arr = _shape_normal(num_steps, **shape_kwargs)
    #     elif shape == 'lorentz':
    #         w1_arr = _shape_cauchy(num_steps, **shape_kwargs)
    #     elif shape == 'sinc':
    #         w1_arr = _shape_sinc(num_steps, **shape_kwargs)
    #     elif shape == 'rect':
    #         w1_arr = np.array((1.0,) * num_steps)
    #     else:
    #         try:
    #             shape_func = eval('_shape_' + shape)
    #             w1_arr = shape_func(num_steps, **shape_kwargs)
    #         except:
    #             raise ValueError('{}: unknown shape'.format(shape))
    #     self = cls(duration, w1_arr, w_c)
    #     self.shape = shape
    #     self.shape_kwargs = shape_kwargs
    #     self.set_flip_angle(flip_angle)
    #     if propagator_mode is not None:
    #         self.propagator_mode = propagator_mode
    #     if propagator_kwargs is not None:
    #         self.propagator_kwargs = propagator_kwargs
    #     return self

    def equilibrium_magnetization(self):
        """
        Generate the equilibrium magnetization vector.

        This is used in conjunction with the propagator operator in order to
        calculate the signal intensity.

        Note that B0 is assumed parallel to z-axis, therefore:
            - the transverse magnetization is zero
            - the longitudinal magnetization is only in the z-axis

        Returns:
            m_eq (ndarray): the equilibrium magnetization vector
        """
        m_eq = np.zeros(self.operator_dim).astype(self.dtype)
        m_eq[0] = 0.5
        num_exact, num_approx = 0, 0
        for m0z, lineshape in zip(self.m0, self.approx):
            # 3: cartesian dims; +1: hom. operator
            if lineshape:
                pos = 1 + num_exact * 3 + num_approx
                num_approx += 1
            else:
                pos = 1 + num_exact * 3 + num_approx + 2
                num_exact += 1
            m_eq[pos] = m0z
        return m_eq

    def detector(
            self,
            phase=pi / 2):
        """
        Generate the detector vector, used to calculate the signal.

        This is used in conjunction with the propagator operator in order to
        calculate the signal intensity.

        Args:
            phase (float): the phase in rad of the detector system.

        Returns:
            det (ndarray): the detector vector
        """
        det = np.zeros(self.operator_dim).astype(complex)
        num_exact, num_approx = 0, 0
        for lineshape in self.approx:
            # 3: cartesian dims; +1: hom. operator
            base = 1 + num_exact * 3 + num_approx
            if lineshape:
                num_approx += 1
            else:
                det[base: base + 2] = 1.0, 1.0j
                num_exact += 1
        det *= exp(1j * phase)
        return det

    def dynamics_operator(self):
        """
        Calculate L_spin: the excitation-independent part of the
        Bloch-McConnell dynamics operator of the spin system.

        Returns:
            l_op (ndarray): The dynamics operator L
        """
        num_pools = len(self.approx)
        l_op = np.zeros(self.operator_base_shape).astype(self.dtype)
        # # ...to make L invertible
        # L[0, 0] = -2.0
        to_remove = []
        for i, lineshape in enumerate(self.approx):
            base = 1 + i * 3
            # Bloch operator core...
            l_op[base:base + 3, base:base + 3] = np.array([
                [self.r2[i], self.w0[i], 0.0],
                [-self.w0[i], self.r2[i], 0.0],
                [0.0, 0.0, self.r1[i]]])
            # ...additional modification for homogeneous form
            l_op[base + 2, 0] = -2.0 * self.r1[i] * self.m0[i]
            # deal with approximations
            if lineshape:
                to_remove.extend([base, base + 1])
        # include pool-pool interaction
        locator = np.diag([0.0, 0.0, 1.0]) \
            if self.ignore_k_transverse else np.eye(3)
        m0_op = np.repeat(self.m0.reshape((-1, 1)), self.num_pools, axis=1)
        l_k_op = self.k_op * m0_op - np.diag(np.dot(self.k_op, self.m0))
        l_op[1:, 1:] -= np.kron(l_k_op, locator)
        # remove transverse components of approximated pools
        l_op = np.delete(l_op, to_remove, 0)
        l_op = np.delete(l_op, to_remove, 1)
        return l_op

    def kinetics_operator(self):
        """
        Calculate the symmetric operator of the pool-pool exchange constants.

        Returns:
            k_op (ndarray):
        """
        indexes = sorted(
            list(itertools.combinations(range(self.num_pools), 2)),
            key=lambda x: x[1])
        k_op = np.zeros((self.num_pools,) * 2).astype(self.dtype)
        for k, index in zip(self.k, indexes):
            k_op[index] = k
            k_op[index[::-1]] = k
        return k_op

    def __repr__(self):
        text = 'SpinModel: '
        names = ['m0', 'w0', 'r1', 'r2', 'k']
        for name in names:
            text += '{}={}  '.format(name, self.__dict__[name])
        return text


# ======================================================================
class PulseExc:
    """
    Pulse excitation interacting with the spin system
    """

    def __init__(
            self,
            duration,
            w1_arr=None,
            w_c=None):
        """
        Base constructor of the pulse excitation class.

        Args:
            duration (float): duration of the pulse in s
            w1_arr (ndarray[complex]): array of the pulse excitation angular
                frequency in rad/s
            w_c (float): carrier angular frequency of the pulse excitation in
                rad/s

        Returns:
            None
        """
        self.shape = 'custom'
        self.duration = duration
        self.w_c = w_c
        self.w1_arr = w1_arr if w1_arr is not None else np.array((1.0,))
        self.num_steps = len(w1_arr)
        self.dt = self.duration / self.num_steps
        if self.num_steps < 1 or self.dt < 0.0:
            raise ValueError('inconsistent pulse excitation')
        self.norm = self.get_norm()
        self.is_real = np.isreal(w1_arr[0])
        self.is_imag = np.isreal(w1_arr[0] / 1j)
        self.propagator_mode = 'exact'
        self.propagator_kwargs = {}

    @property
    def flip_angle(self):
        """
        Returns:
            flip_angle (float): the flip angle in deg of the pulse excitation.

            Mathematically, with the proper units, this is equivalent to the
            absolute norm of the array:
                sum(abs(a_i))
        """
        return np.rad2deg(self.norm)

    @classmethod
    def shaped(
            cls,
            duration,
            flip_angle=90.0,
            num_steps=1,
            shape='rect',
            shape_kwargs=None,
            w_c=None,
            propagator_mode=None,
            propagator_kwargs=None):
        """
        Generate a pulse excitation with a specific shape.

        Args:
            duration (float): duration of the pulse in s
            flip_angle (float): flip angle of the excitation in deg
            num_steps (int): number of sampling point for the pulse
            shape (str|unicode): name of the desired pulse shape.
                Note: a function named '_shape_[SHAPE]' must exist.
                    [normal|cauchy|sinc|cos_sin]
            shape_kwargs (dict|None): keyword arguments for the shape function
            w_c (float): carrier angular frequency of the pulse excitation in
                rad/s
            propagator_mode (str|unicode): calculation mode for the propagator.
                Note: a function named '_propagator_[MODE]' must exist.
                    [sum|sum_order1|sum_sep|poly|interp|linear|reduced]
            propagator_kwargs (dict|None): keyword arguments for the
                propagator function

        Returns:
            pulse_exc (PulseExc): the generated pulse excitation
        """
        if shape_kwargs is None:
            shape_kwargs = {}
        if shape == 'gauss':
            w1_arr = _shape_normal(num_steps, **shape_kwargs)
        elif shape == 'lorentz':
            w1_arr = _shape_cauchy(num_steps, **shape_kwargs)
        elif shape == 'sinc':
            w1_arr = _shape_sinc(num_steps, **shape_kwargs)
        elif shape == 'rect':
            w1_arr = np.ones((num_steps,)).astype(complex)
        elif shape.startswith('_from_'):
            filename = shape[len('_from_'):]
            w1_arr = _shape_from_file(filename)
        else:
            try:
                shape_func = eval('_shape_' + shape)
                w1_arr = shape_func(num_steps, **shape_kwargs)
            except NameError:
                msg = '{}: unknown shape. Fall back to rect'.format(shape)
                warnings.warn(msg)
                w1_arr = np.ones((num_steps,)).astype(complex)
        self = cls(duration, w1_arr, w_c)
        self.shape = shape
        self.shape_kwargs = shape_kwargs
        self.set_flip_angle(flip_angle)
        if propagator_mode is not None:
            self.propagator_mode = propagator_mode
        if propagator_kwargs is not None:
            self.propagator_kwargs = propagator_kwargs
        return self

    def get_norm(self):
        """
        Calculate the norm of the pulse excitation in rad

        Returns:
            norm (float): the norm of the pulse excitation.
        """
        return np.sum(np.abs(self.w1_arr * self.dt))

    def set_norm(self, new_norm):
        """
        Set a new norm for the pulse excitation in rad

        This is equivalent to setting the flip angle (mind a scaling factor).
        Note that this modifies the w1_arr.

        Args:
            new_norm (float): the new norm of the pulse.

        Returns:
            None.
        """
        if self.norm == 0.0:
            self.w1_arr = np.ones_like(self.w1_arr)
            self.norm = self.get_norm()
        self.w1_arr = self.w1_arr * new_norm / self.norm
        self.norm = new_norm

    def get_flip_angle(self):
        """
        Get the flip angle of the pulse excitation in deg

        Returns:
            norm (float): the norm of the pulse excitation.
        """
        return self.flip_angle

    def set_flip_angle(self, new_flip_angle):
        """
        Set a new flip angle for the pulse excitation.

        This is equivalent to setting the norm (mind a scaling factor).
        Note that this modifies the w1_arr.

        Args:
            new_flip_angle (float): the new flip angle of the pulse in deg

        Returns:
            None.
        """
        self.set_norm(np.deg2rad(new_flip_angle))

    def propagator(
            self,
            spin_model,
            *args,
            **kwargs):
        """
        Calculate the Bloch-McConnell propagator: expm(-L * Dt).

        L is the dynamics operator of the rf-excited spin system.
        Dt is a time interval where spin exchange dynamics is negligible.

        Args:
            spin_model (SpinModel): The physical model for the spin pools
            mode (string|None): Approximation to use for faster computation.

        Returns:
            p_op (ndarray): The propagator P.
        """
        if not kwargs:
            kwargs = self.propagator_kwargs
        if self.propagator_mode == 'exact':
            p_op_list = [
                sp.linalg.expm(
                    -self.dt * dynamics_operator(spin_model, self.w_c, w1))
                for w1 in self.w1_arr]
            p_op = mrb.mdot(*p_op_list[::-1])
        else:
            try:
                p_op_func = eval('_propagator_' + self.propagator_mode)
                p_op = p_op_func(self, spin_model, *args, **kwargs)
            except NameError:
                msg = '{}: unknown approximation'.format(self.propagator_mode)
                warnings.warn(msg)
                p_op_list = [
                    sp.linalg.expm(
                        -self.dt * dynamics_operator(spin_model, self.w_c, w1))
                    for w1 in self.w1_arr]
                p_op = mrb.mdot(*p_op_list[::-1])
        return p_op

    def __repr__(self):
        text = '{}: '.format(self.__class__.__name__)
        text += '{}={}|{}  '.format(
            'flip_angle', round(self.flip_angle, 1),
            np.deg2rad(self.flip_angle))
        names = [
            'duration', 'shape', 'shape_kwargs', 'dt', 'num_steps', 'w_c',
            'is_real', 'is_imag', 'propagator_mode', 'propagator_kwargs']
        for name in names:
            if hasattr(self, name):
                text += '{}={}  '.format(name, self.__dict__[name])
        if self.shape == 'custom':
            text += '\n w1_arr={}'.format(self.w1_arr)
        return text


# ======================================================================
class Spoiler:
    def __init__(
            self,
            efficiency=1.0):
        """


        Args:
            efficiency:

        Returns:

        """
        self.efficiency = efficiency

    def propagator(
            self,
            spin_model,
            *args,
            **kwargs):
        """
        Calculate the propagator of a spoiler with a specific efficiency.

        The efficiency is expressed as the fraction of the transverse
        magnetization that is decoherenced as the result of the spoiling.

        Args:
            spin_model:
            *args:
            **kwargs:

        Returns:

        """
        p_op_diag = np.ones(spin_model.operator_dim)
        num_exact, num_approx = 0, 0
        for lineshape in spin_model.approx:
            # 3: cartesian dims; +1: hom. operator
            base = 1 + num_exact * 3 + num_approx
            if lineshape:
                num_approx += 1
            else:
                p_op_diag[base:base + 2] -= self.efficiency
                num_exact += 1
        return np.diag(p_op_diag)

    def __repr__(self):
        text = 'Spoiler: '
        names = ['efficiency']
        for name in names:
            text += '{}={}  '.format(name, self.__dict__[name])
        return text


# ======================================================================
class Delay(PulseExc):
    def __init__(
            self,
            duration,
            w_c=None):
        PulseExc.__init__(self, duration, np.zeros(1), w_c)
        self.shape = 'rect'

    def __repr__(self):
        text = '{}: '.format(self.__class__.__name__)
        names = ['duration', 'w_c']
        for name in names:
            if hasattr(self, name):
                text += '{}={}  '.format(name, self.__dict__[name])
        return text


# ======================================================================
class PulseSequence:
    def __init__(
            self,
            w_c=None,
            gamma=GAMMA,
            b0=B0,
            *args,
            **kwargs):
        self.w_c = gamma * b0 if w_c is None else w_c

    def propagator(
            self,
            *args,
            **kwargs):
        pass

    def signal(
            self,
            spin_model):
        signal = mrb.mdot(
            spin_model.det, self.propagator(spin_model), spin_model.m_eq)
        return np.abs(signal)

    def __repr__(self):
        text = '{}'.format(self.__class__.__name__)
        if hasattr(self, 'num_repetitions'):
            text += ' (num_repetitions={})'.format(self.num_repetitions)
        text += ':\n'
        if hasattr(self, 'pulses'):
            for pulse in self.pulses:
                text += '  {}\n'.format(pulse)
        if hasattr(self, 'kernel'):
            text += '{}\n'.format(self.kernel)
        return text


# ======================================================================
class PulseList(PulseSequence):
    def __init__(
            self,
            pulses,
            *args,
            **kwargs):
        PulseSequence.__init__(self, *args, **kwargs)
        self.pulses = pulses
        for pulse in pulses:
            if hasattr(pulse, 'w_c') and pulse.w_c is None:
                pulse.w_c = self.w_c

    def propagator(
            self,
            spin_model,
            *args,
            **kwargs):
        propagators = [
            pulse.propagator(spin_model, *args, **kwargs)
            for pulse in self.pulses]
        return mrb.mdot(*propagators[::-1])


# ======================================================================
class PulseTrain(PulseSequence):
    def __init__(
            self,
            kernel,
            num_repetitions,
            *args,
            **kwargs):
        PulseSequence.__init__(self, *args, **kwargs)
        self.kernel = kernel
        self.num_repetitions = num_repetitions

    def propagator(
            self,
            spin_model,
            *args,
            **kwargs):
        return scipy.linalg.fractional_matrix_power(
            self.kernel.propagator(spin_model, *args, **kwargs),
            self.num_repetitions)


# ======================================================================
class MtFlash(PulseTrain):
    def __init__(
            self,
            kernel,
            num_repetitions,
            mt_pulse_index=None,
            *args,
            **kwargs):
        """

        Args:
            kernel (PulseList): The list of pulses to be repeated.
            num_repetitions (:
            mt_pulse_index (int|None): Index of the MT pulse in the kernel.
                If None, use the pulse with zero carrier frequency.
            *args:
            **kwargs:

        Returns:

        """
        PulseTrain.__init__(self, kernel, num_repetitions, *args, **kwargs)
        if mt_pulse_index is None:
            for i, item in enumerate(kernel.pulses):
                if hasattr(item, 'w_c') and item.w_c == 0.0:
                    self.mt_pulse_index = i
                    break
        else:
            self.mt_pulse_index = mt_pulse_index

    def set_flip_angle(self, flip_angle):
        self.kernel.pulses[self.mt_pulse_index].set_flip_angle(flip_angle)

    def set_freq(self, freq):
        self.kernel.pulses[self.mt_pulse_index].w_c = \
            self.kernel.w_c + 2 * pi * freq


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
        None
    """
    # todo: make it flexible and working
    w_c, w1 = sym.symbols('w_c w1')
    m0 = [sym.symbols('m0{}'.format())]

    # 2-pool model
    spin_model = SpinModel(
        s0=100,
        mc=(1.0, 0.152),
        w0=((w_c,) * 2),
        r1=(1.8, 1.0),
        r2=(32.2581, 8.4746e4),
        k=(0.05,),
        approx=(None, 'superlorentz_approx'))

    print(spin_model)
    print(spin_model.m_eq)
    print(spin_model.k_op)
    print(spin_model.l_op)
    print(dynamics_operator(spin_model, w_c + 10.0, w1))

    # 3-pool model
    spin_model = SpinModel(
        s0=100,
        mc=(1.0, 0.152),
        w0=((w_c,) * 3),
        r1=(1.8, 1.0, 1.2),
        r2=(32.2581, 8.4746e4, 30.0),
        k=(0.05, 0.5, 0.1),
        approx=(None, 'superlorentz_approx', None))

    print(spin_model)
    print(spin_model.m_eq)
    print(spin_model.k_op)
    print(spin_model.l_op)
    print(dynamics_operator(spin_model, w_c + 10.0, w1))

    # 4-pool model
    spin_model = SpinModel(
        s0=100,
        mc=(1.0, 0.152),
        w0=((w_c,) * 4),
        r1=(1.8, 1.0, 1.2, 2.0),
        r2=(32.2581, 8.4746e4, 30.0, 60.0),
        k=(0.05, 0.5, 0.1, 0.001, 0.4, 0.2),
        approx=(None, 'superlorentz_approx', None, 'gauss'))

    print(spin_model)
    print(spin_model.m_eq)
    print(spin_model.k_op)
    print(spin_model.l_op)
    print(dynamics_operator(spin_model, w_c + 10.0, w1))


# ======================================================================
def test_dynamics_operator():
    """
    Notes: import pi, sin and cos from numpy

    Returns:

    """
    w_c = GAMMA * B0
    w1 = 1.0

    # 2-pool model
    spin_model = SpinModel(
        s0=100,
        mc=(1.0, 0.152),
        w0=((w_c,) * 2),
        r1=(1.8, 1.0),
        r2=(32.2581, 8.4746e4),
        k=(0.3456,),
        approx=(None, 'superlorentz_approx'))

    print(spin_model)
    print(spin_model.m_eq)
    print(spin_model.k_op)
    print(spin_model.l_op)
    print(dynamics_operator(spin_model, w_c + 10.0, w1))

    # 3-pool model
    spin_model = SpinModel(
        m0=[v * 100.0 for v in (1.0, 0.152, 0.3)],
        w0=((w_c,) * 3),
        r1=(1.8, 1.0, 1.2),
        r2=(32.2581, 8.4746e4, 30.0),
        k=(0.05, 0.5, 0.1),
        approx=(None, 'superlorentz_approx', None))

    print(spin_model)
    print(spin_model.m_eq)
    print(spin_model.k_op)
    print(spin_model.l_op)
    print(dynamics_operator(spin_model, w_c + 10.0, w1))

    # 4-pool model
    spin_model = SpinModel(
        m0=[v * 100.0 for v in (1.0, 0.152, 0.3, 0.01)],
        w0=((w_c,) * 4),
        r1=(1.8, 1.0, 1.2, 2.0),
        r2=(32.2581, 8.4746e4, 30.0, 60.0),
        k=(0.05, 0.5, 0.1, 0.001, 0.4, 0.2),
        approx=(None, 'superlorentz_approx', None, 'gauss'))

    print(spin_model)
    print(spin_model.m_eq)
    print(spin_model.k_op)
    print(spin_model.l_op)
    print(dynamics_operator(spin_model, w_c + 10.0, w1))


# ======================================================================
def test_mt_sequence():
    """
    Test for the MT sequence.
    """
    w_c = GAMMA * B0

    spin_model = SpinModel(
        s0=100,
        mc=(1.0, 0.152),
        w0=((w_c,) * 2),
        r1=(1.8, 1.0),
        r2=(32.2581, 8.4746e4),
        k=(0.3456,),
        approx=(None, 'superlorentz_approx'))

    num_repetitions = 300

    mt_flash_kernel = PulseList([
        Delay(10.0e-3),
        Spoiler(1.0),
        PulseExc.shaped(40.0e-3, 220.0, 4000, 'gauss', None,
                        w_c + 50.0, 'poly', {'fit_order': 5}),
        Delay(20.0e-3),
        Spoiler(1.0),
        PulseExc.shaped(10.0e-6, 90.0, 1, 'rect', None),
        Delay(30.0e-3)],
        b0=3.0)
    mt_flash = PulseTrain(mt_flash_kernel, num_repetitions)

    signal = mt_flash.signal(spin_model)

    print(mt_flash)
    print(mt_flash.propagator(spin_model))
    print(signal)


# ======================================================================
def test_approx_propagator(
        spin_model=SpinModel(
            s0=100,
            mc=(0.8681, 0.1319),
            w0=((GAMMA * B0,) * 2),
            r1=(1.8, 1.0),
            r2=(32.2581, 8.4746e4),
            k=(0.3456,),
            approx=(None, 'superlorentz_approx')),
        flip_angle=90.0):
    """
    Test the approximation of propagators - for speeding up.

    Args:
        spin_model (SpinModel):
        flip_angles (float):
    """
    w_c = GAMMA * B0

    modes = ['exact']
    modes += ['linear', 'reduced']
    # modes += ['sum_simple', 'sum_order1', 'sum_sep', 'reduced']
    modes += ['poly_{}'.format(order) for order in range(4, 5)]
    modes += ['interp_{}_{}'.format(mode, num_samples)
              for mode in ['linear', 'cubic'] for num_samples in range(4, 5)]
    modes = {
        'linear': {
            'num_samples': tuple(range(10, 20, 5))},
        'interp': {
            'method': ('linear', 'cubic'),
            'num_samples': tuple(range(10, 20, 3))},
        'reduced': {
            'num_resamples': tuple(range(10, 20, 5))},
        'poly': {
            'fit_order': tuple(range(3, 6))}
    }

    shapes = {
        'gauss': {},
        'lorentz': {},
        'sinc': {},
        # 'fermi': {},
        # 'random': {},
        'cos_sin': {},
    }
    exact_p_ops = {}
    for shape, shape_kwargs in shapes.items():
        pulse = PulseExc.shaped(
            40.0e-3, flip_angle, 4000, shape, shape_kwargs, w_c, 'exact', {})
        exact_p_ops[shape] = pulse.propagator(spin_model)

    for shape, shape_kwargs in shapes.items():
        for mode, mode_params in modes.items():
            kwargs_items = [{}]
            names = mode_params.keys()
            for values in itertools.product(*[mode_params[i] for i in names]):
                kwargs_items.append({k: v for k, v in zip(names, values)})
            for kwargs in kwargs_items:
                pulse = PulseExc.shaped(
                    40.0e-3, flip_angle, 4000, shape, shape_kwargs, w_c,
                    mode, kwargs)
                begin_time = time.time()
                p_op = pulse.propagator(spin_model)
                elapsed = datetime.timedelta(0, time.time() - begin_time)
                rel_error = np.sum(np.abs(exact_p_ops[shape] - p_op)) / \
                            np.sum(np.abs(exact_p_ops[shape]))
                print('{:>8s}, {:>8s}, {:>48s},\t{:.3e}, {}'.format(
                    shape, mode, str(kwargs), rel_error, elapsed))


# ======================================================================
def test_z_spectrum(
        spin_model=SpinModel(
            s0=100,
            mc=(0.8681, 0.1319),
            w0=((GAMMA * B0,) * 2),
            r1=(1.8, 1.0),
            r2=(32.2581, 8.4746e4),
            k=(0.3456,),
            approx=(None, 'superlorentz_approx')),
        freqs=np.round(mrb.sgnlogspace(50, 50000, 16)),
        amplitudes=np.round(mrb.sgnlogspace(50, 5000, 16)),
        plot_data=True,
        save_file=None):
    """
    Test calculation of z-spectra

    Args:

        spin_model (SpinModel):
        freqs (ndarray[float]):
        amplitudes (ndarray[float]):
        plot_data (bool):
        save_file (string):

    Returns:
        freq

    """
    w_c = spin_model.w0[0]

    flip_angles = amplitudes * 11.799 / 50.0

    mt_flash = MtFlash(
        PulseList([
            Delay(10.0e-3),
            Spoiler(1.0),
            PulseExc.shaped(10.0e-3, 90.0, 4000, 'gauss', {},
                            0.0, 'poly', {'fit_order': 3}),
            Delay(10.0e-3),
            Spoiler(1.0),
            PulseExc.shaped(2.1e-3, 11.0, 1, 'rect', {})],
            w_c=w_c),
        300)

    data = np.zeros((len(freqs), len(amplitudes)))
    for j, freq in enumerate(freqs):
        for i, flip_angle in enumerate(flip_angles):
            mt_flash.set_flip_angle(flip_angle)
            mt_flash.set_freq(freq)
            data[j, i] = mt_flash.signal(spin_model)

    # plot results
    if plot_data:
        X, Y = np.meshgrid(amplitudes, np.log10(freqs))
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(
            X, Y, data, cmap=plt.cm.hot,
            rstride=1, cstride=1, linewidth=0.01, antialiased=False)
    if save_file:
        np.savez(save_file, freqs, amplitudes, data)
    return data, freqs, flip_angles


# ======================================================================
def test_fit_spin_model(
        snr_level=20,
        plot_data=True):
    """
    Test calculation of z-spectra

    Args:
        snr_level (float):
        plot_data (bool):

    Returns:
        None
    """
    w_c = GAMMA * B0

    mt_flash = MtFlash(
        PulseList([
            Delay(16610.0e-6),
            Spoiler(0.0),
            Delay(160.0e-6),
            PulseExc.shaped(10000.0e-6, 90.0, 0, '_from_GAUSS5120', {},
                            0.0, 'poly', {'fit_order': 3}),
            Delay(160.0e-6 + 970.0e-6),
            Spoiler(1.0),
            Delay(160.0e-6),
            PulseExc.shaped(100e-6, 11.0, 1, 'rect', {}),
            Delay(4900.0e-6)],
            w_c=w_c),
        300)

    def mt_signal(x_arr, s0, mc_a, r1a, r2a, r2b, k_ab):
        spin_model = SpinModel(
            s0=s0,
            mc=(mc_a, 1.0 - mc_a),
            w0=(w_c, w_c * (1 - 3.5e-6)),
            # w0=((w_c,) * 2),
            r1=(r1a, 1.0),
            r2=(r2a, r2b),
            k=(k_ab,),
            approx=(None, 'superlorentz_approx'))
        y_arr = np.zeros_like(x_arr[:, 0])
        i = 0
        for freq, flip_angle in x_arr:
            mt_flash.set_flip_angle(flip_angle)
            mt_flash.set_freq(freq)
            y_arr[i] = mt_flash.signal(spin_model)
            i += 1
        return y_arr

    # simulate a measurement
    freqs = mrb.sgnlogspace(100.0, 300.0e3, 32)
    flip_angles = np.linspace(1.0, 1100.0, 32)

    x_data = np.array(tuple(itertools.product(freqs, flip_angles)))

    # see: mt_signal
    p_e = 100, 0.8681, 2.0, 32.2581, 8.4746e4, 0.3456
    exact = mt_signal(x_data, *p_e).reshape((len(freqs), len(flip_angles)))
    # num = len(freqs) * len(flip_angles)
    # noise = (np.random.rand(*exact.shape) - 0.5) * np.max(exact) / snr_level
    # measured = exact + noise
    #
    # p0 = 100, 0.5, 5.0, 20.0, 5e4, 0.5
    # bounds = [[50, 1000], [0, 1], [0.1, 10], [10, 50], [1e4, 1e5], [0, 1]]
    # y_data = measured.ravel()
    #
    # def sum_of_squares(params, x_data, m_data):
    #     e_data = mt_signal(x_data, *params)
    #     return np.sum((m_data - e_data) ** 2.0)
    #
    # res = scipy.optimize.minimize(
    #     sum_of_squares, p0, args=(x_data, y_data), method='L-BFGS-B',
    #     bounds=bounds, options={'gtol': 1e-05, 'ftol': 2e-09})
    # print(res.x, res.success, res.message)
    #
    # fitted = mt_signal(x_data, *res.x).reshape(measured.shape)

    # # faked fitted
    # fitted = mt_signal(x_data, *p0).reshape(measured.shape)

    if plot_data:
        X, Y = np.meshgrid(flip_angles, mrb.sgnlog(freqs, 10.0))
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlabel('Pulse Amplitude (flip angle) / deg')
        ax.set_ylabel('Frequency offset / Hz (log10 scale)')
        ax.set_zlabel('Signal Intensity / arb.units')
        ax.plot_surface(
            X, Y, exact, cmap=plt.cm.hot,
            rstride=1, cstride=1, linewidth=0.005, antialiased=False)
        # ax.plot_surface(
        #     X, Y, measured, cmap=plt.cm.ocean,
        #     rstride=1, cstride=1, linewidth=0.01, antialiased=False)
        # ax.plot_surface(
        #     X, Y, fitted, cmap=plt.cm.bone,
        #     rstride=1, cstride=1, linewidth=0.01, antialiased=False)


# ======================================================================
if __name__ == '__main__':
    print(__doc__)
    # test_dynamics_operator_symbolic()
    # elapsed('test_symbolic')
    # test_dynamics_operator()
    # elapsed('test_dynamics_operator')
    # test_mt_sequence()
    # elapsed('test_mt_sequence')
    # test_approx_propagator()
    # elapsed('test_approx_propagator')
    # test_z_spectrum(
    #     SpinModel(100.0, (0.5, 0.3, 0.1, 0.1), (GAMMA * B0,) * 4,
    #               (0.25, 0.8, 0.001, 1.0), (20.0, 60.0, 8e4, 5e4),
    #               (1.0, 0.3, 0.0, 1.0, 0.5, 1.0),
    #               (None, None, 'superlorenz_approx', 'superlorenz_approx')))
    # test_z_spectrum()
    # elapsed('test_z_spectrum')
    test_fit_spin_model()
    elapsed('test_fit_spin_model')

    print_elapsed()
    # profile.run('test_z_spectrum()', sort=1)
    plt.show()
