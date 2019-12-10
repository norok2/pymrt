#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pymrt.sequences.matrix_algebra: numerical solver for Bloch-McConnell equations

Computes the signal by solving the Bloch-McConnell equations for multiple
spin systems with exchange, using the matrix algebra approach.

See Also:
 - Allard P., Helgstrand M., Härd T. (1997), "A Method for Simulation of
   NOESY, ROESY, and Off-Resonance ROESY Spectra", Journal of Magnetic
   Resonance 129 (1):19–29, DOI:10.1006/jmre.1997.1252.
 - Müller D.K., Pampel A., Möller H.E. (2013), "Matrix-Algebra-Based
   Calculations of the Time Evolution of the Binary Spin-Bath Model for
   Magnetization Transfer", Journal of Magnetic Resonance 230 (May):88–97,
   DOI:10.1016/j.jmr.2013.01.013.
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
# import cmath  # Mathematical functions for complex numbers
# import time  # Time access and conversions
# import datetime  # Basic date and time types
# import operator  # Standard operators as functions
# import collections  # Container datatypes
# import argparse  # Parser for command-line options, arguments and subcommands
import itertools  # Functions creating iterators for efficient looping
# import functools  # Higher-order functions and operations on callable objects
# import subprocess  # Subprocess management
# import multiprocessing  # Process-based parallelism
# import csv  # CSV File Reading and Writing [CSV: Comma-Separated Values]
# import json  # JSON encoder and decoder [JSON: JavaScript Object Notation]
import warnings  # Warning control
# import profile  # Deterministic Profiler
import pickle  # Python object serialization

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
import flyingcircus as fc  # Everything you always wanted to have in Python.*

# :: External Imports Submodules
# import matplotlib.pyplot as plt  # Matplotlib's pyplot: MATLAB-like syntax
# import mpl_toolkits.mplot3d as mpl3  # Matplotlib's 3D support
# import scipy.optimize  # SciPy: Optimization
import scipy.integrate  # SciPy: Integration
import scipy.interpolate  # SciPy: Interpolation
# import scipy.spatial  # SciPy: Spatial algorithms and data structures
# import scipy.constants  # SciPy: Constants
# import sp.ndimage  # SciPy: Multidimensional image processing
# import scipy.linalg  # SciPy: Linear Algebra
import scipy.stats  # SciPy: Statistical functions
import scipy.misc  # SciPy: Miscellaneous routines

from numpy import pi, sin, cos, exp, sqrt, sinc
# from sympy import pi, sin, cos, exp, sqrt, sinc
# from sympy import re, im
# from numba import jit

# :: Local Imports
import pymrt as mrt
import pymrt.util

# import raster_geometry  # Create/manipulate N-dim raster geometric shapes.
# import pymrt.plot as pmp
# import pymrt.segmentation

from pymrt import INFO, PATH
from pymrt import VERB_LVL, D_VERB_LVL, VERB_LVL_NAMES
from pymrt import elapsed, report, run_doctests
from pymrt import msg, dbg, fmt, fmtm
from pymrt.config import CFG

from pymrt.constants import GAMMA, GAMMA_BAR

_N_DIMS = 3


# ======================================================================
def superlorentz_integrand(x, t):
    """
    Calculate the integrand of the superlorentzian.

    Args:
        x (ndarray[float]): The independent variable
        t (ndarray[float]): The angle.

    Returns:
        (ndarray[float]): The integrand of the superlorentzian.
    """
    return sqrt(2.0 / pi) * \
           exp(-2.0 * (x / (3 * cos(t) ** 2 - 1)) ** 2.0) * sin(t) / \
           abs(3 * cos(t) ** 2 - 1)


# ======================================================================
@np.vectorize
def superlorentz(x):
    """
    Computing the superlorentzian function.

    Args:
        x (ndarray[float]): The independent variable.

    Returns:
       y (float): The superlorentzian function.
    """
    # sp.integrate.quad returns both the value and the error, here ignored
    return sp.integrate.quad(
        lambda t: superlorentz_integrand(x, t), 0.0, pi / 2.0)[0]


# ======================================================================
def _prepare_superlorentz(
        x=np.logspace(-10.0, 1.8, 2048),
        use_cache=CFG['use_cache']):
    """
    Precomputation of the superlorentzian.

    The computation is performed once and later loaded from cache.

    Args:
        x (ndarray[float]): The input interpolation points.
        use_cache (bool): Use cached computation.

    Returns:
        result (dict): The dictionary
            contains:
             - 'x': The independent interpolation sample points.
             - 'y': The superlorentzian of 'x'.
    """
    cache_filepath = os.path.join(PATH['cache'], 'superlorentz_approx.cache')
    if not os.path.isfile(cache_filepath) or not use_cache:
        result = dict(x=x, y=superlorentz(x))
        with open(cache_filepath, 'wb') as cache_file:
            pickle.dump(result, cache_file)
    else:
        with open(cache_filepath, 'rb') as cache_file:
            result = pickle.load(cache_file)
    return result


_SUPERLORENTZ = _prepare_superlorentz()
fc.base.elapsed('Superlorentz Approx.')


# ======================================================================
def _superlorentz_approx(
        x,
        x_i=_SUPERLORENTZ['x'],
        y_i=_SUPERLORENTZ['y']):
    """
    Interpolates the superlorentzian from sample points.

    Args:
        x (ndarray[float]): The independent variable.
        x_i (ndarray[float]): The independent interpolation sample points.
        y_i (ndarray[float]): The dependent interpolation sample points.

    Returns:
        y (ndarray[float]): The interpolated superlorentzian.
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
    Computation of the Henkelman absorption saturation rate.

    Args:
        r2 (float): The transverse relaxation rates in Hz.
        w0 (float): The resonance angular frequency in rad/s.
        w_c (float): The carrier angular frequency of the pulse excitation in
        rad/s.
        w1 (complex): The pulse excitation angular frequency in rad/s.
        lineshape (str): The lineshape of the Henkelman absorption term.

    Returns:
        saturation_rate (float): The Henkelman absorption saturation rate.
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
    Compute the pulse shape for a truncated normal pulse.

    This is equivalent to the Gaussian pulse shape with mean equal to 0.

    Args:
        num_steps (int): The number sample points.
        truncation (Iterable[float]): The truncation interval.

    Returns:
        y (ndarray[float]): The pulse shape.

    See Also:
        _shape_gaussian()
    """
    x = np.linspace(
        sp.stats.norm.ppf(0.0 + truncation[0]),
        sp.stats.norm.ppf(1.0 - truncation[1]),
        num_steps)
    y = sp.stats.norm.pdf(x)
    return y


# ======================================================================
def _shape_cauchy(
        num_steps,
        truncation=(0.001, 0.001)):
    """
    Compute the pulse shape for a truncated Cauchy pulse.

    This is equivalent to the Lorentzian pulse with scale parameter
    equal to 1.

    Args:
        num_steps (int): The number of sample points.
        truncation (Iterable[float]): The truncation interval.

    Returns:
        y (ndarray[float]): The pulse shape.

    See Also:
        _shape_lorentz()
    """
    x = np.linspace(
        sp.stats.cauchy.ppf(0.0 + truncation[0]),
        sp.stats.cauchy.ppf(1.0 - truncation[1]),
        num_steps)
    y = sp.stats.cauchy.pdf(x)
    return y


# ======================================================================
def _shape_gauss(
        num_steps,
        sigma=0.1,
        mu=0.5):
    """
    Compute the pulse shape for a Gaussian pulse.

    Args:
        num_steps (int): The number of sample points.
        sigma (float): The standard deviation.
        mu (float): The mean value.

    Returns:
        y (ndarray[float]): The pulse shape.
    """
    mu *= num_steps
    sigma *= num_steps
    x = np.arange(num_steps)
    y = exp(-(x - mu) ** 2 / (2 * sigma ** 2))
    return y


# ======================================================================
def _shape_lorentz(
        num_steps,
        gamma=0.1,
        mu=0.5):
    """
    Compute the pulse shape for a Lorentzian pulse.

     Args:
        num_steps (int): The number of sample points.
        gamma (float): The scale parameter.
        mu (float): The mean / location parameter.

    Returns:
        y (ndarray[float]): The pulse shape.

    See Also:
        _shape_cauchy()
    """
    mu *= num_steps
    gamma *= num_steps
    x = np.arange(num_steps)
    y = gamma / (np.pi * ((x - mu) ** 2 + gamma ** 2))
    return y


# ======================================================================
def _shape_fermi(
        num_steps,
        flat=0.2,
        step=0.03,
        mu=0.5):
    """
    Compute the pulse shape for a Fermi pulse.

    Args:
        num_steps (int): The number of sample points.
        flat (float): The width of the center plateau.
        step: The duration of the decaying ramps.
        mu: The location parameter.

    Returns:
        y (ndarray[float]): The pulse shape.
    """
    mu *= num_steps
    flat *= num_steps
    step *= num_steps
    x = np.arange(num_steps)
    y = 1.0 / (1.0 + exp((np.abs(x - mu) - flat) / step))
    return y


# ======================================================================
def _shape_sinc(
        num_steps,
        roots=(3.0, 3.0)):
    """
    Compute the pulse shape for a sinc-shaped pulse.

    Args:
        num_steps (int): The number of sample points.
        roots (ndarray[float]): The number of roots on the pos./neg. abscissa.

    Returns:
        y (ndarray[float]): The pulse shape.
    """
    x = np.linspace(-roots[0] * pi, roots[1] * pi, num_steps)
    y = sinc(x)
    return y


# ======================================================================
def _shape_cos_sin(
        num_steps,
        roots=(1.0, 1.0)):
    """
    Compute the pulse shape for a cos/sin pulse.

    This should be used only for testing.

    Args:
        num_steps (int): The number of sample points.
        roots (ndarray[float]): The number of roots.

    Returns:
        y (complex): The pulse shape.
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
    Loading the pulse shape from a text file.

    Args:
        filename (string): The filename of the input.
        dirpath (string): The base directory.

    Returns:
        y (complex): The pulse shape.
    """
    tmp_dirpaths = [
        fc.base.realpath(dirpath),
        os.path.join(os.path.dirname(__file__), dirpath),
    ]
    for tmp_dirpath in tmp_dirpaths:
        if os.path.isdir(tmp_dirpath):
            dirpath = tmp_dirpath
            break
    filepath = os.path.join(
        dirpath, filename + fc.base.add_extsep(mrt.util.EXT['tab']))
    arr = np.loadtxt(filepath)
    if arr.ndim == 1:
        y_re = arr
        y_im = 0.0
    else:  # if arr.ndim > 1:
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
    Compute the Bloch-McConnell dynamics operator L.

    Args:
        spin_model (SpinModel): The spin model term of the dynamics operator.
        w_c (float): The carrier angular frequency of the excitation in rad/s.
        w1 (complex): The pulse excitation angular frequency in rad/s.

    Returns:
        l_op (ndarray[float]): The dynamics operator L.
    """
    l_op = np.zeros(spin_model._operator_shape).astype(spin_model._dtype)
    num_exact, num_approx = 0, 0
    for i, lineshape in enumerate(spin_model.approx):
        # (1 + ) because hom. operator
        base = 1 + num_exact * _N_DIMS + num_approx
        # w1x, w1y = re(w1), im(w1)  # for symbolic
        w1x, w1y = w1.real, w1.imag
        bloch_core = np.array([
            [0.0, -w_c, -w1y],
            [w_c, 0.0, w1x],
            [w1y, -w1x, 0.0]])
        # same as above
        # bloch_core = fc.extra.to_self_adjoint_matrix(
        #     [-w_c, -w1y, w1x], skew=True)
        if lineshape:
            r_rf = _sat_rate_lineshape(
                spin_model.r2[i], spin_model.w0[i], w_c, w1, lineshape)
            # r_rf = sym.symbols('r_rf')  # for symbolic
            l_op[base, base] += bloch_core[-1, -1] + r_rf
            num_approx += 1
        else:
            l_op[base:base + _N_DIMS, base:base + _N_DIMS] += bloch_core
            num_exact += 1
    return spin_model._l_op + l_op


# ======================================================================
def _propagator_sum(
        pulse_exc,
        spin_model):
    """
    Compute the time-evolution propagator P assuming commutativity.

    The time evolution propagator expm(-L * Dt) can be estimated using:
        expm(sum(M_i)) = prod(expm(M_i))
    which is only valid for commuting operators.

    Args:
        pulse_exc (Pulse): The em pulse excitation to manipulate the spins.
        spin_model (SpinModel): The model for the spin system.

    Returns:
        p_op (np.ndarray): The (approximated) propagator operator P.
    """
    l_op_sum = np.zeros(spin_model._operator_shape)
    for w1 in pulse_exc._w1_arr:
        l_op = dynamics_operator(spin_model, pulse_exc.w_c, w1)
        l_op_sum += pulse_exc.dt * l_op
    return sp.linalg.expm(-l_op_sum)


# ======================================================================
def _propagator_sum_order1(
        pulse_exc,
        spin_model):
    """
    Compute the time-evolution propagator P assuming quasi-commutativity.

    The time evolution propagator expm(-L * Dt) can be estimated using:
        expm(sum(M_i)) = prod(expm(M_i))
    including a pseudo-first-order correction for quasi-commuting operators:
        expm(sum(M_i) + sum([M_i, M_j]/2) = prod(expm(M_i))

    Args:
        pulse_exc (Pulse): The em pulse excitation to manipulate the spins.
        spin_model (SpinModel): The model for the spin system.

    Returns:
        p_op (np.ndarray): The (approximated) propagator operator P.
    """
    l_ops = []
    for w1 in pulse_exc._w1_arr:
        l_op = dynamics_operator(spin_model, pulse_exc.w_c, w1)
        l_ops.append(pulse_exc.dt * l_op)
    l_op_sum = sum(l_ops)
    # pseudo-first-order correction
    comms = [
        fc.extra.commutator(l_ops[i], l_ops[i + 1]) / 2.0
        for i in range(len(l_ops[:-1]))]
    comm_sum = sum(comms)
    return sp.linalg.expm(-(l_op_sum + comm_sum))


# ======================================================================
def _propagator_sum_sep(
        pulse_exc,
        spin_model):
    """
    Compute the time-evolution propagator P using a separated sum
    approximation.

    The time evolution propagator expm(-L * Dt) can be estimated assuming:
        prod(expm(-L_i * Dt)) = powm(expm(L_free)) + expm(sum(L_w1_i * Dt))
        (L_i = L_free + L_w1_i)

    Args:
        pulse_exc (Pulse): The em pulse excitation to manipulate the spins.
        spin_model (SpinModel): The model for the spin system.

    Returns:
        p_op (np.ndarray): The (approximated) propagator operator P.
    """
    p_op_w1_sum = np.zeros(spin_model._operator_shape)
    l_op_free = dynamics_operator(spin_model, pulse_exc.w_c, 0.0)
    for w1 in pulse_exc._w1_arr:
        l_op = dynamics_operator(spin_model, pulse_exc.w_c, w1)
        p_op_w1_sum += (pulse_exc.dt * (l_op - l_op_free))
    # calculate propagators
    p_op_free = sp.linalg.expm(-l_op_free * pulse_exc.dt)
    p_op_pow = sp.linalg.fractional_matrix_power(
        p_op_free, pulse_exc.num_steps)
    p_op_w1 = sp.linalg.expm(-p_op_w1_sum)
    return np.dot(p_op_pow, p_op_w1_sum)


# ======================================================================
def _propagator_poly(
        pulse_exc,
        spin_model,
        fit_order=3,
        num_samples=None):
    """
    Compute the time-evolution propagator P using polynomial approximation.

    The time evolution propagator expm(-L * Dt) is estimated with a (fast)
    element-wise polynomial fit, on exact values calculated over the w1 domain.

    Args:
        pulse_exc (Pulse): The e.m. pulse pulse_exc to manipulate the spins.
        spin_model (SpinModel): The model for the spin system.
        fit_order (int): The order to be used for the polynomial fit.
        num_samples (int): The number of samples used for the polynomial fit.

    Returns:
        p_op (np.ndarray): The (approximated) propagator operator P.
    """
    if not num_samples or num_samples < fit_order + 1:
        num_samples = fit_order + 1
    if pulse_exc.is_real or pulse_exc.is_imag:
        _w1_arr = pulse_exc._w1_arr.real if pulse_exc.is_real else \
            pulse_exc._w1_arr.imag
        # :: calculate samples
        p_op_approx = np.zeros(
            spin_model._operator_shape + (num_samples,))
        w1_approx = np.linspace(
            np.min(_w1_arr), np.max(_w1_arr), num_samples)
        for i, w1 in enumerate(w1_approx):
            l_op = dynamics_operator(spin_model, pulse_exc.w_c, w1)
            p_op_approx[:, :, i] = sp.linalg.expm(-pulse_exc.dt * l_op)
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
        p_arr = np.transpose(p_arr)
        # revert to original shape
        p_arr = p_arr.reshape(list(shape[:support_axis]) + [fit_order + 1])
        # :: approximate all propagators and calculate final result
        p_op_arr = np.zeros(
            (pulse_exc.num_steps,) + spin_model._operator_shape)
        for i in range(spin_model._operator_dim):
            for j in range(spin_model._operator_dim):
                p_op_arr[:, i, j] = np.polyval(p_arr[i, j, :], _w1_arr)
        p_ops = [p_op_arr[j, :, :] for j in range(pulse_exc.num_steps)]
        p_op = fc.extra.mdot(p_ops[::-1])
    else:
        # :: calculate samples
        num_extra_samples = num_samples * num_samples
        re_min = np.min(pulse_exc._w1_arr.real)
        im_min = np.min(pulse_exc._w1_arr.imag)
        re_max = np.max(pulse_exc._w1_arr.real)
        im_max = np.max(pulse_exc._w1_arr.imag)
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
            spin_model._operator_shape + (num_samples,))
        w1_approx = np.zeros(
            num_re_samples * num_im_samples).astype(complex)
        i = 0
        for w1_re in w1_re_approx:
            for w1_im in w1_im_approx:
                l_op = dynamics_operator(
                    spin_model, pulse_exc.w_c, w1_re + 1j * w1_im)
                p_op_approx[:, :, i] = sp.linalg.expm(-pulse_exc.dt * l_op)
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
            (pulse_exc.num_steps,) + spin_model._operator_shape)
        for i in range(spin_model._operator_dim):
            for j in range(spin_model._operator_dim):
                p_op_arr[:, i, j] = np.real(
                    np.polyval(p_arr[i, j, :], pulse_exc._w1_arr))
        p_ops = [p_op_arr[j, :, :] for j in range(pulse_exc.num_steps)]
        p_op = fc.extra.mdot(p_ops[::-1])
    return p_op


# ======================================================================
def _propagator_interp(
        pulse_exc,
        spin_model,
        method='cubic',
        num_samples=5):
    """
    Compute the time-evolution propagator P using interpolation.

    The time evolution propagator expm(-L * Dt) is estimated with a (fast)
    element-wise interpolation approximation, on exact values calculated
    over the w1 domain.

    Args:
        pulse_exc (Pulse): The em pulse excitation to manipulate the spins.
        spin_model (SpinModel): The model for the spin system.
        method (str): The method used by the interpolating function
            'sp.interpolate.griddata'.
        num_samples (int): The number of samples for the interpolation.

    Returns:
        p_op (np.ndarray): The (approximated) propagator operator P.
    """
    if pulse_exc.is_real or pulse_exc.is_imag:
        _w1_arr = pulse_exc._w1_arr.real \
            if pulse_exc.is_real else pulse_exc._w1_arr.imag
        # :: calculate samples
        p_op_approx = np.zeros(spin_model._operator_shape + (num_samples,))
        w1_approx = np.linspace(
            np.min(_w1_arr), np.max(_w1_arr), num_samples)
        for i, w1 in enumerate(w1_approx):
            l_op = dynamics_operator(spin_model, pulse_exc.w_c, w1)
            p_op_approx[:, :, i] = sp.linalg.expm(-pulse_exc.dt * l_op)
        # perform interpolation
        p_op_arr = np.zeros(
            (pulse_exc.num_steps,) + spin_model._operator_shape)
        for i in range(spin_model._operator_dim):
            for j in range(spin_model._operator_dim):
                p_op_arr[:, i, j] = sp.interpolate.griddata(
                    w1_approx, p_op_approx[i, j, :], _w1_arr,
                    method=method, fill_value=0.0)
        p_ops = [p_op_arr[j, :, :] for j in
                 range(pulse_exc.num_steps)]
        p_op = fc.extra.mdot(p_ops[::-1])
    else:
        # :: calculate samples
        num_extra_samples = num_samples * num_samples
        re_min = np.min(pulse_exc._w1_arr.real)
        im_min = np.min(pulse_exc._w1_arr.imag)
        re_max = np.max(pulse_exc._w1_arr.real)
        im_max = np.max(pulse_exc._w1_arr.imag)
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
            spin_model._operator_shape + (num_samples,))
        w1_approx = np.zeros((num_re_samples * num_im_samples, 2))
        i = 0
        for w1_re in w1_re_approx:
            for w1_im in w1_im_approx:
                l_op = dynamics_operator(
                    spin_model, pulse_exc.w_c, w1_re + 1j * w1_im)
                p_op_approx[:, :, i] = sp.linalg.expm(-pulse_exc.dt * l_op)
                w1_approx[i, :] = (w1_re, w1_im)
                i += 1
        # perform interpolation
        p_op_arr = np.zeros(
            (pulse_exc.num_steps,) + spin_model._operator_shape)
        for i in range(spin_model._operator_dim):
            for j in range(spin_model._operator_dim):
                p_op_arr[:, i, j] = sp.interpolate.griddata(
                    w1_approx, p_op_approx[i, j, :],
                    (pulse_exc._w1_arr.real, pulse_exc._w1_arr.imag),
                    method=method, fill_value=0.0)
        p_ops = [p_op_arr[j, :, :] for j in range(pulse_exc.num_steps)]
        p_op = fc.extra.mdot(p_ops[::-1])
    return p_op


# ======================================================================
def _propagator_linear(
        pulse_exc,
        spin_model,
        num_samples=32):
    """
    Compute the time-evolution propagator P using linear interpolation.

    The time evolution propagator expm(-L * Dt) is estimated with a (fast)
    element-wise linear interpolation approximation, on exact values calculated
    over the w1 domain. Assumes real and imaginary part to be independent.

    Args:
        pulse_exc (Pulse): The em pulse excitation to manipulate the spins.
        spin_model (SpinModel): The model for the spin system.
        num_samples (int): Number of samples for the interpolation.

    Returns:
        p_op (np.ndarray): The (approximated) propagator operator P.
    """
    if pulse_exc.is_real or pulse_exc.is_imag:
        _w1_arr = pulse_exc._w1_arr.real \
            if pulse_exc.is_real else pulse_exc._w1_arr.imag
        # :: calculate samples
        p_op_approx = np.zeros(
            spin_model._operator_shape + (num_samples,))
        w1_approx = np.linspace(
            np.min(_w1_arr), np.max(_w1_arr), num_samples)
        for i, w1 in enumerate(w1_approx):
            l_op = dynamics_operator(spin_model, pulse_exc.w_c, w1)
            p_op_approx[:, :, i] = sp.linalg.expm(-pulse_exc.dt * l_op)
        # perform interpolation
        p_op_arr = np.zeros(
            (pulse_exc.num_steps,) + spin_model._operator_shape)
        for i in range(spin_model._operator_dim):
            for j in range(spin_model._operator_dim):
                p_op_arr[:, i, j] = np.interp(
                    _w1_arr, w1_approx, p_op_approx[i, j, :])
        p_ops = [p_op_arr[j, :, :] for j in
                 range(pulse_exc.num_steps)]
        p_op = fc.extra.mdot(p_ops[::-1])
    else:
        # :: calculate samples
        num_extra_samples = num_samples * num_samples
        re_min = np.min(pulse_exc._w1_arr.real)
        im_min = np.min(pulse_exc._w1_arr.imag)
        re_max = np.max(pulse_exc._w1_arr.real)
        im_max = np.max(pulse_exc._w1_arr.imag)
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
            spin_model._operator_shape + (num_re_samples,))
        p_op_im_approx = np.zeros(
            spin_model._operator_shape + (num_im_samples,))
        for i, w1_re in enumerate(w1_re_approx):
            l_op = dynamics_operator(spin_model, pulse_exc.w_c, w1_re)
            p_op_re_approx[:, :, i] = sp.linalg.expm(-pulse_exc.dt * l_op)
        for i, w1_im in enumerate(w1_im_approx):
            l_op = dynamics_operator(spin_model, pulse_exc.w_c, w1_im)
            p_op_re_approx[:, :, i] = sp.linalg.expm(-pulse_exc.dt * l_op)
        # perform interpolation
        p_op_arr = np.zeros(
            (pulse_exc.num_steps,) + spin_model._operator_shape)
        for i in range(spin_model._operator_dim):
            for j in range(spin_model._operator_dim):
                p_op_arr_re = np.interp(
                    pulse_exc._w1_arr.real,
                    w1_re_approx, p_op_re_approx[i, j, :])
                p_op_arr_im = np.interp(
                    pulse_exc._w1_arr.imag,
                    w1_im_approx, p_op_im_approx[i, j, :])
                weighted = \
                    (p_op_arr_re * np.abs(pulse_exc._w1_arr.real) +
                     p_op_arr_im * np.abs(pulse_exc._w1_arr.imag)) / \
                    (np.abs(pulse_exc._w1_arr.real) +
                     np.abs(pulse_exc._w1_arr.imag))
                p_op_arr[:, i, j] = weighted
        p_ops = [p_op_arr[j, :, :] for j in range(pulse_exc.num_steps)]
        p_op = fc.extra.mdot(p_ops[::-1])
    return p_op


# ======================================================================
def _propagator_reduced(
        pulse_exc,
        spin_model,
        num_resamples=32):
    """
    Compute the time-evolution propagator P of a coarser pulse excitation.

    The time evolution propagator expm(-L * Dt) is estimated on an approximated
    pulse excitation, obtained by reducing the number of samples of the
    original pulse excitation.

    Args:
        pulse_exc (Pulse): The em pulse excitation to manipulate the spins.
        spin_model (SpinModel): The model for the spin system.
        num_resamples (int): The number of samples for the reduced resampling.

    Returns:
        p_op (np.ndarray): The (approximated) propagator operator P.
    """
    chunk_marks = np.round(
        np.linspace(0, pulse_exc.num_steps - 1, num_resamples + 1))
    dt_reduced = pulse_exc.dt * pulse_exc.num_steps / num_resamples
    w1_reduced_arr = np.zeros(num_resamples).astype(pulse_exc._w1_arr[0])
    for i in range(num_resamples):
        chunk = slice(chunk_marks[i], chunk_marks[i + 1])
        w1_reduced_arr[i] = np.mean(pulse_exc._w1_arr[chunk])
    p_ops = [
        sp.linalg.expm(
            -dt_reduced *
            dynamics_operator(spin_model, pulse_exc.w_c, w1))
        for w1 in w1_reduced_arr]
    return fc.extra.mdot(p_ops[::-1])


# ======================================================================
class SpinModel(object):
    """
    Model of the spin system.
    """

    # -----------------------------------
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
            s0 (float): The signal magnitude scaling in arb. units.
            mc (ndarray[float]): The concentration ratios in one units.
            w0 (ndarray[float]): The resonance angular frequencies in rad/s.
            r1 (ndarray[float]): The longitudinal relaxation rates in Hz.
            r2 (ndarray[float]): The transverse relaxation rates in Hz.
            k (ndarray[float]): The pool-pool exchange rate constants in Hz.

        Returns:
            None
        """
        self.num_pools = len(mc)
        # exchange at equilibrium between each two pools
        self.num_exchange = \
            fc.base.comb(self.num_pools, 2) if self.num_pools > 1 else 0
        self.approx = approx \
            if approx is not None else [None] * self.num_pools
        self.num_approx = sum(
            [0 if item is None else 1 for item in self.approx])
        self._num_exact = self.num_pools - self.num_approx
        self._operator_dim = 1 + _N_DIMS * self._num_exact + self.num_approx
        self._operator_shape = (self._operator_dim,) * 2
        self._operator_base_dim = 1 + _N_DIMS * self.num_pools
        self._operator_base_shape = (self._operator_base_dim,) * 2
        self._dtype = type(mc[0])
        # simple check on the number of parameters
        if self.num_pools != len(r1) != len(r2) != len(approx) \
                or len(k) != self.num_exchange:
            raise IndexError('inconsistent spin model')

        self.s0 = s0
        self.mc = np.array(mc) / sum(mc)
        self.m0 = self.mc
        self.w0 = np.array(w0)
        self.r1 = np.array(r1)
        self.r2 = np.array(r2)
        self.k = np.array(k)

        self._ignore_k_transverse = True
        self._k_op = self.kinetics_operator()
        self._l_op = self.dynamics_operator()

    # -----------------------------------
    def equilibrium_magnetization(self):
        """
        Generate the equilibrium magnetization vector.

        This is used in conjunction with the propagator operator in order to
        calculate the signal intensity.

        Note that B0 is assumed parallel to z-axis, therefore:
            - the transverse magnetization is zero
            - the longitudinal magnetization is only in the z-axis

        Returns:
            m_eq (np.ndarray): The equilibrium magnetization vector.
        """
        m_eq = np.zeros(self._operator_dim).astype(self._dtype)
        m_eq[0] = 0.5
        num_exact, num_approx = 0, 0
        for m0z, lineshape in zip(self.m0, self.approx):
            if lineshape:
                pos = 1 + num_exact * _N_DIMS + num_approx
                num_approx += 1
            else:
                pos = 1 + num_exact * _N_DIMS + num_approx + 2
                num_exact += 1
            m_eq[pos] = m0z
        return m_eq

    # -----------------------------------
    def detector(
            self,
            phase=pi / 2):
        """
        Generate the detector vector, used to calculate the signal.

        This is used in conjunction with the propagator operator in order to
        calculate the signal intensity.

        Args:
            phase (float): The phase in rad of the detector system.

        Returns:
            result (np.ndarray): The detector vector.
        """
        result = np.zeros(self._operator_dim).astype(complex)
        num_exact, num_approx = 0, 0
        for lineshape in self.approx:
            base = 1 + num_exact * _N_DIMS + num_approx
            if lineshape:
                num_approx += 1
            else:
                result[base: base + 2] = 1.0, 1.0j
                num_exact += 1
        result *= exp(1j * phase)
        return result

    # -----------------------------------
    def dynamics_operator(self):
        """
        Compute L_spin: the excitation-independent part of the
        Bloch-McConnell dynamics operator of the spin system.

        Returns:
            l_op (np.ndarray): The dynamics operator L.
        """
        num_pools = len(self.approx)
        l_op = np.zeros(self._operator_base_shape).astype(self._dtype)
        # # ...to make L invertible
        # L[0, 0] = -2.0
        to_remove = []
        for i, lineshape in enumerate(self.approx):
            base = 1 + i * _N_DIMS
            # Bloch operator core...
            l_op[base:base + _N_DIMS, base:base + _N_DIMS] = np.array([
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
            if self._ignore_k_transverse else np.eye(_N_DIMS)
        m0_op = np.repeat(self.m0.reshape((-1, 1)), self.num_pools, axis=1)
        l_k_op = self._k_op * m0_op - np.diag(np.dot(self._k_op, self.m0))
        l_op[1:, 1:] -= np.kron(l_k_op, locator)
        # remove transverse components of approximated pools
        l_op = np.delete(l_op, to_remove, 0)
        l_op = np.delete(l_op, to_remove, 1)
        return l_op

    # -----------------------------------
    def kinetics_operator(self):
        """
        Compute the symmetric operator of the pool-pool exchange constants.

        Returns:
            k_op (np.ndarray): The kinetics operator K.
        """
        indexes = sorted(
            list(itertools.combinations(range(self.num_pools), 2)),
            key=lambda x: x[1])
        k_op = np.zeros((self.num_pools,) * 2).astype(self._dtype)
        for k, index in zip(self.k, indexes):
            k_op[index] = k
            k_op[index[::-1]] = k
        return k_op

    # -----------------------------------
    def __str__(self):
        text = '{}: '.format(self.__class__.__name__)
        for name in dir(self):
            if not name.startswith('_'):
                text += '{}={}  '.format(name, getattr(self, name))
        return text


# ======================================================================
class SequenceEvent(object):
    """
    Pulse sequence event.
    """

    # -----------------------------------
    def __init__(
            self,
            duration):
        """

        Args:
            duration ():
        """
        self.duration = duration

    # -----------------------------------
    def __str__(self):
        text = '{}: '.format(self.__class__.__name__)
        for name in dir(self):
            if not name.startswith('_'):
                text += '{}={}  '.format(name, getattr(self, name))
        return text


# ======================================================================
class Pulse(SequenceEvent):
    """
    Pulse excitation interacting with the spin system.
    """

    # -----------------------------------
    def __init__(
            self,
            duration=0.0,
            w1_arr=None,
            w_c=None,
            label=None):
        """
        Base constructor of the pulse excitation class.

        Args:
            duration (float): The duration of the pulse in s.
            w1_arr (ndarray[complex]): The modulation of the pulse in rad/s.
            w_c (float): The carrier angular frequency of the pulse in rad/s.

        Returns:
            None
        """
        # todo: add something to memorize shape
        # so that setting flip angle to 0 does not munge the shape
        SequenceEvent.__init__(self, duration)
        self.shape = 'custom'
        self.w_c = w_c if w_c is not None else 0.0
        self._w1_arr = w1_arr if w1_arr is not None else np.array((1.0,))
        self.num_steps = len(w1_arr)
        self.dt = self.duration / self.num_steps
        if self.num_steps < 1 or self.dt < 0.0:
            raise ValueError('inconsistent pulse excitation')
        self.norm = self.get_norm()
        self.is_real = np.isreal(w1_arr[0])
        self.is_imag = np.isreal(w1_arr[0] / 1j)
        self.propagator_mode = 'exact'
        self.propagator_kwargs = dict()
        self.label = label

    # -----------------------------------
    @property
    def flip_angle(self):
        """
        Returns:
            flip_angle (float): The flip angle in deg.

            Mathematically, with the proper units, this is equivalent to the
            absolute norm of the array:
                sum(abs(a_i))
        """
        return np.rad2deg(self.norm)

    # -----------------------------------
    @property
    def carrier_freq(self):
        """
        Returns:
            freq (float): The carrier frequency in Hz.
        """
        return fc.extra.afreq2freq(self.w_c)

    # -----------------------------------
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
            duration (float): The duration of the pulse in s.
            flip_angle (float): The flip angle of the excitation in deg.
            num_steps (int): The number of sampling point for the pulse.
            shape (basestr): The name of the desired pulse shape.
                Note: a function named '_shape_[SHAPE]' must exist.
                    [normal|cauchy|sinc|cos_sin]
            shape_kwargs (dict|None): The keyword arguments for the shape
                function.
            w_c (float|None): Carrier angular frequency of the pulse in
                rad/s.
            propagator_mode (basestr|None): The calculation mode for the
                propagator.
                Note: a function named '_propagator_[MODE]' must exist.
                    [sum|sum_order1|sum_sep|poly|interp|linear|reduced]
            propagator_kwargs (dict|None): The keyword arguments for the
                propagator function.

        Returns:
            pulse_exc (Pulse): The generated pulse excitation.
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
                text = '{}: unknown shape. Fall back to rect'.format(shape)
                warnings.warn(text)
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

    # -----------------------------------
    def get_norm(self):
        """
        Compute the norm of the pulse excitation in rad.

        Returns:
            norm (float): The norm of the pulse excitation.
        """
        return np.sum(np.abs(self._w1_arr * self.dt))

    # -----------------------------------
    def set_norm(self, new_norm):
        """
        Set a new norm for the pulse excitation in rad.

        This is equivalent to setting the flip angle (mind a scaling factor).
        Note that this modifies the w1_arr.

        Args:
            new_norm (float): The new norm of the pulse.

        Returns:
            None.
        """
        if self.norm == 0.0:
            self._w1_arr = np.ones_like(self._w1_arr)
            self.norm = self.get_norm()
        self._w1_arr = self._w1_arr * new_norm / self.norm
        self.norm = new_norm

    # -----------------------------------
    def get_flip_angle(self):
        """
        Get the flip angle of the pulse excitation in deg.

        Returns:
            flip angle (float): The flip angle of the pulse excitation.
        """
        return self.flip_angle

    # -----------------------------------
    def set_flip_angle(self, new_flip_angle):
        """
        Set a new flip angle for the pulse excitation.

        This is equivalent to setting the norm (mind a scaling factor).
        Note that this modifies the w1_arr.

        Args:
            new_flip_angle (float): The new flip angle of the pulse in deg.

        Returns:
            None.
        """
        self.set_norm(np.deg2rad(new_flip_angle))
        return self

    # -----------------------------------
    def set_carrier_freq(self, new_f_c):
        """
        Set a new carrier frequency for the pulse excitation.

        Args:
            new_f_c (float): The new flip angle of the pulse in deg.

        Returns:
            None.
        """
        self.w_c = fc.extra.freq2afreq(new_f_c)
        return self

    # -----------------------------------
    def shift_carrier_freq(self, delta_f_c):
        """
        Shift the carrier frequency for the pulse excitation.

        Note that consecutive calls to this function will accumulate the shift
        with respect to the original carrier frequency.

        Args:
            delta_f_c (float): The new flip angle of the pulse in deg.

        Returns:
            None.
        """
        self.w_c = self.w_c + fc.extra.freq2afreq(delta_f_c)
        return self

    # -----------------------------------
    def propagator(
            self,
            spin_model,
            *_args,
            **_kws):
        """
        Compute the Bloch-McConnell propagator: expm(-L * Dt).

        L is the dynamics operator of the rf-excited spin system.
        Dt is a time interval where spin exchange dynamics is negligible.

        Args:
            spin_model (SpinModel): The physical model for the spin pools.
            *_args: Positional arguments for 'p_op_func()'.
            **_kws: Keyword arguments for 'p_op_func()'.

        Returns:
            p_op (np.ndarray): The propagator P.
        """
        if not _kws:
            _kws = self.propagator_kwargs
        if self.propagator_mode == 'exact':
            p_ops = [
                sp.linalg.expm(
                    -self.dt * dynamics_operator(spin_model, self.w_c, w1))
                for w1 in self._w1_arr]
            p_op = fc.extra.mdot(p_ops[::-1])
        else:
            try:
                p_op_func = eval('_propagator_' + self.propagator_mode)
                p_op = p_op_func(self, spin_model, *_args, **_kws)
            except NameError:
                text = '{}: unknown approximation'.format(self.propagator_mode)
                warnings.warn(text)
                p_ops = [
                    sp.linalg.expm(
                        -self.dt * dynamics_operator(spin_model, self.w_c, w1))
                    for w1 in self._w1_arr]
                p_op = fc.extra.mdot(p_ops[::-1])
        return p_op

    # -----------------------------------
    def __str__(self):
        text = '{}: '.format(self.__class__.__name__)
        text += '{}={}|{}  '.format(
            'flip_angle', round(self.flip_angle, 1),
            np.deg2rad(self.flip_angle))
        names = [
            'duration', 'shape', 'shape_kwargs', 'dt', 'num_steps', 'w_c',
            'is_real', 'is_imag', 'propagator_mode', 'propagator_kwargs']
        for name in names:
            if hasattr(self, name):
                text += '{}={}  '.format(name, getattr(self, name))
        if self.shape == 'custom':
            text += '\n w1_arr={}'.format(self._w1_arr)
        return text


# ======================================================================
class Spoiler(SequenceEvent):
    """
    Spoiler for eliminating the transverse magnetization.
    """

    # -----------------------------------
    def __init__(
            self,
            efficiency=1.0,
            duration=0.0):
        """
        Base constructor of the spoiler class.

        Args:
            efficiency(float): The spoiler effiency.
                Must be  in the [0.0, 1.0] range.

        Returns:
            None
        """
        SequenceEvent.__init__(self, duration)
        if efficiency > 1.0 or efficiency < 0.0:
            raise ValueError(fmtm('Invalid efficiency value: `{efficiency}`'))
        self.efficiency = efficiency

    # -----------------------------------
    def propagator(
            self,
            spin_model):
        """
        Compute the propagator of the spoiler with a specific efficiency.

        The efficiency is expressed as the fraction of the transverse
        magnetization that is decoherenced as the result of the spoiling.

        Args:
            spin_model (SpinModel): The model for the spin system.

        Returns:
            y (ndarray[float]): The propagator.
        """
        p_op_diag = np.ones(spin_model._operator_dim)
        num_exact, num_approx = 0, 0
        for lineshape in spin_model.approx:
            base = 1 + num_exact * _N_DIMS + num_approx
            if lineshape:
                num_approx += 1
            else:
                p_op_diag[base:base + 2] -= self.efficiency
                num_exact += 1
        return np.diag(p_op_diag)

    # -----------------------------------
    def __str__(self):
        text = '{}: '.format(self.__class__.__name__)
        names = ['duration', 'efficiency']
        for name in names:
            if hasattr(self, name):
                text += '{}={}  '.format(name, getattr(self, name))
        return text


# ======================================================================
class Delay(Pulse):
    """
    Delay after the pulse excitation.
    """

    # -----------------------------------
    def __init__(
            self,
            duration,
            w_c=None):
        """
        Base constructor of the pulse delay class.

        Args:
            duration (float): The duration of the excitation pulse in s.
            w_c (float|None): Carrier angular frequency of the pulse in rad/s.

        Returns:
            None
        """
        Pulse.__init__(self, duration, np.zeros(1), w_c)
        self.shape = 'rect'

    # -----------------------------------
    def __str__(self):
        text = '{}: '.format(self.__class__.__name__)
        names = ['duration', 'w_c']
        for name in names:
            if hasattr(self, name):
                text += '{}={}  '.format(name, getattr(self, name))
        return text


# ======================================================================
class PulseExc(Pulse):
    """
    Excitation pulse before detecting the signal.
    """

    # -----------------------------------
    def __init__(
            self,
            *_args,
            **_kws):
        Pulse.__init__(self, *_args, **_kws)


# ======================================================================
class MagnetizationPreparation(Pulse):
    """
    Pulse to prepare the magnetization state of spins.
    """

    # -----------------------------------
    def __init__(
            self,
            *_args,
            **_kws):
        Pulse.__init__(self, *_args, **_kws)


# ======================================================================
class ReadOut(Delay):
    """
    Read out block (a delay containing signal detection).
    """

    # -----------------------------------
    def __init__(
            self,
            duration=None,
            w_c=None):
        """
        Base constructor of the `ReadOut` class.

        Args:
            duration (float|None): The duration of the excitation pulse in s.
            w_c (float|None): Carrier angular frequency of the pulse in rad/s.

        Returns:
            None
        """
        if duration is None:
            duration = 0.0
        Pulse.__init__(self, duration, np.zeros(1), w_c)
        self.shape = 'rect'


# ======================================================================
class PulseSequence(object):
    """
    The pulse excitation sequence.
    """

    # -----------------------------------
    def __init__(
            self,
            pulses,
            w_c=None):
        """
        Base constructor of the pulse sequence class.

        Args:
            w_c (float|None): Carrier angular frequency of the pulse in rad/s.
        """
        self.pulses = pulses
        self.w_c = w_c
        if w_c:
            self.update_carrier_freq()
        self._idx = dict()

    def update_carrier_freq(self):
        for i, pulse in enumerate(self.pulses):
            if hasattr(pulse, 'w_c') and pulse.w_c != self.w_c:
                self.pulses[i].w_c = self.w_c

    # -----------------------------------
    def get_unique_pulses(
            self,
            uniques):
        """
        Ensure certain pulse types are unique.

        Args:
            uniques (Iterable[str]): The class name of the unique pulse types.

        Returns:
            idx (dict): Indices of the unique pulse types.
        """
        idx = dict()
        pulse_types = [pulse.__class__.__name__ for pulse in self.pulses]
        for unique in uniques:
            indices = [i for i, x in enumerate(pulse_types) if x == unique]
            if len(indices) > 1:
                raise ValueError('Only one pulse can be `{}`.'.format(unique))
            elif len(indices) > 0:
                idx[unique] = pulse_types.index(unique)
            else:
                text = 'Pulse `{}` not found in `pulses`.'.format(unique)
                raise ValueError(text)
        return idx

    # -----------------------------------
    @staticmethod
    def _duration(pulses):
        return sum(
            pulse.duration if hasattr(pulse, 'duration') else 0.0
            for pulse in pulses)

    # -----------------------------------
    @property
    def duration(self):
        return self._duration(self.pulses)

    # -----------------------------------
    def propagators(
            self,
            spin_model,
            *_args,
            **_kws):
        """
        Compute the propagator of the pulse sequence.

        Args:
            spin_model (SpinModel): The model for the spin system.
            *_args: Positional arguments for 'pulse_propagator()'.
            **_kws: Keyword arguments for 'pulse_propagator()'.

        Returns:
            p_ops (list(ndarray[float])): The propagators.
        """
        return [
            pulse.propagator(spin_model, *_args, **_kws)
            for pulse in self.pulses]

    # -----------------------------------
    def propagator(
            self,
            spin_model,
            *_args,
            **_kws):
        """
        Compute the propagator of the pulse sequence.

        Args:
            spin_model (SpinModel): The model for the spin system.
            *_args: Positional arguments for 'propagators()'.
            **_kws: Keyword arguments for'propagators()'.

        Returns:
            y (ndarray[float]): The propagator.
        """
        return self._propagator(self.propagators(spin_model, *_args, **_kws))

    # -----------------------------------
    @staticmethod
    def _propagator(
            p_ops,
            num=1):
        """
        Compute the propagator of the pulse sequence.

        Args:
            p_ops (Sequence[np.ndarray]): The propagator operator.
            num (int): The number of times the propagator is repeated.

        Returns:
            y (ndarray[float]): The propagator.
        """
        p_op = fc.extra.mdot(p_ops[::-1])
        if num > 1:
            p_op = sp.linalg.fractional_matrix_power(p_op, num)
        return p_op

    # -----------------------------------
    def signal(
            self,
            spin_model,
            *_args,
            **_kws):
        """
        Compute the signal from the pulse sequence.

        Args:
            spin_model (SpinModel): The model for the spin system.
            *_args: Positional arguments for `propagator()`
            *_kws: Keyword arguments for `propagator()`

        Returns:
            y (ndarray[float]): The signal.

        """
        return spin_model.s0 * np.array(self._signal(
            spin_model, self.propagator(spin_model, *_args, **_kws)))

    # -----------------------------------
    @staticmethod
    def _signal(
            spin_model,
            p_op):
        """
        Compute the signal from the pulse sequence.

        Args:
            spin_model (SpinModel): The model for the spin system.
            p_op (np.ndarray[complex]):

        Returns:
            y (ndarray[float]): The signal.

        """
        return np.abs(fc.extra.mdot((
            spin_model.detector(),
            p_op,
            spin_model.equilibrium_magnetization())))

    # -----------------------------------
    @staticmethod
    def magnetization(
            spin_model,
            p_op):
        """
        Compute the signal from the pulse sequence.

        Args:
            spin_model (SpinModel): The model for the spin system.
            p_op (np.ndarray[complex]):

        Returns:
            y (ndarray[float]): The signal.

        """
        return np.dot(p_op, spin_model.equilibrium_magnetization())

    # -----------------------------------
    @staticmethod
    def _p_ops_subst(p_ops, idx, new_p_op):
        p_ops[idx] = new_p_op
        return p_ops

    # -----------------------------------
    @staticmethod
    def _p_ops_substs(p_ops, substs):
        for idx, new_p_op in substs:
            p_ops[idx] = new_p_op
        return p_ops

    # -----------------------------------
    def __repr__(self):
        text = '{}'.format(self.__class__.__name__)
        text += ':\n'
        if hasattr(self, 'pulses'):
            for pulse in self.pulses:
                text += '  {}\n'.format(pulse)
        return text

    # -----------------------------------
    def __str__(self):
        text = '{}: '.format(self.__class__.__name__)
        _blacklist = ['pulses']
        if hasattr(self, 'pulses'):
            text += 'pulses=[\n'
            for pulse in self.pulses:
                text += '  -{}\n'.format(pulse)
            text += '  ]\n'
        for name in dir(self):
            if not name.startswith('_') and name not in _blacklist:
                text += '{}={}  '.format(name, getattr(self, name))
        return text


# ======================================================================
class SteadyState(PulseSequence):
    # -----------------------------------
    def __init__(
            self,
            te=None,
            tr=None,
            n_r=1,
            *_args,
            **_kws):
        """

        Args:
            kernel (PulseSequence): The list of pulses to be repeated.
            num_repetitions (:
            mt_pulse_index (int|None): Index of the MT pulse in the kernel.
                If None, use the pulse with zero carrier frequency.
            *_args:
            **_kws:

        Returns:

        """
        PulseSequence.__init__(self, *_args, **_kws)
        idx = self.get_unique_pulses(('PulseExc', 'ReadOut'))
        if hasattr(self, '_idx'):
            self._idx.update(idx)
        else:
            self._idx = idx
        # handle repetition time
        if tr is not None:
            self.pulses[self._idx['ReadOut']].duration = \
                self._get_t_ro(self.duration, tr)
        else:
            tr = self.duration
        self.tr = tr
        self._t_ro = self.pulses[self._idx['ReadOut']].duration
        self._t_pexc = self.pulses[self._idx['PulseExc']].duration
        # handle echo times
        if te is None:
            te = self._t_pexc / 2.0
        self.te = te
        # ensure compatible echo_times and repetition_time
        if any(t < 0 for t in self \
                ._pre_post_delays(self.te, self._t_ro, self._t_pexc)):
            text = (
                'Incompatible options '
                '`echo_time={}` and `repetition_time={}`'.format(
                    te, tr))
            raise ValueError(text)
        self.n_r = n_r

    # -----------------------------------
    @staticmethod
    def _get_tr(pulses):
        return sum(
            pulse.duration if hasattr(pulse, 'duration') else 0.0
            for pulse in pulses)

    # -----------------------------------
    @staticmethod
    def _get_t_ro(temp_tr, tr):
        if tr >= temp_tr:
            t_ro = tr - temp_tr
        else:
            text = (
                'Incompatible pulse sequence and '
                '`repetition_time={}`'.format(tr))
            raise ValueError(text)
        return t_ro

    # -----------------------------------
    @staticmethod
    def _pre_post_delays(te, t_ro, t_pexc):
        return t_ro - te + t_pexc / 2.0, te - t_pexc / 2.0

    # -----------------------------------
    @staticmethod
    def _p_ops_reorder(p_ops, idx):
        return p_ops[idx + 1:] + p_ops[:idx]

    # -----------------------------------
    def propagator(
            self,
            spin_model,
            *_args,
            **_kws):
        """
        Compute the propagator of the pulse sequence.

        Args:
            spin_model (SpinModel): The model for the spin system.
            *_args: Positional arguments for 'propagators()'.
            **_kws: Keyword arguments for 'propagators()'.

        Returns:
            p_op (ndarray[float]): The propagator.
        """
        base_p_ops = self.propagators(spin_model, *_args, **_kws)
        pre_t, post_t = self._pre_post_delays(
            self.te, self._t_ro, self._t_pexc)
        p_ops = (
                [Delay(pre_t).propagator(spin_model, *_args, **_kws)] +
                self._p_ops_reorder(base_p_ops, self._idx['ReadOut']) +
                [Delay(post_t).propagator(spin_model, *_args, **_kws)])
        return self._propagator(p_ops, self.n_r)

    # -----------------------------------
    def __str__(self):
        text = '{}: '.format(self.__class__.__name__)
        for attr in ('tr', 'n_r', 'te', 'tes'):
            if hasattr(self, attr):
                text += '{}={}  '.format(attr, getattr(self, attr))
        text += '\n'
        if hasattr(self, 'pulses'):
            for pulse in self.pulses:
                text += '  {}\n'.format(pulse)
        return text


# ======================================================================
class MultiGradEchoSteadyState(SteadyState):
    # -----------------------------------
    def __init__(
            self,
            tes,
            *_args,
            **_kws):
        """

        Args:
            kernel (PulseSequence): The list of pulses to be repeated.
            num_repetitions (:
            mt_pulse_index (int|None): Index of the MT pulse in the kernel.
                If None, use the pulse with zero carrier frequency.
            *_args:
            **_kws:

        Returns:

        """
        SteadyState.__init__(self, None, *_args, **_kws)
        # handle echo times
        if tes is None:
            tes = self.te
        self.tes = fc.base.auto_repeat(tes, 1, False, False)
        # ensure compatible echo_times and repetition_time
        if any(t < 0
               for te in self.tes
               for t in self._pre_post_delays(te, self._t_ro, self._t_pexc)):
            text = (
                'Incompatible options '
                '`echo_time={}` and `repetition_time={}`'.format(
                    tes, self.tr))
            raise ValueError(text)

    # -----------------------------------
    def signal(
            self,
            spin_model,
            *_args,
            **_kws):
        te_p_ops = [
            [Delay(duration).propagator(spin_model, *_args, **_kws)
             for duration in
             self._pre_post_delays(te, self._t_ro, self._t_pexc)]
            for te in self.tes]
        central_p_ops = self._p_ops_reorder(
            self.propagators(spin_model), self._idx['ReadOut'])
        s_arr = np.array([
            self._signal(
                spin_model,
                self._propagator(
                    [te_p_op_i] + central_p_ops + [te_p_op_f],
                    self.n_r))
            for (te_p_op_i, te_p_op_f) in te_p_ops])
        return s_arr

    # -----------------------------------
    def __str__(self):
        text = '{}'.format(self.__class__.__name__)
        for attr in ('tr', 'n_r', 'tes'):
            if hasattr(self, attr):
                text += '{}={}'.format(attr, getattr(self, attr))
        text += ':\n'
        if hasattr(self, 'pulses'):
            for pulse in self.pulses:
                text += '  {}\n'.format(pulse)
        return text


# ======================================================================
# :: Tests

#     The magnetization transfer signal generated by the following sequence:
#
#     RF  __/‾\_____/\_______________
#     Gpe _______/≣\_________________
#     Gsl _______/≣\_________________
#     Gro _____________/‾‾‾‾‾‾‾‾\____
#     ADC______________/‾‾‾‾‾‾‾‾\____
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
elapsed(__file__[len(PATH['base']) + 1:])

# ======================================================================
if __name__ == '__main__':
    run_doctests(__doc__)
