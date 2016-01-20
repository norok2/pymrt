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
import scipy.constants  # SciPy: Constants
# import scipy.ndimage  # SciPy: Multidimensional image processing
import scipy.linalg  # SciPy: Linear Algebra
import scipy.stats  # SciPy: Statistical functions
import scipy.misc  # SciPy: Miscellaneous routines

from numpy import pi, sin, cos, exp, sqrt
# from sympy import pi, sin, cos, exp, sqrt, re, im


# :: Local Imports
import mri_tools.modules.base as mrb

# import mri_tools.modules.geometry as mrg
# import mri_tools.modules.plot as mrp
# import mri_tools.modules.segmentation as mrs

# from mri_tools import INFO
# from mri_tools import VERB_LVL
# from mri_tools import D_VERB_LVL

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
def superlorentz(x):
    # scipy.integrate.quad returns both the value and the error, here ignored
    return sp.integrate.quad(
            lambda t: superlorentz_integrand(x, t), 0.0, pi / 2.0)[0]


# ======================================================================
begin_time = time.time()
_SUPERLORENTZ['x'] = np.logspace(-10.0, 1.7, 256)
_SUPERLORENTZ['y'] = np.vectorize(superlorentz)(_SUPERLORENTZ['x'])
end_time = time.time()
print('Superlorentz Approx. Time: ',
      datetime.timedelta(0, end_time - begin_time))


def superlorentz_approx(
        x,
        x_i=_SUPERLORENTZ['x'],
        y_i=_SUPERLORENTZ['y']):
    return np.interp(np.abs(x), x_i, y_i)


# ======================================================================
class SpinModel:
    def __init__(
            self,
            m0,
            w0,
            r1,
            r2,
            k,
            approx=None):
        """
        Physical model to use for the calculation of the signal level.

        Args:
            m0 (list[float]): magnetization vectors magnitudes in arb.units
            w0 (list[float]): resonance frequencies in Hz
            r1 (list[float]): longitudinal relaxation rates in Hz
            r2 (list[float]): transverse relaxation rates in Hz
            k (list[float]): pool-pool exchange rates in Hz

        Returns:
            None.
        """
        self.num_pools = len(m0)
        # exchange at equilibrium between each two pools
        self.num_exchange = sp.misc.comb(self.num_pools, 2)
        self.approx = approx if approx is not None else [None] * self.num_pools
        self.num_approx = np.sum(
                [0 if pool_approx is None else 1
                 for pool_approx in self.approx])
        self.num_exact = self.num_pools - self.num_approx
        self.dynamics_operator_dim = 1 + 3 * self.num_exact + self.num_approx
        self.object_type = type(m0[0])
        # simple check on the number of parameters
        if self.num_pools != len(r1) != len(r2) != len(approx) \
                or len(k) != self.num_exchange:
            raise IndexError(
                    'inconsistent parameters number in physical model')
        self.m0 = np.array(m0)
        self.w0 = np.array(w0)
        self.r1 = np.array(r1)
        self.r2 = np.array(r2)
        self.k = np.array(k)

    def m_eq(self, approx=True):
        """
        Generate the equilibrium magnetization vector.

        Note that B0 is assumed parallel to z-axis, therefore:
        - the transverse magnetization is zero
        - the longitudinal magnetization is only in the z-axis

        Args:
            approx (bool): approximation to neglect

        Returns:
            m0_arr (ndarray[complex]
        """
        mag_eq_len = 1 + self.num_pools * 3
        mag_eq = np.zeros(mag_eq_len).astype(self.object_type)
        mag_eq[0] = 0.5
        # print(self.num_pools, self.num_approx, self.num_exact)  # DEBUG
        to_remove = []
        for j, (m0, approx) in enumerate(zip(self.m0, self.approx)):
            # print(j, m0, approx)  # DEBUG
            base = 1 + j * 3
            mag_eq[base + 2] = m0
            if approx:
                to_remove.extend([base, base + 1])
        # print(to_approx)  # DEBUG
        mag_eq = np.delete(mag_eq, to_remove, 0)
        return mag_eq

    def detector(
            self,
            phase=pi / 2):
        mag_eq_len = 1 + self.num_pools * 3
        detect = np.zeros(mag_eq_len).astype(complex)
        # print(self.num_pools, self.num_approx, self.num_exact)  # DEBUG
        to_remove = []
        for j, (m0, approx) in enumerate(zip(self.m0, self.approx)):
            base = 1 + j * 3
            # print(j, m0, approx)  # DEBUG
            detect[base:base + 2] = 1.0, 1.0j
            if approx:
                to_remove.extend([base, base + 1])
        # print(to_approx)  # DEBUG
        detect = np.delete(detect, to_remove, 0)
        detect *= exp(1j * phase)
        return detect

    def dynamics_operator(
            self,
            w_rf,
            w1):
        """
        Calculate the Bloch-McConnell dynamics operator, L.

        Args:
            w_rf (float): Modulation frequency in Hz
            w1 (complex): Excitation (carrier) frequency in Hz

        Returns:
            dynamics (ndarray[float]): The dynamics operator L
        """

        # TODO: include the exchange rate constant
        def bloch_core(r1, r2, w0):
            # note: w1 and w_rf are defined from the outer scope
            w1x, w1y = w1.real, w1.imag
            # w1x, w1y = re(w1), im(w1)  # for symbolic
            return np.array([
                [r2, w0 - w_rf, -w1y],
                [w_rf - w0, r2, w1x],
                [w1y, -w1x, r1]])

        def sat_rate_lineshape(r2, w0, lineshape):
            if lineshape == 'superlorentz':
                lineshape_factor = superlorentz((w0 - w_rf) / r2)
            elif lineshape == 'superlorentz_approx':
                lineshape_factor = superlorentz_approx((w0 - w_rf) / r2)
            elif lineshape in ('lorentz', 'cauchy'):
                lineshape_factor = \
                    1.0 / (pi * (1.0 + ((w0 - w_rf) / r2) ** 2.0))
            elif lineshape in ('gauss', 'normal'):
                lineshape_factor = \
                    exp(- ((w0 - w_rf) / r2) ** 2.0 / 2.0) / sqrt(2.0 * pi)
            else:
                lineshape_factor = 1.0
            saturation_rate = pi * w1 * w1.conjugate() * lineshape_factor / r2
            return np.abs(saturation_rate)

        # 3: cartesian dimensions
        # +1 for homogeneous operators
        # *2 because it is a matrix
        operator_shape = [3 * self.num_pools + 1] * 2
        dynamics = np.zeros(operator_shape).astype(self.object_type)
        # # ...to make L invertible
        # L[0, 0] = -2.0
        to_remove = []
        for i, approx in enumerate(self.approx):
            base = 1 + i * 3
            # Bloch operator core...
            dynamics[base:base + 3, base:base + 3] = bloch_core(
                    self.r1[i], self.r2[i], self.w0[i])
            # ...additional modification for homogeneous form
            dynamics[base + 2, 0] = -2.0 * self.r1[i] * self.m0[i]
            # deal with approximations
            if approx:
                to_remove.extend([base, base + 1])
                dynamics[base + 2, base + 2] = sat_rate_lineshape(
                        self.r2[i], self.w0[i], approx)
        # remove transverse components of approximated pools
        dynamics = np.delete(dynamics, to_remove, 0)
        dynamics = np.delete(dynamics, to_remove, 1)
        return dynamics

    def __repr__(self):
        text = 'SpinModel: '
        names = ['m0', 'w0', 'r1', 'r2', 'k']
        for name in names:
            text += '{}={}  '.format(name, self.__dict__[name])
        return text


# ======================================================================
class RfPulse:
    def __init__(
            self,
            w1_arr,
            t_arr,
            propagator_mode='exact'):
        if len(w1_arr) != len(t_arr):
            raise IndexError('inconsistent RfPulse value / time')
        self.wave_form = 'custom'
        self.w1_arr = np.array(w1_arr)
        self.t_arr = np.array(t_arr)
        self.num_steps = len(self.w1_arr)
        self.duration = np.sum(self.t_arr)
        self.time_step = self.duration / self.num_steps
        self.flip_angle = np.sum(self.w1_arr * self.t_arr)
        self.propagator_mode = propagator_mode
        self.propagator_mode_list = (
            None, 'sum', 'sum_pseudo-order-1', 'poly_abs-w1')

    def set_flip_angle(self, flip_angle):
        self.w1_arr = self.w1_arr * flip_angle / self.flip_angle
        self.flip_angle = flip_angle

    def propagator(
            self,
            spin_model,
            w_rf,
            mode=None):
        """
        Calculate the Bloch-McConnell propagator: expm(-L * Dt).

        L is the dynamics operator of the rf-excited system.
        Dt is a time interval where spin exchange dynamics is negligible.

        Args:
            spin_model (SpinModel): The physical model for the spin pools
            w_rf (float):
            mode (string|None): Approximation to use for faster computation.

        Returns:
            P (ndarray): The propagator P.
        """
        if mode is None:
            approx = self.propagator_mode
        if mode == 'sum':
            expm_sum = np.zeros((spin_model.dynamics_operator_dim,) * 2)
            for w1, t in zip(self.w1_arr, self.t_arr):
                expm_sum += spin_model.dynamics_operator(w_rf, w1)
            propagator_operator = scipy.linalg.expm(-expm_sum)
        elif mode == 'sum_pseudo-order-1':
            expm_list = []
            for w1, t in zip(self.w1_arr, self.t_arr):
                dynamics_operator = spin_model.dynamics_operator(w_rf, w1)
                expm_list.append(dynamics_operator * t)
            expm_sum = sum(expm_list)
            # pseudo-first-order correction
            comm_list = [
                mrb.commutator(expm_list[i], expm_list[i + 1]) / 2.0
                for i in range(len(expm_list[:-1]))]
            comm_sum = sum(comm_list)
            propagator_operator = scipy.linalg.expm(-(expm_sum + comm_sum))
        elif mode == 'poly_abs-w1':
            # todo: polish code
            num_samples = 5
            fit_order = 3
            selected = np.round(
                    np.linspace(0.0, self.num_steps - 1, num_samples))
            if fit_order is None:
                fit_order = num_samples - 1
            P_approx = np.zeros(
                    (spin_model.dynamics_operator_dim,) * 2 + (num_samples,))
            w1_approx = np.zeros(num_samples)
            E_list, L_list = [], []
            for j, i in enumerate(selected):
                L = spin_model.dynamics_operator(w_rf, self.w1_arr[i])
                L_list.append(L)
                E_list.append(L * self.t_arr[i])
                P_approx[:, :, j] = scipy.linalg.expm(L * self.t_arr[i])
                w1_approx[j] = np.abs(self.w1_arr[i])

            x_arr = w1_approx
            y_arr = P_approx

            support_axis = -1
            shape = y_arr.shape
            support_size = shape[support_axis]
            y_arr = y_arr.reshape((-1, support_size))
            # polyfit requires to change matrix orientation using transpose
            p_arr = np.polyfit(x_arr, y_arr.transpose(), fit_order)
            # transpose the results back
            p_arr = p_arr.transpose()

            P_arr = np.zeros((p_arr.shape[0], self.num_steps))
            for i in range(p_arr.shape[0]):
                P_arr[i, :] = np.polyval(p_arr[i], np.abs(self.w1_arr))
            P_list = [
                P_arr[:, i].reshape(list(shape[:support_axis]))
                for i in range(self.num_steps)]
            propagator_operator = mrb.mdot(*P_list[::-1])
        elif mode == 'sep_w1':
            # todo: polish code
            E_list, F_list = [], []
            L_free = spin_model.dynamics_operator(w_rf, 0.0)
            for w1, t in zip(self.w1_arr, self.t_arr):
                L = spin_model.dynamics_operator(w_rf, w1)
                F_list.append((L - L_free) * t)
                E_list.append(L * t)
            # calculate propagators
            P_free = scipy.linalg.expm(-L_free * np.mean(self.t_arr))
            P_pow = scipy.linalg.fractional_matrix_power(
                    P_free, self.num_steps)
            P_w1 = scipy.linalg.expm(-sum(F_list))
            return np.dot(P_pow, P_w1)
        elif mode == 'decimate':
            decimated_size = 3
            chunk_marks = np.round(
                    np.linspace(0, self.num_steps - 1, decimated_size + 1))
            t_arr = np.zeros(decimated_size)
            w1_arr = np.zeros(decimated_size).astype(complex)
            for i in range(decimated_size):
                chunk = slice(chunk_marks[i], chunk_marks[i + 1])
                t_arr[i] = np.sum(self.t_arr[chunk])
                w1_arr[i] = np.mean(self.w1_arr[chunk])
            propagator_list = [
                scipy.linalg.expm(-spin_model.dynamics_operator(w_rf, w1) * t)
                for w1, t in zip(w1_arr, t_arr)]
            propagator_operator = mrb.mdot(*propagator_list[::-1])
        else:  # no approximation
            propagator_list = [
                scipy.linalg.expm(-spin_model.dynamics_operator(w_rf, w1) * t)
                for w1, t in zip(self.w1_arr, self.t_arr)]
            propagator_operator = mrb.mdot(*propagator_list[::-1])
        return propagator_operator

    def __repr__(self):
        text = 'RfPulse: '
        names = [
            'wave_form', 'time_step', 'num_steps', 'duration']
        for name in names:
            text += '{}={}  '.format(name, self.__dict__[name])
        text += '{}={} ({})  '.format('flip_angle', self.flip_angle,
                                      np.rad2deg(self.flip_angle))
        return text


# ======================================================================
class NoRfPulse(RfPulse):
    def __init__(
            self,
            duration):
        RfPulse.__init__(self, (0.0,), (duration,))
        self.wave_form = 'const'


# ======================================================================
class RfPulseRect(RfPulse):
    def __init__(
            self,
            duration,
            flip_angle=pi / 2):
        w1 = flip_angle / duration
        RfPulse.__init__(self, (w1,), (duration,))
        self.wave_form = 'const'


# ======================================================================
class RfPulseGauss(RfPulse):
    def __init__(
            self,
            num_steps,
            duration,
            flip_angle=pi / 2,
            truncation=(0.001, 0.001),
            propagator_mode='exact'):
        # todo: double check
        if truncation[0] == truncation[1]:
            self.symmetric = True
        support = np.linspace(
                scipy.stats.norm.ppf(0.0 + truncation[0]),
                scipy.stats.norm.ppf(1.0 - truncation[1]),
                num_steps)
        w1_arr = scipy.stats.norm.pdf(support)
        t_arr = np.ones_like(w1_arr) * duration / num_steps
        RfPulse.__init__(self, w1_arr, t_arr, propagator_mode)
        self.set_flip_angle(flip_angle)
        self.wave_form = 'gauss'
        self.truncation = truncation

    def __repr__(self):
        text = RfPulse.__repr__(self)
        names = ['truncation']
        for name in names:
            text += '{}={}  '.format(name, self.__dict__[name])
        text += '\n{}={}'.format('w1_arr', self.w1_arr)
        return text


# ======================================================================
class Spoiler:
    def __init__(
            self,
            efficiency=1.0):
        self.efficiency = efficiency
        self.propagator_mode = 'exact'

    def propagator(
            self,
            spin_model,
            w_rf,
            mode=None):
        return np.diag([1.0 if val != 0.0 else 1.0 - self.efficiency
                        for val in spin_model.m_eq()])

    def __repr__(self):
        text = 'Spoiler: '
        names = ['efficiency']
        for name in names:
            text += '{}={}  '.format(name, self.__dict__[name])
        return text


# ======================================================================
class PulseSequence:
    def __init__(
            self,
            name,
            w_rf=None,
            gamma=GAMMA,
            b0=B0):
        self.name = name
        self.w_rf = GAMMA * B0 if w_rf is None else w_rf

    def propagator(
            self,
            spin_model,
            w_rf=None,
            mode=None):
        pass

    def signal(
            self,
            spin_model,
            w_rf=None):
        if w_rf is None:
            w_rf = self.w_rf
        signal = mrb.mdot(
                spin_model.detector(),
                self.propagator(spin_model, w_rf),
                spin_model.m_eq())
        return np.abs(signal)

    def __repr__(self):
        text = 'PulseSequence: '
        names = ['name']
        for name in names:
            text += '{}={}  '.format(name, self.__dict__[name])
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

    def propagator(
            self,
            spin_model,
            w_rf=None,
            mode=None):
        if w_rf is None:
            w_rf = self.w_rf
        propagators = [
            pulse.propagator(spin_model, w_rf, pulse.propagator_mode)
            for pulse in self.pulses]
        return mrb.mdot(*propagators[::-1])


# ======================================================================
class PulseTrain(PulseSequence):
    def __init__(
            self,
            pulse_sequence,
            num_repetitions,
            *args,
            **kwargs):
        PulseSequence.__init__(self, *args, **kwargs)
        self.pulse_sequence = pulse_sequence
        self.num_repetition = num_repetitions

    def propagator(
            self,
            spin_model,
            w_rf=None,
            mode=None):
        if w_rf is None:
            w_rf = self.w_rf
        return scipy.linalg.fractional_matrix_power(
                self.pulse_sequence.propagator(spin_model, w_rf),
                self.num_repetition)


# # ======================================================================
# def magnetization_transfer_signal(physical_model, pulse_sequence_params):
#     """
#     The magnetization transfer signal generated by the following sequence:
#
#     RF  _()_/â€¾\____()_/\________________
#     Gpe ___________/â‰£\____________
#     Gsl ___________/â‰£\____________
#     Gro ______________/â€¾â€¾â€¾â€¾â€¾â€¾â€¾â€¾\__
#     ADC ______________/â€¾â€¾â€¾â€¾â€¾â€¾â€¾â€¾\__
#
#     Î´x:     Delay
#     Ïƒx:     Spoiler, _()_
#     Pp:     Preparation (MT) pulse
#     Ep:     Exc
#
#     RF:     RadioFrequency signal
#     Gpe:    Gradient for phase encoding
#     Gsl:    Gradient for slice selection
#     Gro:    Gradient for readout
#
#     /|      inversion pulse
#     /\      Gaussian pulse
#     &       Sinc pulse
#     |\      selective pulse
#     TODO:
#     """
#     # mtloop=protocol parameter
#     # meq := equilibrium magnetization
#     dynamic_operator_arr = dynamic_operator(physical_model)
#     propagator_spoil_arr = propagator_spoil(physical_model['pools'])
#     propagator_detection_arr = propagator_detection(physical_model['pools'])
#     propagator_delay1_arr, propagator_delay2_arr, propagator_delay3_arr = [
#         propagator_delay(dynamic_operator_arr, pulse_sequence_params[key])
#         for key in ['delay_1', 'delay_2', 'delay_3']]
#     propagator_readout_arr, propagator_preparation_arr = [
#         propagator_pulse(physical_model, pulse_sequence_params[key])
#         for key in ['excitation_pulse', 'preparation_pulse']]
#
#     # delay3 * readoutpulse * spoil * delay2 * mtpulse * spoil * delay1
#     propagator_sequence = mdot_l(
#             propagator_delay1_arr,
#             propagator_spoil_arr,
#             propagator_preparation_arr,
#             propagator_delay2_arr,
#             propagator_spoil_arr,
#             propagator_readout_arr,
#             propagator_delay3_arr)
#
#     # detection * (sequence ^ num_loops) * equilibrium_magnetization
#     signal = mdot_r(
#             propagator_detection_arr,
#             np.linalg.matrix_power(
#                     propagator_sequence,
#                     pulse_sequence['num_preparation_loops']),
#             physical_model['equilibrium_magnetization'])
#     return signal


# ======================================================================
def test_symbolic():
    """
    Notes: import pi, sin and cos from sympy

    Returns:

    """
    m0a, m0b, m0c, w0, r1a, r1b, r1c, r2a, r2b, r2c, k_ab, k_ac, k_bc = \
        sym.symbols('m0a m0b m0c w0 r1a r1b r1c r2a r2b r2c k_ab k_ac k_bc')
    spin_model = SpinModel(
            m0=[m0a, m0b, m0c], w0=[w0, w0, w0],
            r1=[r1a, r1b, r1c], r2=[r2a, r2b, r2c],
            k=[k_ab, k_ac, k_bc],
            approx=(None, None, None))

    w_rf, w1 = sym.symbols('w_rf w1')

    L = spin_model.dynamics_operator(w_rf, w1)
    print(spin_model.m_eq())
    print(spin_model)
    print(L)


# ======================================================================
def test_simple():
    """
    Notes: import pi, sin and cos from numpy

    Returns:

    """
    m0a, m0b, w0, r1a, r1b, r2a, r2b, k = \
        1000.0, 100.0, GAMMA * B0, 0.1, 0.2, 0.01, 0.02, 0.05
    spin_model = SpinModel(
            m0=[m0a, m0b], w0=[w0, w0], r1=[r1a, r1b], r2=[r2a, r2b], k=[k])

    w_rf = GAMMA * B0
    w1 = cmath.rect(100.0, 0.0)

    L = spin_model.dynamics_operator(w_rf, w1)

    print(spin_model.m_eq())
    print(spin_model)
    print(L)


# ======================================================================
def test_mt_sequence():
    """
    Notes: import pi, sin and cos from numpy

    Returns:

    """
    m0a, m0b, w0, r1a, r1b, r2a, r2b, k = \
        1000.0, 150.0, GAMMA * B0, 1.8, 1.0, 32.0, 8.5e4, 0.0
    spin_model = SpinModel(
            m0=[m0a, m0b], w0=[w0, w0], r1=[r1a, r1b], r2=[r2a, r2b], k=[k])

    num_repetitions = 300

    delays = [NoRfPulse(10.0e-3), NoRfPulse(20.0e-3), NoRfPulse(30.0e-3)]
    readout_pulse = RfPulseRect(10.0e-6)
    spoilers = [Spoiler(1.0), Spoiler(1.0)]
    mt_pulse = RfPulseGauss(4000, 40.0e-3, np.deg2rad(220.0))

    pulse_sequence = PulseTrain(
            PulseList(
                    [delays[0],
                     spoilers[0],
                     mt_pulse,
                     delays[1],
                     spoilers[1],
                     readout_pulse,
                     delays[2]],
                    name='FlashMtKernel'),
            num_repetitions,
            name='FlashMt')

    signal = pulse_sequence.signal(spin_model)

    print(spin_model)
    print(delays[0])
    print(readout_pulse)
    print(mt_pulse)
    print(spoilers[0])
    print(signal)
    print(spin_model.detector())


# ======================================================================
def test_approx_propagator(powers=(1.0,)):
    m0a, m0b, w0, r1a, r1b, r2a, r2b, k = \
        1.0, 0.15, GAMMA * B0, 1.8, 1.0, 32.0, 8.5e4, 0.05
    spin_model = SpinModel(
            m0=[m0a], w0=[w0],
            r1=[r1a], r2=[r2a],
            k=[],
            approx=None)
    w_rf = GAMMA * B0

    for power in powers:
        rf_pulse = RfPulseGauss(
                4000, 40.0e-3, np.deg2rad(90.0 * power),
                propagator_mode='sum')
        P_exact = rf_pulse.propagator(
                spin_model, w_rf, 'x')
        P_sum = rf_pulse.propagator(
                spin_model, w_rf)
        P_sum_comm = rf_pulse.propagator(
                spin_model, w_rf, 'sum_pseudo-order-1')
        P_sum_pow = rf_pulse.propagator(
                spin_model, w_rf, 'sep_w1')
        P_poly = rf_pulse.propagator(
                spin_model, w_rf, 'poly')
        print(P_exact)
        print(np.sum(np.abs(P_exact)), np.sum(np.abs(P_exact - P_exact)))
        print(P_sum)
        print(np.sum(np.abs(P_sum)), np.sum(np.abs(P_exact - P_sum)))
        print(P_sum_comm)
        print(np.sum(np.abs(P_sum_comm)), np.sum(np.abs(P_exact - P_sum_comm)))
        print(P_sum_pow)
        print(np.sum(np.abs(P_sum_pow)), np.sum(np.abs(P_exact - P_sum_pow)))
        print(P_poly)
        print(np.sum(np.abs(P_poly)), np.sum(np.abs(P_exact - P_poly)))


# ======================================================================
def test_z_spectrum(
        freqs=np.logspace(0, 4, 40) * 5,
        powers=np.logspace(-1, 1, 5),
        save_file='/tmp/mri_tools/z_spectrum_approx.npz'):
    """

    Args:
        freqs:
        powers:
        save_file:

    Returns:

    """
    m0a, m0b, w0, r1a, r1b, r2a, r2b, k = \
        1000.0, 100.0, GAMMA * B0, 1.8, 1.0, 32.0, 8.5e4, 0.05
    spin_model = SpinModel(
            m0=[m0a, m0b], w0=[w0, w0],
            r1=[r1a, r1b], r2=[r2a, r2b],
            k=[k],
            approx=[None, 'superlorentz_approx'])

    num_repetitions = 300

    data = np.zeros((len(freqs), len(powers)))

    delays = [NoRfPulse(10.0e-3), NoRfPulse(10.0e-3), NoRfPulse(0.0)]
    readout_pulse = RfPulseRect(10.0e-6)
    spoilers = [Spoiler(1.0), Spoiler(1.0)]

    # could be rewritten using numpy.vectorize

    for i, power in enumerate(powers):
        for j, freq in enumerate(freqs):
            mt_pulse = RfPulseGauss(
                    40, 40.0e-3, np.deg2rad(90.0 * power),
                    propagator_mode='decimate')

            sequence_kernel = PulseList(
                    [delays[0],
                     spoilers[0],
                     mt_pulse,
                     delays[1],
                     spoilers[1],
                     readout_pulse,
                     delays[2]],
                    name='FlashMtKernel')
            pulse_sequence = PulseTrain(
                    sequence_kernel, num_repetitions, name='FlashMt')

            data[j, i] = pulse_sequence.signal(
                    spin_model, pulse_sequence.w_rf + freq)

    # plot results
    X, Y = np.meshgrid(powers, range(len(freqs)))
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X, Y, data)
    np.savez(save_file, freqs, powers, data)
    # plt.show()


# ======================================================================
if __name__ == '__main__':
    print(__doc__)
    # test_symbolic()
    # test_simple()
    # test_mt_sequence()
    # test_approx_propagator()
    # test_z_spectrum()

    profile.run('test_z_spectrum()', sort=2)
