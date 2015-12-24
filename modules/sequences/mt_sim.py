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
# import time  # Time access and conversions
# import datetime  # Basic date and time types
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

# :: External Imports Submodules
import matplotlib.pyplot as plt  # Matplotlib's pyplot: MATLAB-like syntax
# import scipy.optimize  # SciPy: Optimization
# import scipy.integrate  # SciPy: Integration
import scipy.constants  # SciPy: Constants
# import scipy.ndimage  # SciPy: Multidimensional image processing
import scipy.linalg  # SciPy: Linear Algebra
import scipy.stats  # SciPy: Statistical functions
import scipy.misc  # SciPy: Miscellaneous routines

from numpy import pi, sin, cos, exp
# from sympy import pi, sin, cos

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
GAMMA = \
    sp.constants.physical_constants['proton gyromag. ratio'][0]
GAMMA_BAR = \
    sp.constants.physical_constants['proton gyromag. ratio over 2 pi'][0]


# ======================================================================
class SpinModel:
    def __init__(
            self,
            m0,
            w0,
            r1,
            r2,
            k):
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
        # simple check on the number of parameters
        if self.num_pools != len(r1) != len(r2) or len(k) != self.num_exchange:
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
        m0_list = [
            np.array([0.0, 0.0, val])
            if not approx or i < 1 else np.array([val])
            for i, val in enumerate(self.m0)]
        return np.concatenate([np.array([0.5])] + m0_list)

    def detector(
            self,
            phase=pi / 2):
        m_eq = self.m_eq()
        detect = np.zeros_like(m_eq).astype(complex)
        counter = 0
        for i, val in enumerate(m_eq):
            if val == 0.0:
                elem = 1.0 if counter % 2 == 0 else 1.0j
                counter += 1
                detect[i] = elem
        detect *= np.exp(1j * phase)
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
            L (ndarray[float]): The L operator
        """
        # TODO: include the saturation rate constant
        if self.num_pools > 2:
            raise ValueError('operator not implemented yet with n_pools > 2')
        L = np.array([
            [0, 0, 0, 0, 0],
            [0, self.r2[0], (self.w0[0] - w_rf), -w1.imag, 0],
            [0, -(self.w0[0] - w_rf), self.r2[0], w1.real, 0],
            [-2 * self.r1[0], w1.imag, -w1.real,
             self.r1[0] + self.k[0] * self.m0[1], -self.k[0] * self.m0[0]],
            [-2 * self.r1[1], 0, 0, -self.k[0] * self.m0[1],
             self.r1[1] + self.k[0] * self.m0[0]]])
        return L

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
            t_arr):
        if len(w1_arr) != len(t_arr):
            raise IndexError('inconsistent RfPulse value / time')
        self.wave_form = 'custom'
        self.w1_arr = np.array(w1_arr)
        self.t_arr = np.array(t_arr)
        self.num_steps = len(self.w1_arr)
        self.duration = np.sum(self.t_arr)
        self.time_step = self.duration / self.num_steps
        self.flip_angle = np.sum(self.w1_arr * self.t_arr)

    def set_flip_angle(self, flip_angle):
        self.w1_arr = self.w1_arr * flip_angle / self.flip_angle
        self.flip_angle = flip_angle

    def propagator(
            self,
            spin_model,
            w_rf):
        """
        Calculate the Bloch-McConnell propagator P.

        Args:
            spin_model (SpinModel): The physical model for the spin pools

        Returns:
            P (ndarray): The propagator P.
        """
        propagators = [
            scipy.linalg.expm(-spin_model.dynamics_operator(w1, w_rf) * t)
            for w1, t in zip(self.w1_arr, self.t_arr)]
        return mrb.mdot(*propagators[::-1])

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
            truncation=0.001):
        # todo: double check
        support = np.linspace(
                scipy.stats.norm.ppf(0.0 + truncation),
                scipy.stats.norm.ppf(1.0 - truncation),
                num_steps)
        w1_arr = scipy.stats.norm.pdf(support)
        t_arr = np.ones_like(w1_arr) * duration / num_steps
        RfPulse.__init__(self, w1_arr, t_arr)
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

    def propagator(
            self,
            spin_model,
            w_rf):
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
    def __init__(self, name):
        self.name = name

    def propagator(
            self,
            spin_model,
            w_rf):
        pass

    def signal(
            self,
            spin_model,
            w_rf):
        signal = mrb.mdot(
                spin_model.detector(),
                self.propagator(spin_model, w_rf), spin_model.m_eq())
        return np.abs(signal)

    def __repr__(self):
        text = 'PulseSequence: '
        names = ['name']
        for name in names:
            text += '{}={}  '.format(name, self.__dict__[name])
        return text


# ======================================================================
class PulseList(PulseSequence):
    def __init__(self, pulses):
        self.pulses = pulses

    def propagator(
            self,
            spin_model,
            w_rf):
        propagators = [
            pulse.propagator(spin_model, w_rf) for pulse in self.pulses]
        return mrb.mdot(*propagators[::-1])


# ======================================================================
class PulseTrain(PulseSequence):
    def __init__(
            self,
            pulse_sequence,
            num_repetitions):
        self.pulse_sequence = pulse_sequence
        self.num_repetition = num_repetitions

    def propagator(
            self,
            spin_model,
            w_rf):
        return scipy.linalg.fractional_matrix_power(
                self.pulse_sequence.propagator(spin_model, w_rf),
                self.num_repetition)


# # ======================================================================
# def propagator(
#         dynamics_operator,
#         time_step=0e-3):
#     """
#     Calculate the propagator associated to delay.
#
#     Parameters
#     ----------
#     dynamic_operator_arr: ndarray
#
#     """
#     return scipy.linalg.expm(-dynamics_operator * time_step)
#     # numpy.linalg.matrix_power(M, n)


# # ======================================================================
# def propagator_detection(
#         physical_model_pools,
#         phase=(np.pi / 2.0)):
#     """
#     Calculate the propagator associated to detection.
#
#     Parameters
#     ----------
#     physical_model_pools: list
#
#     TODO:
#     """
#     propagator = np.ones()
#     return propagator
#
#
# # ======================================================================
# def propagator_spoil(
#         physical_model_pools,
#         scale=0):
#     """
#     Calculate the propagator associated to spoil.
#
#     Parameters
#     ----------
#     operator_L_arr : ndarray
#     TODO:
#
#     """
#     propagator = np.ones()
#     return propagator
#
#
# # ======================================================================
# def propagator_pulse(
#         physical_model,
#         pulse,
#         approximation='polynomial'):
#     """
#     Calculate the propagator associated to spoil.
#
#     Parameters
#     ----------
#     physical_model : SpinModel
#         Physical model used in the Magnetization Transfer experiment.
#     pulse : Pulse
#
#     """
#     # TODO:
#     propagator = np.ones()
#     return propagator
#
#
# # ======================================================================
# def magnetization_transfer_signal(physical_model, pulse_sequence_params):
#     """
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
    m0a, m0b, r1a, r1b, r2a, r2b, k = sym.symbols('m0a m0b r1a r1b r2a r2b k')
    spin_model = SpinModel(m0=[m0a, m0b], r1=[r1a, r1b], r2=[r2a, r2b], k=[k])

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
    B0 = 7.0  # T
    m0a, m0b, w0, r1a, r1b, r2a, r2b, k = \
        1000.0, 10.0, GAMMA * B0, 0.1, 0.2, 0.01, 0.02, 0.05
    spin_model = SpinModel(
            m0=[m0a, m0b], w0=[w0, w0], r1=[r1a, r1b], r2=[r2a, r2b], k=[k])

    w_rf = GAMMA * B0
    w1 = cmath.rect(100.0, 0.0)

    time_step = 10e6  # 10 µs
    L = spin_model.dynamics_operator(w_rf, w1)
    # P = propagator(L, time_step)

    print(spin_model.m_eq())
    print(spin_model)
    print(L)
    # print(P)


# ======================================================================
def test_mt_sequence():
    """
    Notes: import pi, sin and cos from numpy

    Returns:

    """
    B0 = 7.0  # T
    m0a, m0b, w0, r1a, r1b, r2a, r2b, k = \
        1000.0, 10.0, GAMMA * B0, 0.1, 0.2, 0.01, 0.02, 0.05
    spin_model = SpinModel(
            m0=[m0a, m0b], w0=[w0, w0], r1=[r1a, r1b], r2=[r2a, r2b], k=[k])

    w_rf = GAMMA * B0
    num_repetitions = 300

    delay1 = NoRfPulse(10.0e-3)
    delay2 = NoRfPulse(20.0e-3)
    delay3 = NoRfPulse(30.0e-3)
    readout_pulse = RfPulseRect(10.0e-6)
    spoiler = Spoiler(1.0)
    mt_pulse = RfPulseGauss(4000, 40.0e-3, np.deg2rad(220.0))

    pulse_sequence = PulseTrain(
            PulseList(
                    [delay1,
                     spoiler,
                     mt_pulse,
                     delay2,
                     spoiler,
                     readout_pulse,
                     delay3]),
            num_repetitions)

    signal = pulse_sequence.signal(spin_model, w_rf)

    print(spin_model)
    print(delay1)
    print(readout_pulse)
    print(mt_pulse)
    print(spoiler)
    print(signal)
    print(spin_model.detector())


# ======================================================================
def test_z_spectrum(
        freqs=np.logspace(0, 4, 400) * 50,
        powers=np.logspace(-1, 1, 20),
        save_file='z_spectrum.npz'):
    """
    Notes: import pi, sin and cos from numpy

    Returns:

    """
    B0 = 7.0  # T
    m0a, m0b, w0, r1a, r1b, r2a, r2b, k = \
        1000.0, 10.0, GAMMA * B0, 0.1, 0.2, 0.01, 0.02, 0.05
    spin_model = SpinModel(
            m0=[m0a, m0b], w0=[w0, w0], r1=[r1a, r1b], r2=[r2a, r2b], k=[k])

    w_rf = GAMMA * B0
    num_repetitions = 300

    delay1 = NoRfPulse(10.0e-3)
    delay2 = NoRfPulse(20.0e-3)
    delay3 = NoRfPulse(30.0e-3)
    readout_pulse = RfPulseRect(10.0e-6)
    spoiler = Spoiler(1.0)
    data = np.zeros((len(freqs), len(powers)))
    for i, power in enumerate(powers):
        mt_pulse = RfPulseGauss(4000, 40.0e-3, np.deg2rad(90.0 * power))

        pulse_sequence = PulseTrain(
                PulseList(
                        [delay1,
                         spoiler,
                         mt_pulse,
                         delay2,
                         spoiler,
                         readout_pulse,
                         delay3]),
                num_repetitions)

        s_func = np.vectorize(pulse_sequence.signal)
        s = s_func(spin_model, w_rf + freqs)
        data[:, i] = s
    np.savez(save_file, freqs, powers, data)

# ======================================================================
if __name__ == '__main__':
    print(__doc__)
    # test_symbolic()
    # test_simple()
    # test_mt_sequence()
    # test_z_spectrum()
    import cProfile

    cProfile.run('test_z_spectrum()', sort=2)
