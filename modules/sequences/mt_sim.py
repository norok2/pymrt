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
# import matplotlib.pyplot as plt  # Matplotlib's pyplot: MATLAB-like syntax
# import scipy.optimize  # SciPy: Optimization
# import scipy.integrate  # SciPy: Integration
import scipy.constants  # SciPy: Constants
# import scipy.ndimage  # SciPy: Multidimensional image processing
import scipy.linalg  # SciPy: Linear Algebra
import scipy.misc  # SciPy: Miscellaneous routines

from numpy import pi, sin, cos
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
            k (list[float]): pool exchange rates in Hz

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
        self.names = ('m0', 'w0', 'r1', 'r2', 'k')
        for name in self.names:
            self.__dict__[name] = np.array(locals()[name])

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

    def __repr__(self):
        text = ''
        for name in self.names:
            text += '{}:{}  '.format(name, self.__dict__[name])
        return text


class RfPulse:
    wave_form = 'custom'
    num_steps = 1
    duration = 0

    def __init__(
            self,
            w1,
            time_step=None):
        try:
            iter(time_step)
        except TypeError:
            time_step = np.ones_like(w1) * time_step
        self.w1 = np.array(w1)
        self.time_step = time_step
        self.num_steps = len(w1)
        self.duration = np.sum(time_step)
        self.flip_angle = 0


class NoRfPulse(RfPulse):
    def __init__(
            self,
            duration):
        self.wave_form


class PulseSequence:
    def __init__(self, pulses):
        self.pulses = pulses

    def propagator(self):
        propagator =
        return


# ======================================================================
def dynamics_operator(
        spin_model,
        w1,
        w_rf,
        approx=True):
    """


    Args:
        spin_model:
        w1:
        w_rf:


    Returns:

    """
    # TODO: include the saturation rate constant
    if spin_model.num_pools > 2:
        raise ValueError('operator not implemented yet with n_pools > 2')
    if approx:
        L = np.array([
            [0, 0, 0, 0, 0],
            [0, spin_model.r2[0], (spin_model.w0[0] - w_rf), -w1.imag, 0],
            [0, -(spin_model.w0[0] - w_rf), spin_model.r2[0], w1.real, 0],
            [-2 * spin_model.r1[0], w1.imag, -w1.real,
             spin_model.r1[0] + spin_model.k[0] * spin_model.m0[1],
             -spin_model.k[0] * spin_model.m0[0]],
            [-2 * spin_model.r1[1], 0, 0, -spin_model.k[0] * spin_model.m0[1],
             spin_model.r1[1] + spin_model.k[0] * spin_model.m0[0]]])
    else:
        L = 2
    return L


# ======================================================================
def propagator(
        dynamics_operator,
        time_step=0e-3):
    """
    Calculate the propagator associated to delay.

    Parameters
    ----------
    dynamic_operator_arr: ndarray

    """
    return scipy.linalg.expm(-dynamics_operator * time_step)


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


# ======================================================================
def magnetization_transfer_signal(physical_model, pulse_sequence_params):
    """
    The magnetization transfer signal generated by the following sequence:

    RF  _()_/‾\____()_/\________________
    Gpe ___________/≣\____________
    Gsl ___________/≣\____________
    Gro ______________/‾‾‾‾‾‾‾‾\__
    ADC ______________/‾‾‾‾‾‾‾‾\__

    δx:     Delay
    σx:     Spoiler, _()_
    Pp:     Preparation (MT) pulse
    Ep:     Exc

    RF:     RadioFrequency signal
    Gpe:    Gradient for phase encoding
    Gsl:    Gradient for slice selection
    Gro:    Gradient for readout

    /|      inversion pulse
    /\      Gaussian pulse
    &       Sinc pulse
    |\      selective pulse
    TODO:
    """
    # mtloop=protocol parameter
    # meq := equilibrium magnetization
    dynamic_operator_arr = dynamic_operator(physical_model)
    propagator_spoil_arr = propagator_spoil(physical_model['pools'])
    propagator_detection_arr = propagator_detection(physical_model['pools'])
    propagator_delay1_arr, propagator_delay2_arr, propagator_delay3_arr = [
        propagator_delay(dynamic_operator_arr, pulse_sequence_params[key])
        for key in ['delay_1', 'delay_2', 'delay_3']]
    propagator_readout_arr, propagator_preparation_arr = [
        propagator_pulse(physical_model, pulse_sequence_params[key])
        for key in ['excitation_pulse', 'preparation_pulse']]

    # delay3 * readoutpulse * spoil * delay2 * mtpulse * spoil * delay1
    propagator_sequence = mdot_l(
            propagator_delay1_arr,
            propagator_spoil_arr,
            propagator_preparation_arr,
            propagator_delay2_arr,
            propagator_spoil_arr,
            propagator_readout_arr,
            propagator_delay3_arr)

    # detection * (sequence ^ num_loops) * equilibrium_magnetization
    signal = mdot_r(
            propagator_detection_arr,
            np.linalg.matrix_power(
                    propagator_sequence,
                    pulse_sequence['num_preparation_loops']),
            physical_model['equilibrium_magnetization'])
    return signal


# ======================================================================
def test_symbolic():
    """
    Notes: import pi, sin and cos from sympy

    Returns:

    """
    m0a, m0b, r1a, r1b, r2a, r2b, k = sym.symbols('m0a m0b r1a r1b r2a r2b k')
    spin_model = SpinModel(m0=[m0a, m0b], r1=[r1a, r1b], r2=[r2a, r2b], k=[k])

    b1t, phi, w, gamma = sym.symbols('b1t phi w g')
    rf_excitation = RfExcitation(b1t, phi, frequency_shift=w, gamma=gamma)

    L = dynamics_operator(spin_model, rf_excitation)
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

    w1, w_rf = cmath.rect(100.0, 0.0), GAMMA * B0

    time_step = 10e6  # 10 µs
    L = dynamics_operator(spin_model, w1, w_rf)
    P = propagator(L, time_step)

    print(spin_model.m_eq())
    print(spin_model)
    print(L)
    print(P)


# ======================================================================
def test_rf_excitation():
    """
    Notes: import pi, sin and cos from numpy

    Returns:

    """
    # todo: create a complex pulse shape and calculate the
    B0 = 7.0  # T
    m0a, m0b, w0a, w0b, r1a, r1b, r2a, r2b, k = \
        1000.0, 10.0, GAMMA * B0, GAMMA * B0, 0.1, 0.2, 0.01, 0.02, 0.05
    spin_model = SpinModel(m0=[m0a, m0b], r1=[r1a, r1b], r2=[r2a, r2b], k=[k])

    w1, w_rf = cmath.rect(100.0, 0.0), GAMMA * B0

    time_step = 10e6  # 10 µs
    L = dynamics_operator(spin_model, w1, w_rf)
    P = propagator(L, time_step)

    print(spin_model.m_eq())
    print(spin_model)
    print(L)
    print(P)


# ======================================================================
if __name__ == '__main__':
    print(__doc__)
    # test_symbolic()
    test_simple()
    # test_rf_excitation()
