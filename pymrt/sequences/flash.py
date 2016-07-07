#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
FLASH pulse sequence library.

Calculate the analytical expression of the FLASH pulse sequence signal and
other quantities related to the FLASH pu
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
# import time  # Time access and conversions
# import datetime  # Basic date and time types
# import operator  # Standard operators as functions
# import collections  # High-performance container datatypes
# import itertools  # Functions creating iterators for efficient looping
# import functools  # Higher-order functions and operations on callable objects
# import argparse  # Parser for command-line options, arguments and subcommands
# import subprocess  # Subprocess management
# import multiprocessing  # Process-based parallelism
# import csv  # CSV File Reading and Writing [CSV: Comma-Separated Values]
# import json  # JSON encoder and decoder [JSON: JavaScript Object Notation]
# import warnings  # Warning control
# import unittest  # Unit testing framework
import doctest  # Test interactive Python examples

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
# import scipy.optimize  # SciPy: Optimization Algorithms
# import scipy.integrate  # SciPy: Integrations facilities
# import scipy.constants  # SciPy: Mathematical and Physical Constants
# import scipy.ndimage  # SciPy: ND-image Manipulation
import scipy.constants  # SciPy: Constants


# :: Local Imports
# import pymrt.modules.base as mrb
# import pymrt.modules.plot as mrp
# from pymrt import INFO
# from pymrt import VERB_LVL
# from pymrt import D_VERB_LVL
# from pymrt import get_first_line


# ======================================================================
def signal():
    pass


# ======================================================================
def rotation(
        angle=sym.Symbol('a'),
        axes=(0, 1),
        num_dim=3):
    rot_mat = sym.eye(num_dim)
    rot_mat[axes[0], axes[0]] = sym.cos(angle)
    rot_mat[axes[1], axes[1]] = sym.cos(angle)
    rot_mat[axes[0], axes[1]] = -sym.sin(angle)
    rot_mat[axes[1], axes[0]] = sym.sin(angle)
    return rot_mat


# ======================================================================
def resonance_offset_evolution(
        time=sym.Symbol('t'),
        magnetic_field_variation=sym.Symbol('B_Delta'),
        gamma=sp.constants.physical_constants['proton gyromagn. ratio'][0]):
    return gamma * magnetic_field_variation * time


# ======================================================================
def evolution(
        initial_magnetization,
        flip_angle=sym.Symbol('a'),
        rotation_plane=(1, 2),
        duration=sym.Symbol('t'),
        relaxation_longitudinal=sym.Symbol('R_1'),
        relaxation_transverse=sym.Symbol('R_2'),
        resonance_offset=0,
        equilibrium_magnetization=sym.Symbol('M_eq')):
    decay = rotation(resonance_offset, (0, 1))
    decay[0:2, 0:2] *= sym.exp(-duration * relaxation_transverse)
    decay[-1, -1] *= sym.exp(-duration * relaxation_longitudinal)
    recovery = sym.Matrix(
        [0, 0, equilibrium_magnetization *
         (1 - sym.exp(-duration * relaxation_longitudinal))])
    excitation = rotation(flip_angle, rotation_plane)
    final_magnetization = decay * excitation * initial_magnetization + recovery
    return final_magnetization, excitation


# ======================================================================
def evolution_flash(
        initial_magnetization,
        repetition_time=sym.Symbol('T_R'),
        flip_angle=sym.Symbol('a')):
    num_dim = 3

    relaxation_longitudinal = sym.Symbol('R_1')
    relaxation_transverse = sym.Symbol('R_2')
    resonance_offset = 0
    equilibrium_magnetization = sym.Symbol('M_eq')

    final_magnetization, first_excitation = evolution(
        initial_magnetization, flip_angle, (1, 2), repetition_time,
        relaxation_longitudinal, relaxation_transverse, resonance_offset,
        equilibrium_magnetization)

    # attempt some simplification
    for i in range(num_dim):
        final_magnetization[i] = sym.trigsimp(final_magnetization[i])
    return final_magnetization, first_excitation


# ======================================================================
def steady_state(
        evolution_func,
        *evolution_args,
        **evolution_kwargs):
    steady_state_magnetization_minus = magnetization('ss')
    final_magnetization, first_excitation = evolution_func(
        steady_state_magnetization_minus,
        *evolution_args, **evolution_kwargs)
    eqn_steady_state = sym.Eq(
        steady_state_magnetization_minus, final_magnetization)
    steady_state_solution = sym.solve(
        eqn_steady_state, steady_state_magnetization_minus)
    steady_state_magnetization_minus = sym.Matrix(
        [steady_state_solution[item]
         for item in steady_state_magnetization_minus])
    steady_state_magnetization_plus = \
        first_excitation * steady_state_magnetization_minus
    return steady_state_magnetization_minus, steady_state_magnetization_plus


# ======================================================================
def magnetization(
        label='',
        num_dim=3):
    mag_vec = sym.Matrix(
        [sym.Symbol('M_{}_{}'.format(label, i))
         for i in range(num_dim)])
    return mag_vec


# ======================================================================
def ernst_calc(
        t1=None,
        tr=None,
        fa=None):
    """
    Calculate optimal T1, TR or FA (given the other two) for FLASH sequence.

    Args:
        t1 (float|None): Longitudinal relaxation time T1 in ms
        tr (float|None): Repetition time TE in ms
        fa (float|None): flip angle FA in deg

    Returns:
        val (float): Either T1, TR or FA fulfilling Ernst condition

    Examples:
        >>> ernst_calc(100.0, 30.0)
        (42.198837866408269, 'FA', 'deg')
        >>> ernst_calc(100.0, fa=42)
        (29.686433336996988, 'TR', 'ms')
        >>> ernst_calc(None, 30.0, 42)
        (101.05626250025875, 'T1', 'ms')
    """
    if t1 and tr:
        fa = np.rad2deg(np.arccos(np.exp(-tr / t1)))
        val, name, units = fa, 'FA', 'deg'
    elif tr and fa:
        t1 = -tr / np.log(np.cos(np.deg2rad(fa)))
        val, name, units = t1, 'T1', 'ms'
    elif t1 and fa:
        tr = -t1 * np.log(np.cos(np.deg2rad(fa)))
        val, name, units = tr, 'TR', 'ms'
    else:
        val = name = units = None
    return val, name, units


# ======================================================================
if __name__ == '__main__':
    print(__doc__)
    doctest.testmod()

    ss = steady_state(evolution_flash)
    signal = sym.sqrt(sym.trigsimp(ss[1][0] * ss[1][0] + ss[1][1] * ss[1][1]))
    print('\nSteady-State before excitation:')
    print(ss[0])
    print('\nSteady-State after excitation:')
    print(ss[1])
    print('\nSignal expression:')
    print(signal)
