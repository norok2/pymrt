#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Turbo Spin-Echo (TSE) signal expression library.

Calculate the analytical expression of the Turbo Spin-Echo (TSE) signal.
"""

# ======================================================================
# :: Future Imports
from __future__ import(
    division, absolute_import, print_function, unicode_literals)
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
# :: External Imports
# import numpy as np  # NumPy (multidimensional numerical arrays library)
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
# import pymrt.modules.base as pmu
# import pymrt.modules.plot as pmp
# from pymrt import INFO
# from pymrt import VERB_LVL, D_VERB_LVL
from pymrt import msg, dbg


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
def evolution_tse(
        initial_magnetization,
        refocus_interval=sym.Symbol('T_E_Delta'),
        echo_time=sym.Symbol('T_E'),
        repetition_time=sym.Symbol('T_R'),
        turbo_factor=1,
        flip_angle=sym.Symbol('a')):
    num_dim = 3

    relaxation_longitudinal = sym.Symbol('R_1')
    relaxation_transverse = sym.Symbol('R_2')
    resonance_offset = 0
    equilibrium_magnetization = sym.Symbol('M_eq')

    tmp_magnetization, first_excitation = evolution(
            initial_magnetization, flip_angle, (1, 2), refocus_interval,
            relaxation_longitudinal, relaxation_transverse, resonance_offset,
            equilibrium_magnetization)

    for i in range(turbo_factor - 1):
        tmp_magnetization, tmp_excitation = evolution(
                tmp_magnetization, 2 * flip_angle, (1, 2), refocus_interval,
                relaxation_longitudinal, relaxation_transverse,
                resonance_offset, equilibrium_magnetization)
    final_magnetization, tmp_excitation = evolution(
            tmp_magnetization, (2 * flip_angle), (1, 2),
            (repetition_time - turbo_factor * refocus_interval),
            relaxation_longitudinal, relaxation_transverse, resonance_offset,
            equilibrium_magnetization)
    # attempt some simplification
    # for i in range(num_dim):
    #     final_magnetization[i] = sym.factor(final_magnetization[i])
    return final_magnetization, first_excitation


# ======================================================================
def steady_state(
        evolution_func,
        *evolution_args,
        **evolution_kwargs):
    steady_state_magnetization_minus = magnetization('ss')
    evolution, first_excitation = evolution_func(
            steady_state_magnetization_minus,
            *evolution_args, **evolution_kwargs)
    eqn_steady_state = sym.Eq(steady_state_magnetization_minus, evolution)
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
            [sym.Symbol('M_{}_{}'.format(label, i)) for i in
             range(num_dim)])
    return mag_vec


# ======================================================================
if __name__ == '__main__':
    msg(__doc__.strip())
    ss = steady_state(evolution_tse, turbo_factor=6)
    print('\nSteady-State before excitation:')
    print(ss[0])
    print('\nSteady-State after excitation:')
    print(ss[1])

    # signal = sym.sqrt(sym.trigsimp(ss[1][0] * ss[1][0] + ss[1][1] * ss[1][
    # 1]))
    signal = sym.sqrt(ss[1][0] * ss[1][0] + ss[1][1] * ss[1][1])
    print('\nSignal expression:')
    print(signal)
