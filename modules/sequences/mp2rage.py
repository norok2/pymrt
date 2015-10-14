#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
mr_lib: MP2RAGE signal expression library.

Calculate the analytical expression of MP2RAGE signal.
Two different set of parameters (direct and indirect) are accepted.
Direct:
- T1: longitudinal relaxation time
- eff : efficiency eff of the adiabatic inversion pulse
- n : number of pulses in each GRE block
- TR_GRE : repetition time of GRE pulses in ms
- TA : time between inversion pulse and first GRE block in ms
- TB : time between first and second GRE blocks in ms
- TC : time after second GRE block in ms
- A1 : flip angle of the first GRE block in deg
- A2 : flip angle a2 of the second GRE block in deg
Indirect:
- T1: longitudinal relaxation time
- eff : efficiency eff of the adiabatic inversion pulse
- n : number of pulses in each GRE block
- TR_GRE : repetition time of GRE pulses in ms
- TR_SEQ : total repetition time of the MP2RAGE sequence in ms
- TI1 : inversion time (at center of k-space) of the first GRE blocks in ms
- TI2 : inversion time (at center of k-space) of the second GRE blocks in ms
- A1 : flip angle of the first GRE block in deg
- A2 : flip angle a2 of the second GRE block in deg
WARNING: when using indirect parameters, remember to check that TA, TB and TC
timing parameters are positive.

[ref: J. P. Marques at al., NeuroImage 49 (2010) 1271-1281]
"""


# ======================================================================
# :: Future Imports
from __future__ import division
from __future__ import absolute_import
from __future__ import print_function
#from __future__ import unicode_literals


__version__ = '0.0.0.0'
# $Source$


# ======================================================================
# :: Custom Module Details
AUTHOR = 'Riccardo Metere'
CONTACT = 'metere@cbs.mpg.de'
DATE_INFO = {'day': 18, 'month': 'Sep', 'year': 2014}
DATE = ' '.join([str(v) for k, v in sorted(DATE_INFO.items())])
LICENSE = 'License GPLv3: GNU General Public License version 3'
COPYRIGHT = 'Copyright (C) ' + str(DATE_INFO['year'])
# first non-empty line of __doc__
DOC_FIRSTLINE = [line for line in __doc__.splitlines() if line][0]


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

# :: External Imports
import numpy as np  # NumPy (multidimensional numerical arrays library)
# import scipy as sp  # SciPy (signal and image processing library)
# import matplotlib as mpl  # Matplotlib (2D/3D plotting library)
import sympy as sym  # SymPy (symbolic CAS library)
# import PIL  # Python Image Library (image manipulation toolkit)
# import SimpleITK as sitk  # Image ToolKit Wrapper
# import nibabel as nib  # NiBabel (NeuroImaging I/O Library)
# import nipy  # NiPy (NeuroImaging in Python)
# import nipype  # NiPype (NiPy Pipelines and Interfaces)

# :: External Imports Submodules
# import matplotlib.pyplot as plt  # Matplotlib's pyplot: MATLAB-like syntax
# import mayavi.mlab as mlab  # Mayavi's mlab: MATLAB-like syntax
# import scipy.optimize  # SciPy: Optimization Algorithms
# import scipy.integrate  # SciPy: Integrations facilities
# import scipy.constants  # SciPy: Mathematal and Physical Constants
# import scipy.ndimage  # SciPy: ND-image Manipulation


# ======================================================================
# :: supported verbosity levels (level 4 skipped on purpose)
VERB_LVL = {'none': 0, 'low': 1, 'medium': 2, 'high': 3, 'debug': 5}


# ======================================================================
# :: Default values
D_T1_RANGE = (400.0, 4000.0)
D_EFF = 1.0
D_N = 160
D_TR_GRE = 7.0
D_A1 = 4.0
D_A2 = 5.0
D_TR_SEQ = 8000.0
D_TI1 = 1000.0
D_TI2 = 3300.0
D_TA = 440.0
D_TB = 1180.0
D_TC = 4140.0

# signal ranges
#NAT_RANGE = (-0.5, 0.5)
STD_RANGE = (-0.5, 0.5)
DICOM_RANGE = (0, 4096)


# ======================================================================
# :: Calculate the analytical expression of MP2RAGE signal
from sympy import exp, sin, cos

T_1, EFF, NUM, TR_GRE, M_0, T_A, T_B, T_C, A_1, A_2 = \
    sym.symbols('T1 eff n TR_GRE M0 TA TB TC A1 A2')
MZ_SS_ = sym.symbols('mz_ss')


def mz_nrf(mz0, t_1, num, tr_gre, alpha, m_0):
    """
    Magnetization during the GRE block
    """
    return mz0 * (cos(alpha) * exp(-tr_gre / t_1)) ** num + \
        m_0 * (1 - exp(-tr_gre / t_1)) * \
        (1 - (cos(alpha) * exp(-tr_gre / t_1)) ** num) / \
        (1 - cos(alpha) * exp(-tr_gre / t_1))


def mz_0rf(mz0, t_1, time, m_0):
    """
    Magnetization during the period with no pulses
    """
    return mz0 * exp(-time / t_1) + m_0 * (1 - exp(-time / t_1))


def mz_i(mz0, eff):
    """
    Magnetization after adiabatic inversion pulse
    """
    return -eff * mz0


# :: steady state magnetization
EQN_MZ_SS = sym.Eq(
    MZ_SS_,
    mz_0rf(
    mz_nrf(
    mz_0rf(
    mz_nrf(
    mz_0rf(
    mz_i(MZ_SS_, EFF),
    T_1, T_A, M_0),
    T_1, NUM, TR_GRE, A_1, M_0),
    T_1, T_B, M_0),
    T_1, NUM, TR_GRE, A_2, M_0),
    T_1, T_C, M_0))
MZ_SS = sym.solve(EQN_MZ_SS, MZ_SS_)[0]
MZ_SS = sym.factor(MZ_SS)

# convenient exponentials
E_1 = exp(-TR_GRE / T_1)
E_A = exp(-T_A / T_1)
E_B = exp(-T_B / T_1)
E_C = exp(-T_C / T_1)

# signal for TI1 image (omitted factor: B1R * E_2 * M_0)
GRE_TI1 = sin(A_1) * ((-EFF * MZ_SS_ / M_0 * E_A + \
    (1 - E_A)) * (cos(A_1) * E_1) ** (NUM / 2 - 1) + \
    ((1 - E_1) * (1 - (cos(A_1) * E_1) ** (NUM / 2 - 1)) / \
    (1 - cos(A_1) * E_1)))

# signal for TI2 image (omitted factor: B1R * E_2 * M_0)
GRE_TI2 = sin(A_2) * (((MZ_SS_ / M_0) - (1 - E_C)) / \
    (E_C * (cos(A_2) * E_1) ** (NUM / 2)) - (1 - E_1) * \
    ((cos(A_2) * E_1) ** (-NUM / 2) - 1) / (1 - cos(A_2) * E_1))

# T1 map as a function of steady state signal
MAP_MZ_SS = (GRE_TI1 * GRE_TI2) / (GRE_TI1 ** 2 + GRE_TI2 ** 2)

# calculated T1 map using experimental parameters
MAP = MAP_MZ_SS.subs(MZ_SS_, MZ_SS)

# convert SymPy expression to Python function
MAP_FUNC = sym.lambdify(
    (T_1, EFF, NUM, TR_GRE, T_A, T_B, T_C, A_1, A_2), MAP)

# make the function NumPy-aware
MAP_NUMPY = np.vectorize(MAP_FUNC)


# ======================================================================
def calc_signal(t_1, eff, num, tr_gre, t_a, t_b, t_c, a_1, a_2):
    """
    Calculate MP2RAGE intensity from direct parameters (NumPy-aware).

    Parameters
    ==========
    t_1 : float
        T1 time in ms.
    eff : float
        efficiency eff of the adiabatic inversion pulse.
    num : int
        number n of pulses in each GRE block.
    tr_gre : float
        TR_GRE repetition time of GRE pulses in ms.
    t_a : float
        time TA between inversion pulse and first GRE block in ms.
    t_b : float
        time TB between first and second GRE blocks in ms.
    t_c : float
        time TC after second GRE block in ms.
    a_1 : float
        flip angle a1 of the first GRE block in deg.
    a_2 : float
        flip angle a2 of the second GRE block in deg.

    Returns
    =======
    sii : float
        signal intensity of the MP2RAGE sequence

    """
    a_1 = np.deg2rad(a_1)
    a_2 = np.deg2rad(a_2)
    return MAP_NUMPY(t_1, eff, num, tr_gre, t_a, t_b, t_c, a_1, a_2)


# ======================================================================
def calc_tr_seq(eff, num, tr_gre, t_a, t_b, t_c, a_1, a_2):
    """Calculate TR_SEQ for MP2RAGE sequence."""
    return t_a + t_b + t_c + 2 * num * tr_gre


# ======================================================================
def calc_ti1(eff, num, tr_gre, t_a, t_b, t_c, a_1, a_2):
    """ Calculate TI1 for MP2RAGE sequence."""
    return t_a + (1 / 2) * num * tr_gre


# ======================================================================
def calc_ti2(eff, num, tr_gre, t_a, t_b, t_c, a_1, a_2):
    """Calculate TI2 for MP2RAGE sequenc.e"""
    return t_a + t_b + (3 / 2) * num * tr_gre


# ======================================================================
def calc_signal2(t_1, eff, num, tr_gre, tr_seq, ti_1, ti_2, a_1, a_2):
    """
    Calculate MP2RAGE intensity from indirect parameters (NumPy-aware).

    Parameters
    ==========
    t_1 : float
        T1 time in ms
    eff : float
        efficiency eff of the adiabatic inversion pulse.
    num : int
        number n of pulses in each GRE block.
    tr_gre : float
        TR_GRE repetition time of GRE pulses in ms.
    tr_seq : float
        total repetition time of the MP2RAGE sequence in ms.
    ti_1 : float
        inversion time (at center of k-space) of the first GRE blocks in ms.
    ti_2 : float
        inversion time (at center of k-space) of the second GRE blocks in ms
    a_1 : float
        flip angle a1 of the first GRE block in deg.
    a_2 : float
        flip angle a2 of the second GRE block in deg.

    Returns
    =======
    sii : float
        signal intensity of the MP2RAGE sequence.

    """
    a_1 = np.deg2rad(a_1)
    a_2 = np.deg2rad(a_2)
    t_a = calc_ta(eff, num, tr_gre, tr_seq, ti_1, ti_2, a_1, a_2)
    t_b = calc_tb(eff, num, tr_gre, tr_seq, ti_1, ti_2, a_1, a_2)
    t_c = calc_tc(eff, num, tr_gre, tr_seq, ti_1, ti_2, a_1, a_2)
    return MAP_NUMPY(t_1, eff, num, tr_gre, t_a, t_b, t_c, a_1, a_2)


# ======================================================================
def calc_ta(eff, num, tr_gre, tr_seq, ti1, ti2, a_1, a_2):
    """Calculate TA for MP2RAGE sequence."""
    return (2.0 * ti1 - num * tr_gre) / 2.0


# ======================================================================
def calc_tb(eff, num, tr_gre, tr_seq, ti1, ti2, a_1, a_2):
    """ Calculate TB for MP2RAGE sequence."""
    return ti2 - ti1 - num * tr_gre


# ======================================================================
def calc_tc(eff, num, tr_gre, tr_seq, ti1, ti2, a_1, a_2):
    """Calculate TC for MP2RAGE sequence."""
    return tr_seq - ti2 - num * tr_gre / 2.0


# ======================================================================
if __name__ == '__main__':
    #todo: refresh this code (at some point)
    print(__doc__)
