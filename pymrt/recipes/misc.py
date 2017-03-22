#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
pymrt.recipes.b0: dB0 computation algorithms.
"""

# ======================================================================
# :: Future Imports
from __future__ import (
    division, absolute_import, print_function, unicode_literals)

# ======================================================================
# :: Python Standard Library Imports
# import itertools  # Functions creating iterators for efficient looping
import warnings  # Warning control
import collections  # Container datatypes

# :: External Imports
import numpy as np  # NumPy (multidimensional numerical arrays library)

# :: Local Imports
import pymrt.utils as pmu

# from pymrt import VERB_LVL, D_VERB_LVL, VERB_LVL_NAMES
# from pymrt import elapsed, print_elapsed
# from pymrt import msg, dbg

from pymrt.constants import GAMMA, GAMMA_BAR
from pymrt.config import _B0

from pymrt.recipes import phs


# ======================================================================
def time_to_rate(
        arr,
        in_units='ms',
        out_units='Hz'):
    k = 1.0
    if in_units == 'ms':
        k *= 1.0e3
    if out_units == 'kHz':
        k *= 1.0e-3
    arr[arr != 0.0] = k / arr[arr != 0.0]
    return arr


# ======================================================================
def rate_to_time(
        arr,
        in_units='Hz',
        out_units='ms'):
    k = 1.0
    if in_units == 'kHz':
        k *= 1.0e3
    if out_units == 'ms':
        k *= 1.0e-3
    arr[arr != 0.0] = k / arr[arr != 0.0]
    return arr


# ======================================================================
def fix_phase_interval(arr):
    """
    Ensure that the range of values is interpreted as valid phase information.

    This is useful for DICOM-converted images (without post-processing).

    Args:
        arr (np.ndarray): Array to be processed.

    Returns:
        array (np.ndarray): An array scaled to (-pi,pi).

    Examples:
        >>> fix_phase_interval(np.arange(8))
        array([-3.14159265, -2.24399475, -1.34639685, -0.44879895,  0.44879895,
                1.34639685,  2.24399475,  3.14159265])
        >>> fix_phase_interval(np.array([-10, -5, 0, 5, 10]))
        array([-3.14159265, -1.57079633,  0.        ,  1.57079633,  3.14159265])
        >>> fix_phase_interval(np.array([-10, 10, 1, -3]))
        array([-3.14159265,  3.14159265,  0.31415927, -0.9424778 ])
    """
    if not pmu.is_in_range(arr, (-np.pi, np.pi)):
        arr = pmu.scale(arr.astype(float), (-np.pi, np.pi))
    return arr


# ======================================================================
def polar2complex(mag_arr, phs_arr=None, fix_phase=True):
    """
    Convert magnitude and phase arrays into a complex array.

    It can automatically correct for arb.units in the phase.

    Args:
        mag_arr (np.ndarray): The magnitude image array in arb.units.
        phs_arr (np.ndarray): The phase image array in rad or arb.units.
            The values range is automatically corrected to radians.
            The wrapped data is expected.
            If units are radians, i.e. data is in the [-π, π) range,
            no conversion is performed.
            If None, only magnitude data is used.
        fix_phase (bool): Fix the phase interval / units.
            If True, `phs_arr` is corrected with `fix_phase_interval()`.

    Returns:
        arr (np.ndarray):

    See Also:
        pymrt.computation.fix_phase_interval
    """
    cx_arr = pmu.polar2complex(
        mag_arr.astype(float), fix_phase_interval(phs_arr)
        if fix_phase else phs_arr) \
        if phs_arr is not None else mag_arr.astype(float)
    return cx_arr
