#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pymrt.recipes.b0: dB0 magnetic field variation computation.
"""

# ======================================================================
# :: Future Imports
from __future__ import (
    division, absolute_import, print_function, unicode_literals, )

# ======================================================================
# :: Python Standard Library Imports
# import itertools  # Functions creating iterators for efficient looping
import warnings  # Warning control
import collections  # Container datatypes

# :: External Imports
import numpy as np  # NumPy (multidimensional numerical arrays library)

# :: Local Imports
import pymrt as mrt  # Python Magnetic Resonance Tools: the multi-tool of MRI
# import pymrt.utils
# import pymrt.computation as pmc

from pymrt import INFO, PATH
from pymrt import VERB_LVL, D_VERB_LVL, VERB_LVL_NAMES
from pymrt import elapsed, report, run_doctests
from pymrt import msg, dbg, fmt, fmtm

from pymrt.constants import GAMMA, GAMMA_BAR

from pymrt.recipes import phs
from pymrt.recipes.phs import phs_to_dphs, dphs_to_phs


# ======================================================================
def phs_to_db0(
        phs_arr,
        b0,
        tis,
        time_units='ms'):
    """
    Convert single echo phase to magnetic field variation.

    This assumes linear phase evolution.

    Args:
        phs_arr (np.ndarray): The input unwrapped phase array in rad.
        b0 (float): Main Magnetic Field B0 Strength in T.
        tis (Iterable): The sampling times Ti in time units.
            The number of points must match the last shape size of arr.
        time_units (str|float|int): Units of measurement of Ti.
            If str, any valid SI time unit (e.g. `ms`, `ns`) will be accepted.
            If int or float, the conversion factor will be multiplied to `ti`.

    Returns:
        db0_arr (np.ndarray): The relative field array in ppb
    """
    return dphs_to_db0(
        phs_to_dphs(phs_arr, tis, tis_mask=None, time_units=time_units), b0=b0)


# ======================================================================
def db0_to_phs(
        db0_arr,
        b0,
        tis,
        phs0_arr=0,
        time_units='ms'):
    """
    Convert magnetic field variation to single echo phase.

    This assumes linear phase evolution.

    Args:
        db0_arr (np.ndarray): The magnetic field variation in ppb.
        b0 (float): Main Magnetic Field B0 Strength in T
        tis (Iterable): The sampling times Ti in time units.
            The number of points must match the last shape size of arr.
        phs0_arr (np.ndarray|int|float): The initial phase offset.
            If int or float, a constant offset is used.
        time_units (str|float|int): Units of measurement of Ti.
            If str, any valid SI time unit (e.g. `ms`, `ns`) will be accepted.
            If int or float, the conversion factor will be multiplied to `ti`.

    Returns:
        phs_arr (np.ndarray): The input unwrapped phase array in rad.
    """

    return dphs_to_phs(db0_to_dphs(db0_arr, b0=b0), tis, phs0_arr, time_units)


# ======================================================================
def dphs_to_db0(
        dphs_arr,
        b0):
    """
    Convert phase variation to magnetic field variation.

    Args:
        dphs_arr (np.ndarray): The phase variation in rad/s.
        b0 (float): Main Magnetic Field B0 Strength in T.

    Returns:
        db0_arr (np.ndarray): The relative field array in ppb
    """
    ppb_factor = 1e9
    db0_arr = ppb_factor * dphs_arr / (GAMMA['1H'] * b0)
    return db0_arr


# ======================================================================
def db0_to_dphs(
        db0_arr,
        b0):
    """
    Convert magnetic field variation to phase variation.

    Args:
        db0_arr (np.ndarray): The magnetic field variation in ppb.
        b0 (float): Main Magnetic Field B0 Strength in T

    Returns:
        dphs_arr (np.ndarray): The phase variation in rad/s.
    """
    ppb_factor = 1e-9
    dphs_arr = ppb_factor * db0_arr * (GAMMA['1H'] * b0)
    return dphs_arr


# ======================================================================
def fit_phase(
        phs_arr,
        b0,
        tis,
        tis_mask=None,
        time_units='ms'):
    """
    Calculate magnetic field variation from phase evolution.

    The sampling times correspond to the echo times T_E in a
    Gradient-Recalled Echo (GRE) type pulse sequence.

    Args:
        phs_arr (np.ndarray): The input array in arb. units.
            The sampling time Ti varies in the last dimension.
        b0 (float): Main Magnetic Field B0 Strength in T.
        tis (Iterable): The sampling times Ti in time units.
            The number of points must match the last shape size of arr.
        tis_mask (Iterable[bool]|None): Determine the sampling times Ti to use.
            If None, all will be used.
        time_units (str|float|int): Units of measurement of Ti.
            If str, any valid SI time unit (e.g. `ms`, `ns`) will be accepted.
            If int or float, the conversion factor will be multiplied to `ti`.

    Returns:
        db0_arr (np.ndarray): The relative field array in ppb
    """
    return dphs_to_db0(phs.phs_to_dphs(
        phs_arr, tis, tis_mask, time_units=time_units), b0)


# ======================================================================
elapsed(__file__[len(PATH['base']) + 1:])

# ======================================================================
if __name__ == '__main__':
    run_doctests(__doc__)
