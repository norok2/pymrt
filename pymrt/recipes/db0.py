#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pymrt.recipes.b0: dB0 magnetic field variation computation.
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
import pymrt as mrt
# import pymrt.utils
# import pymrt.computation as pmc

# from pymrt import VERB_LVL, D_VERB_LVL, VERB_LVL_NAMES
# from pymrt import elapsed, report
# from pymrt import msg, dbg

from pymrt.constants import GAMMA, GAMMA_BAR
from pymrt.config import _B0

from pymrt.recipes import phs
from pymrt.recipes.phs import phs_to_dphs, dphs_to_phs

# ======================================================================
_TE = 20.0  # ms


# ======================================================================
def phs_to_db0(
        phs_arr,
        b0=_B0,
        tis=_TE,
        units='ms'):
    """
    Convert single echo phase to magnetic field variation.

    This assumes linear phase evolution.

    Args:
        phs_arr (np.ndarray): The input unwrapped phase array in rad.
        b0 (float): Main Magnetic Field B0 Strength in T.
        tis (float): Echo Time in ms.

    Returns:
        db0_arr (np.ndarray): The relative field array in ppb
    """
    return dphs_to_db0(
        phs_to_dphs(phs_arr, tis, tis_mask=None, units=units), b0=b0)


# ======================================================================
def db0_to_phs(
        db0_arr,
        b0=_B0,
        te=_TE):
    """
    Convert magnetic field variation to single echo phase.

    This assumes linear phase evolution.

    Args:
        db0_arr (np.ndarray): The magnetic field variation in ppb.
        b0 (float): Main Magnetic Field B0 Strength in T
        te (float): Echo Time in ms

    Returns:
        phs_arr (np.ndarray): The input unwrapped phase array in rad.
    """
    return dphs_to_phs(db0_to_dphs(db0_arr, b0=b0), tis=te)


# ======================================================================
def dphs_to_db0(
        dphs_arr,
        b0=_B0):
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
        b0=_B0):
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
        tis,
        tis_mask=None,
        b0=_B0):
    """
    Calculate magnetic field variation from phase evolution.

    The sampling times correspond to the echo times T_E in a
    Gradient-Recalled Echo (GRE) type pulse sequence.

    Args:
        phs_arr (np.ndarray): The input array in arb. units.
            The sampling time Ti varies in the last dimension.
        tis (Iterable): The sampling times Ti in time units.
            The number of points must match the last shape size of arr.
        tis_mask (Iterable[bool]|None): Determine the sampling times Ti to use.
            If None, all will be used.
        b0 (float): Main Magnetic Field B0 Strength in T

    Returns:
        db0_arr (np.ndarray): The relative field array in ppb
    """
    return dphs_to_db0(phs.phs_to_dphs(phs_arr, tis, tis_mask), b0)

import pymrt as mrt
import pymrt.input_output
from pymrt.recipes.generic import fix_phase_interval
s = '/home/raid1/metere/hd4/data/siemens/RME^ECM_Exp03/PMB_Thal_L+R_2017-04-04_16-30_7T_Magnetom/niz/s011__ME-FLASH-3D_res333_fast.nii.gz'
o = '/home/raid1/metere/data_sandbox/ecm_mri/exp03/b0.nii.gz'
flash_tes = [3.32, 10.27, 17.22, 24.17, 31.12, 38.07]  # ms
b0 = 297218209 / GAMMA_BAR['1H']
phs_arr, meta = mrt.input_output.load(s, meta=True)
phs_arr = fix_phase_interval(phs_arr)
db0_arr = fit_phase(phs_arr, flash_tes, b0=b0)
mrt.input_output.save(o, db0_arr, **meta)
print(o)


import pymrt as mrt
import pymrt.input_output
import pymrt.geometry
from pymrt.recipes.generic import fix_phase_interval
s = '/home/raid1/metere/hd4/data/siemens/RME^ECM_Exp03/PMB_Thal_L+R_2017-04-04_16-30_7T_Magnetom/niz/s039__B0-ME-FLASH-3D_res1.00.nii.gz'
o = '/home/raid1/metere/data_sandbox/ecm_mri/exp03/b0_i.nii.gz'
flash_tes = [1.31, 4.03]  # ms
b0 = 297218209 / GAMMA_BAR['1H']
base_shape = (192,) * 3
phs_arr, meta = mrt.input_output.load(s, meta=True)
phs_arr = fix_phase_interval(phs_arr)
db0_arr = fit_phase(phs_arr[..., :2], flash_tes[:2], b0=b0)
db0_arr = mrt.geometry.resample(db0_arr, base_shape)
mrt.input_output.save(o, db0_arr, **meta)
print(o)

