#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
pymrt.recipes.b1: B1 pulse excitation magnetic field computation.
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

# from pymrt import VERB_LVL, D_VERB_LVL, VERB_LVL_NAMES
# from pymrt import elapsed, print_elapsed
# from pymrt import msg, dbg


# ======================================================================
def actual_flip_angle(
        arr,
        repetition_times,
        flip_angle,
        zero_cutoff=np.spacing(1)):
    """
    Calculate the flip angle efficiency from Actual Flip Angle (AFI) data.

    The efficiency factor :math:`\eta_{\\alpha}` is defined by:

    .. math:
        \eta_{\\alpha} =
        \\frac{\\alpha_{\mathrm{meas.}}}{\\alpha_{\mathrm{nom.}}}

    This is a closed-form solution.

    Args:
        arr (np.ndarray): The input array in arb.units.
            The repetition time T_R varies in the last dimension.
            Only the first two T_R images are used.
            If no phase data is provided, then the maximum measured flip angle
            must be 90°, otherwise a flip angle efficiency above 1 cannot be
            measured.
        repetition_times (iterable[int|float]): The repetition times T_R in ms.
            The number of points must match the last shape size of the data
            mag_arr (and phs_arr if not None).
            Only the first two T_R values are used.
            The measuring units are irrelevant as only the ratio is used.
            It assumes that values are sorted increasingly.
        flip_angle (int|float): The nominal flip angle in deg.
        zero_cutoff (float|None): The threshold value for masking zero values.

    Returns:
        results (dict): The calculated information.
            `fa` contains the measured flip angle map in deg.
            `eff` contains the flip angle efficiency factor.
    """
    valid = np.abs(arr[..., 1]) > zero_cutoff

    ratio_arr = np.zeros_like(arr[..., 1])
    ratio_arr[valid] = arr[..., 0][valid] / arr[..., 1][valid]

    tr_ratio = repetition_times[1] / repetition_times[0]

    valid *= np.abs(tr_ratio - ratio_arr) > zero_cutoff

    ratio_arr = (ratio_arr * tr_ratio - 1) / (tr_ratio - ratio_arr)
    flip_angle_arr = np.rad2deg(np.real(np.arccos(ratio_arr)))

    result = {
        'fa': flip_angle_arr,
        'eff': flip_angle_arr / flip_angle}

    return result


# ======================================================================
def double_angle_flash(
        arr1,
        arr2,
        flip_angle,
        tr=None,
        t1_arr=None,
        short_tr=False):
    """
    Calculate the flip angle efficiency from double-angle FLASH acquisitions.

    Uses the ratio between two FLASH images provided that the nominal flip 
    angles are known and they are one double the other.

    For optimal results: 30° < flip_angle < 60°.

    Here, :math:`s_1` is `arr1`, :math:`s_2` is `arr2` and
    :math:`r = \\frac{s_1}{s_2}`.

    If under the "long TR" approximation :math:`T_R \\gg \\max(T_1)`):
    
    .. math::
        \\cos(\\alpha) = \\frac{1}{2 r}
    
    The error is proportional to :math:`e^{-\\frac{T_R}{T_1}}`, with
    the largest bias determined by the maximum :math:`T_1`, but
    the acquisition time will be long.

    If under the "short TR" approximation :math:`T_R \\ll \\min(T_1)`):
    
    .. math::
        \\cos(\\alpha) = \\frac{r \pm |r - 2|}{2 (r - 1)}
    
    The error is proportional to :math:`\\frac{T_R}{T_1}`, with
    the largest bias determined by the minimum :math:`T_1`, but
    the acquisition time will be short.

    If TR is known, and a T1 map is available:
    
    .. math::
        \\cos(\\alpha) =
        \\frac{r \pm \sqrt{2 (1 - r) E_1^2 + 2 (1 - r) E_1 + r^2}}
        {2 (r - 1) E_1}
    
    with :math:`E_1 = e^{-\\frac{T_R}{T_1}}`.

    This is a closed-form solution.

    Args:
        arr1 (np.ndarray): The input array in arb.units.
            This is a FLASH image acquired with the nominal flip angle
            specified in the `flip_angle` parameter.
        arr2 (np.ndarray): The input double array in arb.units.
            This is a FLASH image acquired with the double the nominal
            flip angle specified in the `flip_angle` parameter.
        flip_angle (int|float): The nominal flip angle in deg.
            This is the nominal flip angle corresponding to `arr1`.
            The nominal flip angle corresponding to `arr2` must be double
            this value.
        tr (float|None): The repetition time in time units.
            If None, an approximated formula is used, depending on the
            value of `short_tr`.
            Otherwise, units must match those of `t1_arr`.
        t1_arr (np.ndarray|None): The T1 map in time units.
            If None, an approximated formula is used, depending on the
            value of `short_tr`.
            Otherwise, units must match those of `tr`.
        short_tr (bool): Determine the approximation to use.
            If True, the "short TR" approximation is used.
            If False, the "long TR" approximation is used.
            If both `tr` and `t1_arr` are defined, this parameter is ignored,
            and the exact formula is used.

    Returns:
        results (dict): The calculated information.
            `fa` contains the measured flip angle map in deg.
            `eff` contains the flip angle efficiency factor.
    """
    from numpy import exp, sqrt, arccos

    if t1_arr and tr:  # no approximation
        sgn = 1  # choose between the two separate solutions
        with np.errstate(divide='ignore', invalid='ignore'):
            ratio_arr = arr1 / arr2
            ratio_arr = \
                (ratio_arr + sgn * sqrt(
                    ratio_arr ** 2
                    + 2 * (ratio_arr - 1) * exp(-tr / t1_arr) ** 2
                    + 2 * (ratio_arr - 1) * exp(-tr / t1_arr))) \
                / (2 * (ratio_arr - 1) * exp(-tr / t1_arr))

    elif short_tr:  # short TR approximation (E1 = exp(-TR/T1) = 1)
        sgn = 1  # choose between the two separate solutions
        with np.errstate(divide='ignore', invalid='ignore'):
            ratio_arr = arr1 / arr2
            ratio_arr = \
                (ratio_arr + sgn * np.abs(ratio_arr - 2)) \
                / (2 * (ratio_arr - 1))

    else:  # long TR approximation (E1 = exp(-TR/T1) = 0)
        with np.errstate(divide='ignore', invalid='ignore'):
            ratio_arr = arr2 / arr1 / 2

    flip_angle_arr = np.rad2deg(np.real(arccos(ratio_arr)))

    result = {
        'fa': flip_angle_arr,
        'eff': flip_angle_arr / flip_angle}

    return result


# ======================================================================
def multi_angle_flash(
        arr,
        arr2,
        flip_angle,
        zero_cutoff=np.spacing(1)):
    warnings.warn('Not implemented yet')


# ======================================================================
def power_two_angles_flash(
        arrs,
        flip_angle,
        zero_cutoff=np.spacing(1)):
    warnings.warn('Not implemented yet')
