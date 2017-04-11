#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
pymrt.recipes.t1: T1 longitudinal relaxation computation.
"""

# ======================================================================
# :: Future Imports
from __future__ import (
    division, absolute_import, print_function, unicode_literals)

# ======================================================================
# :: Python Standard Library Imports
# import itertools  # Functions creating iterators for efficient looping
# import warnings  # Warning control
# import collections  # Container datatypes

# :: External Imports
import numpy as np  # NumPy (multidimensional numerical arrays library)

# :: Local Imports
import pymrt as mrt
# import pymrt.utils

# from pymrt import VERB_LVL, D_VERB_LVL, VERB_LVL_NAMES
# from pymrt import elapsed, print_elapsed
# from pymrt import msg, dbg
import pymrt.utils
from pymrt.recipes import generic
from pymrt.recipes.generic import fix_phase_interval


# ======================================================================
def fit_mp2rage_rho(
        inv1m_arr,
        inv1p_arr,
        inv2m_arr,
        inv2p_arr,
        regularization=np.spacing(1),
        values_interval=None):
    """
    Calculate the rho image from an MP2RAGE acquisition.
    
    This is also referred to as the uniform images, because it should be free
    from low-spatial frequency biases.

    Args:
        inv1m_arr (float|np.ndarray): Magnitude of the first inversion image.
        inv1p_arr (float|np.ndarray): Phase of the first inversion image.
        inv2m_arr (float|np.ndarray): Magnitude of the second inversion image.
        inv2p_arr (float|np.ndarray): Phase of the second inversion image.
        regularization (float|int): Parameter for the regularization.
            This parameter is added to the denominator of the rho expression
            for normalization purposes, therefore should be much smaller than
            the average of the magnitude images.
            Larger values of this parameter will have the side effect of
            denoising the background.
        values_interval (tuple[float|int]|None): The output values interval.
            The standard values are linearly converted to this range.
            If None, the natural [-0.5, 0.5] interval will be used.

    Returns:
        rho_arr (float|np.ndarray): The calculated rho (uniform) image.
    """
    if not regularization:
        regularization = 0
    inv1m_arr = inv1m_arr.astype(float)
    inv2m_arr = inv2m_arr.astype(float)
    inv1p_arr = fix_phase_interval(inv1p_arr)
    inv2p_arr = fix_phase_interval(inv2p_arr)
    inv1_arr = mrt.utils.polar2complex(inv1m_arr, inv1p_arr)
    inv2_arr = mrt.utils.polar2complex(inv2m_arr, inv2p_arr)
    rho_arr = np.real(inv1_arr.conj() * inv2_arr /
                      (inv1m_arr ** 2 + inv2m_arr ** 2 + regularization))
    if values_interval:
        rho_arr = mrt.utils.scale(rho_arr, values_interval, (-0.5, 0.5))
    return rho_arr


# ======================================================================
def fit_mp2rage_rho_to_t1(
        rho_arr,
        eff_arr=None,
        t1_values_range=(100, 5000),
        t1_num=512,
        eff_num=32,
        **acq_params_kws):
    """
    Calculate the T1 map from an MP2RAGE acquisition.

    Args:
        rho_arr (float|np.ndarray): Magnitude of the first inversion image.
        eff_arr (float|np.array|None): Efficiency of the RF pulse excitation.
            This is equivalent to the normalized B1T field.
            Note that this must have the same spatial dimensions as the images
            acquired with MP2RAGE.
            If None, no correction for the RF efficiency is performed.
        t1_values_range (tuple[float]): The T1 value range to consider.
            The format is (min, max) where min < max.
            Values should be positive.
        t1_num (int): The base number of sampling points of T1.
            The actual number of sampling points is usually smaller, because of
            the removal of non-bijective branches.
            This affects the precision of the MP2RAGE estimation.
        eff_num (int): The base number of sampling points for the RF efficiency.
            This affects the precision of the RF efficiency correction.
        **acq_params_kws (dict): The acquisition parameters.
            This should match the signature of: `mp2rage.acq_to_seq_params`.

    Returns:
        t1_arr (float|np.ndarray): The calculated T1 map.
    """
    from pymrt.sequences import mp2rage
    if eff_arr:
        # todo: implement B1T correction
        raise NotImplementedError('B1T correction is not yet implemented')
    else:
        # determine the rho expression
        t1 = np.linspace(t1_values_range[0], t1_values_range[1], t1_num)
        rho = mp2rage.rho(
            t1, **mp2rage.acq_to_seq_params(**acq_params_kws)[0])
        # remove non-bijective branches
        bijective_slice = mrt.utils.bijective_part(rho)
        t1 = t1[bijective_slice]
        rho = rho[bijective_slice]
        if rho[0] > rho[-1]:
            rho = rho[::-1]
            t1 = t1[::-1]
        # check that rho values are strictly increasing
        if not np.all(np.diff(rho) > 0):
            raise ValueError('MP2RAGE look-up table was not properly prepared.')

        # fix values range for rho
        if not mrt.utils.is_in_range(rho_arr, mp2rage.RHO_INTERVAL):
            rho_arr = mrt.utils.scale(rho_arr, mp2rage.RHO_INTERVAL)

        t1_arr = np.interp(rho_arr, rho, t1)
    return t1_arr


# ======================================================================
def fit_mp2rage(
        inv1m_arr,
        inv1p_arr,
        inv2m_arr,
        inv2p_arr,
        regularization=np.spacing(1),
        eff_arr=None,
        t1_values_range=(100, 5000),
        t1_num=512,
        eff_num=32,
        **acq_param_kws):
    """
    Calculate the T1 map from an MP2RAGE acquisition.

    Args:
        inv1m_arr (float|np.ndarray): Magnitude of the first inversion image.
        inv1p_arr (float|np.ndarray): Phase of the first inversion image.
        inv2m_arr (float|np.ndarray): Magnitude of the second inversion image.
        inv2p_arr (float|np.ndarray): Phase of the second inversion image.
        regularization (float|int): Parameter for the regularization.
            This parameter is added to the denominator of the rho expression
            for normalization purposes, therefore should be much smaller than
            the average of the magnitude images.
            Larger values of this parameter will have the side effect of
            denoising the background.
        eff_arr (float|np.array|None): Efficiency of the RF pulse excitation.
            This is equivalent to the normalized B1T field.
            Note that this must have the same spatial dimensions as the images
            acquired with MP2RAGE.
            If None, no correction for the RF efficiency is performed.
        t1_values_range (tuple[float]): The T1 value range to consider.
            The format is (min, max) where min < max.
            Values should be positive.
        t1_num (int): The base number of sampling points of T1.
            The actual number of sampling points is usually smaller, because of
            the removal of non-bijective branches.
            This affects the precision of the MP2RAGE estimation.
        eff_num (int): The base number of sampling points for the RF efficiency.
            This affects the precision of the RF efficiency correction.
        **acq_param_kws (dict): The acquisition parameters.
            This should match the signature of:  `mp2rage.acq_to_seq_params`.

    Returns:
        t1_arr (float|np.ndarray): The calculated T1 map.
    """
    rho_arr = fit_mp2rage_rho(
        inv1m_arr, inv1p_arr, inv2m_arr, inv2p_arr, regularization,
        values_interval=None)
    t1_arr = fit_mp2rage_rho_to_t1(
        rho_arr, eff_arr, t1_values_range, t1_num, eff_num, **acq_param_kws)
    return t1_arr


# ======================================================================
def dual_angle_flash(
        arr1,
        arr2,
        eff_arr,
        fa1,
        fa2,
        tr):
    """
    Calculate the T1 map from two FLASH acquisitions.
    
    Solving for T1 the ratio of two FLASH signals
    (where :math:`s_1` is `arr1` and :math:`s_2` is `arr2`):
    
    .. math::
        \\frac{s_1}{s_2} = \\frac{\\sin(\\alpha_1)}{\\sin(
        \\alpha_2)}\\frac{1 - \\cos(\\alpha_2) e^{-\\frac{T_R}{T_1}}}
        {1 - \\cos(\\alpha_1) e^{-\\frac{T_R}{T_1}}}
    
    We obtain:
    
    .. math::
        e^{-\\frac{T_R}{T_1}} = \\frac{\\sin(\\alpha_1) s_2-\\sin(
        \\alpha_2) s_1}{\\sin(\\alpha_1) \\cos(\\alpha_2) s_2-\\cos(\\alpha_1) 
        \\sin(\\alpha_2) s_1} = X
        
    Or:
    
    .. math::
        T_1 = -\\frac{T_R}{\\log(X)}

    This is a closed-form solution.
    
    Args:
        arr1 (np.ndarray): The first input array in arb.units.
            This is a FLASH image acquired with:
                - the nominal flip angle specified in the `fa1` parameter;
                - the repetition time specified in the `tr` parameter.
        arr2 (np.ndarray): The second input array in arb.units.
            This is a FLASH image acquired with:
                - the nominal flip angle specified in the `fa2` parameter;
                - the repetition time specified in the `tr` parameter.
        eff_arr (np.ndarray): The flip angle efficient in #.
        fa1 (float): Flip angle of the first acquisition in deg.
        fa2 (float): Flip angle of the second acquisition in deg.
        tr (float): Repetition time of the acquisitions in time units.
            Both acquisitions must have the same TR.
            Units of TR determine the units of T1.

    Returns:
        t1_arr (float|np.ndarray): The calculated T1 map.
    """
    from numpy import log, sin, cos
    fa1 = np.deg2rad(fa1)
    fa2 = np.deg2rad(fa2)
    # t1_arr = -tr / log(
    #     (sin(fa1) * arr2 - sin(fa2) * arr1) /
    #     (sin(fa1) * cos(fa2) * arr2 - cos(fa1) * sin(fa2) * arr1))
    with np.errstate(divide='ignore', invalid='ignore'):
        t1_arr = -tr / log(
            (sin(fa1 * eff_arr) * arr2 - sin(fa2 * eff_arr) * arr1) / (
                sin(fa1 * eff_arr) * cos(fa2 * eff_arr) * arr2 - cos(
                    fa1 * eff_arr) * sin(fa2 * eff_arr) * arr1))
    return t1_arr
