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
import warnings  # Warning control
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
from pymrt.recipes.generic import fix_noise_mean, fix_phase_interval


# ======================================================================
def mp2rage_cx_to_rho(
        inv1_arr,
        inv2_arr,
        regularization=np.spacing(1),
        values_interval=None):
    """
    Calculate the rho signal from an MP2RAGE acquisition.

    This is also referred to as the uniform arrays, because it should be free
    from low-spatial frequency biases.

    Args:
        inv1_arr (float|np.ndarray): Complex array of the first inversion.
        inv2_arr (float|np.ndarray): Complex array of the second inversion.
        regularization (float|int): Parameter for the regularization.
            This parameter is added to the denominator of the rho expression
            for normalization purposes, therefore should be much smaller than
            the average of the magnitude arrays.
            Larger values of this parameter will have the side effect of
            denoising the background.
        values_interval (tuple[float|int]|None): The output values interval.
            The standard values are linearly converted to this range.
            If None, the natural [-0.5, 0.5] interval will be used.

    Returns:
        rho_arr (float|np.ndarray): The calculated rho (uniform) array.
    """
    rho_arr = np.real(
        inv1_arr.conj() * inv2_arr /
        (np.abs(inv1_arr) + np.abs(inv2_arr) + regularization))
    if values_interval:
        rho_arr = mrt.utils.scale(rho_arr, values_interval, (-0.5, 0.5))
    return rho_arr


# ======================================================================
def mp2rage_mag_phs_to_rho(
        inv1m_arr,
        inv1p_arr,
        inv2m_arr,
        inv2p_arr,
        regularization=np.spacing(1),
        values_interval=None):
    """
    Calculate the rho signal from an MP2RAGE acquisition.
    
    This is also referred to as the uniform arrays, because it should be free
    from low-spatial frequency biases.

    Args:
        inv1m_arr (float|np.ndarray): Magnitude of the first inversion.
        inv1p_arr (float|np.ndarray): Phase of the first inversion.
        inv2m_arr (float|np.ndarray): Magnitude of the second inversion.
        inv2p_arr (float|np.ndarray): Phase of the second inversion.
        regularization (float|int): Parameter for the regularization.
            This parameter is added to the denominator of the rho expression
            for normalization purposes, therefore should be much smaller than
            the average of the magnitude arrays.
            Larger values of this parameter will have the side effect of
            denoising the background.
        values_interval (tuple[float|int]|None): The output values interval.
            The standard values are linearly converted to this range.
            If None, the natural [-0.5, 0.5] interval will be used.

    Returns:
        rho_arr (float|np.ndarray): The calculated rho (uniform) array.
    """
    inv1m_arr = inv1m_arr.astype(float)
    inv2m_arr = inv2m_arr.astype(float)
    inv1p_arr = fix_phase_interval(inv1p_arr)
    inv2p_arr = fix_phase_interval(inv2p_arr)
    inv1_arr = mrt.utils.polar2complex(inv1m_arr, inv1p_arr)
    inv2_arr = mrt.utils.polar2complex(inv2m_arr, inv2p_arr)
    rho_arr = mp2rage_cx_to_rho(
        inv1_arr, inv2_arr, regularization, values_interval)
    return rho_arr


# ======================================================================
def mp2rage_rho_to_t1(
        rho_arr,
        eta_fa_arr=None,
        t1_values_range=(100, 5000),
        t1_num=512,
        eta_fa_values_range=(0.1, 2),
        eta_fa_num=512,
        **params_kws):
    """
    Calculate the T1 map from an MP2RAGE acquisition.

    Args:
        rho_arr (float|np.ndarray): MP2RAGE signal (uniform) array.
        eta_fa_arr (float|np.array|None): Flip angle efficiency.
            This is equivalent to the normalized B1T field.
            Note that this must have the same spatial dimensions as the arrays
            acquired with MP2RAGE.
            If None, no correction for the flip angle efficiency is performed.
        t1_values_range (tuple[float]): The T1 range.
            The format is (min, max) where min < max.
            Values should be positive.
        t1_num (int): The number of samples for T1.
            The actual number of sampling points is usually smaller, because of
            the removal of non-bijective branches.
            This affects the precision of the estimation.
        eta_fa_values_range (tuple[float]): The flip angle efficiency range.
            The format is (min, max) where min < max.
            Values should be positive.
        eta_fa_num (int): The number of samples for flip angle efficiency.
            The actual number of sampling points is usually smaller, because of
            the removal of non-bijective branches.
            This affects the precision of the estimation.
        **params_kws (dict): The acquisition parameters.
            This should match the signature of: `mp2rage.acq_to_seq_params`.

    Returns:
        t1_arr (float|np.ndarray): The calculated T1 map.
    """
    from pymrt.sequences import mp2rage


    if eta_fa_arr:
        # todo: implement B1T correction
        raise NotImplementedError('B1T correction is not yet implemented')
    else:
        # determine the rho expression
        t1 = np.linspace(t1_values_range[0], t1_values_range[1], t1_num)
        for k, v in params_kws.items():
            pass
        # todo: split acq_params and seq_params
        seq_pars = mp2rage.acq_to_seq_params(**params_kws)[0]
        rho = mp2rage.rho(t1=t1, **seq_pars)
        # remove non-bijective branches
        bijective_slice = mrt.utils.bijective_part(rho)
        t1 = t1[bijective_slice]
        rho = rho[bijective_slice]
        if rho[0] > rho[-1]:
            rho = rho[::-1]
            t1 = t1[::-1]
        # check that rho values are strictly increasing
        if not np.all(np.diff(rho) > 0):
            raise ValueError(
                'MP2RAGE look-up table was not properly prepared.')

        # fix values range for rho
        if not mrt.utils.is_in_range(rho_arr, mp2rage.RHO_INTERVAL):
            rho_arr = mrt.utils.scale(rho_arr, mp2rage.RHO_INTERVAL)

        t1_arr = np.interp(rho_arr, rho, t1)
    return t1_arr


# ======================================================================
def mp2rage_t1(
        inv1m_arr,
        inv1p_arr,
        inv2m_arr,
        inv2p_arr,
        regularization=np.spacing(1),
        eta_fa_arr=None,
        t1_values_range=(100, 5000),
        t1_num=512,
        eta_fa_values_range=(0.1, 2),
        eta_fa_num=512,
        **acq_param_kws):
    """
    Calculate the T1 map from an MP2RAGE acquisition.

    Args:
        inv1m_arr (float|np.ndarray): Magnitude of the first inversion.
        inv1p_arr (float|np.ndarray): Phase of the first inversion.
        inv2m_arr (float|np.ndarray): Magnitude of the second inversion.
        inv2p_arr (float|np.ndarray): Phase of the second inversion.
        regularization (float|int): Parameter for the regularization.
            This parameter is added to the denominator of the rho expression
            for normalization purposes, therefore should be much smaller than
            the average of the magnitude arrays.
            Larger values of this parameter will have the side effect of
            denoising the background.
        eta_fa_arr (float|np.array|None): Flip angle efficiency.
            This is equivalent to the normalized B1T field.
            Note that this must have the same spatial dimensions as the arrays
            acquired with MP2RAGE.
            If None, no correction for the flip angle efficiency is performed.
        t1_values_range (tuple[float]): The T1 range.
            The format is (min, max) where min < max.
            Values should be positive.
        t1_num (int): The number of samples for T1.
            The actual number of sampling points is usually smaller, because of
            the removal of non-bijective branches.
            This affects the precision of the estimation.
        eta_fa_values_range (tuple[float]): The flip angle efficiency range.
            The format is (min, max) where min < max.
            Values should be positive.
        eta_fa_num (int): The number of samples for flip angle efficiency.
            The actual number of sampling points is usually smaller, because of
            the removal of non-bijective branches.
            This affects the precision of the estimation.
        **acq_param_kws (dict): The acquisition parameters.
            This should match the signature of:  `mp2rage.acq_to_seq_params`.

    Returns:
        t1_arr (float|np.ndarray): The calculated T1 map.
    """
    rho_arr = mp2rage_mag_phs_to_rho(
        inv1m_arr, inv1p_arr, inv2m_arr, inv2p_arr, regularization,
        values_interval=None)
    t1_arr = mp2rage_rho_to_t1(
        rho_arr, eta_fa_arr,
        t1_values_range, t1_num,
        eta_fa_values_range, eta_fa_num, **acq_param_kws)
    return t1_arr


# ======================================================================
def dual_flash(
        arr1,
        arr2,
        fa1,
        fa2,
        tr1,
        tr2,
        eta_fa_arr=None,
        approx=None,
        prepare=fix_noise_mean):
    """
    Calculate the T1 map from two FLASH acquisitions.
    
    Solving for T1 the combination of two FLASH signals
    (where :math:`s_1` is `arr1` and :math:`s_2` is `arr2`).

    .. math::
        \\frac{s_1}{s_2} = \\frac{\\sin(\\alpha_1)}{\\sin(
        \\alpha_2)}\\frac{1 - \\cos(\\alpha_2) e^{-\\frac{T_R}{T_1}}}
        {1 - \\cos(\\alpha_1) e^{-\\frac{T_R}{T_1}}}
    
    which becomes:
    
    .. math::
        e^{-\\frac{T_R}{T_1}} = \\frac{\\sin(\\alpha_1) s_2-\\sin(
        \\alpha_2) s_1}{\\sin(\\alpha_1) \\cos(\\alpha_2) s_2-\\cos(\\alpha_1) 
        \\sin(\\alpha_2) s_1} = X
        
    Or:
    
    .. math::
        T_1 = -\\frac{T_R}{\\log(X)}

    This is a closed-form solution.

    Array units must be consistent.
    Time units must be consistent.
    
    Args:
        arr1 (np.ndarray): The first input array in arb.units.
            This is a FLASH array acquired with:
                - the nominal flip angle specified in the `fa1` parameter;
                - the repetition time specified in the `tr` parameter.
        arr2 (np.ndarray): The second input array in arb.units.
            This is a FLASH array acquired with:
                - the nominal flip angle specified in the `fa2` parameter;
                - the repetition time specified in the `tr` parameter.
        fa1 (int|float): The first nominal flip angle in deg.
        fa2 (int|float): The second nominal flip angle in deg.
        tr1 (int|float): The first repetition time in time units.
        tr2 (int|float): The second repetition time in time units.
        eta_fa_arr (np.ndarray|None): The flip angle efficiency in #.
            If None, a significant bias may still be present.
        approx (str|None): Determine the approximation to use.
            Accepted values:
             - `short_tr`: assumes tr1, tr2 << min(T1)
        prepare (callable|None): Input array preparation.
            Must have the signature: f(np.ndarray) -> np.ndarray

    Returns:
        t1_arr (float|np.ndarray): The calculated T1 map.
    """
    if eta_fa_arr is None:
        eta_fa_arr = 1
    if approx:
        approx = approx.lower()

    fa1 = np.deg2rad(fa1)
    fa2 = np.deg2rad(fa2)

    tr = tr1
    fa = fa1
    n_tr = tr2 / tr1  # the repetition times ratio
    # m_fa = fa2 / fa1  # the flip angles ratio
    same_tr = np.isclose(tr1, tr2)
    same_fa = np.isclose(fa1, fa2)

    if eta_fa_arr is None:
        eta_fa_arr = 1
    fa1 *= eta_fa_arr
    fa2 *= eta_fa_arr

    if same_tr:
        if approx == 'short_tr':
            with np.errstate(divide='ignore', invalid='ignore'):
                t1_arr = tr * n_tr * (
                    (np.sin(fa1) * np.cos(fa2) * arr2 -
                     np.cos(fa1) * np.sin(fa2) * arr1) /
                    ((np.sin(fa1) * np.cos(fa2) - np.sin(fa1)) * arr2 +
                     (1 - np.cos(fa1)) * np.sin(fa2) * n_tr * arr1))
        else:
            with np.errstate(divide='ignore', invalid='ignore'):
                t1_arr = -tr / np.log(
                    (np.sin(fa1) * arr2 -
                     np.sin(fa2) * arr1) /
                    (np.sin(fa1) * np.cos(fa2) * arr2 -
                     np.cos(fa1) * np.sin(fa2) * arr1))

    else:
        if approx == 'short_tr' and same_fa:
            with np.errstate(divide='ignore', invalid='ignore'):
                t1_arr = tr * n_tr * (
                    np.cos(fa) * (arr2 - arr1) /
                    ((np.cos(fa) - 1) * arr2 +
                     (1 - np.cos(fa)) * n_tr * arr1))
        else:
            warnings.warn('Unsupported fa1, fa2, tr1, tr2 combination')
            t1_arr = np.ones_like(arr1)
    return t1_arr


# ======================================================================
def multi_flash(
        arrs,
        fas,
        trs,
        eta_fa_arr=None,
        prepare=fix_noise_mean):
    """


    Args:
        arrs (iterable[np.ndarray]): The input signal arrays in arb.units
        fas (iterable[int|float]): The flip angles in deg.
        trs (iterable[int|float]): The repetition times in time units.
        eta_fa_arr (np.ndarray|None): The flip angle efficiency in #.
            If None, a significant bias may still be present.
        prepare (callable|None): Input array preparation.
            Must have the signature: f(np.ndarray) -> np.ndarray

    Returns:

    References:
        - Helms, G., Dathe, H., Weiskopf, N., Dechent, P., 2011.
          Identification of signal bias in the variable flip angle method by
          linear display of the algebraic ernst equation. Magn. Reson. Med.
          66, 669–677. doi:10.1002/mrm.22849
    """
    assert (len(arrs) == len(fas) == len(trs))
    fas = [np.deg2rad(fa) for fa in fas]
    arrs = [prepare(arr) if prepare else arr.astype(float) for arr in arrs]

    if eta_fa_arr is None:
        eta_fa_arr = np.ones(arrs[0].shape, dtype=float)

    index = -1
    num = len(arrs)

    s_arr = np.stack(arrs, index).astype(float)
    tau_arr = np.stack([2 * np.tan(fa * eta_fa_arr / 2) for fa in fas], index)

    tr = np.mean(trs)
    same_tr = all([np.isclose(x, tr) for x in trs])

    with np.errstate(divide='ignore', invalid='ignore'):
        if same_tr:
            t1_arr = \
                (np.sum(s_arr * tau_arr, index) ** 2 -
                 num * np.sum(s_arr ** 2 * tau_arr ** 2, index)) / \
                ((num * np.sum(s_arr ** 2, index) -
                  np.sum(s_arr * tau_arr, index) *
                  np.sum(s_arr / tau_arr, index)) * 2)
            t1_arr = tr / np.log((2 + t1_arr) / (2 - t1_arr))
        else:
            warnings.warn('Using approximation: TR << T1')
            tr_arr = np.stack(
                [x * np.ones(arrs[0].shape, dtype=float) for x in trs],
                index)
            t1_arr = \
                ((num * np.sum(s_arr ** 2 / tr_arr, index) -
                  np.sum(s_arr * tau_arr / tr_arr, index) *
                  np.sum(s_arr / tau_arr, index)) * 2) / \
                (np.sum(s_arr * tau_arr / tr_arr, index) ** 2 -
                 num * np.sum(s_arr ** 2 * tau_arr ** 2 / tr_arr ** 2, index))
        xi_arr = \
            (np.sum(s_arr ** 2 * tau_arr ** 2, index) *
             np.sum(s_arr / tau_arr, index) -
             np.sum(s_arr * tau_arr, index) *
             np.sum(s_arr ** 2, index)) / \
            (num * np.sum(s_arr ** 2 * tau_arr ** 2, index) -
             np.sum(s_arr * tau_arr, index) ** 2)

    return t1_arr, xi_arr
