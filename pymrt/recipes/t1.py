#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pymrt.recipes.t1: T1 longitudinal relaxation computation.
"""

# ======================================================================
# :: Future Imports
from __future__ import (
    division, absolute_import, print_function, unicode_literals, )

# ======================================================================
# :: Python Standard Library Imports
# import itertools  # Functions creating iterators for efficient looping
import warnings  # Warning control
# import collections  # Container datatypes

# :: External Imports
import numpy as np  # NumPy (multidimensional numerical arrays library)
import scipy as sp  # SciPy (signal and image processing library)
import flyingcircus as fc  # Everything you always wanted to have in Python*
import flyingcircus_numeric as fcn  # FlyingCircus with NumPy/SciPy

# :: External Imports Submodules
import scipy.interpolate  # Scipy: Interpolation

# :: Local Imports
import pymrt as mrt  # Python Magnetic Resonance Tools: the multi-tool of MRI
import pymrt.correction

from pymrt import INFO, PATH
from pymrt import VERB_LVL, D_VERB_LVL, VERB_LVL_NAMES
from pymrt import elapsed, report, run_doctests
from pymrt import msg, dbg, fmt, fmtm

from pymrt.recipes.generic import (
    fix_phase_interval, rate_to_time, time_to_rate,
    mag_phase_2_combine, cx_2_combine)
import pymrt.recipes.multi_flash
from pymrt.sequences import mp2rage


# ======================================================================
def mp2rage_rho(
        rho_arr,
        eta_fa_arr=1,
        t1_values_range=(100, 5000),
        t1_num=512,
        eta_fa_values_range=(0.001, 2.0),
        eta_fa_num=64,
        mode='pseudo-ratio',
        inverted=False,
        **params_kws):
    """
    Calculate the T1 map from an MP2RAGE acquisition.
    
    This also supports SA2RAGE and NO2RAGE.

    Args:
        rho_arr (float|np.ndarray): MP2RAGE rho signal array in one units.
            Its interpretation depends on the `mode` parameter.
        eta_fa_arr (int|float|np.array): Flip angle efficiency in one units.
            This is equivalent to the normalized B1T field.
            If np.ndarray, it must have the same shape as `rho_arr`.
            If int or float, the flip angle efficiency is assumed to be
            constant over `rho_arr`.
        t1_values_range (tuple[float]): The T1 range in ms.
            The format is (min, max) where min < max.
            Values should be positive.
        t1_num (int): The number of samples for T1.
            The actual number of sampling points is usually smaller, because of
            the removal of non-bijective branches.
            This parameter may affect the precision of the estimation.
        eta_fa_values_range (tuple[float]): The flip angle efficiency range.
            The format is (min, max) where min < max.
            Values should be positive.
        eta_fa_num (int): The number of samples for flip angle efficiency.
            The actual number of sampling points is usually smaller, because of
            the removal of non-bijective branches.
            This parameter may affect the precision of the estimation.
        mode (str): Select the array combination mode.
            See `pymrt.sequences.mp2rage.rho()` for more info.
        inverted (bool): Invert results to convert times to rates.
            Assumes that units of time is ms and units of rates is Hz.
        **params_kws: The acquisition parameters.
            This is filtered through `fc.split_func_kws()` for
            `sequences.mp2rage.acq_to_seq_params()` and the result
            is passed to `sequences.mp2rage.rho()`.
            Its (key, value) pairs must be accepted by either
             `sequences.mp2rage.acq_to_seq_params()` or
             `sequences.mp2rage.rho()`.

    Returns:
        t1_arr (np.ndarray): The T1 map in ms.

    References:
        1) Marques, J.P., Kober, T., Krueger, G., van der Zwaag, W.,
           Van de Moortele, P.-F., Gruetter, R., 2010. MP2RAGE, a self
           bias-field corrected sequence for improved segmentation and
           T1-mapping at high field. NeuroImage 49, 1271–1281.
           doi:10.1016/j.neuroimage.2009.10.002
        2) Metere, R., Kober, T., Möller, H.E., Schäfer, A.,
        2017. Simultaneous
           Quantitative MRI Mapping of T1, T2* and Magnetic Susceptibility
           with
           Multi-Echo MP2RAGE. PLOS ONE 12, e0169265.
           doi:10.1371/journal.pone.0169265
        3) Eggenschwiler, F., Kober, T., Magill, A.W., Gruetter, R.,
           Marques, J.P., 2012. SA2RAGE: A new sequence for fast B1+-mapping.
           Magnetic Resonance Medicine 67, 1609–1619. doi:10.1002/mrm.23145

    See Also:
        sequences.mp2rage
    """
    # determine the sequence parameters
    try:
        acq_kws, kws = fc.split_func_kws(
            mp2rage.acq_to_seq_params, params_kws)
        seq_kws, extra_info = mp2rage.acq_to_seq_params(**acq_kws)
        seq_kws.update(kws)
    except TypeError:
        seq_kws, kws = fc.split_func_kws(mp2rage.rho, params_kws)
        if len(kws) > 0:
            warnings.warn('Unrecognized parameters: {}'.format(kws))

    if isinstance(eta_fa_arr, (int, float)):
        # determine the rho expression
        t1 = np.linspace(t1_values_range[0], t1_values_range[1], t1_num)
        rho = mp2rage.rho(t1=t1, eta_fa=eta_fa_arr, mode=mode, **seq_kws)
        # remove non-bijective branches
        bijective_slice = fcn.bijective_part(rho)
        t1 = t1[bijective_slice]
        rho = rho[bijective_slice]
        if rho[0] > rho[-1]:
            rho = rho[::-1]
            t1 = t1[::-1]
        # check that rho values are strictly increasing
        if not np.all(np.diff(rho) > 0):
            raise ValueError(
                'MP2RAGE look-up table was not properly prepared.')
        t1_arr = np.interp(rho_arr, rho, t1)

    else:
        # determine the rho expression
        t1 = np.linspace(
            t1_values_range[0], t1_values_range[1], t1_num).reshape(-1, 1)
        eta_fa = np.linspace(
            eta_fa_values_range[0], eta_fa_values_range[1],
            eta_fa_num).reshape(1, -1)
        rho = mp2rage.rho(t1=t1, eta_fa=eta_fa, mode=mode, **seq_kws)
        # remove non bijective branches
        for i in range(eta_fa_num):
            bijective_slice = fcn.bijective_part(rho[:, i])
            non_bijective_slice = tuple(fc.complement(
                range(t1_num), bijective_slice))
            rho[non_bijective_slice, i] = -1
        # use griddata for interpolation
        t1_arr = sp.interpolate.griddata(
            (rho.ravel(), (np.zeros_like(rho) + eta_fa).ravel()),
            (np.zeros_like(rho) + t1).ravel(),
            (rho_arr.ravel(), eta_fa_arr.ravel()))
        t1_arr = t1_arr.reshape(rho_arr.shape)

    if inverted:
        t1_arr = time_to_rate(t1_arr, 'ms', 'Hz')
    return t1_arr


# ======================================================================
def double_flash(
        arr1,
        arr2,
        fa1,
        fa2,
        tr1,
        tr2,
        eta_fa_arr=1,
        approx=None,
        inverted=False,
        prepare=mrt.correction.fix_bias_rician):
    """
    Calculate the T1 map from two FLASH acquisitions.

    This method is sensitive to the actual flip angle.

    Solving for T1 using the combination of two FLASH signals
    (where :math:`s_1` is `arr1` and :math:`s_2` is `arr2`).

    If TR is the same:

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

    Note: a similar expression may be derived for different repetition times
    and identical flip angles.

    This is a closed-form solution.

    Array units must be consistent.
    Time units must be consistent.
    
    Args:
        arr1 (np.ndarray): The first input array in arb. units.
            This is a FLASH array acquired with:
                - the nominal flip angle specified in the `fa1` parameter;
                - the repetition time specified in the `tr` parameter.
        arr2 (np.ndarray): The second input array in arb. units.
            This is a FLASH array acquired with:
                - the nominal flip angle specified in the `fa2` parameter;
                - the repetition time specified in the `tr` parameter.
        fa1 (int|float): The first nominal flip angle in deg.
        fa2 (int|float): The second nominal flip angle in deg.
        tr1 (int|float): The first repetition time in time units.
        tr2 (int|float): The second repetition time in time units.
        eta_fa_arr (int|float|np.ndarray): The flip angle efficiency in one
        units.
            If int or float, it is assumed constant throught the inputs.
            If np.ndarray, its shape must match the shape of both `arr1` and
            `arr2`.
        approx (str|None): Determine the approximation to use.
            Accepted values:
             - `short_tr`: assumes tr1, tr2 << min(T1)
        inverted (bool): Invert results to convert times to rates.
            Assumes that units of time is ms and units of rates is Hz.
        prepare (callable|None): Input array preparation.
            Must have the signature: f(np.ndarray) -> np.ndarray.
            Useful for data pre-whitening, including for example the
            correction of magnitude data from Rician mean bias.

    Returns:
        t1_arr (np.ndarray): The calculated T1 map in time units.
    """
    arr1 = prepare(arr1) if prepare else arr1.astype(float)
    arr2 = prepare(arr2) if prepare else arr2.astype(float)

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
        warnings.warn('This method is sensitive to B1+.')
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
                t1_arr = tr / np.log(
                    (np.sin(fa1) * np.cos(fa2) * arr2 -
                     np.cos(fa1) * np.sin(fa2) * arr1) /
                    (np.sin(fa1) * arr2 -
                     np.sin(fa2) * arr1))

    else:
        if approx == 'short_tr' and same_fa:
            with np.errstate(divide='ignore', invalid='ignore'):
                t1_arr = tr * n_tr * (
                        np.cos(fa) * (arr2 - arr1) /
                        ((np.cos(fa) - 1) * arr2 +
                         (1 - np.cos(fa)) * n_tr * arr1))
        else:
            warnings.warn(
                'Unsupported parameter combination. '
                'Fallback to `t1_arr=1`.')
            t1_arr = np.ones_like(arr1)

    if inverted:
        t1_arr = time_to_rate(t1_arr, 'ms', 'Hz')
    return t1_arr


# ======================================================================
def multi_flash(
        arrs,
        fas,
        trs,
        eta_fa_arr=None,
        method='vfa',
        inverted=False,
        prepare=mrt.correction.fix_bias_rician):
    """
    Calculate the T1 map using multiple FLASH acquisitions.

    Assumes that the flip angles are small (must be below 45°).
    If the different TR are used for the acquisitions, assumes all TRs << T1.

    Args:
        arrs (Iterable[np.ndarray]): The input signal arrays in arb. units.
        fas (Iterable[int|float]): The flip angles in deg.
        trs (Iterable[int|float]): The repetition times in time units.
        eta_fa_arr (np.ndarray|None): The flip angle efficiency in one units.
            If None, a significant bias may still be present.
        method (str): Determine the fitting method to use.
            Accepted values are:
             - 'auto': determine an optimal method by inspecting the data.
             - 'vfa': use the variable fli-angle method (closed form solution),
               very fast and accurate but very sensitive to flip angle
               efficiency (or B1+).
             - 'leasq': use non-linear least square fit, slow but accurate,
               and requires many acquisitions to be accurate.
        inverted (bool): Invert results to convert times to rates.
            Assumes that units of time is ms and units of rates is Hz.
        prepare (callable|None): Input array preparation.
            Must have the signature: f(np.ndarray) -> np.ndarray.
            Useful for data pre-whitening, including for example the
            correction of magnitude data from Rician mean bias.

    Returns:
        t1_arr (np.ndarray): The calculated T1 map in time units.

    See Also:
        recipes.multi_flash.vfa()

    References:
        - Helms, G., Dathe, H., Weiskopf, N., Dechent, P., 2011.
          Identification of signal bias in the variable flip angle method by
          linear display of the algebraic ernst equation. Magn. Reson. Med.
          66, 669–677. doi:10.1002/mrm.22849
    """
    methods = ('vfa', 'leasq')

    if method == 'vfa':
        t1_arr, xi_arr = mrt.recipes.multi_flash.vfa(
            arrs, fas, trs, eta_fa_arr=eta_fa_arr, prepare=prepare)
    elif method == 'leasq':
        t1_arr, eta_fa_arr, xi_arr = mrt.recipes.multi_flash.fit_leasq(
            arrs, fas, trs, prepare=prepare)
    else:
        raise ValueError(
            'valid methods are: {} (given: {})'.format(methods, method))
    if inverted:
        t1_arr = time_to_rate(t1_arr, 'ms', 'Hz')
    return t1_arr


# ======================================================================
elapsed(__file__[len(PATH['base']) + 1:])

# ======================================================================
if __name__ == '__main__':
    run_doctests(__doc__)
