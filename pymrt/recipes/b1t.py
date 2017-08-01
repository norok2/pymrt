#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
pymrt.recipes.b1t: relative B1+ (or flip angle efficiency) computation.
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
import pymrt.utils

from pymrt.recipes.generic import fix_noise_mean
# from pymrt import VERB_LVL, D_VERB_LVL, VERB_LVL_NAMES
# from pymrt import elapsed, print_elapsed
# from pymrt import msg, dbg


# ======================================================================
def afi(
        arr1,
        arr2,
        n_tr,
        fa):
    """
    Calculate the flip angle efficiency from Actual Flip Angle (AFI) data.

    The flip angle is obtained by solving:

    .. math::
        \\cos(\\alpha) = \\frac{r n - 1}{n - r}

    where :math:`s_1` is `arr1` acquired with :math:`T_{R,1}`,
    :math:`s_2` is `arr2` acquired with :math:`T_{R,2}`,
    :math:`n = \\frac{T_{R,2}}{T_{R,1}}` and  :math:`r = \\frac{s_2}{s_1}`

    (assuming :math:`T_{R,1}<T_{R,2}`)

    This is a closed-form solution.

    If no phase data is provided, then the maximum measured flip angle must be
    90°, otherwise a flip angle efficiency above 1 cannot be measured.

    Args:
        arr1 (np.ndarray): The first input array in arb.units.
            Contains the signal with :math:`T_{R,1}`.
        arr2 (np.ndarray): The second input array in arb.units.
            Contains the signal with :math:`T_{R,2}`.
        n_tr (int|float): The repetition times ratio in #.
            This is defined as :math:`n = \\frac{T_{R,2}}{T_{R,1}}`.
        fa (int|float): The flip angle in deg.

    Returns:
        eta_fa_arr (np.ndarray): The flip angle efficiency in #.
            This is the :math:`\\eta_\\alpha` factor.

    See Also:
        - Yarnykh, V.L., 2007. Actual flip-angle imaging in the pulsed steady
          state: A method for rapid three-dimensional mapping of the
          transmitted radiofrequency field. Magn. Reson. Med. 57, 192–200.
          doi:10.1002/mrm.21120
    """
    fa = np.deg2rad(fa)
    with np.errstate(divide='ignore', invalid='ignore'):
        eta_fa_arr = arr2 / arr1
        eta_fa_arr = (eta_fa_arr * n_tr - 1) / (n_tr - eta_fa_arr)

    eta_fa_arr = np.real(np.arccos(eta_fa_arr))
    return eta_fa_arr / fa


# ======================================================================
def dual_flash(
        arr1,
        arr2,
        fa1,
        fa2,
        tr1,
        tr2,
        t1_arr=None,
        approx=None,
        sign=1,
        prepare=fix_noise_mean):
    """
    Calculate the flip angle efficiency from two FLASH acquisitions.

    Uses the ratio between two FLASH images provided some conditions are met.

    Most notably, the following cases are covered:
     - `tr1, tr2 >> max(T1)`, `tr1 == tr2` and `fa2 == 2 * fa1`.
     - `tr1, tr2 << min(T1)`, `tr1 == tr2` and `fa2 == 2 * fa1`
       (an estimated average T1 value must be specified).
     - `tr1, tr2 << min(T1)`, `tr1 != tr2` and `fa2 == fa1`
       (an estimated average T1 value must be specified).
     - `tr1 == tr2` and `fa2 == 2 * fa1` (T1 must be known).
     - `tr1 != tr2` and `fa2 == fa1` (T1 must be known).

    This is a closed-form solution.

    Array units must be consistent.
    Time units must be consistent.

    Args:
        arr1 (np.ndarray): The second input array in arb.units.
            This is a FLASH image acquired with the nominal flip angle
            specified in the `flip_angle` parameter.
            `arr1` and `arr2` shapes must match.
        arr2 (np.ndarray): The first input array in arb.units.
            This is a FLASH image acquired with the double the nominal
            flip angle specified in the `flip_angle` parameter.
            `arr1` and `arr2` shapes must match.
        fa1 (int|float): The first nominal flip angle in deg.
        fa2 (int|float): The second nominal flip angle in deg.
        tr1 (int|float): The first repetition time in time units.
        tr2 (int|float): The second repetition time in time units.
        t1_arr (int|float|np.ndarray|None): The T1 value in time units.
            If None, one of the following conditions must be met:
             a. `approx == 'long_tr'`, `tr1 == tr2` and `fa2 == 2 * fa1`.
            If int or float, one of the following conditions must be met:
             a. `approx == 'short_tr'`, `tr1 == tr2` and `fa2 == 2 * fa1`.
             b. `approx == 'short_tr'`, `tr1 != tr2` and `fa1 == fa2`.
            If np.ndarray, one of the following conditions must be met:
             a. `tr1 == tr2` and `fa2 == 2 * fa1`.
             b. `tr1 != tr2` and `fa2 == fa1`.
            If np.ndarray, `t1`, `arr1` and `arr2` shapes must match.
            If not None, units must match those of both `tr1` and `tr2`.
        approx (str|None): Determine the approximation to use.
            Accepted values:
             - `long_tr`: tr1, tr2 >> max(T1) => exp(-tr/t1) = 0
             - `short_tr`: tr1, tr2 << min(T1) => exp(-tr/t1) = 1 - tr/t1
        sign (int): Select one of the two solutions for the equations.
            Must be either +1 or -1.
        prepare (callable|None): Input array preparation.
            Must have the signature: f(np.ndarray) -> np.ndarray.
            Useful for data pre-whitening, including for example the
            correction of magnitude data from Rician mean bias.

    Returns:
        eta_fa_arr (np.ndarray): The flip angle efficiency in #.
            This is the :math:`\\eta_\\alpha` factor.
    """
    if approx:
        approx = approx.lower()

    fa1 = np.deg2rad(fa1)
    fa2 = np.deg2rad(fa2)

    tr = tr1
    fa = fa1
    n_tr = tr2 / tr1  # the repetition times ratio
    m_fa = fa2 / fa1  # the flip angles ratio
    same_tr = np.isclose(tr1, tr2)
    same_fa = np.isclose(fa1, fa2)
    double_fa = np.isclose(2, m_fa)

    # double angle methods
    with np.errstate(divide='ignore', invalid='ignore'):
        if double_fa and same_tr:  # no approximation
            if approx == 'long_tr':
                eta_fa_arr = arr2 / arr1 / 2
            elif approx == 'short_tr' and t1_arr is not None:
                sign = 1  # choose between the two separate solutions
                eta_fa_arr = (arr1 / arr2)
                eta_fa_arr = (
                    (eta_fa_arr + sign * np.sqrt(
                        (eta_fa_arr - 2) ** 2 +
                        6 * (eta_fa_arr - 1) * (tr / t1_arr) +
                        2 * (1 - eta_fa_arr) * (tr / t1_arr) ** 2)) /
                    (2 * (1 - eta_fa_arr) * (1 + tr / t1_arr)))
            elif t1_arr is not None:
                sign = 1  # choose between the two separate solutions
                eta_fa_arr = (arr1 / arr2)
                eta_fa_arr = (
                    (eta_fa_arr + sign * np.sqrt(
                        eta_fa_arr ** 2
                        + 2 * (eta_fa_arr - 1) * np.exp(-tr / t1_arr)
                        + 2 * (eta_fa_arr - 1) * np.exp(-tr / t1_arr) ** 2)) /
                    (2 * (eta_fa_arr - 1) * np.exp(-tr / t1_arr)))

        # same-angle method (variable tr)
        elif same_fa:
            if approx == 'short_tr':
                eta_fa_arr = (arr1 / arr2)
                eta_fa_arr = (
                    (1 - n_tr * eta_fa_arr) /
                    (n_tr * (eta_fa_arr - 1) * (tr / t1_arr) +
                     1 - n_tr * eta_fa_arr))
            elif t1_arr:
                eta_fa_arr = (
                    (arr2 * np.exp(tr2 / t1_arr) -
                     arr1 * np.exp(tr1 / t1_arr) +
                     (arr1 - arr2) * np.exp((tr1 + tr2) / t1_arr)) /
                    (arr2 - arr1 +
                     arr1 * np.exp(tr2 / t1_arr) -
                     arr2 * np.exp(tr1 / t1_arr)))

        else:
            warnings.warn(
                'Unsupported fa1, fa2, tr1, tr2 combination. Fallback to 1.')
            eta_fa_arr = np.ones_like(arr1) * fa

    eta_fa_arr = np.real(np.arccos(eta_fa_arr))
    return eta_fa_arr / fa


# ======================================================================
def sa2rage_rho_to_eta_fa(
        rho_arr,
        t1_arr=None,
        eta_fa_values_range=(0.1, 2),
        eta_fa_num=512,
        t1_values_range=(1200, 2000),
        t1_num=512,
        **acq_params_kws):
    """
    Calculate the T1 map from an MP2RAGE acquisition.

    Args:
        rho_arr (float|np.ndarray): Magnitude of the first inversion image.
        eta_fa_arr (float|np.array|None): Efficiency of the RF pulse excitation.
            This is equivalent to the normalized B1T field.
            Note that this must have the same spatial dimensions as the images
            acquired with MP2RAGE.
            If None, no correction for the RF efficiency is performed.
        t1_values_range (tuple[float]): The T1 value range to consider.
            The format is (min, max) where min < max.
            Values should be positive.
        t1_num (int): The number of sampling points of T1.
            The actual number of sampling points is usually smaller, because of
            the removal of non-bijective branches.
            This affects the precision of the MP2RAGE estimation.
        eff_num (int): The number of sampling points for flip angle efficiency.
            The actual number of sampling points is usually smaller, because of
            the removal of non-bijective branches.
            This affects the precision of the MP2RAGE estimation.
        **acq_params_kws (dict): The acquisition parameters.
            This should match the signature of: `mp2rage.acq_to_seq_params`.

    Returns:
        t1_arr (float|np.ndarray): The calculated T1 map.
    """
    # todo: implement correctly
    from pymrt.sequences import mp2rage
    if t1_arr:
        # todo: implement T1 correction
        raise NotImplementedError('T1 correction is not yet implemented')
    else:
        # determine the rho expression
        b1t = np.linspace(
            eta_fa_values_range[0], eta_fa_values_range[1], eta_fa_num)
        rho = mp2rage.rho(
            t1_arr, **mp2rage.acq_to_seq_params(**acq_params_kws)[0])
        # remove non-bijective branches
        bijective_slice = mrt.utils.bijective_part(rho)
        t1 = t1_arr[bijective_slice]
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
def mu2rage(
        rho_arr,
        fa1,
        fa2,
        tr_gre,
        n_gre,
        t1=2000,
        eta_fa_values_range=(0.01, 2.0),
        eta_fa_num=512):
    """
    Calculate the flip angle efficiency from a MU2RAGE acquisitions.

    MU2RAGE: Magnetization Unprepared 2 RApid Gradient Echo

    Args:
        rho_arr (np.ndarray): The input array.

    """
    # todo: implement correctly
    from pymrt.sequences import mp2rage
    eta_fa = np.linspace(
        eta_fa_values_range[0], eta_fa_values_range[1], eta_fa_num)
    rho = mp2rage.rho(
        t1, n_gre, tr_gre, 0, 0, 0, fa1, fa2, 0, eta_fa)
    # remove non-bijective branches
    bijective_slice = mrt.utils.bijective_part(rho)
    eta_fa = eta_fa[bijective_slice]
    rho = rho[bijective_slice]
    if rho[0] > rho[-1]:
        rho = rho[::-1]
        eta_fa = eta_fa[::-1]
    # check that rho values are strictly increasing
    if not np.all(np.diff(rho) > 0):
        raise ValueError('MP2RAGE look-up table was not properly prepared.')

    # fix values range for rho
    if not mrt.utils.is_in_range(rho_arr, mp2rage.RHO_INTERVAL):
        rho_arr = mrt.utils.scale(rho_arr, mp2rage.RHO_INTERVAL)

    eta_fa_arr = np.interp(rho_arr, rho, eta_fa)
    return eta_fa_arr
