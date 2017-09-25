#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pymrt.recipes.multi_flash: multiple simultaneous computation from FLASH signal.
"""

# ======================================================================
# :: Future Imports
from __future__ import (
    division, absolute_import, print_function, unicode_literals)

# ======================================================================
# :: Python Standard Library Imports
# import os  # Miscellaneous operating system interfaces
# import itertools  # Functions creating iterators for efficient looping
import warnings  # Warning control
import collections  # Container datatypes
# import pickle  # Python object serialization

# :: External Imports
import numpy as np  # NumPy (multidimensional numerical arrays library)
# import sympy as sym  # SymPy (symbolic CAS library)


# :: Local Imports
import pymrt as mrt
import pymrt.utils
import pymrt.segmentation

from pymrt.recipes.generic import fix_magnitude_bias, voxel_curve_fit
from pymrt.recipes import t1, b1t


# from pymrt import VERB_LVL, D_VERB_LVL, VERB_LVL_NAMES
# from pymrt import elapsed, report
# from pymrt import msg, dbg


# ======================================================================
def _flash_signal(
        fa,
        tr,
        t1_,
        xi,
        eta_fa=1.0):
    """
    FLASH signal (no TE dependency) to use with `scipi.optimize.curve_fit()`.

    Args:
        fa (int|float|np.ndarray): The flip angle in rad.
        tr (int|float|np.ndarray): The repetition time in time units.
            Units must be the same as `t1`.
        t1_ (int|float|np.ndarray): The longitudinal relaxation in time units.
            The units match those of `tr`.
        xi (int|float|np.ndarray): The amplitude factor in arb.units.
            Contains information on the spin density `m0`, the coil
            sensitivity (proportional to `b1r`) and units transformation
            factors.
        eta_fa (int|float|np.ndarray): The flip angle efficiency in #.

    Returns:
        s_arr (np.ndarray): The signal array in arb.units.
            The shape of the array is (N), where N is the number of signals
            to fit.
    """
    return xi * np.sin(fa * eta_fa) * (1.0 - np.exp(-tr / t1_)) / \
           (1.0 - np.cos(fa * eta_fa) * np.exp(-tr / t1_))


# ======================================================================
def _flash_signal_fit(
        fa_tr,
        t1_,
        xi,
        eta_fa):
    """
    FLASH signal (no TE dependency) to use with `scipi.optimize.curve_fit()`.

    Args:
        fa_tr (np.ndarray): The independent variables.
            The shape of the array is (2, N), where N is the number of signals
            to fit. Upon assignment, must expand to: `fa, tr = fa_tr` and
            `fa` are the flip angles in radians, and `tr` is the repetition
            times in time units (matching those of `t1`).
        t1_ (float): The longitudinal relaxation in time units.
            The units match those of `tr`.
        xi (float): The amplitude factor in arb.units.
            Contains information on the spin density `m0`, the coil
            sensitivity (proportional to `b1r`) and units transformation
            factors.
        eta_fa (float): The flip angle efficiency in #.

    Returns:
        s_arr (np.ndarray): The signal array in arb.units.
            The shape of the array is (N), where N is the number of signals
            to fit.
    """
    fa, tr = fa_tr
    return xi * np.sin(fa * eta_fa) * (1.0 - np.exp(-tr / t1_)) / \
           (1.0 - np.cos(fa * eta_fa) * np.exp(-tr / t1_))


# ======================================================================
def triple_special1(
        arr1,
        arr2,
        arr3,
        fa,
        tr,
        n_tr=2,
        sign=1,
        prepare=fix_magnitude_bias):
    """
    Calculate the parameters of the FLASH signal at fixed echo time.

    This method is not stable and should not be used.

    Obtains T1, the flip angle efficiency (proportional to the coil transmit
    field) and the apparent spin density (modulated by coil sensitivity).

    The (fa, tr) combinations required are:
     - arr1: (    fa,        tr)
     - arr2: (2 * fa,        tr)
     - arr3: (    fa, n_tr * tr)

    Assumes that the `tr` (and `n_tr * tr`) is much smaller compared to `t1`.

    Given the following expression for the FLASH signal:

    .. math::
        s = \\eta_{m_0} m_0 \\sin(\\eta_\\alpha \\alpha)
        e^{-\\frac{T_E}{T_2^*}}
        \\frac{1 - e^{-\\frac{T_R}{T_1}}}
        {1 - \\cos(\\eta_\\alpha \\alpha) e^{-\\frac{T_R}{T_1}}}
        
    where
    :math:`m_0` is the spin density,
    :math:`\\eta_{m_0}` is an the receive efficiency
    (proportional to the coil receive field :math:`B_1^-`),
    :math:`\\alpha` is the flip angle of the RF excitation,
    :math:`\\eta_\\alpha` is the flip angle efficiency
    (proportional to the coil transmit field :math:`B_1^+`),
    :math:`T_E` is the echo time,
    :math:`T_2^*` is the reduced transverse relaxation time,
    :math:`T_R` is the repetition time, and
    :math:`T_1` is the longitudinal relaxation time.

    The obtained FLASH signal parameters are :math:`T_1`,
    :math:`\\eta_\\alpha` and
    :math:`\\xi = \\eta_{m_0} m_0 e^{-\\frac{T_E}{T_2^*}`.

    This is a closed-form solution.

    Args:
        arr1 (np.ndarray): The input array for the first FLASH signal.
            The flip angle must be `fa` and the repetition time `tr`.
        arr2 (np.ndarray): The input array for the second FLASH signal.
            The flip angle must be `2 * fa` and the repetition time `tr`.
        arr3 (np.ndarray): The input array for the third FLASH signal.
            The flip angle must be `fa` and the repetition time `n * tr`.
        fa (int|float): The base flip angle in deg.
            This is the flip angle for `arr1` and `arr3`, while the flip
            angle for `arr2` must be `2 * fa`.
        tr (int|float): The base repetition time in time units.
            This is the repetition time for `arr1` and `arr2`, while the
            repetition time for `arr3` must be `n * tr`.
        n_tr (int|float): The repetition times ratio in #.
            This is the repetition times ratio, obtained dividing the
            repetition time of `arr3` by the repetition time of `arr1`.
        sign (int|float): Select one of the two solution for the equations.
            Must be either +1 or -1.
        prepare (callable|None): Input array preparation.
            Must have the signature: f(np.ndarray) -> np.ndarray.
            Useful for data pre-whitening, including for example the
            correction of magnitude data from Rician mean bias.

    Returns:
        result (tuple[np.ndarray]): The tuple
            contains:
             - t1_arr (np.ndarray): The longitudinal relaxation in time units.
               The units of `t1_arr` are defined by the units of `tr`.
             - xi_arr (np.ndarray): The signal factor in arb. units.
               This is :math:`\\eta_{m_0} m_0 e^{-\\frac{T_E}{T_2^*}`.
             - eta_fa_arr (np.ndarray): The flip angle efficiency in #.
               This is proportional to the coil transmit field :math:`B_1^+`.
    """
    fa = np.deg2rad(fa)
    arr1 = prepare(arr1) if prepare else arr1.astype(float)
    arr2 = prepare(arr2) if prepare else arr2.astype(float)
    arr3 = prepare(arr3) if prepare else arr3.astype(float)
    with np.errstate(divide='ignore', invalid='ignore'):
        eta_fa_arr = (n_tr * arr2 * arr3 - n_tr * arr1 * arr2 - sign *
                      np.sqrt(
                          arr2 * (
                              n_tr ** 2 * arr1 ** 2 * arr2 -
                              2.0 * n_tr ** 2 * arr1 ** 2 * arr3 +
                              n_tr ** 2 * arr2 * arr3 ** 2 +
                              2.0 * n_tr * arr1 ** 2 * arr3 -
                              2.0 * n_tr * arr1 * arr2 * arr3 +
                              2.0 * n_tr * arr1 * arr3 ** 2 -
                              2.0 * n_tr * arr2 * arr3 ** 2 -
                              2.0 * arr1 * arr3 ** 2 +
                              2.0 * arr2 * arr3 ** 2))) / (
                         2 * arr3 * (
                             n_tr * arr1 - n_tr * arr2 - arr1 + arr2))
        eta_fa_arr = np.real(np.arccos(eta_fa_arr))

        t1_arr = n_tr * tr * (arr1 - arr3) * (
            n_tr * arr2 * (arr1 - arr3) *
            (n_tr * arr1 - n_tr * arr2 - arr1 + arr2) +
            arr2 * (n_tr - 1.0) * (arr1 - arr2) * (n_tr * arr1 - arr3) +
            sign * (n_tr - 1.0) * (arr1 - arr2) * np.sqrt(
                arr2 * (
                    n_tr ** 2 * arr1 ** 2 * arr2 -
                    2.0 * n_tr ** 2 * arr1 ** 2 * arr3 +
                    n_tr ** 2 * arr2 * arr3 ** 2 +
                    2.0 * n_tr * arr1 ** 2 * arr3 -
                    2.0 * n_tr * arr1 * arr2 * arr3 +
                    2.0 * n_tr * arr1 * arr3 ** 2 -
                    2.0 * n_tr * arr2 * arr3 ** 2 -
                    2.0 * arr1 * arr3 ** 2 +
                    2.0 * arr2 * arr3 ** 2))) / (
                     (n_tr - 1.0) * (arr1 - arr2) * (n_tr * arr1 - arr3) *
                     (3.0 * n_tr * arr1 * arr2 +
                      2.0 * n_tr * arr1 * arr3 -
                      4.0 * n_tr * arr2 * arr3 -
                      2.0 * arr1 * arr3 + arr2 * arr3))

        xi_arr = arr3 / (
            np.sin(fa * eta_fa_arr) * (1.0 - np.exp(-n_tr * tr / t1_arr)) /
            (1.0 - np.cos(fa * eta_fa_arr) * np.exp(-n_tr * tr / t1_arr)))
    return t1_arr, xi_arr, eta_fa_arr


# ======================================================================
def triple_special2(
        arr1,
        arr2,
        arr3,
        fa,
        tr,
        n_tr=2,
        sign=1,
        max_iter=64,
        threshold=1e-8,
        prepare=fix_magnitude_bias):
    """
    Calculate the parameters of the FLASH signal at fixed echo time.

    This method is not stable and should not be used.

    Obtains T1, the flip angle efficiency (proportional to the coil transmit
    field) and the apparent spin density (modulated by coil sensitivity).

    The (fa, tr) combinations required are:
     - arr1: (    fa,        tr)
     - arr2: (2 * fa,        tr)
     - arr3: (2 * fa, n_tr * tr)

    Assumes that the `tr` (and `n_tr * tr`) is much smaller compared to `t1`.

    Given the following expression for the FLASH signal:

    .. math::
        s = \\eta_{m_0} m_0 \\sin(\\eta_\\alpha \\alpha)
        e^{-\\frac{T_E}{T_2^*}}
        \\frac{1 - e^{-\\frac{T_R}{T_1}}}
        {1 - \\cos(\\eta_\\alpha \\alpha) e^{-\\frac{T_R}{T_1}}}

    where
    :math:`m_0` is the spin density,
    :math:`\\eta_{m_0}` is an the receive efficiency
    (proportional to the coil receive field :math:`B_1^-`),
    :math:`\\alpha` is the flip angle of the RF excitation,
    :math:`\\eta_\\alpha` is the flip angle efficiency
    (proportional to the coil transmit field :math:`B_1^+`),
    :math:`T_E` is the echo time,
    :math:`T_2^*` is the reduced transverse relaxation time,
    :math:`T_R` is the repetition time, and
    :math:`T_1` is the longitudinal relaxation time.

    The obtained FLASH signal parameters are :math:`T_1`,
    :math:`\\eta_\\alpha` and
    :math:`\\xi = \\eta_{m_0} m_0 e^{-\\frac{T_E}{T_2^*}`.

    Assumes that the `tr` (and `n * tr`) is much smaller compared to `t1`.

    This is an iterative solution.

    Args:
        arr1 (np.ndarray): The input array for the first FLASH signal.
            The flip angle must be `fa` and the repetition time `tr`.
        arr2 (np.ndarray): The input array for the second FLASH signal.
            The flip angle must be `2 * fa` and the repetition time `tr`.
        arr3 (np.ndarray): The input array for the third FLASH signal.
            The flip angle must be `2 * fa` and the repetition time `n * tr`.
        fa (int|float): The base flip angle in deg.
            This is the flip angle for `arr1`, while the flip
            angle for `arr2` and `arr3` must be `2 * fa`.
        tr (int|float): The base repetition time in time units.
            This is the repetition time for `arr1` and `arr2`, while the
            repetition time for `arr3` must be `n * tr`.
        n_tr (int|float): The repetition times ratio in #.
            This is the repetition times ratio, obtained dividing the
            repetition time of `arr3` by the repetition time of `arr1`.
        sign (int|float): Select one of the two solution for the equations.
            Must be either +1 or -1.
        max_iter (int): Maximum number of iterations.
            If `threshold` > 0, the algorithm may stop earlier.
        threshold (float): Threshold for next iteration.
            If the next iteration globally modifies the sensitivity by less
            than `threshold`, the algorithm stops.
        prepare (callable|None): Input array preparation.
            Must have the signature: f(np.ndarray) -> np.ndarray.
            Useful for data pre-whitening, including for example the
            correction of magnitude data from Rician mean bias.

    Returns:
        result (tuple[np.ndarray]): The tuple
            contains:
             - t1_arr (np.ndarray): The longitudinal relaxation in time units.
               The units of `t1_arr` are defined by the units of `tr`.
             - xi_arr (np.ndarray): The signal factor in arb. units.
               This is :math:`\\eta_{m_0} m_0 e^{-\\frac{T_E}{T_2^*}`.
             - eta_fa_arr (np.ndarray): The flip angle efficiency in #.
               This is proportional to the coil transmit field :math:`B_1^+`.
    """
    fa = np.deg2rad(fa)
    arr1 = prepare(arr1) if prepare else arr1.astype(float)
    arr2 = prepare(arr2) if prepare else arr2.astype(float)
    arr3 = prepare(arr3) if prepare else arr3.astype(float)
    fa1, fa2, fa3 = fa, 2 * fa, 2 * fa
    tr1, tr2, tr3 = tr, tr, n_tr * tr

    with np.errstate(divide='ignore', invalid='ignore'):
        t1_arr = np.full_like(arr1, tr)
        eta_fa_arr = np.ones_like(arr1)
        mask = np.ones_like(arr1, dtype=bool)
        mask *= arr1 >= mrt.segmentation.threshold_otsu(arr1)
        mask *= arr2 >= mrt.segmentation.threshold_otsu(arr2)
        mask *= arr3 >= mrt.segmentation.threshold_otsu(arr3)
        print('Mask: {:.0%}'.format(np.sum(mask) / mask.size))
        for i in range(max_iter):
            t1_last = t1_arr.copy() if threshold > 0 else 0
            eta_fa_last = eta_fa_arr.copy() if threshold > 0 else 0
            t1_arr = t1.double_flash(
                arr3, arr2, fa3, fa2, tr3, tr2, eta_fa_arr, approx='short_tr')
            eta_fa_arr = b1t.double_flash(
                arr1, arr2, fa1, fa2, tr1, tr2, t1_arr, approx='short_tr')
            print('Mean eta_fa: {}'.format(np.nanmean(eta_fa_arr)))
            if threshold > 0:
                delta = np.nansum(
                    np.abs(t1_arr[mask] - t1_last[mask]) +
                    np.abs(eta_fa_arr[mask] - eta_fa_last[mask]))
                if delta < threshold:
                    break
                else:
                    print('iter: {}, delta: {}'.format(i, delta))

        arrs = arr1, arr2, arr3
        fas = fa1, fa2, fa3
        trs = tr1, tr2, tr3
        xi_arr = np.zeros_like(arr1, dtype=float)
        for arr, fa, tr in zip(arrs, fas, trs):
            xi_arr += arr / (
                np.sin(fa * eta_fa_arr) * (1.0 - np.exp(-tr / t1_arr)) /
                (1.0 - np.cos(fa * eta_fa_arr) * np.exp(-tr / t1_arr)))
        xi_arr /= 3

    return t1_arr, xi_arr, eta_fa_arr


# ======================================================================
def triple(
        arr1,
        arr2,
        arr3,
        fa1,
        fa2,
        fa3,
        tr1,
        tr2,
        tr3,
        sign=1,
        prepare=fix_magnitude_bias):
    """
    Calculate the parameters of the FLASH signal at fixed echo time.

    This method is not stable and should not be used.

    Obtains T1, the flip angle efficiency (proportional to the coil transmit
    field) and the apparent spin density (modulated by coil sensitivity).

    Assumes that the following approximations are valid:
     - `tr` (`tr1`, `tr2`, `tr3`) is much smaller compared to `t1`;
     - `fa` (`fa1, `fa2`, `fa3`) is close to zero.

    Given the following expression for the FLASH signal:

    .. math::
        s = \\eta_{m_0} m_0 \\sin(\\eta_\\alpha \\alpha)
        e^{-\\frac{T_E}{T_2^*}}
        \\frac{1 - e^{-\\frac{T_R}{T_1}}}
        {1 - \\cos(\\eta_\\alpha \\alpha) e^{-\\frac{T_R}{T_1}}}

    where
    :math:`m_0` is the spin density,
    :math:`\\eta_{m_0}` is an the receive efficiency
    (proportional to the coil receive field :math:`B_1^-`),
    :math:`\\alpha` is the flip angle of the RF excitation,
    :math:`\\eta_\\alpha` is the flip angle efficiency
    (proportional to the coil transmit field :math:`B_1^+`),
    :math:`T_E` is the echo time,
    :math:`T_2^*` is the reduced transverse relaxation time,
    :math:`T_R` is the repetition time, and
    :math:`T_1` is the longitudinal relaxation time.

    The obtained FLASH signal parameters are :math:`T_1`,
    :math:`\\eta_\\alpha` and :math:`\\eta_{m_0} m_0 e^{-\\frac{T_E}{T_2^*}`.

    Requires that the combination of flip angles and repetition times to be
    unique (not necessarily that all flip angles and all repetition times are
    different).

    This is a closed-form solution.

    Args:
        arr1 (np.ndarray): The input array for the first FLASH signal.
        arr2 (np.ndarray): The input array for the second FLASH signal.
        arr3 (np.ndarray): The input array for the third FLASH signal.
        fa1 (int|float): The first flip angle in deg.
        fa2 (int|float): The second flip angle in deg.
        fa3 (int|float): The third flip angle in deg.
        tr1 (int|float): The first repetition time in time units.
        tr2 (int|float): The first repetition time in time units.
        tr3 (int|float): The first repetition time in time units.
        sign (int): Select one of the two solutions for the equations.
            Must be either +1 or -1.
        prepare (callable|None): Input array preparation.
            Must have the signature: f(np.ndarray) -> np.ndarray.
            Useful for data pre-whitening, including for example the
            correction of magnitude data from Rician mean bias.

    Returns:
        result (tuple[np.ndarray]): The tuple
            contains:
             - t1_arr (np.ndarray): The longitudinal relaxation in time units.
               The units of `t1_arr` are defined by the units of `tr`.
             - xi_arr (np.ndarray): The signal factor in arb. units.
               This is :math:`\\eta_{m_0} m_0 e^{-\\frac{T_E}{T_2^*}`.
             - eta_fa_arr (np.ndarray): The flip angle efficiency in #.
               This is proportional to the coil transmit field :math:`B_1^+`.
    """
    fa1 = np.deg2rad(fa1)
    fa2 = np.deg2rad(fa2)
    fa3 = np.deg2rad(fa3)
    arr1 = (prepare(arr1) if prepare else arr1).astype(np.double)
    arr2 = (prepare(arr2) if prepare else arr2).astype(np.double)
    arr3 = (prepare(arr3) if prepare else arr3).astype(np.double)
    with np.errstate(divide='ignore', invalid='ignore'):
        t1_arr = tr1 * tr2 * tr3 * (
            (fa2 ** 2 - fa1 ** 2) * fa3 * arr1 * arr2 + arr3 * (
                (fa1 * fa3 ** 2 - fa1 * fa2 ** 2) * arr2 +
                (fa1 ** 2 * fa2 - fa2 * fa3 ** 2) * arr1)) / (
                     ((fa2 ** 2 * fa3 * arr1 * arr2 -
                       fa1 * fa2 ** 2 * arr2 * arr3) * tr1 +
                      (fa1 ** 2 * fa2 * arr1 * arr3 -
                       fa1 ** 2 * fa3 * arr1 * arr2) * tr2) * tr3 +
                     (fa1 * fa3 ** 2 * arr2 -
                      fa2 * fa3 ** 2 * arr1) * arr3 * tr1 * tr2)

        eta_fa_arr = (
            fa1 * fa3 * arr1 * arr3 * tr2 * tr3 -
            fa1 * fa2 * arr1 * arr2 * tr2 * tr3 -
            fa2 * fa3 * arr2 * arr3 * tr1 * tr3 +
            fa1 * fa2 * arr1 * arr2 * tr1 * tr3 +
            fa2 * fa3 * arr2 * arr3 * tr1 * tr2 -
            fa1 * fa3 * arr1 * arr3 * tr1 * tr2)
        eta_fa_arr = sign * np.sqrt(2) * np.sqrt(
            (fa1 ** 2 * fa2 * arr1 * arr3 * tr2 * tr3) / eta_fa_arr -
            (fa1 ** 2 * fa3 * arr1 * arr2 * tr2 * tr3) / eta_fa_arr -
            (fa1 * fa2 ** 2 * arr2 * arr3 * tr1 * tr3) / eta_fa_arr +
            (fa2 ** 2 * fa3 * arr1 * arr2 * tr1 * tr3) / eta_fa_arr +
            (fa1 * fa3 ** 2 * arr2 * arr3 * tr1 * tr2) / eta_fa_arr -
            (fa2 * fa3 ** 2 * arr1 * arr3 * tr1 * tr2) / eta_fa_arr) / (
                         np.sqrt(fa1) * np.sqrt(fa2) * np.sqrt(fa3))

        arrs = arr1, arr2, arr3
        fas = fa1, fa2, fa3
        trs = tr1, tr2, tr3
        xi_arr = np.zeros_like(arr1, dtype=float)
        for arr, fa, tr in zip(arrs, fas, trs):
            xi_arr += arr / (
                np.sin(fa * eta_fa_arr) * (1.0 - np.exp(-tr / t1_arr)) /
                (1.0 - np.cos(fa * eta_fa_arr) * np.exp(-tr / t1_arr)))
        xi_arr /= 3

    return t1_arr, xi_arr, eta_fa_arr


# ======================================================================
def vfa(
        arrs,
        fas,
        trs,
        eta_fa_arr=None,
        prepare=fix_magnitude_bias):
    """
    Calculate the parameters of the FLASH signal using variable flip angles.

    Obtains T1 and the apparent spin density (modulated by coil sensitivity).

    Assumes that the flip angles are small (must be below 45°).
    If the different TR are used for the acquisitions, assumes all TRs << T1.

    Given the following expression for the FLASH signal:

    .. math::
        s = \\eta_{m_0} m_0 \\sin(\\eta_\\alpha \\alpha)
        e^{-\\frac{T_E}{T_2^*}}
        \\frac{1 - e^{-\\frac{T_R}{T_1}}}
        {1 - \\cos(\\eta_\\alpha \\alpha) e^{-\\frac{T_R}{T_1}}}

    where
    :math:`m_0` is the spin density,
    :math:`\\eta_{m_0}` is an the receive efficiency
    (proportional to the coil receive field :math:`B_1^-`),
    :math:`\\alpha` is the flip angle of the RF excitation,
    :math:`\\eta_\\alpha` is the flip angle efficiency
    (proportional to the coil transmit field :math:`B_1^+`),
    :math:`T_E` is the echo time,
    :math:`T_2^*` is the reduced transverse relaxation time,
    :math:`T_R` is the repetition time, and
    :math:`T_1` is the longitudinal relaxation time.

    The obtained FLASH signal parameters are :math:`T_1` and
    :math:`\\eta_{m_0} m_0 e^{-\\frac{T_E}{T_2^*}`.

    This is a closed-form solution.

    Args:
        arrs (iterable[np.ndarray]): The input signal arrays in arb.units
        fas (iterable[int|float]): The flip angles in deg.
        trs (iterable[int|float]): The repetition times in time units.
        eta_fa_arr (np.ndarray|None): The flip angle efficiency in #.
            If None, a significant bias may still be present.
        prepare (callable|None): Input array preparation.
            Must have the signature: f(np.ndarray) -> np.ndarray.
            Useful for data pre-whitening, including for example the
            correction of magnitude data from Rician mean bias.

    Returns:
        result (tuple[np.ndarray]): The tuple
            contains:
             - t1_arr (np.ndarray): The longitudinal relaxation in time units.
               The units of `t1_arr` are defined by the units of `tr`.
             - xi_arr (np.ndarray): The signal factor in arb. units.
               This is :math:`\\eta_{m_0} m_0 e^{-\\frac{T_E}{T_2^*}`.

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
            # warnings.warn('Using approximation: TR << T1')
            tr_arr = np.stack(
                [np.full_like(arrs[0], tr, dtype=float) for tr in trs],
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


# ======================================================================
def fit_multipolyfit(
        arrs,
        fas,
        trs,
        prepare=fix_magnitude_bias,
        full=False):
    """
    Fit the parameters of the FLASH signal at fixed echo time.

    This is an iterative optimization fit.

    Args:
        arrs (iterable[np.ndarray]): The input signal arrays in arb.units
        fas (iterable[int|float]): The flip angles in deg.
        trs (iterable[int|float]): The repetition times in time units.


    Returns:

    """
    assert (len(arrs) == len(fas) == len(trs))
    fas = [np.deg2rad(fa) for fa in fas]
    arrs = [prepare(arr) if prepare else arr.astype(float) for arr in arrs]

    raise NotImplementedError


# ======================================================================
def fit_leasq(
        arrs,
        fas,
        trs,
        eta_fa_arr=None,
        prepare=fix_magnitude_bias,
        optim='trf',
        init=(1000, 1e4, 1.0),
        bounds=((10.0, 1e-2, 0.01), (5000.0, 1e10, 1.99)),
        full=False):
    """
    Fit the parameters of the FLASH signal at fixed echo time.

    Obtains T1, the flip angle efficiency (proportional to the coil transmit
    field) and the apparent spin density (modulated by coil sensitivity).

    No assumptions on the flip angles and the repetition times are expected.
    However, reasonable coverage of both flip angles is expected and at least
    three different flip angles and three different repetition times are
    recommended.

    Given the following expression for the FLASH signal:

    .. math::
        s = \\eta_{m_0} m_0 \\sin(\\eta_\\alpha \\alpha)
        e^{-\\frac{T_E}{T_2^*}}
        \\frac{1 - e^{-\\frac{T_R}{T_1}}}
        {1 - \\cos(\\eta_\\alpha \\alpha) e^{-\\frac{T_R}{T_1}}}

    where
    :math:`m_0` is the spin density,
    :math:`\\eta_{m_0}` is an the receive efficiency
    (proportional to the coil receive field :math:`B_1^-`),
    :math:`\\alpha` is the flip angle of the RF excitation,
    :math:`\\eta_\\alpha` is the flip angle efficiency
    (proportional to the coil transmit field :math:`B_1^+`),
    :math:`T_E` is the echo time,
    :math:`T_2^*` is the reduced transverse relaxation time,
    :math:`T_R` is the repetition time, and
    :math:`T_1` is the longitudinal relaxation time.

    The obtained FLASH signal parameters are :math:`T_1` and
    :math:`\\eta_{m_0} m_0 e^{-\\frac{T_E}{T_2^*}`.

    This is an iterative optimization fit.

    Args:
        arrs (iterable[np.ndarray]): The input signal arrays in arb.units
        fas (iterable[int|float]): The flip angles in deg.
        trs (iterable[int|float]): The repetition times in time units.


    Returns:
        result (tuple[np.ndarray]): The tuple
            contains:
             - t1_arr (np.ndarray): The longitudinal relaxation in time units.
               The units of `t1_arr` are defined by the units of `tr`.
             - xi_arr (np.ndarray): The signal factor in arb. units.
               This is :math:`\\eta_{m_0} m_0 e^{-\\frac{T_E}{T_2^*}`.
             - eta_fa_arr (np.ndarray): The flip angle efficiency in #.
               This is proportional to the coil transmit field :math:`B_1^+`.
    """
    assert (len(arrs) == len(fas) == len(trs))
    fas = [np.deg2rad(fa) for fa in fas]
    arrs = [prepare(arr) if prepare else arr.astype(float) for arr in arrs]

    axis = -1
    y_arr = np.array(np.stack(arrs, axis))
    x_arr = np.stack(
        (np.array(fas), np.array(trs)), 0).astype(float)

    num_params = 3 if eta_fa_arr is None else 2

    if init is None:
        init = [1] * num_params

    if eta_fa_arr is None:
        init = init[:num_params]
        bounds = [part_bounds[:num_params] for part_bounds in bounds]

    method_kws = dict(method=optim, bounds=bounds)

    p_arr = voxel_curve_fit(
        y_arr, x_arr, _flash_signal_fit, init,
        method='curve_fit_parallel_map',
        method_kws=method_kws)

    shape = p_arr.shape[:axis]
    p_arrs = [arr.reshape(shape) for arr in np.split(p_arr, num_params, axis)]

    if eta_fa_arr is None:
        t1_arr, xi_arr, eta_fa_arr = p_arrs
    else:
        t1_arr, xi_arr = p_arrs
    # names = 't1', 'xi', 'eta_fa'
    # results = collections.OrderedDict(
    #     tuple((name, p_arr) for name, p_arr in zip(names, p_arrs)))

    if full:
        warnings.warn('E: Not implemented yet!')

    # return results
    return t1_arr, xi_arr, eta_fa_arr
