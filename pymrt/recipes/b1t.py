#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pymrt.recipes.b1t: relative B1+ (or flip angle efficiency) computation.
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
import scipy as sp  # SciPy (signal and image processing library)

import scipy.interpolate  # Scipy: Interpolation

# :: Local Imports
import pymrt as mrt
import pymrt.utils
import pymrt.correction

from pymrt.sequences import mp2rage


# from pymrt import VERB_LVL, D_VERB_LVL, VERB_LVL_NAMES
# from pymrt import elapsed, report
# from pymrt import msg, dbg


# ======================================================================
def afi(
        arr1,
        arr2,
        n_tr,
        fa,
        prepare=None):
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
        arr1 (np.ndarray): The first input array in arb. units.
            Contains the signal with :math:`T_{R,1}`.
        arr2 (np.ndarray): The second input array in arb. units.
            Contains the signal with :math:`T_{R,2}`.
        n_tr (int|float): The repetition times ratio in one units.
            This is defined as :math:`n = \\frac{T_{R,2}}{T_{R,1}}`.
        fa (int|float): The flip angle in deg.
        prepare (callable|None): Input array preparation.
            Must have the signature: f(np.ndarray) -> np.ndarray.
            Useful for data pre-whitening, including for example the
            correction of magnitude data from Rician mean bias.

    Returns:
        eta_fa_arr (np.ndarray): The flip angle efficiency in one units.
            This is the :math:`\\eta_\\alpha` factor.

    References:
        - Yarnykh, V.L., 2007. Actual flip-angle imaging in the pulsed steady
          state: A method for rapid three-dimensional mapping of the
          transmitted radiofrequency field. Magn. Reson. Med. 57, 192–200.
          doi:10.1002/mrm.21120
        - Steichter, M., 2018.
    """
    arr1 = prepare(arr1) if prepare else arr1.astype(complex)
    arr2 = prepare(arr2) if prepare else arr2.astype(complex)

    fa = np.deg2rad(fa)
    with np.errstate(divide='ignore', invalid='ignore'):
        eta_fa_arr = arr2 / arr1
        eta_fa_arr = (eta_fa_arr * n_tr - 1) / (n_tr - eta_fa_arr)

    eta_fa_arr = np.real(np.arccos(eta_fa_arr))
    return eta_fa_arr / fa


# ======================================================================
def double_rare(
        arr1,
        arr2,
        fa,
        sign=1,
        prepare=mrt.correction.fix_bias_rician):
    """
    Calculate the flip angle efficiency from two RARE acquisitions.

    The flip angle is obtained by solving:

    .. math::
        \\cos(\\alpha) = \\frac{r\\pm\\sqrt{r^2 + 8}}{4}

    where :math:`s_1` is the signal of the first acquisition (`arr1`)
    obtained with a preparation flip angle of :math:`\\alpha`,
    :math:`s_2` is  the signal of the second acquisition (`arr2`)
    acquired with a preparation flip angle :math:`2\\alpha`,
    and  :math:`r = \\frac{s_2}{s_1}`.

    The sign of the equation should be `-` only if the actual flip angle
    exceeds 90° (180°) in the first (second) acquisition.

    Assumes long :math:`T_R` regime: :math:`T_R > 5 * T_1`.

    A nominal flip angle combination of 30°/60° is suggested.

    The method works also for two double-angle MPRAGE where the double angle
    refers to the flip angle of the magnetization preparation (with
    identical gradient-redout blocks).

    This is a closed-form solution.

    Args:
        arr1 (np.ndarray): The first input array in arb. units.
            Contains the signal with flip angle `fa`.
        arr2 (np.ndarray): The second input array in arb. units.
            Contains the signal with flip angle `2 * fa`.
        fa (int|float): The flip angle in deg.
            This is the flip angle of the first acquisition.
        sign (int): Select one of the two solutions for the equations.
            Must be either +1 or -1.
        prepare (callable|None): Input array preparation.
            Must have the signature: f(np.ndarray) -> np.ndarray.
            Useful for data pre-whitening, including for example the
            correction of magnitude data from Rician mean bias.

    Returns:
        eta_fa_arr (np.ndarray): The flip angle efficiency in one units.
            This is the :math:`\\eta_\\alpha` factor.

    Notes:
        The same information can be obtained with two MPRAGE-like acquisition
        where the magnetization preparation plays the role of the initial
        exicitation in the RARE pulse sequence (assumes long :math:`T_R`
        regime - for :math:`T_{R,\mathrm{seq}}`, not for
        :math:`T_{R,\mathrm{GRE}}`).

    References:
        - Sled, J.G., Pike, G.B., 2000. Correction for B1 and B0 variations
          in quantitative T2 measurements using MRI. Magn. Reson. Med. 43,
          589–593.
          doi:10.1002/(SICI)1522-2594(200004)43:4<589::AID-MRM14>3.0.CO;2-2
    """
    arr1 = prepare(arr1) if prepare else arr1.astype(complex)
    arr2 = prepare(arr2) if prepare else arr2.astype(complex)

    fa = np.deg2rad(fa)
    with np.errstate(divide='ignore', invalid='ignore'):
        eta_fa_arr = arr2 / arr1
        eta_fa_arr = (eta_fa_arr + sign * np.sqrt(eta_fa_arr ** 2 + 8)) / 4
        # eta_fa_arr_ = (eta_fa_arr - sign * np.sqrt(eta_fa_arr ** 2 + 8)) / 4
        # eta_fa_arr[eta_fa_arr > 1] = eta_fa_arr_[eta_fa_arr > 1]

    eta_fa_arr = np.real(np.arccos(eta_fa_arr))
    return eta_fa_arr / fa


# ======================================================================
def double_flash(
        arr1,
        arr2,
        fa1,
        fa2,
        tr1,
        tr2,
        t1_arr=None,
        approx=None,
        sign=1,
        prepare=mrt.correction.fix_bias_rician):
    """
    Calculate the flip angle efficiency from two FLASH acquisitions.

    This method is sensitive to T1 except for tr1, tr2 >> T1.

    Uses the ratio between two FLASH images provided some conditions are met.

    Most notably, the following cases are covered:
     - `tr1, tr2 >> max(T1)`, `tr1 == tr2` and `fa2 == 2 * fa1`.
     - `tr1, tr2 << min(T1)`, `tr1 == tr2` and `fa2 == 2 * fa1`
       (T1 must be known).
     - `tr1, tr2 << min(T1)`, `tr1 != tr2` and `fa2 == fa1`
       (T1 must be known).
     - `tr1 == tr2` and `fa2 == 2 * fa1` (T1 must be known).
     - `tr1 != tr2` and `fa2 == fa1` (T1 must be known).

    This method, unless the `long_tr` approximation is used, requires an
    accurate T1 map.

    This is a closed-form solution.

    Array units must be consistent.
    Time units must be consistent.

    Args:
        arr1 (np.ndarray): The second input array in arb. units.
            This is a FLASH image acquired with the nominal flip angle
            specified in the `flip_angle` parameter.
            `arr1` and `arr2` shapes must match.
        arr2 (np.ndarray): The first input array in arb. units.
            This is a FLASH image acquired with the double the nominal
            flip angle specified in the `flip_angle` parameter.
            `arr1` and `arr2` shapes must match.
        fa1 (int|float): The first nominal flip angle in deg.
        fa2 (int|float): The second nominal flip angle in deg.
        tr1 (int|float): The first repetition time in time units.
        tr2 (int|float): The second repetition time in time units.
        t1_arr (int|float|np.ndarray|None): The T1 value in time units.
            If None, the following conditions must be met:
             - `approx == 'long_tr'`, `tr1 == tr2` and `fa2 == 2 * fa1`.
            If int or float, one of the following conditions must be met:
             - `approx == 'short_tr'`, `tr1 == tr2` and `fa2 == 2 * fa1`.
             - `approx == 'short_tr'`, `tr1 != tr2` and `fa1 == fa2`.
            If np.ndarray, one of the following conditions must be met:
             - `tr1 == tr2` and `fa2 == 2 * fa1`.
             - `tr1 != tr2` and `fa2 == fa1`.
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
        eta_fa_arr (np.ndarray): The flip angle efficiency in one units.
            This is the :math:`\\eta_\\alpha` factor.
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
    m_fa = fa2 / fa1  # the flip angles ratio
    same_tr = np.isclose(tr1, tr2)
    same_fa = np.isclose(fa1, fa2)
    double_fa = np.isclose(2, m_fa)

    eta_fa_arr = None
    with np.errstate(divide='ignore', invalid='ignore'):
        # double angle methods
        if double_fa and same_tr:
            if approx == 'long_tr':
                eta_fa_arr = arr2 / arr1 / 2
            elif approx == 'short_tr' and t1_arr is not None:
                eta_fa_arr = (np.sqrt(
                    arr1 ** 2 * (1 + (2 * tr) / t1_arr) +
                    (2 * arr2 * (arr2 - arr1)) * (1 + tr / t1_arr) +
                    2 * arr2 * (arr2 - arr1)) - arr1 * (1 + tr / t1_arr)) / \
                             (2 * (arr2 - arr1))
            elif t1_arr is not None:
                eta_fa_arr = (np.sqrt(
                    arr1 ** 2 * np.exp((2 * tr) / t1_arr) +
                    (2 * arr2 * (arr2 - arr1)) * np.exp(tr / t1_arr) +
                    2 * arr2 * (arr2 - arr1)) - arr1 * np.exp(tr / t1_arr)) / \
                             (2 * (arr2 - arr1))

        # same-angle methods (variable tr)
        elif same_fa:
            if approx == 'short_tr' and t1_arr is not None:
                eta_fa_arr = (arr1 / arr2)
                eta_fa_arr = (
                    (1 - n_tr * eta_fa_arr) /
                    (n_tr * (eta_fa_arr - 1) * (tr / t1_arr) +
                     1 - n_tr * eta_fa_arr))
            elif t1_arr is not None:
                eta_fa_arr = (
                    (arr2 * np.exp(tr2 / t1_arr) -
                     arr1 * np.exp(tr1 / t1_arr) +
                     (arr1 - arr2) * np.exp((tr1 + tr2) / t1_arr)) /
                    (arr2 - arr1 +
                     arr1 * np.exp(tr2 / t1_arr) -
                     arr2 * np.exp(tr1 / t1_arr)))

        if eta_fa_arr is None:
            text = 'Unsupported parameter combination. ' \
                   'Fallback to `eta_fa_arr=1`.'
            warnings.warn(text)
            eta_fa_arr = np.full_like(arr1, fa)

    eta_fa_arr = np.real(np.arccos(eta_fa_arr))
    return eta_fa_arr / fa


# ======================================================================
def mp2rage_rho(
        rho_arr,
        t1_arr=1500,
        t1_values_range=(100, 5000),
        t1_num=512,
        eta_fa_values_range=(0.001, 2.0),
        eta_fa_num=512,
        mode='ratio',
        **params_kws):
    """
    Calculate the flip angle efficiency from an MP2RAGE acquisition.

    This also supports SA2RAGE and NO2RAGE.

    Args:
        rho_arr (float|np.ndarray): MP2RAGE signal array.
            Its interpretation depends on `use_ratio`.
            If `use_ratio` is False, the pseudo-ratio s1*s2/(s1^2+s2^2) is
            used and the values must be in the (-0.5, 0.5) range.
            If `use_ratio` is True, the ratio s1/s2 is used and the values are
            not bound.
        t1_arr (int|float|np.array): Longitudinal relaxation in ms.
            If np.ndarray, it must have the same shape as `rho_arr`.
            If int or float, the longitudinal relaxation is assumed to be
            constant over `rho_arr`.
        t1_values_range (tuple[float]): The T1 range.
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
        **params_kws: The acquisition parameters.
            This is filtered through `pymrt.utils.split_func_kws()` for
            `sequences.mp2rage.acq_to_seq_params()` and the result
            is passed to `sequences.mp2rage.rho()`.
            Its (key, value) pairs must be accepted by either
             `sequences.mp2rage.acq_to_seq_params()` or
             `sequences.mp2rage.rho()`.

    Returns:
        eta_fa_arr (np.ndarray): The flip angle efficiency in one units.

    References:
        1) Marques, J.P., Kober, T., Krueger, G., van der Zwaag, W.,
           Van de Moortele, P.-F., Gruetter, R., 2010. MP2RAGE, a self
           bias-field corrected sequence for improved segmentation and
           T1-mapping at high field. NeuroImage 49, 1271–1281.
           doi:10.1016/j.neuroimage.2009.10.002
        2) Metere, R., Kober, T., Möller, H.E., Schäfer, A., 2017. Simultaneous
           Quantitative MRI Mapping of T1, T2* and Magnetic Susceptibility with
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
        acq_kws, kws = mrt.utils.split_func_kws(
            mp2rage.acq_to_seq_params, params_kws)
        seq_kws, extra_info = mp2rage.acq_to_seq_params(**acq_kws)
        seq_kws.update(kws)
    except TypeError:
        seq_kws, kws = mrt.utils.split_func_kws(mp2rage.rho, params_kws)
        if len(kws) > 0:
            warnings.warn('Unrecognized parameters: {}'.format(kws))

    if isinstance(t1_arr, (int, float)):
        # determine the rho expression
        eta_fa = np.linspace(
            eta_fa_values_range[0], eta_fa_values_range[1], t1_num)
        rho = mp2rage.rho(t1=t1_arr, eta_fa=eta_fa, mode=mode, **seq_kws)
        # remove non-bijective branches
        bijective_slice = mrt.utils.bijective_part(rho)
        eta_fa = eta_fa[bijective_slice]
        rho = rho[bijective_slice]
        if rho[0] > rho[-1]:
            rho = rho[::-1]
            eta_fa = eta_fa[::-1]
        # check that rho values are strictly increasing
        if not np.all(np.diff(rho) > 0):
            raise ValueError(
                'MP2RAGE look-up table was not properly prepared.')
        eta_fa_arr = np.interp(rho_arr, rho, eta_fa)

    else:
        # determine the rho expression
        t1 = np.linspace(
            t1_values_range[0], t1_values_range[1], t1_num).reshape(-1, 1)
        eta_fa = np.linspace(
            eta_fa_values_range[0], eta_fa_values_range[1],
            eta_fa_num).reshape(1, -1)
        rho = mp2rage.rho(t1=t1, eta_fa=eta_fa, mode=mode, **seq_kws)
        # todo: remove non bijective branches
        # use griddata for interpolation
        eta_fa_arr = sp.interpolate.griddata(
            (rho.ravel(), (np.zeros_like(rho) + t1).ravel()),
            (np.zeros_like(rho) + eta_fa).ravel(),
            (rho_arr.ravel(), t1_arr.ravel()))
        eta_fa_arr = eta_fa_arr.reshape(rho_arr.shape)

    return eta_fa_arr
