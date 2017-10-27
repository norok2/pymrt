#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MP2RAGE pulse sequence library.

Calculate the analytical expression of MP2RAGE signal rho and related
functions.

- T1: longitudinal relaxation time
- eff : efficiency eff of the preparation pulse
- n_GRE : number of pulses in each GRE block
- TR_GRE : repetition time of GRE pulses in ms
- TD0 : time between preparation pulse and first GRE block in ms
- TD1 : time between first and second GRE blocks in ms
- TD2 : time after second GRE block in ms
- A1 : flip angle of the first GRE block in deg
- A2 : flip angle of the second GRE block in deg

Additionally, Conversion from acquisition to sequence parameters is supported.

See Also:
1) Marques, J.P., Kober, T., Krueger, G., van der Zwaag, W., Van de Moortele,
   P.-F., Gruetter, R., 2010. MP2RAGE, a self bias-field corrected sequence
   for improved segmentation and T1-mapping at high field. NeuroImage 49,
   1271–1281. doi:10.1016/j.neuroimage.2009.10.002
2) Metere, R., Kober, T., Möller, H.E., Schäfer, A., 2017. Simultaneous
   Quantitative MRI Mapping of T1, T2* and Magnetic Susceptibility with
   Multi-Echo MP2RAGE. PLOS ONE 12, e0169265. doi:10.1371/journal.pone.0169265
3) Eggenschwiler, F., Kober, T., Magill, A.W., Gruetter, R., Marques, J.P.,
   2012. SA2RAGE: A new sequence for fast B1+-mapping. Magnetic Resonance
   Medicine 67, 1609–1619. doi:10.1002/mrm.23145
"""

# ======================================================================
# :: Future Imports
from __future__ import (
    division, absolute_import, print_function, unicode_literals)

# ======================================================================
# :: Python Standard Library Imports
import os  # Miscellaneous operating system interfaces
import pickle  # Python object serialization

# :: External Imports
import numpy as np  # NumPy (multidimensional numerical arrays library)
# import scipy as sp  # SciPy (signal and image processing library)
import sympy as sym  # SymPy (symbolic CAS library)
from sympy import pi, exp, sin, cos

# :: External Imports Submodules
# import scipy.signal  # SciPy: Signal Processing

# :: Local Imports
import pymrt as mrt
import pymrt.utils
from pymrt import DIRS
from pymrt.config import CFG
from pymrt import msg

# ======================================================================
# :: Default values
_SEQ_PARAMS = dict(
    eta_p=1.0,  # #
    n_gre=160,  # #
    tr_gre=7.0,  # ms
    fa1=4.0,  # deg
    fa2=5.0,  # deg
    ta=440.0,  # ms
    tb=1180.0,  # ms
    tc=4140.0,  # ms
    eta_fa=1.0,  # #
    fa_p=180,  # deg
)

# rho ranges
PSEUDO_RATIO_INTERVAL = (-0.5, 0.5)


# ======================================================================
def _mz_nrf(mz0, t1, n_gre, tr_gre, fa, m0, eta_fa):
    """Magnetization during the GRE block"""
    return mz0 * (cos(fa * eta_fa) * exp(-tr_gre / t1)) ** n_gre + (
        m0 * (1 - exp(-tr_gre / t1)) *
        (1 - (cos(fa * eta_fa) * exp(-tr_gre / t1)) ** n_gre) /
        (1 - cos(fa * eta_fa) * exp(-tr_gre / t1)))


# ======================================================================
def _mz_0rf(mz0, t1, t, m0):
    """Magnetization during the period with no pulses"""
    return mz0 * exp(-t / t1) + m0 * (1 - exp(-t / t1))


# ======================================================================
def _mz_p(mz0, fa_p, eta_p):
    """Magnetization after preparation pulse"""
    return mz0 * cos(fa_p * eta_p)


# ======================================================================
def _prepare_mp2rage(use_cache=CFG['use_cache']):
    """Solve the MP2RAGE signal expressions analytically."""

    cache_filepath = os.path.join(DIRS['cache'], 'mp2rage.cache')
    if not os.path.isfile(cache_filepath) or not use_cache:
        print('Solving MP2RAGE equations. May take some time.')
        print('Caching results: {}'.format(use_cache))
        s1, s2 = sym.symbols('s1 s2')
        m0, mz_ss = sym.symbols('m0 mz_ss')
        n_gre, tr_gre = sym.symbols('n_gre tr_gre')
        fa1, fa2 = sym.symbols('fa1 fa2')
        td0, td1, td2 = sym.symbols('td0 td1 td2')
        fa_p, eta_p = sym.symbols('fa_p eta_p')
        t1, eta_fa = sym.symbols('t1 eta_fa')

        # steady-state magnetization
        eqn_mz_ss = sym.Eq(
            mz_ss,
            _mz_0rf(
                _mz_nrf(
                    _mz_0rf(
                        _mz_nrf(
                            _mz_0rf(
                                _mz_p(mz_ss, fa_p, eta_p),
                                t1, td0, m0),
                            t1, n_gre, tr_gre, fa1, m0, eta_fa),
                        t1, td1, m0),
                    t1, n_gre, tr_gre, fa2, m0, eta_fa),
                t1, td2, m0))
        mz_ss_ = sym.factor(sym.solve(eqn_mz_ss, mz_ss)[0])
        print('mz_ss: {}'.format(mz_ss_))

        # convenient exponentials
        e1 = exp(-tr_gre / t1)
        # e2 = exp(-te / t2s)  # * exp(-1j * d_omega * te)
        ea = exp(-td0 / t1)
        # eb = exp(-td1 / t1)
        ec = exp(-td2 / t1)

        # : signal for TI1
        # eqn_s1 = sym.Eq(
        #     s1,
        #     _mz_nrf(
        #         _mz_0rf(
        #             _mz_p(mz_ss, fa_p, eta_p),
        #             t1, td0, m0),
        #         t1, n_gre / 2 - 1, tr_gre, fa1, m0, eta_fa))
        # s1_ = sym.simplify(sym.solve(eqn_s1, s1)[0])

        # expression from paper (omitted: b1r * e2 * m0 * sin(fa1 * eta_fa))
        s1_ = (
            (_mz_p(mz_ss, fa_p, eta_p) / m0 * ea +
             (1 - ea)) * (cos(fa1 * eta_fa) * e1) ** (n_gre / 2 - 1) + (
                (1 - e1) * (1 - (cos(fa1 * eta_fa) * e1) ** (n_gre / 2 - 1)) /
                (1 - cos(fa1 * eta_fa) * e1)))
        print('s1: {}'.format(s1_))

        # : signal for TI2
        # eqn_s2 = sym.Eq(
        #     s2,
        #     _mz_nrf(
        #         _mz_0rf(
        #             _mz_nrf(
        #                 _mz_0rf(
        #                     _mz_p(mz_ss, fa_p, eta_p),
        #                     t1, td0, m0),
        #                 t1, n_gre, tr_gre, fa1, m0, eta_fa),
        #             t1, td1, m0),
        #         t1, n_gre / 2 - 1, tr_gre, fa2, m0, eta_fa))
        # s2_ = sym.simplify(sym.solve(eqn_s2, s2)[0])

        # expression from paper (omitted: b1r * e2 * m0 * sin(fa2 * eta_fa))
        s2_ = (
            ((mz_ss / m0) - (1 - ec)) /
            (ec * (cos(fa2 * eta_fa) * e1) ** (n_gre / 2)) -
            (1 - e1) * ((cos(fa2 * eta_fa) * e1) ** (-n_gre / 2) - 1) /
            (1 - cos(fa2 * eta_fa) * e1))
        print('s2: {}'.format(s2_))

        # include factors that do not vanish in the ratio
        # (still omitted factors: b1r * e2 * m0)
        s1_ = sin(fa1 * eta_fa) * s1_
        s2_ = sin(fa2 * eta_fa) * s2_

        # T1 map as a function of steady state rho
        # p_ratio_ = 1 / ((s1 / s2) + (s2 / s1))
        p_ratio_ = (s1 * s2) / (s1 ** 2 + s2 ** 2)
        p_ratio_ = (
            p_ratio_.subs({s1: s1_, s2: s2_}).subs({mz_ss: mz_ss_}))
        ratio_ = s1 / s2
        ratio_ = sym.factor(
            ratio_.subs({s1: s1_, s2: s2_}).subs({mz_ss: mz_ss_}))
        r_ratio_ = s2 / s1
        r_ratio_ = sym.factor(
            r_ratio_.subs({s1: s1_, s2: s2_}).subs({mz_ss: mz_ss_}))

        print('pseudo-ratio: {}'.format(p_ratio_))
        print('ratio: {}'.format(ratio_))
        print('reverse-ratio: {}'.format(r_ratio_))

        params = (
            n_gre, tr_gre, fa1, fa2, td0, td1, td2, fa_p, eta_p, t1, eta_fa)
        with open(cache_filepath, 'wb') as cache_file:
            pickle.dump((params, p_ratio_, ratio_, r_ratio_), cache_file)
    else:
        with open(cache_filepath, 'rb') as cache_file:
            params, p_ratio_, ratio_, r_ratio_ = pickle.load(cache_file)
    p_ratio_ = sym.lambdify(params, p_ratio_)
    ratio_ = sym.lambdify(params, ratio_)
    r_ratio_ = sym.lambdify(params, r_ratio_)
    return p_ratio_, ratio_, r_ratio_


# ======================================================================
# :: defines the mp2rage signal expression
_p_ratio, _ratio, _r_ratio = _prepare_mp2rage()


# ======================================================================
def _bijective_part(arr, mask_val=np.nan):
    """
    Mask the largest bijective part of an array.

    Args:
        arr (np.ndarray): The input array.
        mask_val (float): The mask value for the non-bijective part.

    Returns:
        arr (np.ndarray): The larger bijective portion of arr.
            Non-bijective parts are masked.
    """
    bijective_part = mrt.utils.bijective_part(arr)
    if bijective_part.start:
        arr[:bijective_part.start] = mask_val
    if bijective_part.stop:
        arr[bijective_part.stop:] = mask_val
    return arr


# ======================================================================
def rho(
        n_gre,
        tr_gre,
        fa1,
        fa2,
        td0,
        td1,
        td2,
        fa_p,
        eta_p,
        t1,
        eta_fa,
        bijective=False,
        mode='pseudo-ratio'):
    """
    Calculate the MP2RAGE signal ratio from the sequence parameters.

    .. math::
        \\rho = \\frac{s_1}{s_2}

    This expression can also be used for SA2RAGE and NO2RAGE.

    This function is NumPy-aware.

    Args:
        n_gre (int|np.ndarray): Number n of r.f. pulses in each GRE block.
        tr_gre (float|np.ndarray): repetition time of GRE pulses in ms.
        fa1 (float|np.ndarray): Flip angle of the first GRE block in deg.
        fa2 (float|np.ndarray): Flip angle fa2 of the second GRE block in deg.
        td0 (float|np.ndarray): First delay in ms.
             This is the time between the beginning of the preparation pulse
             and the beginning of the first GRE block.
        td1 (float|np.ndarray): Second delay in ms.
            This is the time between the end of the first and the beginning
            of the second GRE blocks.
        td2 (float|np.ndarray): Third delay in ms.
            This is the time between the end of the second GRE block and the
            beginning of the preparation pulse.
        fa_p (float|np.ndarray): Flip angle of the preparation pulse in deg.
        eta_p (float|np.ndarray): Efficiency of the preparation pulse in #.
        t1 (float|np.ndarray): Longitudinal relaxation time in ms.
        eta_fa (float|np.ndarray): Efficiency of the RF excitation.
            This only affects the excitation inside the GRE blocks.
            Equivalent to B1+ efficiency.
        bijective (bool): Force the rho expression to be bijective.
            Non-bijective parts of rho are masked out (set to NaN).
        mode (str): Signal calculation mode.
            Accepted values are:
             - 'p-ratio', 'uni', 'pseudo-ratio', 'uniform': use the pseudo-ratio
               :math:`\\rho = \\frac{s_1 s_2}{s_1^2 + s_2^2}`
             - 'ratio', 'div': calculate the ratio
               :math:`\\rho = \\frac{s_1}{s_2}`
             - 'r-ratio', 'r-div', 'reverse-ratio': use the reverse ratio
               :math:`\\rho = \\frac{s_2}{s_1}`

    Returns:
        ratio (float|np.ndarray): ratio intensity of the MP2RAGE sequence.
    """
    fa1 = np.deg2rad(fa1)
    fa2 = np.deg2rad(fa2)
    fa_p = np.deg2rad(fa_p)
    if eta_p is None:
        eta_p = eta_fa
    mode = mode.lower()
    if mode in ('p-ratio', 'uni', 'pseudo-ratio', 'uniform'):
        rho_func = _p_ratio
    elif mode in ('ratio', 'div'):
        rho_func = _ratio
    elif mode in ('r-ratio', 'r-div', 'reverse-ratio'):
        rho_func = _r_ratio
    else:
        rho_func = None
    result = rho_func(
        n_gre, tr_gre, fa1, fa2, td0, td1, td2, fa_p, eta_p, t1, eta_fa)
    if bijective:
        result = _bijective_part(result)
    return result


# ======================================================================
def acq_to_seq_params(
        ti,
        tr_gre,
        tr_seq,
        matrix_sizes,
        grappa_factors=1,
        grappa_refs=0,
        part_fourier_factors=1.0,
        n_dim=3,
        sl_pe_swap=False,
        pe_correction=(np.pi / 4, 1),
        center_k_correction=0.5,
        bandwidths=None):
    """
    Determine the sequence parameters from the acquisition parameters.

    Args:
        ti (tuple[int]): The inversion times TI of the sequence in ms.
        tr_gre (float): The repetition time TR_GRE of the GRE block in ms.
        tr_seq (int): The repetition time TR_seq of the sequence in ms.
        matrix_sizes (int|tuple[int]):
        grappa_factors (int|tuple[int]):
        grappa_refs (tuple[int]
        part_fourier_factors (tuple[float]):
        sl_pe_swap (bool):
        n_dim (int|None):
        pe_correction (tuple[float]): Correct for the number of k-space lines.
            This factor determines how many k-space lines are actually acquired
            in the k-space for the phase encoding directions.
            This could be different from 1, for example with elliptical k-space
            coverage.
        center_k_correction (float): Correct for the k-space center position.
            This factor determines where the k-space central line is actually
            acquired within the GRE block.
            This parameter affects the accessible inversion times.
        bandwidths (tuple[int]|None): readout bandwidth in Hz/px


    Returns:
        result (tuple[dict]): The tuple
            contains:
             - seq_params (dict): The sequence parameters for rho calculation.
             - extra_info (dict): Additional sequence information.

    Examples:
        >>> seq_kws, extra_info = acq_to_seq_params(
        ...     ti=(900, 3300),
        ...     tr_gre=20.0,
        ...     tr_seq=8000,
        ...     matrix_sizes=256,
        ...     grappa_factors=(1, 2, 1),
        ...     grappa_refs=(0, 24, 0),
        ...     part_fourier_factors=(1.0, 6 / 8, 6 / 8),
        ...     sl_pe_swap=True,
        ...     n_dim=None,
        ...     pe_correction=(np.pi / 4, 1),
        ...     center_k_correction=0.5,
        ...     bandwidths=None)
        >>> print(sorted(seq_kws.items()))
        [('n_gre', 87), ('td0', 573.75), ('td1', 660.0), ('td2', 3286.25),\
 ('tr_gre', 20.0)]
        >>> print(sorted(extra_info.items()))
        [('t_acq', 1536.0)]
    """

    def k_space_lines(size, part_fourier, grappa, grappa_refs):
        return int(size / grappa * part_fourier) + \
               int(np.ceil(grappa_refs * (grappa - 1) / grappa))

    if len(ti) != 2:
        raise ValueError('Exactly two inversion times must be used.')

    if not n_dim:
        n_dim = mrt.utils.combine_iter_len(
            (matrix_sizes, grappa_factors, grappa_refs, part_fourier_factors))
    # check compatibility of given parameters
    matrix_sizes = mrt.utils.auto_repeat(
        matrix_sizes, n_dim, check=True)
    grappa_factors = mrt.utils.auto_repeat(
        grappa_factors, n_dim, check=True)
    grappa_refs = mrt.utils.auto_repeat(
        grappa_refs, n_dim, check=True)
    part_fourier_factors = mrt.utils.auto_repeat(
        part_fourier_factors, n_dim, check=True)

    if n_dim == 3:
        pe1 = 1 if sl_pe_swap else 2
        pe2 = 2 if sl_pe_swap else 1
    else:
        pe1 = 1
        pe2 = None

    n_gre = k_space_lines(
        int(matrix_sizes[pe1] * pe_correction[0]),
        part_fourier_factors[pe1],
        grappa_factors[pe1], grappa_refs[pe1])
    if bandwidths:
        min_tr_gre = round(
            sum([1 / bw * 2 * matrix_sizes[0] for bw in bandwidths]), 2)
        assert (tr_gre >= min_tr_gre)
    t_gre_block = n_gre * tr_gre
    center_k = part_fourier_factors[pe1] / 2 * center_k_correction
    td = ((ti[0] - center_k * t_gre_block),) + \
         tuple(np.diff(ti) - t_gre_block) + \
         ((tr_seq - ti[-1] - (1 - center_k) * t_gre_block),)
    seq_params = dict(
        n_gre=n_gre, tr_gre=tr_gre, td0=td[0], td1=td[1], td2=td[2])
    extra_info = dict(
        t_acq=tr_seq * 1e-3 * (k_space_lines(
            int(matrix_sizes[pe2] * pe_correction[1]),
            part_fourier_factors[pe2],
            grappa_factors[pe2], grappa_refs[pe2]) if pe2 else 1))

    if any(x < 0.0 for x in td):
        raise ValueError('Invalid sequence parameters: {}'.format(seq_params))
    return seq_params, extra_info


# ======================================================================
def test_signal():
    import matplotlib.pyplot as plt

    t1 = np.linspace(50, 5000, 5000)
    s = rho(t1=t1, **_SEQ_PARAMS, bijective=True)
    plt.plot(s, t1)
    plt.show()
    eta_fa = np.array([0.9, 1.0, 1.1])
    s0 = rho(
        t1=100,
        eta_p=1.0,  # #
        n_gre=160,  # #
        tr_gre=7.0,  # ms
        fa1=4.0,  # deg
        fa2=5.0,  # deg
        # tr_seq': 8000.0,  # ms
        # ti1': 1000.0,  # ms
        # ti2': 3300.0,  # ms
        td0=440.0,  # ms
        td1=1180.0,  # ms
        td2=4140.0,  # ms
        eta_fa=eta_fa,  # #
        fa_p=180,  # deg
        bijective=False)
    print(s0)
    return


# ======================================================================
if __name__ == '__main__':
    test_signal()
    msg(__doc__.strip())
