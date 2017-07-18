#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MP2RAGE pulse sequence library.

Calculate the analytical expression of MP2RAGE signal rho and related
functions.

- T1: longitudinal relaxation time
- eff : efficiency eff of the preparation pulse
- n_GRE : number of pulses in each GRE block
- TR_GRE : repetition time of GRE pulses in ms
- TA : time between preparation pulse and first GRE block in ms
- TB : time between first and second GRE blocks in ms
- TC : time after second GRE block in ms
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
RHO_INTERVAL = (-0.5, 0.5)


# ======================================================================
def _mz_nrf(mz0, t1, n_gre, tr_gre, fa, m0, eta_fa):
    """Magnetization during the GRE block"""
    return mz0 * (cos(fa * eta_fa) * exp(-tr_gre / t1)) ** n_gre + (
        m0 * (1 - exp(-tr_gre / t1)) *
        (1 - (cos(fa * eta_fa) * exp(-tr_gre / t1)) ** n_gre) /
        (1 - cos(fa * eta_fa) * exp(-tr_gre / t1)))


def _mz_0rf(mz0, t1, t, m0):
    """Magnetization during the period with no pulses"""
    return mz0 * exp(-t / t1) + m0 * (1 - exp(-t / t1))


def _mz_i(mz0, fa_p, eta_p):
    """Magnetization after preparation pulse"""
    return mz0 * cos(fa_p * eta_p)


# ======================================================================
def _prepare(use_cache=CFG['use_cache']):
    """Solve the MP2RAGE rho expression analytically."""

    cache_filepath = os.path.join(DIRS['cache'], 'mp2rage.cache')
    if not os.path.isfile(cache_filepath) or not use_cache:
        m0, mz_ss = sym.symbols('m0 mz_ss')
        n_gre, tr_gre = sym.symbols('n_gre tr_gre')
        fa1, fa2 = sym.symbols('fa1 fa2')
        ta, tb, tc = sym.symbols('ta tb tc')
        fa_p, eta_p = sym.symbols('fa_p eta_p')
        t1, eta_fa = sym.symbols('t1 eta_fa')

        eqn_mz_ss = sym.Eq(
            mz_ss,
            _mz_0rf(
                _mz_nrf(
                    _mz_0rf(
                        _mz_nrf(
                            _mz_0rf(
                                _mz_i(mz_ss, fa_p, eta_p),
                                t1, ta, m0),
                            t1, n_gre, tr_gre, fa1, m0, eta_fa),
                        t1, tb, m0),
                    t1, n_gre, tr_gre, fa2, m0, eta_fa),
                t1, tc, m0))
        mz_ss_ = sym.factor(sym.solve(eqn_mz_ss, mz_ss)[0])

        # convenient exponentials
        e1 = exp(-tr_gre / t1)
        ea = exp(-ta / t1)
        # eb = exp(-tb / t1)
        ec = exp(-tc / t1)

        # rho for TI1 image (omitted factor: b1r * e2 * m0)
        gre_ti1 = sin(fa1 * eta_fa) * (
            (_mz_i(mz_ss, fa_p, eta_p) / m0 * ea +
             (1 - ea)) * (cos(fa1 * eta_fa) * e1) ** (n_gre / 2 - 1) + (
                (1 - e1) * (1 - (cos(fa1* eta_fa) * e1) ** (n_gre / 2 - 1)) /
                (1 - cos(fa1 * eta_fa) * e1)))

        # rho for TI2 image (omitted factor: b1r * e2 * m0)
        gre_ti2 = sin(fa2 * eta_fa) * (
            ((mz_ss / m0) - (1 - ec)) /
            (ec * (cos(fa2 * eta_fa) * e1) ** (n_gre / 2)) -
            (1 - e1) * ((cos(fa2 * eta_fa) * e1) ** (-n_gre / 2) - 1) /
            (1 - cos(fa2 * eta_fa) * e1))

        # T1 map as a function of steady state rho
        s = (gre_ti1 * gre_ti2) / (gre_ti1 ** 2 + gre_ti2 ** 2)
        s = s.subs(mz_ss, mz_ss_)

        pickles = (
            (n_gre, tr_gre, fa1, fa2, ta, tb, tc, fa_p, eta_p, t1, eta_fa), s)
        with open(cache_filepath, 'wb') as cache_file:
            pickle.dump(pickles, cache_file)
    else:
        with open(cache_filepath, 'rb') as cache_file:
            pickles = pickle.load(cache_file)
    result = np.vectorize(sym.lambdify(*pickles))
    return result


# ======================================================================
# :: defines the mp2rage_t1 signal expression
_rho = _prepare()


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
        ta,
        tb,
        tc,
        fa_p,
        eta_p,
        t1,
        eta_fa,
        bijective=False):
    """
    Calculate MP2RAGE signal rho from the sequence parameters.

    This function is NumPy-aware.

    Args:
        n_gre (int): Number n of r.f. pulses in each GRE block.
        tr_gre (float): repetition time of GRE pulses in ms.
        fa1 (float): Flip angle fa1 of the first GRE block in deg.
        fa2 (float): Flip angle fa2 of the second GRE block in deg.
        ta (float): Time TA between preparation pulse and first GRE block in
        ms.
        tb (float): Time TB between first and second GRE blocks in ms.
        tc (float): Time TC after second GRE block in ms.
        fa_p (float): Flip angle fa_p of the preparation pulse.
        eta_p (float): Efficiency of the preparation pulse.
        t1 (float): T1 time in ms.
        eta_fa (float): Efficiency of the RF excitation in the GRE block.
            Equivalent to B1+ efficiency.
        bijective (bool): Force the rho to be bijective.
            Non-bijective parts of rho are masked out (using NaN).

    Returns:
        rho (float): rho intensity of the MP2RAGE sequence.
    """
    fa1 = np.deg2rad(fa1)
    fa2 = np.deg2rad(fa2)
    fa_p = np.deg2rad(fa_p)
    result = _rho(
        n_gre, tr_gre, fa1, fa2, ta, tb, tc, fa_p, eta_p, t1, eta_fa)
    if bijective:
        result = _bijective_part(result)
    return result


# ======================================================================
def acq_to_seq_params(
        matrix_sizes=(256, 256, 256),
        grappa_factors=(1, 2, 1),
        grappa_refs=(0, 24, 0),
        part_fourier_factors=(1.0, 6 / 8, 6 / 8),
        bandwidths=None,
        sl_pe_swap=True,
        pe_correction=(np.pi / 4, 1),
        center_k_correction=0.5,
        tr_seq=8000,
        ti=(900, 3300),
        tr_gre=20.0):
    """
    Determine the sequence parameters from the acquisition parameters.

    Args:
        matrix_sizes (tuple[int]):
        grappa_factors (tuple[int]):
        grappa_refs (tuple[int]
        part_fourier_factors (tuple[float]):
        bandwidths (tuple[int]|None): readout bandwidth in Hz/px
        sl_pe_swap (bool):
        pe_correction (tuple[float]): Correct for the number of k-space lines.
            This factor determines how many k-space lines are actually acquired
            in the k-space for the phase encoding directions.
            This could be different from 1, for example with elliptical k-space
            coverage.
        center_k_correction (float): Correct for the k-space center position.
            This factor determines where the k-space central line is actually
            acquired within the GRE block.
            This parameter affects the accessible inversion times.
        tr_seq (int): repetition time TR_seq of the sequence
        ti (tuple[int]):
        fa (tuple[int]):
        eta_p (float):
        tr_gre (float):

    Returns:
        result (tuple[dict]): The tuple
            contains:
             - seq_params (dict): The sequence parameters for rho calculation.
             - extra_info (dict): Additional sequence information.
    """

    def k_space_lines(size, part_fourier, grappa, grappa_refs):
        return int(size / grappa * part_fourier) + \
               int(np.ceil(grappa_refs * (grappa - 1) / grappa))

    if len(ti) != 2:
        raise ValueError('Exactly two inversion times must be used.')

    pe1 = 1 if sl_pe_swap else 2
    pe2 = 2 if sl_pe_swap else 1

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
        n_gre=n_gre,
        tr_gre=tr_gre,
        ta=td[0],
        tb=td[1],
        tc=td[2])
    extra_info = dict(
        t_acq=tr_seq * 1e-3 * k_space_lines(
            int(matrix_sizes[pe2] * pe_correction[1]),
            part_fourier_factors[pe2],
            grappa_factors[pe2], grappa_refs[pe2]))

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
        ta=440.0,  # ms
        tb=1180.0,  # ms
        tc=4140.0,  # ms
        eta_fa=eta_fa,  # #
        fa_p=180,  # deg
        bijective=False)
    print(s0)
    return


# ======================================================================
if __name__ == '__main__':
    test_signal()
    msg(__doc__.strip())
