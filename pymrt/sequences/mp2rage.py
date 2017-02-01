#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MP2RAGE pulse sequence library.

Calculate the analytical expression of MP2RAGE signal.

- T1: longitudinal relaxation time
- eff : efficiency eff of the adiabatic inversion pulse
- n_GRE : number of pulses in each GRE block
- TR_GRE : repetition time of GRE pulses in ms
- TA : time between inversion pulse and first GRE block in ms
- TB : time between first and second GRE blocks in ms
- TC : time after second GRE block in ms
- A1 : flip angle of the first GRE block in deg
- A2 : flip angle a2 of the second GRE block in deg

Additionally, Conversion from acquisition to sequence parameters is supported.

[ref: J. P. Marques at al., NeuroImage 49 (2010) 1271-1281]
"""

# ======================================================================
# :: Future Imports
from __future__ import (
    division, absolute_import, print_function, unicode_literals)

# ======================================================================
# :: Python Standard Library Imports

# :: External Imports
import numpy as np  # NumPy (multidimensional numerical arrays library)
import scipy as sp  # SciPy (signal and image processing library)
import sympy as sym  # SymPy (symbolic CAS library)

# :: External Imports Submodules
import scipy.signal  # SciPy: Signal Processing

# :: Local Imports
import pymrt.utils as pmu
from pymrt import msg

# ======================================================================
# :: Default values
T1_INTERVAL = (50.0, 5000.0)
D_SEQ_PARAMS = dict(
    eff=1.0,  # #
    n_gre=160,  # #
    tr_gre=7.0,  # ms
    a1=4.0,  # deg
    a2=5.0,  # deg
    # tr_seq': 8000.0,  # ms
    # ti1': 1000.0,  # ms
    # ti2': 3300.0,  # ms
    ta=440.0,  # ms
    tb=1180.0,  # ms
    tc=4140.0,  # ms
)

# signal ranges
UNIFORM_INTERVAL = (-0.5, 0.5)
DICOM_INTERVAL = (0, 4095)


# ======================================================================
def _mz_nrf(mz0, t1, n_gre, tr_gre, alpha, m0):
    """Magnetization during the GRE block"""
    from sympy import exp, cos
    return mz0 * (cos(alpha) * exp(-tr_gre / t1)) ** n_gre + \
           m0 * (1 - exp(-tr_gre / t1)) * \
           (1 - (cos(alpha) * exp(-tr_gre / t1)) ** n_gre) / \
           (1 - cos(alpha) * exp(-tr_gre / t1))


def _mz_0rf(mz0, t1, t, m0):
    """Magnetization during the period with no pulses"""
    from sympy import exp
    return mz0 * exp(-t / t1) + m0 * (1 - exp(-t / t1))


def _mz_i(mz0, eff):
    """Magnetization after adiabatic inversion pulse"""
    return -eff * mz0


# ======================================================================
# :: solve the MP2RAGE signal expression analytically
def _prepare():
    from sympy import exp, sin, cos

    t1, eff, n_gre, tr_gre, m0, ta, tb, tc, a1, a2, mz_ss = \
        sym.symbols('T1 eff n_GRE TR_GRE M0 TA TB TC A1 A2 Mz_ss')

    eqn_mz_ss = sym.Eq(
        mz_ss,
        _mz_0rf(
            _mz_nrf(
                _mz_0rf(
                    _mz_nrf(
                        _mz_0rf(
                            _mz_i(mz_ss, eff),
                            t1, ta, m0),
                        t1, n_gre, tr_gre, a1, m0),
                    t1, tb, m0),
                t1, n_gre, tr_gre, a2, m0),
            t1, tc, m0))
    mz_ss_ = sym.factor(sym.solve(eqn_mz_ss, mz_ss)[0])

    # convenient exponentials
    e1 = exp(-tr_gre / t1)
    ea = exp(-ta / t1)
    # eb = exp(-tb / t1)
    ec = exp(-tc / t1)

    # signal for TI1 image (omitted factor: b1r * e2 * m0)
    gre_ti1 = sin(a1) * (
        (-eff * mz_ss / m0 * ea +
         (1 - ea)) * (cos(a1) * e1) ** (n_gre / 2 - 1) + (
            (1 - e1) * (1 - (cos(a1) * e1) ** (n_gre / 2 - 1)) /
            (1 - cos(a1) * e1)))

    # signal for TI2 image (omitted factor: b1r * e2 * m0)
    gre_ti2 = sin(a2) * (
        ((mz_ss / m0) - (1 - ec)) / (ec * (cos(a2) * e1) ** (n_gre / 2))
        - (1 - e1) * ((cos(a2) * e1) ** (-n_gre / 2) - 1) / (1 - cos(a2) * e1))

    # T1 map as a function of steady state signal
    s = (gre_ti1 * gre_ti2) / (gre_ti1 ** 2 + gre_ti2 ** 2)
    s = s.subs(mz_ss, mz_ss_)

    return np.vectorize(
        sym.lambdify((t1, eff, n_gre, tr_gre, ta, tb, tc, a1, a2), s))


# ======================================================================
# :: defines the mp2rage signal expression
_signal = _prepare()


# ======================================================================
def signal(
        t1,
        eff,
        n_gre,
        tr_gre,
        ta,
        tb,
        tc,
        a1,
        a2,
        bijective=False):
    """
    Calculate MP2RAGE intensity from direct parameters (NumPy-aware).

    Args:
        t1 (float|np.ndarray): T1 time in ms.
        eff (float): Efficiency eff of the adiabatic inversion pulse.
        n_gre (int): Number n of r.f. pulses in each GRE block.
        tr_gre (float): repetition time of GRE pulses in ms.
        ta (float): Time TA between inversion pulse and first GRE block in ms.
        tb (float): Time TB between first and second GRE blocks in ms.
        tc (float): Time TC after second GRE block in ms.
        a1 (float): Flip angle a1 of the first GRE block in deg.
        a2 (float): Flip angle a2 of the second GRE block in deg.
        bijective (bool): Force the signal to be bijective.
            Non-bijective parts of the signal are masked out (using NaN).

    Returns:
        s (float|np.ndarray): signal intensity of the MP2RAGE sequence.
    """
    a1 = np.deg2rad(a1)
    a2 = np.deg2rad(a2)
    s = _signal(t1, eff, n_gre, tr_gre, ta, tb, tc, a1, a2)
    if bijective:
        s = _bijective_part(s)
    return s


# ======================================================================
def _calc_tr_seq(eff, n_gre, tr_gre, ta, tb, tc, a1, a2):
    """Calculate TR_SEQ for MP2RAGE sequence."""
    return ta + tb + tc + 2 * n_gre * tr_gre


# ======================================================================
def _calc_ti1(eff, n_gre, tr_gre, ta, tb, tc, a1, a2):
    """ Calculate TI1 for MP2RAGE sequence."""
    return ta + (1 / 2) * n_gre * tr_gre


# ======================================================================
def _calc_ti2(eff, n_gre, tr_gre, ta, tb, tc, a1, a2):
    """Calculate TI2 for MP2RAGE sequenc.e"""
    return ta + tb + (3 / 2) * n_gre * tr_gre


# ======================================================================
def _signal2(t1, eff, n_gre, tr_gre, tr_seq, ti1, ti2, a1, a2):
    """
    Calculate MP2RAGE intensity from indirect parameters (NumPy-aware).

    Args:
        t1 (float|np.ndarray): T1 time in ms
        eff (float): efficiency eff of the adiabatic inversion pulse.
        n_gre (int): number n of pulses in each GRE block.
        tr_gre (float): TR_GRE repetition time of GRE pulses in ms.
        tr_seq (float
        total repetition time of the MP2RAGE sequence in ms.
        t1 (float): T1 time in ms.
        eff (float): efficiency eff of the adiabatic inversion pulse.
        n_gre (int): number n of r.f. pulses in each GRE block.
        tr_gre (float): repetition time of GRE pulses in ms.
        ti1 (float): inversion time of the center of first GRE blocks in ms.
        ti2 (float): inversion time of the center of second GRE blocks in ms.
        a1 (float): flip angle a1 of the first GRE block in deg.
        a2 (float): flip angle a2 of the second GRE block in deg.

    Returns:
        s (float|np.ndarray): signal intensity of the MP2RAGE sequence.
    """
    a1 = np.deg2rad(a1)
    a2 = np.deg2rad(a2)
    ta = _calc_ta(eff, n_gre, tr_gre, tr_seq, ti1, ti2, a1, a2)
    tb = _calc_tb(eff, n_gre, tr_gre, tr_seq, ti1, ti2, a1, a2)
    tc = _calc_tc(eff, n_gre, tr_gre, tr_seq, ti1, ti2, a1, a2)
    return _signal(t1, eff, n_gre, tr_gre, ta, tb, tc, a1, a2)


# ======================================================================
def acq_to_seq_params(
        matrix_sizes=(256, 256, 256),
        grappa_factors=(1, 2, 1),
        grappa_refs=(0, 24, 0),
        part_fourier_factors=(1.0, 6 / 8, 6 / 8),
        bandwidths=(250,) * 4,
        sl_pe_swap=True,
        tr_seq=8000,
        ti=(900, 3300),
        alpha=(3.0, 5.0),
        eff=0.95,
        tr_gre=20.0):
    """
    Determine the sequence parameters from the acquisition parameters.

    Args:
        matrix_sizes (tuple[int]):
        grappa_factors (tuple[int]):
        grappa_refs (tuple[int]
        part_fourier_factors (tuple[float]):
        tr_gre (float):
        bandwidths (tuple[int]|None): readout bandwidth in Hz/px
        sl_pe_swap (bool):
        tr_seq (int): repetition time TR_seq of the sequence
        ti (tuple[int]):
        alpha (tuple[int]):

    Returns:
        seq_params (dict): The sequence parameters for signal calculation.
        extra_info (dict): Additional sequence information.
    """

    def k_space_lines(size, grappa, part_fourier, grappa_refs):
        return int(size / grappa * part_fourier) + \
               int(np.ceil(grappa_refs * (grappa - 1) / grappa))

    if len(ti) != 2:
        raise ValueError('Exactly two inversion times must be used.')
    if len(ti) != len(alpha):
        raise ValueError('Number of inversions and flip angles must match.')
    pe1 = 1 if sl_pe_swap else 2
    pe2 = 2 if sl_pe_swap else 1
    n_gre = k_space_lines(
        matrix_sizes[pe1], grappa_factors[pe1], part_fourier_factors[pe1],
        grappa_refs[pe1])
    if bandwidths:
        min_tr_gre = round(
            sum([1 / bw * 2 * matrix_sizes[0] for bw in bandwidths]), 2)
        assert (tr_gre >= min_tr_gre)
    t_gre_block = n_gre * tr_gre
    center_k = part_fourier_factors[pe1] / 2
    td = ((ti[0] - center_k * t_gre_block),) + \
         tuple(np.diff(ti) - t_gre_block) + \
         ((tr_seq - ti[-1] - (1 - center_k) * t_gre_block),)
    seq_params = dict(
        eff=eff,
        n_gre=n_gre,
        tr_gre=tr_gre,
        ta=td[0],
        tb=td[1],
        tc=td[2],
        a1=alpha[0],
        a2=alpha[1])
    extra_info = dict(
        t_acq=tr_seq * 1e-3 * k_space_lines(
            matrix_sizes[pe2], grappa_factors[pe2], part_fourier_factors[pe2],
            grappa_refs[pe2]))

    if any(x < 0.0 for x in td):
        raise ValueError('Invalid sequence parameters: {}'.format(seq_params))
    return seq_params, extra_info


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
    bijective_part = pmu.bijective_part(arr)
    if bijective_part.start:
        arr[:bijective_part.start] = mask_val
    if bijective_part.stop:
        arr[bijective_part.stop:] = mask_val
    return arr


# ======================================================================
def _calc_ta(eff, n_gre, tr_gre, tr_seq, ti1, ti2, a1, a2):
    """Calculate TA for MP2RAGE sequence."""
    return (2.0 * ti1 - n_gre * tr_gre) / 2.0


# ======================================================================
def _calc_tb(eff, n_gre, tr_gre, tr_seq, ti1, ti2, a1, a2):
    """ Calculate TB for MP2RAGE sequence."""
    return ti2 - ti1 - n_gre * tr_gre


# ======================================================================
def _calc_tc(eff, n_gre, tr_gre, tr_seq, ti1, ti2, a1, a2):
    """Calculate TC for MP2RAGE sequence."""
    return tr_seq - ti2 - n_gre * tr_gre / 2.0


# ======================================================================
def test_signal():
    import matplotlib.pyplot as plt
    t1 = np.linspace(50, 5000, 5000)
    s = signal(t1, **D_SEQ_PARAMS, bijective=True)
    plt.plot(s, t1)
    plt.show()
    eff = np.array([0.9, 1.0, 1.1])
    s0 = signal(
        100,
        eff=1.0,  # #
        n_gre=160,  # #
        tr_gre=7.0,  # ms
        a1=4.0 * eff,  # deg
        a2=5.0,  # deg
        # tr_seq': 8000.0,  # ms
        # ti1': 1000.0,  # ms
        # ti2': 3300.0,  # ms
        ta=440.0,  # ms
        tb=1180.0,  # ms
        tc=4140.0,  # ms
        bijective=True)
    print(s0)


# ======================================================================
if __name__ == '__main__':
    test_signal()
    msg(__doc__.strip())
