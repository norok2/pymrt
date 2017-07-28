#!/usr/bin/env python
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
# import itertools  # Functions creating iterators for efficient looping
import warnings  # Warning control
import collections  # Container datatypes

# :: External Imports
import numpy as np  # NumPy (multidimensional numerical arrays library)

# :: Local Imports
import pymrt as mrt
import pymrt.utils

from pymrt.recipes import t1
# from pymrt import VERB_LVL, D_VERB_LVL, VERB_LVL_NAMES
# from pymrt import elapsed, print_elapsed
# from pymrt import msg, dbg


# # ======================================================================
# def _prepare_rho_mp2rage(use_cache=CFG['use_cache']):
#     """Solve the MP2RAGE rho expression analytically."""
#
#     cache_filepath = os.path.join(DIRS['cache'], 'mp2rage.cache')
#     if not os.path.isfile(cache_filepath) or not use_cache:
#         m0, mz_ss = sym.symbols('m0 mz_ss')
#         n_gre, tr_gre = sym.symbols('n_gre tr_gre')
#         fa1, fa2 = sym.symbols('fa1 fa2')
#         ta, tb, tc = sym.symbols('ta tb tc')
#         fa_p, eta_p = sym.symbols('fa_p eta_p')
#         t1, eta_fa = sym.symbols('t1 eta_fa')
#
#         eqn_mz_ss = sym.Eq(
#             mz_ss,
#             _mz_0rf(
#                 _mz_nrf(
#                     _mz_0rf(
#                         _mz_nrf(
#                             _mz_0rf(
#                                 _mz_i(mz_ss, fa_p, eta_p),
#                                 t1, ta, m0),
#                             t1, n_gre, tr_gre, fa1, m0, eta_fa),
#                         t1, tb, m0),
#                     t1, n_gre, tr_gre, fa2, m0, eta_fa),
#                 t1, tc, m0))
#         mz_ss_ = sym.factor(sym.solve(eqn_mz_ss, mz_ss)[0])
#
#         # convenient exponentials
#         e1 = exp(-tr_gre / t1)
#         ea = exp(-ta / t1)
#         # eb = exp(-tb / t1)
#         ec = exp(-tc / t1)
#
#         # rho for TI1 image (omitted factor: b1r * e2 * m0)
#         gre_ti1 = sin(fa1 * eta_fa) * (
#             (_mz_i(mz_ss, fa_p, eta_p) / m0 * ea +
#              (1 - ea)) * (cos(fa1 * eta_fa) * e1) ** (n_gre / 2 - 1) + (
#                 (1 - e1) * (1 - (cos(fa1* eta_fa) * e1) ** (n_gre / 2 - 1)) /
#                 (1 - cos(fa1 * eta_fa) * e1)))
#
#         # rho for TI2 image (omitted factor: b1r * e2 * m0)
#         gre_ti2 = sin(fa2 * eta_fa) * (
#             ((mz_ss / m0) - (1 - ec)) /
#             (ec * (cos(fa2 * eta_fa) * e1) ** (n_gre / 2)) -
#             (1 - e1) * ((cos(fa2 * eta_fa) * e1) ** (-n_gre / 2) - 1) /
#             (1 - cos(fa2 * eta_fa) * e1))
#
#         # T1 map as a function of steady state rho
#         s = (gre_ti1 * gre_ti2) / (gre_ti1 ** 2 + gre_ti2 ** 2)
#         s = s.subs(mz_ss, mz_ss_)
#
#         pickles = (
#             (n_gre, tr_gre, fa1, fa2, ta, tb, tc, fa_p, eta_p, t1, eta_fa), s)
#         with open(cache_filepath, 'wb') as cache_file:
#             pickle.dump(pickles, cache_file)
#     else:
#         with open(cache_filepath, 'rb') as cache_file:
#             pickles = pickle.load(cache_file)
#     result = np.vectorize(sym.lambdify(*pickles))
#     return result
#
#
# # ======================================================================
# # :: defines the mp2rage signal expression
# _rho = _prepare_rho_mp2rage()


# ======================================================================
def multi_flash(
        arrs,
        flip_angles,
        repetition_times,):
    """
    Calculate the parameters of the FLASH signal at fixed echo time.

    In particular, the following are obtained:
     - T1: the longitudinal relaxation time
     - FA_eff: the flip-angle efficiency (proportional to B1+)
     - M0: the (apparent) spin density (modulated by B1-)

    This is a closed-form solution.

    Args:
        arrs ():
        flip_angles ():
        repetition_times ():
        calc_fa_eff ():
        calc_t1 ():
        calc_m0 ():

    Returns:

    """
    assert(len(arrs) == len(flip_angles) == len(repetition_times))
    warnings.warn('Not implemented yet')


# ======================================================================
def fit_multi_flash(
        arrs,
        flip_angles,
        repetition_times,
        fit_fa_eff=True,
        fit_t1=True,
        fit_m0=True):
    """

    This is an iterative optimization fit.

    Args:
        arrs ():
        flip_angles ():
        repetition_times ():
        fit_fa_eff ():
        fit_t1 ():
        fit_m0 ():

    Returns:

    """
    assert(len(arrs) == len(flip_angles) == len(repetition_times))
    warnings.warn('Not implemented yet')
