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


# ======================================================================
def multi_flash(
        arrs,
        flip_angles,
        repetition_times,
        fit_fa_eff=True,
        fit_t1=True,
        fit_m0=True):
    """

    This is a closed-form solution.

    Args:
        arrs ():
        flip_angles ():
        repetition_times ():
        fit_fa_eff ():
        fit_t1 ():
        fit_m0 ():

    Returns:

    """
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
    warnings.warn('Not implemented yet')
