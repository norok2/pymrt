#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pymrt.recipes.t2: T2 transverse relaxation computation.
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
# import pymrt.utils

# from pymrt import VERB_LVL, D_VERB_LVL, VERB_LVL_NAMES
# from pymrt import elapsed, print_elapsed
# from pymrt import msg, dbg

from pymrt.recipes.generic import fit_exp_loglin


# ======================================================================
def fit_monoexp(
        arr, echo_times, echo_times_mask=None, mode='loglin'):
    if mode == 'loglin':
        return fit_exp_loglin(arr, echo_times, echo_times_mask)['tau']
    else:
        warnings.warn('Unknonw mode `{mode}`'.format_map(locals()))
    return
