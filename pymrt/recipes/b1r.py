#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pymrt.recipes.b1t: relative B1- (or coil sensitivity) computation.

EXPERIMENTAL!

Note: this may have some overlap with `pymrt.recipes.coils`.
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

# :: Local Imports
import pymrt as mrt

from pymrt.recipes import t1
from pymrt import INFO, PATH
from pymrt import VERB_LVL, D_VERB_LVL, VERB_LVL_NAMES
from pymrt import elapsed, report, run_doctests
from pymrt import msg, dbg


# ======================================================================
def receive_profile(arr):
    """
    EXPERIMENTAL!

    Args:
        arr (np.ndarray): The input array.

    Returns:
        arr (np.ndarray): The output array.
    """
    raise NotImplementedError


# ======================================================================
elapsed(__file__[len(PATH['base']) + 1:])

# ======================================================================
if __name__ == '__main__':
    run_doctests(__doc__)
