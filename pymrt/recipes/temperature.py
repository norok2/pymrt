#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pymrt.recipes.temperature: temperature computation.
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
import pymrt.utils
# import pymrt.geometry
