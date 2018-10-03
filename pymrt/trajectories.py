#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pymrt.geometry: create and manipulate 2D, 3D and N-D geometries.

The 2D geometrical shapes currently available are:
- square
- rectangle
- rhombus
- circle
- ellipse

The 3D geometrical shapes currently available are:
- cube
- cuboid
- rhomboid
- sphere
- ellipsoid
- cylinder

The N-D geometrical shapes currently available are:
- cuboid: sum[abs(x_n/a_n)^inf] < 1
- superellipsoid: sum[abs(x_n/a_n)^k] < 1
- prism: stack (N-1)-D rendered objects on given axis

Running this file directly will run some tests.
"""

# ======================================================================
# :: Future Imports
from __future__ import (
    division, absolute_import, print_function, unicode_literals, )

# ======================================================================
# :: Python Standard Library Imports
# import os  # Miscellaneous operating system interfaces
import itertools  # Functions creating iterators for efficient looping
import functools  # Higher-order functions and operations on callable objects
import warnings  # Warning control
import random  # Generate pseudo-random numbers
import doctest  # Test interactive Python examples

# :: External Imports
import numpy as np  # NumPy (multidimensional numerical arrays library)
import scipy as sp  # SciPy (signal and image processing library)
import flyingcircus as fc  # Everything you always wanted to have in Python.*

# :: External Imports Submodules
import scipy.ndimage  # SciPy: ND-image Manipulation
import flyingcircus.util  # FlyingCircus: generic basic utilities
import flyingcircus.num  # FlyingCircus: generic numerical utilities

# :: Local Imports
import pymrt as mrt
import pymrt.utils
import pymrt.plot

from pymrt import INFO, PATH
from pymrt import elapsed, report
from pymrt import msg, dbg


# ======================================================================
def zig_zag_cartesian_2d(
        t=100,
        initial=(-1, -1),
        final=(1, 1),
        blips=0.2,
        num_blips=None,
        mask_blips=True):
    """


    Args:
        t (np.ndarray|int): The trajectory parameter.

        initial:
        final:
        blips:
        num_blips:
        mask_blips:

    Returns:
        x:
        y:
        mask:
    """
    if mask_blips:
        raise NotImplementedError
    else:
        mask = np.ones(t.shape, dtype=bool)
    if isinstance(t, int):
        t = np.linspace(0.0, 1.0, t)

    return x, y, mask
