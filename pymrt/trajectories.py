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
import scipy.signal  # SciPy: Signal Processing
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
        t=0.01,
        initial=(0, 0),
        final=(1, 1),
        scales=(1, 1),
        blip_size=None,
        num_blips=None):
    """


    Args:
        t (np.ndarray|int): The trajectory parameter.

        initial:
        final:
        blips:
        num_blips:
        mask_blips:

    Returns:
        result (tuple): The tuple
            contains:
             - x_i (np.ndarray):
             - mask (np.ndarray):
    """
    if isinstance(t, int):
        t = np.linspace(0.0, 1.0, t)
    elif isinstance(t, float):
        t = np.arange(0.0, 1.0 + np.spacing(1.0), t)
    t_res = t[1] - t[0]
    if not blip_size and not num_blips:
        blip_size = t_res
    if not blip_size:
        blip_size = (final[1] - initial[1]) / num_blips
    else:  # if not num_blips
        num_blips = (final[1] - initial[1]) / blip_size
    num_trains = num_blips + 1
    train_size = final[0] - initial[0]
    traj_size = train_size * num_trains + blip_size * num_blips
    print(blip_size, num_blips, train_size, num_trains, traj_size)
    x_i = np.zeros((2, t.size))
    duty = blip_size / train_size
    print(duty, num_trains, t_res)
    # assert(duty / num_trains >= t_res)
    print(duty, t[1] - t[0], num_trains)
    blips_mask = (1.0 - sp.signal.square(
        (num_trains - 2 * duty) * 2.0 * np.pi * t, 1.0 - duty)) / 2
    print(blips_mask)
    print(x_i)
    assert(t_re)
    return x_i

import matplotlib.pyplot as plt

x_i = zig_zag_cartesian_2d(num_blips=6)

fig, ax = plt.subplots()
ax.plot(x_i[0], x_i[1])

plt.show()
