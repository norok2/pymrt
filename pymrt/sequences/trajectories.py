#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pymrt.sequences.trajectories: create and manipulate 2D, 3D and N-D trajectories

This is useful for pulse sequences evaluation.
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
def reframe(
        trajectory,
        bounds=(-1, 1)):
    n_dims = trajectory.shape[0]
    try:
        len(bounds)
    except (IndexError, TypeError):
        bounds = (0, bounds)
    try:
        [len(x) for x in bounds]
    except (IndexError, TypeError):
        bounds = fc.util.auto_repeat(bounds, n_dims, True, True)
    if any(len(x) != 2 for x in bounds):
        text = 'Invalid `bounds` format.'
        raise ValueError(text)
    trajectory = trajectory.astype(float)
    for i in range(n_dims):
        trajectory[i] = fc.num.scale(trajectory[i], bounds[i])
    return trajectory


# ======================================================================
def zig_zag_cartesian_2d(
        train_size,
        blip_size=1,
        num_trains=None,
        num_blips=None):
    """
    Generate a zig-zag cartesian trajectory.

    Args:
        train_size:
        blip_size:
        num_trains:
        num_blips:

    Returns:
        result (tuple): The tuple
            contains:
             - x_i (np.ndarray): The coordinates of the trajectory.
             - mask (np.ndarray): The mask for the zig-zag trains.

    Examples:
        >>>
        todo: finish docs
    """
    if not num_trains and not num_blips:
        text = 'At least one of `num_trains` and `num_blips` must be not None'
        raise ValueError(text)
    elif not num_trains:
        num_trains = num_blips + 1
    elif not num_blips:
        num_blips = num_trains - 1
    train = np.arange(0, abs(train_size)) * np.sign(train_size)
    blip = np.arange(1, abs(blip_size)) * np.sign(blip_size)
    x, y, mask = [], [], []
    y_offset = 0
    for j in range(num_trains):
        slicing = slice(None, None, -1) if j % 2 else slice(None, None, 1)
        x.extend(train[slicing].tolist())
        y.extend((np.zeros_like(train) + y_offset).tolist())
        mask.extend(np.ones_like(train, dtype=bool).tolist())
        if j < num_blips:
            x.extend([train[0 if j % 2 else -1]] * blip.size)
            y.extend((blip + y_offset).tolist())
            mask.extend(np.zeros_like(blip, dtype=bool).tolist())
        y_offset += blip_size
    return np.array([x, y]), mask


# ======================================================================
def zig_zag_linear_2d(
        train_size,
        num_trains):
    """
    Generate a zig-zag linear trajectory.

    This is defined as a train of points with monotonic y-axis
    increase/decrease, and alternate monotonic x-axis oscillations.

    Args:
        train_size (int): The number of points of the train.
            Its sign determines the direction of the train: if positive goes
            in the direction of positive x-axis, otherwise goes in the
            other direction.
        num_trains (int): The number of trains.
            Its sign determines the direction of subsequent trains: if
            positive goes in the direction of positive y-axis, otherwise goes
            in the other direction.

    Returns:
        result (tuple): The tuple
            contains:
             - x_i (np.ndarray): The coordinates of the trajectory.
             - mask (np.ndarray): The mask for the zig-zag trains.

    Examples:
        >>> traj, mask = zig_zag_linear_2d(6, 2)
        >>> print(traj)
        [[ 0  1  2  3  4  5  4  3  2  1  0]
         [ 0  1  2  3  4  5  6  7  8  9 10]]
        >>> print(mask)
        [ True  True  True  True  True  True  True  True  True  True  True]
        >>> traj, mask = zig_zag_linear_2d(4, 3)
        >>> print(traj)
        [[0 1 2 3 2 1 0 1 2 3]
         [0 1 2 3 4 5 6 7 8 9]]
        >>> print(mask)
        [ True  True  True  True  True  True  True  True  True  True]
        >>> traj, mask = zig_zag_linear_2d(-4, 3)
        >>> print(traj)
        [[ 0 -1 -2 -3 -2 -1  0 -1 -2 -3]
         [ 0  1  2  3  4  5  6  7  8  9]]
        >>> print(mask)
        [ True  True  True  True  True  True  True  True  True  True]
        >>> traj, mask = zig_zag_linear_2d(4, -3)
        >>> print(traj)
        [[ 0  1  2  3  2  1  0  1  2  3]
         [ 0 -1 -2 -3 -4 -5 -6 -7 -8 -9]]
        >>> print(mask)
        [ True  True  True  True  True  True  True  True  True  True]
        >>> traj, mask = zig_zag_linear_2d(-4, -3)
        >>> print(traj)
        [[ 0 -1 -2 -3 -2 -1  0 -1 -2 -3]
         [ 0 -1 -2 -3 -4 -5 -6 -7 -8 -9]]
        >>> print(mask)
        [ True  True  True  True  True  True  True  True  True  True]
    """
    num_points = (abs(train_size) - 1) * abs(num_trains) + 1
    train = np.arange(0, abs(train_size)) * np.sign(train_size)
    x = [0]
    for j in range(abs(num_trains)):
        slicing = slice(-2, None, -1) if j % 2 else slice(1, None, 1)
        x.extend(train[slicing].tolist())
    y = np.arange(0, num_points) * np.sign(num_trains)
    mask = np.ones(num_points, dtype=bool)
    return np.stack([x, y]), mask


# ======================================================================
elapsed(__file__[len(PATH['base']) + 1:])

# ======================================================================
if __name__ == '__main__':
    import doctest  # Test interactive Python examples

    msg(__doc__.strip())
    doctest.testmod()
    msg(report())
