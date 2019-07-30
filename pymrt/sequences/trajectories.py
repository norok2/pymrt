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

# :: Local Imports
import pymrt as mrt
import pymrt.util
import pymrt.plot

from pymrt import INFO, PATH
from pymrt import VERB_LVL, D_VERB_LVL, VERB_LVL_NAMES
from pymrt import elapsed, report, run_doctests
from pymrt import msg, dbg, fmt, fmtm


# ======================================================================
def reframe(
        traj,
        bounds=(-1, 1)):
    """
    Scale the coordinates of the trajectory to be within the specified bounds.

    Args:
        traj (np.ndarray): The coordinates of the trajectory.
            The shape is: (n_dim, n_points).
        bounds (Iterable[int|float|Iterable[int|float]: The scaling bounds.
            If Iterable of int or float, must have size 2, corresponding to
            the min and max bounds for all dimensions.
            If Iterable of Iterable, the outer Iterable must match the
            dimensions of the trajectory, while the inner Iterables must have
            size 2, corresponding to the min and max bounds for each
            dimensions.

    Returns:
        traj (np.ndarray): The coordinates of the trajectory.
            This is scaled to fit in the specified bounds.

    Examples:
        >>> traj, mask = zig_zag_blipped_2d(5, 1, 2)
        >>> print(traj)
        [[0 1 2 3 4 4 3 2 1 0]
         [0 0 0 0 0 1 1 1 1 1]]

        >>> print(reframe(traj, (-1, 1)))
        [[-1.  -0.5  0.   0.5  1.   1.   0.5  0.  -0.5 -1. ]
         [-1.  -1.  -1.  -1.  -1.   1.   1.   1.   1.   1. ]]

        >>> print(reframe(traj, ((0, 8), (0, 3))))
        [[0. 2. 4. 6. 8. 8. 6. 4. 2. 0.]
         [0. 0. 0. 0. 0. 3. 3. 3. 3. 3.]]

        >>> print(reframe(traj, (2, 3, 1)))
        Traceback (most recent call last):
            ...
        ValueError: Invalid `bounds` format.

        >>> print(reframe(traj, ((0, 1, 8), (0, 3))))
        Traceback (most recent call last):
            ...
        ValueError: Invalid `bounds` format.
    """
    n_dims = traj.shape[0]
    try:
        [len(x) for x in bounds]
    except (IndexError, TypeError):
        bounds = fc.base.auto_repeat(bounds, n_dims, True, True)
    if any(len(x) != 2 for x in bounds):
        text = 'Invalid `bounds` format.'
        raise ValueError(text)
    traj = traj.astype(float)
    for i in range(n_dims):
        traj[i] = fc.extra.scale(traj[i], bounds[i])
    return traj


# ======================================================================
def zoom(
        traj,
        factors=1):
    """
    Scale the coordinates of the trajectory by the specified factors.

    Args:
        traj (np.ndarray): The coordinates of the trajectory.
            The shape is: (n_dim, n_points).
        factors (int|float|Iterable[int|float]): The scaling factor(s).
            If int or float, the same factor is used for all dimensions.
            If Iterable, must match the dimensions of the trajectory.

    Returns:
        traj (np.ndarray): The coordinates of the trajectory.
            The shape is: (n_dim, n_points).
            The values are scaled according to the specified factors.

    Examples:
        >>> traj, mask = zig_zag_blipped_2d(5, 1, 2)
        >>> print(traj)
        [[0 1 2 3 4 4 3 2 1 0]
         [0 0 0 0 0 1 1 1 1 1]]

        >>> print(zoom(traj, 2))
        [[0 2 4 6 8 8 6 4 2 0]
         [0 0 0 0 0 2 2 2 2 2]]

        >>> print(zoom(traj, (2, 3)))
        [[0 2 4 6 8 8 6 4 2 0]
         [0 0 0 0 0 3 3 3 3 3]]

        >>> print(zoom(traj, (2, 3, 1)))
        Traceback (most recent call last):
            ...
        AssertionError
    """
    n_dims = traj.shape[0]
    factors = fc.base.auto_repeat(factors, n_dims, False, True)
    factors = np.array(factors).reshape(-1, 1)
    traj = traj * factors
    return traj


# ======================================================================
def to_nd(
        traj,
        new_dims=None):
    """
    Convert any trajectory to an N-dim trajectory.

    Args:
        traj (np.ndarray): The coordinates of the trajectory.
            The shape is: (n_dim, n_points).
            This can represent any dim trajectory.
            If n_dim < new_dims, the other dimensions are set to 0.
            If n_dim == new_dims, the trajectory is left untouched.
            If n_dim > new_dims, only the first `new_dims` dims of the
            trajectory are kept.
        new_dims (int|None): The number of dims of the new trajectory.
            If larger than 0, a new trajectory with the specified number of
            dims is obtained.
            If equal to 0, an empty trajectory is obtained.
            If smaller than 0, the number of dims is reduced by the specified
            amount.
            If None, the trajectory is left untouched.

    Returns:
        traj (np.ndarray): The coordinates of the trajectory.
            The shape is: (new_dims, n_points).
            This represents the N-dim trajectory.

    Examples:
        >>> traj, mask = zig_zag_blipped_2d(5, 1, 2)
        >>> print(traj.shape)
        (2, 10)
        >>> print(traj)
        [[0 1 2 3 4 4 3 2 1 0]
         [0 0 0 0 0 1 1 1 1 1]]
        >>> print(to_nd(traj))
        [[0 1 2 3 4 4 3 2 1 0]
         [0 0 0 0 0 1 1 1 1 1]]
        >>> print(to_nd(traj, 3))
        [[0 1 2 3 4 4 3 2 1 0]
         [0 0 0 0 0 1 1 1 1 1]
         [0 0 0 0 0 0 0 0 0 0]]
        >>> print(to_nd(to_nd(traj, 3), 3))
        [[0 1 2 3 4 4 3 2 1 0]
         [0 0 0 0 0 1 1 1 1 1]
         [0 0 0 0 0 0 0 0 0 0]]
        >>> print(to_nd(np.concatenate((traj, traj), axis=0), 3))
        [[0 1 2 3 4 4 3 2 1 0]
         [0 0 0 0 0 1 1 1 1 1]
         [0 1 2 3 4 4 3 2 1 0]]
        >>> print(to_nd(traj, 5))
        [[0 1 2 3 4 4 3 2 1 0]
         [0 0 0 0 0 1 1 1 1 1]
         [0 0 0 0 0 0 0 0 0 0]
         [0 0 0 0 0 0 0 0 0 0]
         [0 0 0 0 0 0 0 0 0 0]]
        >>> print(to_nd(to_nd(traj, 10), 2))
        [[0 1 2 3 4 4 3 2 1 0]
         [0 0 0 0 0 1 1 1 1 1]]
        >>> print(to_nd(traj, 1))
        [[0 1 2 3 4 4 3 2 1 0]]
        >>> print(to_nd(traj, 0))
        []
        >>> print(to_nd(to_nd(traj, 5), -2))
        [[0 1 2 3 4 4 3 2 1 0]
         [0 0 0 0 0 1 1 1 1 1]
         [0 0 0 0 0 0 0 0 0 0]]
    """
    n_dims, n_points = traj.shape
    if new_dims is None:
        new_dims = n_dims
    if n_dims < new_dims:
        return np.concatenate(
            (traj,
             np.zeros((new_dims - n_dims, n_points), dtype=traj.dtype)),
            axis=0)
    elif n_dims > new_dims:
        return traj[0:new_dims, ...]
    else:
        return traj


# ======================================================================
def zig_zag_blipped_2d(
        train_size,
        blip_size=1,
        num_trains=None,
        num_blips=None,
        modulation=np.zeros_like,
        modulation_kws=None):
    """
    Generate a zig-zag blipped trajectory.

    Args:
        train_size (int): The number of points of the train.
            Its sign determines the direction of the train: if positive goes
            in the direction of positive x-axis, otherwise goes in the
            other direction.
        blip_size (int): The number of points of the blip.
            Its sign determines the direction of blip: if positive goes in the
            direction of positive y-axis, otherwise goes in the other
            direction.
        num_trains (int|None): The number of trains.
            If None, it is calculated from the number of blips.
            The following equation must hold: `num_trains - num_blips == 1`.
        num_blips (int|None): The number of blips.
            If None, it is calculated from the number of trains.
            The following equation must hold: `num_trains - num_blips == 1`.
        modulation (callable): The modulation of the trains.
            Must have the following signature:
            func(np.ndarray, **modulation_kws) -> np.ndarray
        modulation_kws (dict|tuple|None): Keyword arguments for `modulation`.
            If tuple, must produce a valid dict upon casting.

    Returns:
        result (tuple): The tuple
            contains:
             - traj (np.ndarray): The coordinates of the trajectory.
                    The shape is: (n_dim, n_points).
             - mask (np.ndarray): The mask for the zig-zag trains.
                    The shape is: (n_dim, n_points).

    Examples:
        >>> traj, mask = zig_zag_blipped_2d(5, 1, 2)
        >>> print(traj)
        [[0 1 2 3 4 4 3 2 1 0]
         [0 0 0 0 0 1 1 1 1 1]]
        >>> print(mask)
        [ True  True  True  True  True  True  True  True  True  True]

        >>> traj, mask = zig_zag_blipped_2d(3, 2, 3)
        >>> print(traj)
        [[0 1 2 2 2 1 0 0 0 1 2]
         [0 0 0 1 2 2 2 3 4 4 4]]
        >>> print(mask)
        [ True  True  True False  True  True  True False  True  True  True]

        >>> traj, mask = zig_zag_blipped_2d(-3, 2, 3)
        >>> print(traj)
        [[ 0 -1 -2 -2 -2 -1  0  0  0 -1 -2]
         [ 0  0  0  1  2  2  2  3  4  4  4]]
        >>> print(mask)
        [ True  True  True False  True  True  True False  True  True  True]

        >>> traj, mask = zig_zag_blipped_2d(3, -2, 3)
        >>> print(traj)
        [[ 0  1  2  2  2  1  0  0  0  1  2]
         [ 0  0  0 -1 -2 -2 -2 -3 -4 -4 -4]]
        >>> print(mask)
        [ True  True  True False  True  True  True False  True  True  True]

        >>> traj, mask = zig_zag_blipped_2d(-3, -2, 3)
        >>> print(traj)
        [[ 0 -1 -2 -2 -2 -1  0  0  0 -1 -2]
         [ 0  0  0 -1 -2 -2 -2 -3 -4 -4 -4]]
        >>> print(mask)
        [ True  True  True False  True  True  True False  True  True  True]

        >>> traj, mask = zig_zag_blipped_2d(3, 2, num_blips=2)
        >>> print(traj)
        [[0 1 2 2 2 1 0 0 0 1 2]
         [0 0 0 1 2 2 2 3 4 4 4]]
        >>> print(mask)
        [ True  True  True False  True  True  True False  True  True  True]

        >>> traj, mask = zig_zag_blipped_2d(3, 2, 5, 2)
        Traceback (most recent call last):
            ...
        AssertionError
    """
    if not num_trains and not num_blips:
        text = 'At least one of `num_trains` and `num_blips` must be not None'
        raise ValueError(text)
    elif not num_trains:
        num_trains = num_blips + 1
    elif not num_blips:
        num_blips = num_trains - 1
    assert (num_trains - num_blips == 1)
    modulation_kws = {} if modulation_kws is None else dict(modulation_kws)
    train = np.arange(0, abs(train_size)) * np.sign(train_size)
    blip = np.arange(1, abs(blip_size)) * np.sign(blip_size)
    x, y, mask = [], [], []
    y_offset = 0
    for j in range(num_trains):
        slicing = slice(None, None, -1) if j % 2 else slice(None, None, 1)
        x.extend(train[slicing].tolist())
        y.extend((modulation(train, **modulation_kws) + y_offset).tolist())
        mask.extend(np.ones_like(train, dtype=bool).tolist())
        if j < num_blips:
            x.extend([train[0 if j % 2 else -1]] * blip.size)
            y.extend((blip + y_offset).tolist())
            mask.extend(np.zeros_like(blip, dtype=bool).tolist())
        y_offset += blip_size
    return np.array([x, y]), np.array(mask)


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
             - traj (np.ndarray): The coordinates of the trajectory.
                    The shape is: (n_dim, n_points).
             - mask (np.ndarray): The mask for the trains.
                    The shape is: (n_dim, n_points).

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
    n_points = (abs(train_size) - 1) * abs(num_trains) + 1
    train = np.arange(0, abs(train_size)) * np.sign(train_size)
    x = [0]
    for j in range(abs(num_trains)):
        slicing = slice(-2, None, -1) if j % 2 else slice(1, None, 1)
        x.extend(train[slicing].tolist())
    y = np.arange(0, n_points) * np.sign(num_trains)
    mask = np.ones(n_points, dtype=bool)
    return np.stack([x, y]), mask


# ======================================================================
def zig_zag_blipped_sinusoidal_2d(
        train_size,
        blip_size=1,
        num_trains=None,
        num_blips=None,
        wavelength=None,
        amplitude=None,
        phase=0.0):
    """
    Generate a zig-zag blipped trajectory with harmonic trains.

    This is generated using a `sin(x)` function.

    Args:
        train_size (int): The number of points of the train.
            Its sign determines the direction of the train: if positive goes
            in the direction of positive x-axis, otherwise goes in the
            other direction.
        blip_size (int): The number of points of the blip.
            Its sign determines the direction of blip: if positive goes in the
            direction of positive y-axis, otherwise goes in the other
            direction.
        num_trains (int|None): The number of trains.
            If None, it is calculated from the number of blips.
            The following equation must hold: `num_trains - num_blips == 1`.
        num_blips (int|None): The number of blips.
            If None, it is calculated from the number of trains.
            The following equation must hold: `num_trains - num_blips == 1`.
        wavelength (int|float|None):
        amplitude (int|float|None):
        phase (int|float): The phase of the sinusoidal modulation in rad.

    Returns:
        result (tuple): The tuple
            contains:
             - traj (np.ndarray): The coordinates of the trajectory.
                    The shape is: (n_dim, n_points).
             - mask (np.ndarray): The mask for the trains.
                    The shape is: (n_dim, n_points).
    Examples:

    """
    if not wavelength:
        wavelength = train_size
    if not amplitude:
        amplitude = blip_size / 2.0
    if not phase:
        phase = 0.0

    def modulation(x):
        return amplitude * np.sin(2.0 * np.pi * x / wavelength + phase)

    return zig_zag_blipped_2d(
        train_size, blip_size, num_trains, num_blips, modulation)


# ======================================================================
def density(
        traj,
        mask=None):
    """
    Compute the spatial density.

    This is the number of points divided by the min-max hyper-volume.

    Args:
        traj (np.ndarray): The coordinates of the trajectory.
            The shape is: (n_dim, n_points).
        mask (np.ndarray|None): The mask for the trains.
            The shape is: (n_dim, n_points).

    Returns:
        result (float): The density.
    """
    n_dims, n_points = traj.shape
    if mask is None:
        mask = slice(None)
        n_valid_points = n_points
    else:
        n_valid_points = np.sum(mask)
    bounds = tuple(fc.extra.minmax(traj[i, mask]) for i in range(n_dims))
    bound_sizes = tuple(np.ptp(interval) for interval in bounds)
    return n_valid_points / fc.base.prod(bound_sizes)


# ======================================================================
def coverage(
        traj,
        shape,
        mask=None):
    """
    Compute the histogram density coverage.

    This is the percent of bins that are covered by the trajectory,
    i.e. the bins that contain at least one point.

    Args:
        traj (np.ndarray): The coordinates of the trajectory.
            The shape is: (n_dim, n_points).
        shape (Iterable[int]): The shape of the array.
        mask (np.ndarray|None): The mask for the trains.
            The shape is: (n_dim, n_points).

    Returns:
        result (float): The coverage.
    """
    if mask is None:
        mask = slice(None)
    hist, edges = np.histogramdd(traj[mask].T, shape)
    max_hist = hist.size
    nonzero_hist = np.sum(hist > 0)
    return nonzero_hist / max_hist


# ======================================================================
def sampling_mask(
        traj,
        shape,
        factors=1,
        fit=True):
    """
    Generate a sampling mask of given shape from a trajectory.

    Args:
        traj (np.ndarray): The coordinates of the trajectory.
            The shape is: (n_dim, n_points).
        shape (Iterable[int]): The shape of the array.
        factors (int|float|Iterable[int|float]): The scaling factor(s).
            The
            If int or float, the same factor is used for all dimensions.
            If Iterable, must match the dimensions of the trajectory.
        fit (bool): Fit the entire trajectory within the shape.

    Returns:
        result (np.ndarray[bool]): The sampling mask.
            This can be applied to any `nd.array` with a matching shape to
            sample the specified trajectory.

    Examples:
        >>> traj = np.array(((0, 0), (1, 1), (2, 2))).T
        >>> print(sampling_mask(traj, (3, 3)))
        [[ True False False]
         [False  True False]
         [False False  True]]
        >>> arr = np.arange(3 * 3).reshape((3, 3)) + 1
        >>> print(arr * sampling_mask(traj, arr.shape))
        [[1 0 0]
         [0 5 0]
         [0 0 9]]
        >>> print(sampling_mask(traj, (3, 3), factors=2))
        [[ True False False False False False]
         [False False False False False False]
         [False False  True False False False]
         [False False False False False False]
         [False False False False False False]
         [False False False False False  True]]
        >>> print(sampling_mask(traj, (3, 3), factors=3).astype(int))
        [[1 0 0 0 0 0 0 0 0]
         [0 0 0 0 0 0 0 0 0]
         [0 0 0 0 0 0 0 0 0]
         [0 0 0 0 0 0 0 0 0]
         [0 0 0 0 1 0 0 0 0]
         [0 0 0 0 0 0 0 0 0]
         [0 0 0 0 0 0 0 0 0]
         [0 0 0 0 0 0 0 0 0]
         [0 0 0 0 0 0 0 0 1]]
    """
    n_dim = len(shape)
    factors = fc.base.auto_repeat(factors, n_dim, False, True)
    shape = tuple(int(size * factor) for size, factor in zip(shape, factors))
    if fit:
        traj = reframe(traj, tuple((0, size - 1) for size in shape))
    result = np.zeros(shape, dtype=bool)
    result[tuple(x for x in np.round(traj).astype(int))] = True
    return result


# ======================================================================
elapsed(__file__[len(PATH['base']) + 1:])

# ======================================================================
if __name__ == '__main__':
    run_doctests(__doc__)
