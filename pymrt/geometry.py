#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pymrt.geometry: create and manipulate 2D, 3D and N-D geometries.

The 2D geometrical shapes currently available are:
- square
- rectangle
- rhombus
- circle
- ellipsis

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
- prism: stack (N-1)-D mask on given axis

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

# :: External Imports Submodules
import scipy.ndimage  # SciPy: ND-image Manipulation

# :: Local Imports
import pymrt as mrt
import pymrt.utils
import pymrt.plot

from pymrt import INFO, DIRS
from pymrt import elapsed, report
from pymrt import msg, dbg


# todo: support for Iterable relative position/size in nd_cuboid, etc.

# ======================================================================
def rel2abs(shape, size=0.5):
    """
    Calculate the absolute size from a relative size for a given shape.

    Args:
        shape (int|Iterable[int]): The shape of the mask in px.
        size (float|tuple[float]): Relative position (to the lowest edge).
            Each element of the tuple should be in the range [0, 1].

    Returns:
        position (float|tuple[float]): Absolute position inside the shape.
            Each element of the tuple should be in the range [0, dim - 1],
            where dim is the corresponding dimension of the shape.

    Examples:
        >>> rel2abs((100, 100, 101, 101), (0.0, 1.0, 0.0, 1.0))
        (0.0, 99.0, 0.0, 100.0)
        >>> rel2abs((100, 99, 101))
        (49.5, 49.0, 50.0)
        >>> rel2abs((100, 200, 50, 99, 37), (0.0, 1.0, 0.2, 0.3, 0.4))
        (0.0, 199.0, 9.8, 29.4, 14.4)
        >>> rel2abs((100, 100, 100), (1.0, 10.0, -1.0))
        (99.0, 990.0, -99.0)
        >>> shape = (100, 100, 100, 100, 100)
        >>> abs2rel(shape, rel2abs(shape, (0.0, 0.25, 0.5, 0.75, 1.0)))
        (0.0, 0.25, 0.5, 0.75, 1.0)
    """
    size = mrt.utils.auto_repeat(size, len(shape), check=True)
    return tuple((s - 1.0) * p for p, s in zip(size, shape))


# ======================================================================
def abs2rel(shape, position=0):
    """
    Calculate the relative size from an absolute size for a given shape.

    Args:
        shape (int|Iterable[int]): The shape of the mask in px.
        position (float|tuple[float]): Absolute position inside the shape.
            Each element of the tuple should be in the range [0, dim - 1],
            where dim is the corresponding dimension of the shape.

    Returns:
        position (float|tuple[float]): Relative position (to the lowest edge).
            Each element of the tuple should be in the range [0, 1].

    Examples:
        >>> abs2rel((100, 100, 101, 99), (0, 100, 100, 100))
        (0.0, 1.0101010101010102, 1.0, 1.0204081632653061)
        >>> abs2rel((100, 99, 101))
        (0.0, 0.0, 0.0)
        >>> abs2rel((412, 200, 37), (30, 33, 11.7))
        (0.072992700729927, 0.1658291457286432, 0.32499999999999996)
        >>> abs2rel((100, 100, 100), (250, 10, -30))
        (2.525252525252525, 0.10101010101010101, -0.30303030303030304)
        >>> shape = (100, 100, 100, 100, 100)
        >>> abs2rel(shape, rel2abs(shape, (0, 25, 50, 75, 100)))
        (0.0, 25.0, 50.0, 75.0, 100.0)
    """
    position = mrt.utils.auto_repeat(position, len(shape), check=True)
    return tuple(p / (s - 1.0) for p, s in zip(position, shape))


# ======================================================================
def render(
        mask,
        fill=(1, 0),
        dtype=float):
    """
    Render a mask as an image.

    Args:
        mask (np.ndarray): Mask to be rendered.
        fill (tuple[bool|int|float]): Values to render the mask with.
            The first value is used for inside the mask.
            The second value is used for outside the mask.
        dtype (data-type): Desired output data-type.
            See `np.ndarray()` for more.

    Returns:
        img (np.ndarray):
            Image of the mask rendered with specified values and data type.
    """
    img = np.empty_like(mask, dtype)
    img[mask] = fill[0]
    img[~mask] = fill[1]
    return img


# ======================================================================
def square(
        shape,
        side,
        position=0.5):
    """
    Generate a mask whose shape is a square.

    Args:
        shape (int|Iterable[int]): The shape of the mask in px.
        side (float): The side of the square in px.
        position (float|Iterable[float]): The position of the center.
            Values are relative to the lowest edge.
            They are interpreted as relative to the shape.

    Returns:
        mask (np.ndarray): Array of boolean describing the geometrical object.

    Examples:
        >>> square(4, 2)
        array([[False, False, False, False],
               [False,  True,  True, False],
               [False,  True,  True, False],
               [False, False, False, False]], dtype=bool)
        >>> square(5, 3)
        array([[False, False, False, False, False],
               [False,  True,  True,  True, False],
               [False,  True,  True,  True, False],
               [False,  True,  True,  True, False],
               [False, False, False, False, False]], dtype=bool)
    """
    return nd_cuboid(
        shape, side / 2.0, position, 2,
        rel_position=True, rel_sizes=False)


# ======================================================================
def rectangle(
        shape,
        semisides,
        position=0.5):
    """
    Generate a mask whose shape is a rectangle.

    Args:
        shape (int|Iterable[int]): The shape of the mask in px.
        semisides (tuple[float]): The semisides of the rectangle in px.
        position (float|Iterable[float]): The position of the center.
            Values are relative to the lowest edge.
            They are interpreted as relative to the shape.

    Returns:
        mask (np.ndarray): Array of boolean describing the geometrical object.

    >>> rectangle(6, (2, 1))
    array([[False, False, False, False, False, False],
           [False, False,  True,  True, False, False],
           [False, False,  True,  True, False, False],
           [False, False,  True,  True, False, False],
           [False, False,  True,  True, False, False],
           [False, False, False, False, False, False]], dtype=bool)
    >>> rectangle(5, (2, 1))
    array([[False,  True,  True,  True, False],
           [False,  True,  True,  True, False],
           [False,  True,  True,  True, False],
           [False,  True,  True,  True, False],
           [False,  True,  True,  True, False]], dtype=bool)
    >>> rectangle(4, (1, 0.5), 0)
    array([[ True, False, False, False],
           [ True, False, False, False],
           [False, False, False, False],
           [False, False, False, False]], dtype=bool)
    >>> rectangle(4, (2, 1), 0)
    array([[ True,  True, False, False],
           [ True,  True, False, False],
           [ True,  True, False, False],
           [False, False, False, False]], dtype=bool)
    """
    return nd_cuboid(
        shape, semisides, position, 2,
        rel_position=True, rel_sizes=False)


# ======================================================================
def rhombus(
        shape,
        semidiagonals,
        position=0.5):
    """
    Generate a mask whose shape is a rhombus.

    Args:
        shape (int|Iterable[int]): The shape of the mask in px.
        semidiagonals (float|tuple[float]): The rhombus semidiagonas in px.
        position (float|Iterable[float]): The position of the center.
            Values are relative to the lowest edge.
            They are interpreted as relative to the shape.

    Returns:
        mask (np.ndarray): Array of boolean describing the geometrical object.

    Examples:
        >>> rhombus(5, 2)
        array([[False, False,  True, False, False],
               [False,  True,  True,  True, False],
               [ True,  True,  True,  True,  True],
               [False,  True,  True,  True, False],
               [False, False,  True, False, False]], dtype=bool)
        >>> rhombus(5, (2, 1))
        array([[False, False,  True, False, False],
               [False, False,  True, False, False],
               [False,  True,  True,  True, False],
               [False, False,  True, False, False],
               [False, False,  True, False, False]], dtype=bool)
    """
    return nd_superellipsoid(
        shape, semidiagonals, 1.0, position, 2,
        rel_position=True, rel_sizes=False)


# ======================================================================
def circle(
        shape,
        radius,
        position=0.5):
    """
    Generate a mask whose shape is a circle.

    Args:
        shape (int|Iterable[int]): The shape of the mask in px.
        radius (float): The radius of the circle in px.
        position (float|Iterable[float]): The position of the center.
            Values are relative to the lowest edge.
            They are interpreted as relative to the shape.

    Returns:
        mask (np.ndarray): Array of boolean describing the geometrical object.

    Examples:
        >>> circle(5, 1)
        array([[False, False, False, False, False],
               [False, False,  True, False, False],
               [False,  True,  True,  True, False],
               [False, False,  True, False, False],
               [False, False, False, False, False]], dtype=bool)
        >>> circle(6, 2)
        array([[False, False, False, False, False, False],
               [False, False,  True,  True, False, False],
               [False,  True,  True,  True,  True, False],
               [False,  True,  True,  True,  True, False],
               [False, False,  True,  True, False, False],
               [False, False, False, False, False, False]], dtype=bool)
        >>> circle(4, 2, 0)
        array([[ True,  True,  True, False],
               [ True,  True, False, False],
               [ True, False, False, False],
               [False, False, False, False]], dtype=bool)
    """
    return nd_superellipsoid(
        shape, radius, 2.0, position, 2,
        rel_position=True, rel_sizes=False)


# ======================================================================
def ellipsis(
        shape,
        semiaxes,
        position=0.5):
    """
    Generate a mask whose shape is an ellipsis.

    Args:
        shape (int|Iterable[int]): The shape of the mask in px.
        semiaxes (float|tuple[float]): The semiaxes of the ellipsis in px.
        position (float|Iterable[float]): The position of the center.
            Values are relative to the lowest edge.
            They are interpreted as relative to the shape.

    Returns:
        mask (np.ndarray): Array of boolean describing the geometrical object.

    Examples:
        >>> ellipsis(6, (2, 3))
        array([[False, False, False, False, False, False],
               [False,  True,  True,  True,  True, False],
               [ True,  True,  True,  True,  True,  True],
               [ True,  True,  True,  True,  True,  True],
               [False,  True,  True,  True,  True, False],
               [False, False, False, False, False, False]], dtype=bool)
        >>> ellipsis(6, (5, 3), 0)
        array([[ True,  True,  True,  True, False, False],
               [ True,  True,  True, False, False, False],
               [ True,  True,  True, False, False, False],
               [ True,  True,  True, False, False, False],
               [ True,  True, False, False, False, False],
               [ True, False, False, False, False, False]], dtype=bool)
    """
    return nd_superellipsoid(
        shape, semiaxes, 2.0, position, 2,
        rel_position=True, rel_sizes=False)


# ======================================================================
def polygon(
        shape,
        vertices,
        position=0.5):
    """
    Generate a mask whose shape is a simple polygon.

    Args:
        shape (int|Iterable[int]): The shape of the mask in px.
        vertices (Iterable[Iterable[float]): Coordinates of the vertices in px.
        position (float|Iterable[float]): The position of the center.
            Values are relative to the lowest edge.
            They are interpreted as relative to the shape.

    Returns:
        mask (np.ndarray): Array of boolean describing the geometrical object.
    """
    raise NotImplementedError


# ======================================================================
def cube(
        shape,
        side,
        position=0.5):
    """
    Generate a mask whose shape is a cube.

    Args:
        shape (int|Iterable[int]): The shape of the mask in px.
        side (float): The side of the cube in px.
        position (float|Iterable[float]): The position of the center.
            Values are relative to the lowest edge.
            They are interpreted as relative to the shape.

    Returns:
        mask (np.ndarray): Array of boolean describing the geometrical object.

    Examples:
        >>> cube(4, 2)
        array([[[False, False, False, False],
                [False, False, False, False],
                [False, False, False, False],
                [False, False, False, False]],
        <BLANKLINE>
               [[False, False, False, False],
                [False,  True,  True, False],
                [False,  True,  True, False],
                [False, False, False, False]],
        <BLANKLINE>
               [[False, False, False, False],
                [False,  True,  True, False],
                [False,  True,  True, False],
                [False, False, False, False]],
        <BLANKLINE>
               [[False, False, False, False],
                [False, False, False, False],
                [False, False, False, False],
                [False, False, False, False]]], dtype=bool)
    """
    return nd_cuboid(
        shape, side / 2.0, position, 3,
        rel_position=True, rel_sizes=False)


# ======================================================================
def cuboid(
        shape,
        semisides,
        position=0.5):
    """
    Generate a mask whose shape is a cuboid.

    Args:
        shape (int|Iterable[int]): The shape of the mask in px.
        semisides (tuple[float]): The semisides of the cuboid in px.
        position (float|Iterable[float]): The position of the center.
            Values are relative to the lowest edge.
            They are interpreted as relative to the shape.

    Returns:
        mask (np.ndarray): Array of boolean describing the geometrical object.

    Examples:
        >>> cuboid((3, 4, 6), (0.5, 2, 1))
        array([[[False, False, False, False, False, False],
                [False, False, False, False, False, False],
                [False, False, False, False, False, False],
                [False, False, False, False, False, False]],
        <BLANKLINE>
               [[False, False,  True,  True, False, False],
                [False, False,  True,  True, False, False],
                [False, False,  True,  True, False, False],
                [False, False,  True,  True, False, False]],
        <BLANKLINE>
               [[False, False, False, False, False, False],
                [False, False, False, False, False, False],
                [False, False, False, False, False, False],
                [False, False, False, False, False, False]]], dtype=bool)
    """
    return nd_cuboid(
        shape, semisides, position, 3,
        rel_position=True, rel_sizes=False)


# ======================================================================
def rhomboid(
        shape,
        semidiagonals,
        position=0.5):
    """
    Generate a mask whose shape is a rhomboid.

    Args:
        shape (int|Iterable[int]): The shape of the mask in px.
        semidiagonals (tuple[float]): The semidiagonals of the rhomboid in px.
        position (float|Iterable[float]): The position of the center.
            Values are relative to the lowest edge.
            They are interpreted as relative to the shape.

    Returns:
        mask (np.ndarray): Array of boolean describing the geometrical object.

    Examples:
        >>> rhomboid((3, 5, 7), (1, 1, 2))
        array([[[False, False, False, False, False, False, False],
                [False, False, False, False, False, False, False],
                [False, False, False,  True, False, False, False],
                [False, False, False, False, False, False, False],
                [False, False, False, False, False, False, False]],
        <BLANKLINE>
               [[False, False, False, False, False, False, False],
                [False, False, False,  True, False, False, False],
                [False,  True,  True,  True,  True,  True, False],
                [False, False, False,  True, False, False, False],
                [False, False, False, False, False, False, False]],
        <BLANKLINE>
               [[False, False, False, False, False, False, False],
                [False, False, False, False, False, False, False],
                [False, False, False,  True, False, False, False],
                [False, False, False, False, False, False, False],
                [False, False, False, False, False, False, False]]],\
 dtype=bool)
    """
    return nd_superellipsoid(
        shape, semidiagonals, 1.0, position, 3,
        rel_position=True, rel_sizes=False)


# ======================================================================
def sphere(
        shape,
        radius,
        position=0.5):
    """
    Generate a mask whose shape is a sphere.

    Args:
        shape (int|Iterable[int]): The shape of the mask in px.
        radius (float): The radius of the sphere in px.
        position (float|Iterable[float]): The position of the center.
            Values are relative to the lowest edge.
            They are interpreted as relative to the shape.

    Returns:
        mask (np.ndarray): Array of boolean describing the geometrical object.

    Examples:
        >>> sphere(3, 1)
        array([[[False, False, False],
                [False,  True, False],
                [False, False, False]],
        <BLANKLINE>
               [[False,  True, False],
                [ True,  True,  True],
                [False,  True, False]],
        <BLANKLINE>
               [[False, False, False],
                [False,  True, False],
                [False, False, False]]], dtype=bool)
        >>> sphere(5, 2)
        array([[[False, False, False, False, False],
                [False, False, False, False, False],
                [False, False,  True, False, False],
                [False, False, False, False, False],
                [False, False, False, False, False]],
        <BLANKLINE>
               [[False, False, False, False, False],
                [False,  True,  True,  True, False],
                [False,  True,  True,  True, False],
                [False,  True,  True,  True, False],
                [False, False, False, False, False]],
        <BLANKLINE>
               [[False, False,  True, False, False],
                [False,  True,  True,  True, False],
                [ True,  True,  True,  True,  True],
                [False,  True,  True,  True, False],
                [False, False,  True, False, False]],
        <BLANKLINE>
               [[False, False, False, False, False],
                [False,  True,  True,  True, False],
                [False,  True,  True,  True, False],
                [False,  True,  True,  True, False],
                [False, False, False, False, False]],
        <BLANKLINE>
               [[False, False, False, False, False],
                [False, False, False, False, False],
                [False, False,  True, False, False],
                [False, False, False, False, False],
                [False, False, False, False, False]]], dtype=bool)
    """
    return nd_superellipsoid(
        shape, radius, 2.0, position, 3,
        rel_position=True, rel_sizes=False)


# ======================================================================
def ellipsoid(
        shape,
        semiaxes,
        position=0.5):
    """
    Generate a mask whose shape is an ellipsoid.

    Args:
        shape (int|Iterable[int]): The shape of the mask in px.
        semiaxes (float|tuple[float]): The semiaxes of the ellipsoid in px.
        position (float|Iterable[float]): The position of the center.
            Values are relative to the lowest edge.
            They are interpreted as relative to the shape.

    Returns:
        mask (np.ndarray): Array of boolean describing the geometrical object.

    Examples:
        >>> ellipsoid(5, (1., 2., 1.5))
        array([[[False, False, False, False, False],
                [False, False, False, False, False],
                [False, False, False, False, False],
                [False, False, False, False, False],
                [False, False, False, False, False]],
        <BLANKLINE>
               [[False, False, False, False, False],
                [False, False, False, False, False],
                [False, False,  True, False, False],
                [False, False, False, False, False],
                [False, False, False, False, False]],
        <BLANKLINE>
               [[False, False,  True, False, False],
                [False,  True,  True,  True, False],
                [False,  True,  True,  True, False],
                [False,  True,  True,  True, False],
                [False, False,  True, False, False]],
        <BLANKLINE>
               [[False, False, False, False, False],
                [False, False, False, False, False],
                [False, False,  True, False, False],
                [False, False, False, False, False],
                [False, False, False, False, False]],
        <BLANKLINE>
               [[False, False, False, False, False],
                [False, False, False, False, False],
                [False, False, False, False, False],
                [False, False, False, False, False],
                [False, False, False, False, False]]], dtype=bool)
    """
    return nd_superellipsoid(
        shape, semiaxes, 2.0, position, 3,
        rel_position=True, rel_sizes=False)


# ======================================================================
def cylinder(
        shape,
        height,
        radius,
        axis=-1,
        position=0.5):
    """
    Generate a mask whose shape is a cylinder.

    Args:
        shape (int|Iterable[int]): The shape of the mask in px.
        height (float): The height of the cylinder in px.
        radius (float): The radius of the cylinder in px.
        axis (int): Orientation of the cylinder in the N-dim space.
        position (float|Iterable[float]): The position of the center.
            Values are relative to the lowest edge.
            They are interpreted as relative to the shape.

    Returns:
        mask (np.ndarray): Array of boolean describing the geometrical object.

    Examples:
        >>> cylinder(4, 2, 2, 0)
        array([[[False, False, False, False],
                [False, False, False, False],
                [False, False, False, False],
                [False, False, False, False]],
        <BLANKLINE>
               [[False,  True,  True, False],
                [ True,  True,  True,  True],
                [ True,  True,  True,  True],
                [False,  True,  True, False]],
        <BLANKLINE>
               [[False,  True,  True, False],
                [ True,  True,  True,  True],
                [ True,  True,  True,  True],
                [False,  True,  True, False]],
        <BLANKLINE>
               [[False, False, False, False],
                [False, False, False, False],
                [False, False, False, False],
                [False, False, False, False]]], dtype=bool)
    """
    n_dim = 3
    shape = mrt.utils.auto_repeat(shape, n_dim)
    position = mrt.utils.auto_repeat(position, n_dim)
    # generate base
    base_shape = tuple(
        dim for i, dim in enumerate(shape) if axis % n_dim != i)
    base_position = tuple(
        dim for i, dim in enumerate(position) if axis % n_dim != i)
    base = circle(base_shape, radius, base_position)
    # use n-dim function
    return nd_prism(
        base, shape[axis], axis, height, position[axis],
        rel_position=True, rel_sizes=False)


# ======================================================================
def polyhedron(
        shape,
        vertices,
        position=0.5):
    """
    Generate a mask whose shape is a simple polyhedron.

    Args:
        shape (int|Iterable[int]): The shape of the mask in px.
        vertices (Iterable[Iterable[float]): Coordinates of the vertices in px.
        position (float|Iterable[float]): The position of the center.
            Values are relative to the lowest edge.
            They are interpreted as relative to the shape.

    Returns:
        mask (np.ndarray): Array of boolean describing the geometrical object.
    """
    raise NotImplementedError


# ======================================================================
def extrema_to_semisizes_position(minima, maxima, num=None):
    """
    Compute semisizes and positions from extrema.

    Args:
        minima (float|Iterable[float]): The minimum extrema.
            These are the lower bound of the object.
            If both `minima` and `maxima` are Iterable, their length must
            match.
        maxima (float|Iterable[float]): The maximum extrema.
            These are the upper bound of the object.
            If both `minima` and `maxima` are Iterable, their length must
            match.
        num (int|None): The number of extrema.
            If None, it is guessed from the `minima` and/or `maxima`.
            If int, must match the length of `minima` and/or `maxima`
            (if they are Iterable).

    Returns:
        result (tuple): The tuple
            contains:
             - semisizes (Iterable[float]): The N-dim cuboid semisides sizes.
             - position (Iterable[float]): The position of the center.
               Values are relative to the lowest edge.

    Examples:
        >>> extrema_to_semisizes_position(0.1, 0.5)
        ([0.2], [0.3])
        >>> extrema_to_semisizes_position(0.2, 0.5)
        ([0.15], [0.35])
        >>> extrema_to_semisizes_position(0.2, 0.5, 2)
        ([0.15, 0.15], [0.35, 0.35])
        >>> extrema_to_semisizes_position((0.2, 0.1), 0.5)
        ([0.15, 0.2], [0.35, 0.3])
        >>> extrema_to_semisizes_position((0.1, 0.2), (0.9, 0.5))
        ([0.4, 0.15], [0.5, 0.35])
    """
    if not num:
        num = mrt.utils.combine_iter_len((minima, maxima, num))
    # check compatibility of given parameters
    minima = mrt.utils.auto_repeat(minima, num, check=True)
    maxima = mrt.utils.auto_repeat(maxima, num, check=True)
    semisizes, position = [], []
    for min_val, max_val in zip(minima, maxima):
        semisizes.append((max_val - min_val) / 2.0)
        position.append((max_val + min_val) / 2.0)
    return semisizes, position


# ======================================================================
def nd_cuboid(
        shape,
        semisizes=0.5,
        position=0.5,
        n_dim=None,
        rel_position=True,
        rel_sizes=True):
    """
    Generate a mask whose shape is an N-dim cuboid (hypercuboid).

    The cartesian equations are:

    .. math::
        \\sum[\\abs(\\frac{x_n}{a_n})^{\\inf}] < 1.0

    where :math:`n` runs through the dims, :math:`x` are the cartesian
    coordinate, :math:`a` are the semi-sizes (semi-axes) and
    :math:`\\inf` is infinity.

    Args:
        shape (int|Iterable[int]): The shape of the mask in px.
        semisizes (float|Iterable[float]): The N-dim cuboid semisides sizes.
            The values interpretation depend on `rel_sizes`.
        position (float|Iterable[float]): The position of the center.
            Values are relative to the lowest edge.
            The values interpretation depend on `rel_position`.
        n_dim (int|None): The number of dimensions.
            If None, the number of dims is guessed from the other parameters.
        rel_position (bool): Interpret positions as relative values.
            If True, position values are interpreted as relative,
            i.e. they are scaled for `shape` values.
            Otherwise, they are interpreted as absolute (in px).
            Uses `pymrt.utils.grid_coord()` internally.
        rel_sizes (bool): Interpret sizes as relative values.
            If True, `semisizes` values are interpreted as relative,
            i.e. they are scaled for `shape` values.
            Otherwise, they are interpreted as absolute (in px).
            Uses `pymrt.geometry.rel2abs()` internally.

    Returns:
        mask (np.ndarray): Array of boolean describing the geometrical object.
    """
    if not n_dim:
        n_dim = mrt.utils.combine_iter_len((shape, position, semisizes))
    # check compatibility of given parameters
    shape = mrt.utils.auto_repeat(shape, n_dim, check=True)
    position = mrt.utils.auto_repeat(position, n_dim, check=True)
    semisizes = mrt.utils.auto_repeat(semisizes, n_dim, check=True)
    # fix relative units
    if rel_sizes:
        semisizes = rel2abs(shape, semisizes)
    position = mrt.utils.grid_coord(
        shape, position, is_relative=rel_position, use_int=False)
    # create the mask
    mask = np.ones(shape, dtype=bool)
    for x_i, semisize in zip(position, semisizes):
        mask *= (np.abs(x_i) <= semisize)
    return mask


# ======================================================================
def nd_superellipsoid(
        shape,
        semisizes=0.5,
        indexes=2,
        position=0.5,
        n_dim=None,
        rel_position=True,
        rel_sizes=True):
    """
    Generate a mask whose shape is an N-dim superellipsoid.

    The cartesian equations are:

    .. math::
        \\sum[\\abs(\\frac{x_n}{a_n})^{k_n}] < 1.0

    where :math:`n` runs through the dims, :math:`x` are the cartesian
    coordinate, :math:`a` are the semi-sizes (semi-axes) and
    :math:`k` are the indexes.

    When the index is 2, an ellipsoid (hyperellipsoid) is generated.

    Args:
        shape (int|Iterable[int]): The shape of the mask in px.
        semisizes (float|Iterable[float]): The N-dim superellipsoid axes sizes.
            The values interpretation depend on `rel_sizes`.
        position (float|Iterable[float]): The position of the center.
            Values are relative to the lowest edge.
            The values interpretation depend on `rel_position`.
        indexes (float|tuple[float]): The exponent of the summed terms.
            If 2, generates n-dim ellipsoids.
        n_dim (int|None): The number of dimensions.
            If None, the number of dims is guessed from the other parameters.
        rel_position (bool): Interpret positions as relative values.
            If True, position values are interpreted as relative,
            i.e. they are scaled for `shape` using `pymrt.utils.grid_coord()`.
            Otherwise, they are interpreted as absolute (in px).
        rel_sizes (bool): Interpret sizes as relative values.
            If True, `semisizes` values are interpreted as relative,
            i.e. they are scaled for `shape` using `pymrt.utils.grid_coord()`.
            Otherwise, they are interpreted as absolute (in px).

    Returns:
        mask (np.ndarray): Array of boolean describing the geometrical object.
    """
    if not n_dim:
        n_dim = mrt.utils.combine_iter_len(
            (shape, position, semisizes, indexes))

    # check compatibility of given parameters
    shape = mrt.utils.auto_repeat(shape, n_dim, check=True)
    position = mrt.utils.auto_repeat(position, n_dim, check=True)
    semisizes = mrt.utils.auto_repeat(semisizes, n_dim, check=True)
    indexes = mrt.utils.auto_repeat(indexes, n_dim, check=True)

    # get correct position
    if rel_sizes:
        semisizes = rel2abs(shape, semisizes)
    # print('Semisizes: {}'.format(semisizes))  # DEBUG
    # print('Shape: {}'.format(shape))  # DEBUG
    position = mrt.utils.grid_coord(
        shape, position, is_relative=rel_position, use_int=False)
    # print('Position: {}'.format(position))  # DEBUG

    # create the mask
    mask = np.zeros(shape, dtype=float)
    for x_i, semisize, index in zip(position, semisizes, indexes):
        mask += (np.abs(x_i / semisize) ** index)
    mask = mask <= 1.0

    return mask


# ======================================================================
def nd_prism(
        base,
        extra_dim,
        axis=-1,
        size=0.5,
        position=0.5,
        rel_position=True,
        rel_sizes=True):
    """
    Generate a mask whose shape is a N-dim prism.

    Args:
        base (np.ndarray): Base (N-1)-dim mask to stack as prism.
        extra_dim (int): Size of the new dimension to be added.
        axis (int): Orientation of the prism in the N-dim space.
        size (float): The size of the prism height.
            The values interpretation depend on `rel_size`.
        position (float): The relative position of the center.
            Values are relative to the lowest edge.
            The values interpretation depend on `rel_position`.
            This setting only affects the extra shape dimension.
        rel_position (bool): Interpret positions as relative value.
            If True, position values are interpreted as relative,
            i.e. they are scaled for `shape` using `pymrt.utils.grid_coord()`.
            Otherwise, they are interpreted as absolute (in px).
        rel_sizes (bool): Interpret sizes as relative value.
            If True, `size` values are interpreted as relative,
            i.e. they are scaled for `shape` using `pymrt.utils.grid_coord()`.
            Otherwise, they are interpreted as absolute (in px).

    Returns:
        mask (np.ndarray): Array of boolean describing the geometrical object.
    """
    n_dim = base.ndim + 1
    if axis > n_dim:
        raise ValueError(
            'axis of orientation must not exceed the number of dimensions')
    # get correct position
    if rel_sizes:
        size = rel2abs((extra_dim,), size)
    position = mrt.utils.grid_coord(
        (extra_dim,), (position,), is_relative=rel_position, use_int=False)[0]
    extra_mask = np.abs(position) <= (size / 2.0)
    # calculate mask shape
    shape = (
        base.shape[:axis] + (extra_dim,) + base.shape[axis:])
    # create indefinite prism
    mask = np.zeros(shape, dtype=bool)
    for i in range(extra_dim):
        if extra_mask[i]:
            index = [slice(None)] * n_dim
            index[axis] = i
            mask[tuple(index)] = base
    return mask


# ======================================================================
def nd_superellipsoidal_prism(
        shape,
        axis=-1,
        semisizes=0.5,
        indexes=2,
        position=0.5,
        n_dim=None,
        rel_position=True,
        rel_sizes=True):
    # todo: ensure n_dim is not none
    # todo: nd_superellipsoidal_come is the same except nd_prims -> nd_cone
    if not n_dim:
        n_dim = mrt.utils.combine_iter_len((shape, position, semisizes))
    axis = axis % n_dim
    # separate shape/dims
    base_shape = tuple(dim for i, dim in enumerate(shape) if i != axis)
    extra_dim = shape[axis]
    # separate position
    position = mrt.utils.auto_repeat(position, n_dim, False, True)
    base_position = tuple(x for i, x in enumerate(position) if i != axis)
    extra_position = position[axis]
    # separate semisizes
    semisizes = mrt.utils.auto_repeat(semisizes, n_dim, False, True)
    base_semisizes = tuple(x for i, x in enumerate(semisizes) if i != axis)
    extra_semisize = semisizes[axis]
    # generate prism base
    base = nd_superellipsoid(
        base_shape, base_semisizes, indexes, base_position, n_dim - 1,
        rel_position, rel_sizes)
    # generate final prism
    mask = nd_prism(
        base, extra_dim, axis, extra_semisize * 2, extra_position,
        rel_position, rel_sizes)
    return mask


# ======================================================================
def nd_cone(
        base,
        extra_dim,
        axis=-1,
        size=0.5,
        position=0.5,
        rel_position=True,
        rel_sizes=True):
    """
    Generate a mask whose shape is a N-dim cone.

    Args:
        base (np.ndarray): Base (N-1)-dim mask to stack as prism.
        extra_dim (int): Size of the new dimension to be added.
        axis (int): Orientation of the prism in the N-dim space.
        size (float): The size of the prism height.
            The values interpretation depend on `rel_size`.
        position (float): The relative position of the center.
            Values are relative to the lowest edge.
            The values interpretation depend on `rel_position`.
            This setting only affects the extra shape dimension.
        rel_position (bool): Interpret positions as relative value.
            If True, position values are interpreted as relative,
            i.e. they are scaled for `shape` using `pymrt.utils.grid_coord()`.
            Otherwise, they are interpreted as absolute (in px).
        rel_sizes (bool): Interpret sizes as relative value.
            If True, `size` values are interpreted as relative,
            i.e. they are scaled for `shape` using `pymrt.utils.grid_coord()`.
            Otherwise, they are interpreted as absolute (in px).

    Returns:
        mask (np.ndarray): Array of boolean describing the geometrical object.
    """
    n_dim = base.ndim + 1
    if axis > n_dim:
        raise ValueError(
            'axis of orientation must not exceed the number of dimensions')
    # get correct position
    if rel_sizes:
        size = rel2abs((extra_dim,), size)
    position = mrt.utils.grid_coord(
        (extra_dim,), (position,), is_relative=rel_position, use_int=False)[0]
    extra_mask = np.abs(position) <= (size / 2.0)
    # calculate mask shape
    shape = (
        base.shape[:axis] + (extra_dim,) + base.shape[axis:])
    # create indefinite prism
    mask = np.zeros(shape, dtype=bool)
    for i in range(extra_dim):
        if extra_mask[i]:
            index = [slice(None)] * n_dim
            index[axis] = i
            mask[tuple(index)] = base
    return mask


# ======================================================================
def nd_superellipsoidal_cone(
        shape,
        semisizes=0.5,
        indexes=2,
        axis=-1,
        size=0.5,
        position=0.5,
        n_dim=None,
        rel_position=True,
        rel_sizes=True):
    raise NotImplementedError


# ======================================================================
def nd_gradient(
        gen_ranges,
        dtype=float,
        dense=False,
        generators=np.linspace,
        generators_kws=None):
    """
    Generate N-dimensional gradient data arrays.

    This is useful for generating simulation data patterns.

    Args:
        gen_ranges (Iterable[Iterable]): Generator ranges and numbers.
            An Iterable of size 3 Iterables: (start, stop, number) where
            start and stop are the extrema of the range to cover, and number
            indicate the number of samples.
        dtype (data-type): Desired output data-type.
            See `np.ndarray()` for more.
        dense (bool): Generate dense results.
            If True, the results have full sizes.
            Otherwise, constant dimensions are keept to size of 1,
            which ensures them being broadcast-safe.
        generators (callable|Iterable[callable]): The range generator(s).
            A generator must have signature:
            f(any, any, int, **kws) -> Iterable
            If callable, the same generator is used for all dimensions.
            If Iterable, each generator generate the data corresponding to its
            position in the Iterable, and its length must match the length of
            `gen_ranges`.
        generators_kws (dict|Iterable[dict|None]|None): Keyword arguments.
            If None, no arguments are passed to the generator.
            If dict, the same object is passed to all instances of
            `generators`.
            If Iterable, each dict (or None) is applied to the corresponding
            generator in `generators`, and its length must match the length of
            `generators`.


    Returns:
        arrs (list[np.ndarray]): The broadcast-safe n-dim gradient arrays.
            The actual shape depend on `dense`.

    Examples:
        >>> for arr in nd_gradient(((0, 1, 2), (-2, 2, 2))):
        ...     print(arr)
        [[ 0.]
         [ 1.]]
        [[-2.  2.]]
        >>> for arr in nd_gradient(((0, 1, 2), (-2, 2, 2)), dense=True):
        ...     print(arr)
        [[ 0.  0.]
         [ 1.  1.]]
        [[-2.  2.]
         [-2.  2.]]
        >>> for arr in nd_gradient(((0, 1, 2), (-2, 2, 2)), int, True):
        ...     print(arr)
        [[ 0.  0.]
         [ 1.  1.]]
        [[-2.  2.]
         [-2.  2.]]
        >>> for arr in nd_gradient(
        ...         ((0, 1, 2), (-2, 2, 2)), float, True, np.logspace):
        ...     print(arr)
        [[  1.   1.]
         [ 10.  10.]]
        [[  1.00000000e-02   1.00000000e+02]
         [  1.00000000e-02   1.00000000e+02]]
        >>> for arr in nd_gradient(
        ...         ((0, 1, 2), (-2, 2, 2)), float, True,
        ...         (np.linspace, np.logspace)):
        ...     print(arr)
        [[ 0.  0.]
         [ 1.  1.]]
        [[  1.00000000e-02   1.00000000e+02]
         [  1.00000000e-02   1.00000000e+02]]
        >>> for arr in nd_gradient(
        ...         ((0, 1, 2), (-1, 1, 3), (-2, 2, 2)), int, True):
        ...     print(arr)
        [[[ 0.  0.]
          [ 0.  0.]
          [ 0.  0.]]
        <BLANKLINE>
         [[ 1.  1.]
          [ 1.  1.]
          [ 1.  1.]]]
        [[[-1. -1.]
          [ 0.  0.]
          [ 1.  1.]]
        <BLANKLINE>
         [[-1. -1.]
          [ 0.  0.]
          [ 1.  1.]]]
        [[[-2.  2.]
          [-2.  2.]
          [-2.  2.]]
        <BLANKLINE>
         [[-2.  2.]
          [-2.  2.]
          [-2.  2.]]]
    """
    num_gens = len(gen_ranges)
    generators = mrt.utils.auto_repeat(
        generators, num_gens, check=True)
    generators_kws = mrt.utils.auto_repeat(
        generators_kws, num_gens, check=True)

    shape = tuple(num for start, stop, num in gen_ranges)

    arrs = []
    for i, ((start, stop, num), generator, generator_kws) in \
            enumerate(zip(gen_ranges, generators, generators_kws)):
        if generator_kws is None:
            generator_kws = {}
        arr = np.array(
            generator(start, stop, num, **generator_kws), dtype=dtype)
        new_shape = tuple(n if j == i else 1 for j, n in enumerate(shape))
        arr = arr.reshape(new_shape)
        if dense:
            arr = np.zeros(shape) + arr
        arrs.append(arr)
    return arrs


# ======================================================================
def nd_dirac_delta(
        shape,
        position=0.5,
        value=np.inf,
        n_dim=None):
    """
    Generate an approximation of Dirac's Delta function.

    Args:
        shape ():
        position ():
        value ():
        n_dim ():

    Returns:

    Examples:
        >>> nd_dirac_delta((5, 5), 0.5, 1)
        array([[ 0.,  0.,  0.,  0.,  0.],
               [ 0.,  0.,  0.,  0.,  0.],
               [ 0.,  0.,  1.,  0.,  0.],
               [ 0.,  0.,  0.,  0.,  0.],
               [ 0.,  0.,  0.,  0.,  0.]])
    """
    if not n_dim:
        n_dim = mrt.utils.combine_iter_len((shape, position))

    # check compatibility of given parameters
    shape = mrt.utils.auto_repeat(shape, n_dim, check=True)
    position = mrt.utils.auto_repeat(position, n_dim, check=True)

    origin = mrt.utils.coord(shape, position, use_int=True)

    mask = np.zeros(shape)
    mask[[slice(i, i + 1) for i in origin]] = value
    return mask


# ======================================================================
def nd_polytope(
        shape,
        vertices,
        position=0.5,
        n_dim=None,
        rel_position=True,
        rel_sizes=True):
    """
    Generate a mask whose shape is a simple polytope.

    Args:
        shape (int|Iterable[int]): The shape of the mask in px.
        vertices (Iterable[Iterable[float]): Coordinates of the vertices.
            The values interpretation depend on `rel_sizes`.
        position (float|Iterable[float]): The position of the center.
            Values are relative to the lowest edge.
            The values interpretation depend on `rel_position`.
        n_dim (int|None): The number of dimensions.
            If None, the number of dims is guessed from the other parameters.
        rel_position (bool): Interpret positions as relative values.
            If True, position values are interpreted as relative,
            i.e. they are scaled for `shape` using `mrt.utils.grid_coord()`.
            Otherwise, they are interpreted as absolute (in px).
        rel_sizes (bool): Interpret sizes as relative values.
            If True, `vertices` values are interpreted as relative,
            i.e. they are scaled for `shape` using `mrt.utils.grid_coord()`.
            Otherwise, they are interpreted as absolute (in px).

    Returns:
        mask (np.ndarray): Array of boolean describing the geometrical object.
    """
    raise NotImplementedError


# ======================================================================
def apply_mask(
        arr,
        mask,
        borders=None,
        background=0.0,
        unsqueeze=True):
    """
    Apply a mask to an array.

    Note: this will not produced a masked array `numpy.ma` object.

    Args:
        arr (np.ndarray): The input array.
        mask (np.ndarray): The mask array.
            The shape of `arr` and `mask` must be identical, broadcastable
            through `np.broadcast_to()`, or unsqueezable using
            `pymrt.utils.unsqueeze()`.
        borders (int|float|tuple[int|float]|None): The border size(s).
            If None, the border is not modified.
            Otherwise, a border is added to the masked array.
            If int, this is in units of pixels.
            If float, this is proportional to the initial array shape.
            If int or float, uses the same value for all dimensions,
            unless `unsqueezing` is set to True, in which case, the same value
            is used only for non-singletons, while 0 is used for singletons.
            If Iterable, the size must match `arr` dimensions.
            If 'use_longest' is True, use the longest dimension for the
            calculations.
        background (int|float): The value used for masked-out pixels.
        unsqueeze (bool): Unsqueeze mask to input.
            If True, use `pymrt.utils.unsqueeze()` on mask.
            Only effective when `arr` and `mask` shapes do not match and
            are not already broadcastable.
            Otherwise, shapes must match or be broadcastable.

    Returns:
        arr (np.ndarray): The output array.
            Values outside of the mask are set to background.
            Array shape may have changed (depending on `borders`).

    Raises:
        ValueError: If the mask and array shapes are not compatible.

    See Also:
        frame()
    """
    mask = mask.astype(bool)
    if arr.ndim > mask.ndim and unsqueeze:
        old_shape = mask.shape
        mask = mrt.utils.unsqueeze(mask, shape=arr.shape)
        if isinstance(borders, (int, float)):
            borders = [borders if dim != 1 else 0 for dim in mask.shape]
        elif borders is not None and len(borders) == len(old_shape):
            borders = list(
                mrt.utils.replace_iter(
                    mask.shape, lambda x: x == 1, borders))
    arr = arr.copy()
    if arr.shape != mask.shape:
        mask = np.broadcast_to(mask, arr.shape)
    if arr.shape == mask.shape:
        arr[~mask] = background
        if borders is not None:
            container = sp.ndimage.find_objects(mask.astype(int))[0]
            if container:
                arr = arr[container]
            arr = frame(arr, borders, background)
    else:
        raise ValueError(
            'Cannot apply mask shaped `{}` to array shaped `{}`.'.format(
                mask.shape, arr.shape))
    return arr


# ======================================================================
def frame(
        arr,
        borders=0.05,
        background=0.0,
        use_longest=True):
    """
    Add a background frame to an array specifying the borders.

    Args:
        arr (np.ndarray): The input array.
        borders (int|float|Iterable[int|float]): The border size(s).
            If int, this is in units of pixels.
            If float, this is proportional to the initial array shape.
            If int or float, uses the same value for all dimensions.
            If Iterable, the size must match `arr` dimensions.
            If 'use_longest' is True, use the longest dimension for the
            calculations.
        background (int|float): The background value to be used for the frame.
        use_longest (bool): Use longest dimension to get the border size.

    Returns:
        result (np.ndarray): The result array with added borders.

    See Also:
        reframe()
    """
    borders = mrt.utils.auto_repeat(borders, arr.ndim)
    if any(borders) < 0:
        raise ValueError('relative border cannot be negative')
    if isinstance(borders[0], float):
        if use_longest:
            dim = max(arr.shape)
            borders = [round(border * dim) for border in borders]
        else:
            borders = [
                round(border * dim) for dim, border in zip(arr.shape, borders)]
    result = np.full(
        [dim + 2 * border for dim, border in zip(arr.shape, borders)],
        background, dtype=arr.dtype)
    inner = [
        slice(border, border + dim, None)
        for dim, border in zip(arr.shape, borders)]
    result[inner] = arr
    return result


# ======================================================================
def reframe(
        arr,
        new_shape,
        position=0.5,
        background=0.0):
    """
    Add a frame to an array by centering the input array into a new shape.

    Args:
        arr (np.ndarray): The input array.
        new_shape (int|Iterable[int]): The shape of the output array.
            If int, uses the same value for all dimensions.
            If Iterable, the size must match `arr` dimensions.
            Additionally, each value of `new_shape` must be greater than or
            equal to the corresponding dimensions of `arr`.
        position (int|float|Iterable[int|float]): Position within new shape.
            Determines the position of the array within the new shape.
            If int or float, it is considered the same in all dimensions,
            otherwise its length must match the number of dimensions of the
            array.
            If int or Iterable of int, the values are absolute and must be
            less than or equal to the difference between the shape of the array
            and the new shape.
            If float or Iterable of float, the values are relative and must be
            in the [0, 1] range.
        background (int|float): The background value to be used for the frame.

    Returns:
        result (np.ndarray): The result array with added borders.

    Raises:
        IndexError: input and output shape sizes must match.
        ValueError: output shape cannot be smaller than the input shape.

    See Also:
        frame()

    Examples:
        >>> arr = np.ones((2, 3))
        >>> reframe(arr, (4, 5))
        array([[ 0.,  0.,  0.,  0.,  0.],
               [ 0.,  1.,  1.,  1.,  0.],
               [ 0.,  1.,  1.,  1.,  0.],
               [ 0.,  0.,  0.,  0.,  0.]])
        >>> reframe(arr, (4, 5), 0)
        array([[ 1.,  1.,  1.,  0.,  0.],
               [ 1.,  1.,  1.,  0.,  0.],
               [ 0.,  0.,  0.,  0.,  0.],
               [ 0.,  0.,  0.,  0.,  0.]])
        >>> reframe(arr, (4, 5), (2, 0))
        array([[ 0.,  0.,  0.,  0.,  0.],
               [ 0.,  0.,  0.,  0.,  0.],
               [ 1.,  1.,  1.,  0.,  0.],
               [ 1.,  1.,  1.,  0.,  0.]])
        >>> reframe(arr, (4, 5), (0.0, 1.0))
        array([[ 0.,  0.,  1.,  1.,  1.],
               [ 0.,  0.,  1.,  1.,  1.],
               [ 0.,  0.,  0.,  0.,  0.],
               [ 0.,  0.,  0.,  0.,  0.]])
    """
    new_shape = mrt.utils.auto_repeat(new_shape, arr.ndim, check=True)
    position = mrt.utils.auto_repeat(position, arr.ndim, check=True)
    if any([old > new for old, new in zip(arr.shape, new_shape)]):
        raise ValueError('new shape cannot be smaller than the old one.')
    position = [
        int(round((new - old) * x_i)) if isinstance(x_i, float) else x_i
        for old, new, x_i in zip(arr.shape, new_shape, position)]
    if any([old + x_i > new
            for old, new, x_i in zip(arr.shape, new_shape, position)]):
        raise ValueError(
            'Incompatible `new_shape`, `array shape` and `position`.')
    result = np.full(new_shape, background)
    inner = [
        slice(offset, offset + dim, None)
        for dim, offset in zip(arr.shape, position)]
    result[inner] = arr
    return result


# ======================================================================
def multi_reframe(
        arrs,
        new_shape=None,
        background=0.0,
        dtype=None):
    """
    Reframe arrays (by adding border) to match the same shape.

    Note that:
     - uses 'reframe' under the hood;
     - the sampling / resolution / voxel size will NOT change;
     - the support space / field-of-view will change.

    Args:
        arrs (Iterable[np.ndarray]): The input arrays.
        new_shape (Iterable[int]): The new base shape of the arrays.
        background (int|float|complex): The background value for the frame.
        dtype (data-type): Desired output data-type.
            If None, its guessed from dtype of arrs.
            See `np.ndarray()` for more.

    Returns:
        result (np.ndarray): The output array.
            It contains all reframed arrays from `arrs`, through the last dim.
            The shape of this array is `new_shape` + `len(arrs)`.
    """
    # calculate new shape
    if new_shape is None:
        shapes = [arr.shape for arr in arrs]
        new_shape = [1] * max([len(shape) for shape in shapes])
        shape_arr = np.ones((len(shapes), len(new_shape))).astype(np.int)
        for i, shape in enumerate(shapes):
            shape_arr[i, :len(shape)] = np.array(shape)
        new_shape = tuple(
            max(*list(shape_arr[:, i]))
            for i in range(len(new_shape)))

    if dtype is None:
        # dtype = functools.reduce(
        #     (lambda x, y: np.promote_types(x, y.dtype)), arrs)
        dtype = bool
        for arr in arrs:
            dtype = np.promote_types(dtype, arr.dtype)

    result = np.array(new_shape + (len(arrs),), dtype=dtype)
    for i, arr in enumerate(arrs):
        # ratio should not be kept: keep_ratio_method=None
        result[..., i] = reframe(arr, new_shape, background=background)
    return result


# ======================================================================
def zoom_prepare(
        zoom_factors,
        shape,
        extra_dim=True,
        fill_dim=True):
    """
    Prepare the zoom and shape tuples to allow for non-homogeneous shapes.

    Args:
        zoom_factors (float|tuple[float]): The zoom factors for each
        directions.
        shape (int|Iterable[int]): The shape of the array to operate with.
        extra_dim (bool): Force extra dimensions in the zoom parameters.
        fill_dim (bool): Dimensions not specified are left untouched.

    Returns:
        zoom (tuple[float]): The zoom factors for each directions.
        shape (int|Iterable[int]): The shape of the array to operate with.
    """
    zoom_factors = list(mrt.utils.auto_repeat(zoom_factors, len(shape)))
    if extra_dim:
        shape = list(shape) + [1] * (len(zoom_factors) - len(shape))
    else:
        zoom_factors = zoom_factors[:len(shape)]
    if fill_dim and len(zoom_factors) < len(shape):
        zoom_factors[len(zoom_factors):] = \
            [1.0] * (len(shape) - len(zoom_factors))
    return zoom_factors, shape


# ======================================================================
def shape2zoom(
        old_shape,
        new_shape,
        aspect=None):
    """
    Calculate zoom (or conversion) factor between two shapes.

    Args:
        old_shape (int|Iterable[int]): The shape of the source array.
        new_shape (int|Iterable[int]): The target shape of the array.
        aspect (callable|None): Function for the manipulation of the zoom.
            Signature: aspect(Iterable[float]) -> float.
            None to leave the zoom unmodified. If specified, the function is
            applied to zoom factors tuple for fine tuning of the aspect.
            Particularly, to obtain specific aspect ratio results:
             - 'min': image strictly contained into new shape
             - 'max': new shape strictly contained into image

    Returns:
        zoom (tuple[float]): The zoom factors for each directions.
    """
    if len(old_shape) != len(new_shape):
        raise IndexError('length of tuples must match')
    zoom_factors = [new / old for old, new in zip(old_shape, new_shape)]
    if aspect:
        zoom_factors = [aspect(zoom_factors)] * len(zoom_factors)
    return zoom_factors


# ======================================================================
def zoom(
        arr,
        factors,
        window=None,
        interp_order=0,
        extra_dim=True,
        fill_dim=True):
    """
    Zoom the array with a specified magnification factor.

    Args:
        arr (np.ndarray): The input array.
        factors (int|float|Iterable[int|float]): The zoom factor(s).
            If int or float, uses isotropic factor along all axes.
            If Iterable, its size must match the number of dims of `arr`.
            Values larger than 1 increase `arr` size along the axis.
            Values smaller than 1 decrease `arr` size along the axis.
        window (int|Iterable[int]|None): Uniform pre-filter window size.
            This is the size of the window for the uniform filter using
            `sp.ndimage.uniform_filter()`.
            If Iterable, its size must match the number of dims of `arr`.
            If int, uses an isotropic window with the specified size.
            If None, the window is calculated automatically from the `zoom`
            parameter.
        interp_order (int): Order of the spline interpolation.
            0: nearest. Accepted range: [0, 5].
        extra_dim (bool): Force extra dimensions in the zoom parameters.
        fill_dim (bool): Dimensions not specified are left untouched.

    Returns:
        result (np.ndarray): The output array.

    See Also:
        geometry.zoom
    """
    factors, shape = zoom_prepare(factors, arr.shape, extra_dim, fill_dim)
    if window is None:
        window = [round(1.0 / (2.0 * x)) for x in factors]
    arr = sp.ndimage.uniform_filter(arr, window)
    arr = sp.ndimage.zoom(
        arr.reshape(shape), factors, order=interp_order)
    return arr


# ======================================================================
def resample(
        arr,
        new_shape,
        aspect=None,
        window=None,
        interp_order=0,
        extra_dim=True,
        fill_dim=True):
    """
    Reshape the array to a new shape (different resolution / pixel size).

    Args:
        arr (np.ndarray): The input array.
        new_shape (Iterable[int|None]): New dimensions of the array.
        aspect (callable|Iterable[callable]|None): Zoom shape manipulation.
            Useful for obtaining specific aspect ratio effects.
            This is passed to `pymrt.geometry.shape2zoom()`.
        window (int|Iterable[int]|None): Uniform pre-filter window size.
            This is the size of the window for the uniform filter using
            `sp.ndimage.uniform_filter()`.
            If Iterable, its size must match the number of dims of `arr`.
            If int, uses an isotropic window with the specified size.
            If None, the window is calculated automatically from `new_shape`.
        interp_order (int|None): Order of the spline interpolation.
            0: nearest. Accepted range: [0, 5].
        extra_dim (bool): Force extra dimensions in the zoom parameters.
        fill_dim (bool): Dimensions not specified are left untouched.

    Returns:
        arr (np.ndarray): The output array.

    See Also:
        geometry.zoom
    """
    factors = shape2zoom(arr.shape, new_shape, aspect)
    factors, shape = zoom_prepare(
        factors, arr.shape, extra_dim, fill_dim)
    arr = zoom(arr, factors, window=window, interp_order=interp_order)
    return arr


# ======================================================================
def multi_resample(
        arrs,
        new_shape=None,
        lossless=False,
        window=None,
        interp_order=0,
        extra_dim=True,
        fill_dim=True,
        dtype=None):
    """
    Resample arrays to match the same shape.

    Note that:
     - uses 'geometry.resample()' internally;
     - the sampling / resolution / voxel size will change;
     - the support space / field-of-view will NOT change.

    Args:
        arrs (Iterable[np.ndarray]): The input arrays,
        new_shape (Iterable[int]): The new base shape of the arrays.
        lossless (bool): allow for lossy resampling.
        window (int|Iterable[int]|None): Uniform pre-filter window size.
            This is the size of the window for the uniform filter using
            `sp.ndimage.uniform_filter()`.
            If Iterable, its size must match the number of dims of `arr`.
            If int, uses an isotropic window with the specified size.
            If None, the window is calculated automatically from `new_shape`.
        interp_order (int|None): Order of the spline interpolation.
            0: nearest. Accepted range: [0, 5].
        extra_dim (bool): Force extra dimensions in the zoom parameters.
        fill_dim (bool): Dimensions not specified are left untouched.
        dtype (data-type): Desired output data-type.
            If None, its guessed from dtype of arrs.
            See `np.ndarray()` for more.

    Returns:
        result (np.ndarray): The output array.
            It contains all reshaped arrays from `arrs`, through the last dim.
            The shape of this array is `new_shape` + `len(arrs)`.
    """
    # calculate new shape
    if new_shape is None:
        shapes = [arr.shape for arr in arrs]
        new_shape = [1] * max([len(shape) for shape in shapes])
        shape_arr = np.ones((len(shapes), len(new_shape))).astype(np.int)
        for i, shape in enumerate(shapes):
            shape_arr[i, :len(shape)] = np.array(shape)
        combiner = mrt.utils.lcm if lossless else max
        new_shape = tuple(
            combiner(*list(shape_arr[:, i]))
            for i in range(len(new_shape)))
    else:
        new_shape = tuple(new_shape)

    # resample images
    if lossless:
        interp_order = 0
        window = None

    if dtype is None:
        # dtype = functools.reduce(
        #     (lambda x, y: np.promote_types(x, y.dtype)), arrs)
        dtype = bool
        for arr in arrs:
            dtype = np.promote_types(dtype, arr.dtype)

    result = np.array(new_shape + (len(arrs),), dtype=dtype)
    for i, arr in enumerate(arrs):
        # ratio should not be kept: keep_ratio_method=None
        result[..., i] = resample(
            arr, new_shape, aspect=None, window=window,
            interp_order=interp_order, extra_dim=extra_dim, fill_dim=fill_dim)
    return result


# ======================================================================
def decode_affine(
        affine):
    """
    Decompose the affine matrix into a linear transformation and a translation.

    Args:
        affine (np.ndarray): The (N+1)-sized affine square matrix.

    Returns:
        linear (np.ndarray): The N-sized linear square matrix.
        shift (np.ndarray): The shift along each axis in px.
    """
    dims = affine.shape
    linear = affine[:dims[0] - 1, :dims[1] - 1]
    shift = affine[:-1, -1]
    return linear, shift


# ======================================================================
def encode_affine(
        linear,
        shift):
    """
    Combine a linear transformation and a translation into the affine matrix.

    PArgs:
        linear (np.ndarray): The N-sized linear square matrix.
        shift (np.ndarray): The shift along each axis in px.

    Returns:
        affine (np.ndarray): The (N+1)-sized affine square matrix.
    """
    dims = linear.shape
    affine = np.eye(dims[0] + 1)
    affine[:dims[0], :dims[1]] = linear
    affine[:-1, -1] = shift
    return affine


# ======================================================================
def num_angles_from_dim(n_dim):
    """
    Calculate the complete number of angles given the dimension.

    Given the dimension of an array, calculate the number of all possible
    cartesian orthogonal planes of rotations, using the formula:

    :math:`N = n * (n - 1) / 2` [ :math:`N = n! / 2! / (n - 2)!` ]
    (N: num of angles, n: num of dim)

    Args:
        n_dim (int): The number of dimensions.

    Returns:
        n_angles (int): The corresponding number of angles.

    See Also:
        pymrt.geometry.num_dim_from_angles()
    """
    return n_dim * (n_dim - 1) // 2


# ======================================================================
def num_dim_from_angles(
        n_angles,
        raise_err=False):
    """
    Computes the number of dimensions from the number of angles.

    This is the solution for `n` to the equation: :math:`n * (n - 1) / 2 = N`
    (N: num of angles, n: num of dim)

    Args:
        n_angles (int): The number of angles.
        raise_err (bool): Raise an exception if invalid number of angles.

    Returns:
        n_dim (int): The corresponding number of dimensions.

    Raises:
        ValueError: if `raise_err == True` and the number of angles is invalid!

    See Also:
        pymrt.geometry.num_angles_from_dim()
    """
    n_dim = ((1 + np.sqrt(1 + 8 * n_angles)) / 2)
    # alternatives: numpy.modf, math.modf
    int_part, dec_part = divmod(n_dim, 1)
    if not np.isclose(dec_part, 0.0) and raise_err:
        raise ValueError('cannot get the dimension from the number of angles')
    return int(np.ceil(n_dim))


# ======================================================================
def angles2linear(
        angles,
        n_dim=None,
        axes_list=None,
        use_degree=True,
        atol=None):
    """
    Calculate the linear transformation relative to the specified rotations.

    Args:
        angles (tuple[float]): The angles to be used for rotation.
        n_dim (int|None): The number of dimensions to consider.
            The number of angles and `n_dim` should satisfy the relation:
            `n_angles = n_dim * (n_dim - 1) / 2`.
            If `len(angles)` is smaller than expected for a given `n_dim`,
            the remaining angles are set to 0.
            If `len(angles)` is larger than expected, the exceeding `angles`
            are ignored.
            If None, n_dim is computed from `len(angles)`.
        axes_list (tuple[tuple[int]]|None): The axes of the rotation plane.
            If not None, for each rotation angle a pair of axes
            (i.e. a 2-tuple of int) must be specified to define the associated
            plane of rotation.
            The number of 2-tuples should match the number of of angles
            `len(angles) == len(axes_list)`.
            If `len(angles) < len(axes_list)` or `len(angles) > len(axes_list)`
            the unspecified rotations are not performed.
            If None, generates `axes_list` using the output of
            `itertools.combinations(range(n_dim), 2)`.
        use_degree (bool): Interpret angles as expressed in degree.
            Otherwise, use radians.
        atol (float|None): Absolute tolerance in the approximation.
            If error tolerance is exceded, a warning is issued.
            If float, the specified number is used as threshold.
            If None, a threshold is computed based on the size of the linear
            transformation matrix: `dim ** 4 * np.finfo(np.double).eps`.

    Returns:
        linear (np.ndarray): The rotation matrix as defined by the angles.

    See Also:
        pymrt.geometry.num_angles_from_dim(),
        pymrt.geometry.num_dim_from_angles(),
        itertools.combinations
    """
    if n_dim is None:
        n_dim = num_dim_from_angles(len(angles))
    if not axes_list:
        axes_list = list(itertools.combinations(range(n_dim), 2))
    lin_mat = np.eye(n_dim).astype(np.double)
    for angle, axes in zip(angles, axes_list):
        if use_degree:
            angle = np.deg2rad(angle)
        rotation = np.eye(n_dim)
        rotation[axes[0], axes[0]] = np.cos(angle)
        rotation[axes[1], axes[1]] = np.cos(angle)
        rotation[axes[0], axes[1]] = -np.sin(angle)
        rotation[axes[1], axes[0]] = np.sin(angle)
        lin_mat = np.dot(lin_mat, rotation)
    # :: check that this is a rotation matrix
    det = np.linalg.det(lin_mat)
    if not atol:
        atol = lin_mat.ndim ** 4 * np.finfo(np.double).eps
    if np.abs(det) - 1.0 > atol:
        text = 'rotation matrix may be inaccurate [det = {}]'.format(repr(det))
        warnings.warn(text)
    return lin_mat


# ======================================================================
def linear2angles(
        linear,
        use_degree=True,
        atol=None):
    # todo: implement the inverse of angles2linear
    raise NotImplementedError


# ======================================================================
def prepare_affine(
        shape,
        lin_mat,
        shift=None,
        origin=None):
    """
    Prepare parameters to be used with `scipy.ndimage.affine_transform()`.

    In particular, it computes the linear matrix and the offset implementing
    an affine transformation followed by a translation on the array.

    The result can be passed directly as `matrix` and `offset` parameters of
    `scipy.ndimage.affine_transform()`.

    Args:
        shape (Iterable): The shape of the array to be transformed.
        lin_mat (np.ndarray): The N-sized linear square matrix.
        shift (np.ndarray|None): The shift along each axis in px.
            If None, no shift is performed.
        origin (np.ndarray|None): The origin of the linear transformation.
            If None, uses the center of the array.

    Returns:
        array (np.ndarray): The transformed array.

    See Also:
        scipy.ndimage.affine_transform
    """
    ndim = len(shape)
    if shift is None:
        shift = 0
    if origin is None:
        origin = np.array(rel2abs(shape, (0.5,) * ndim))
    offset = origin - np.dot(lin_mat, origin + shift)
    return lin_mat, offset


# ======================================================================
def weighted_center(
        arr,
        labels=None,
        index=None):
    """
    Determine the covariance mass matrix with respect to the origin.

    .. math::
        \\sum_i w_i (\\vec{x}_i - \\vec{o}_i)

    for i spanning through all support space.

    Args:
        arr (np.ndarray): The input array.
        labels (np.ndarray|None): Cumulative mask array for the objects.
            The output of `scipy.ndimage.label` is expected.
            The number of dimensions must be the same as `array`.
            Only uses the labels as indicated by `index`.
        index (int|Iterable[int]|None): Labels used for the calculation.
            If an int, uses all labels between 1 and the specified value.
            If a tuple of int, uses only the selected labels.
            If None, uses all positive labels.

    Returns:
        center (np.ndarray): The coordinates of the weighed center.

    See Also:
        pymrt.geometry.tensor_of_inertia(),
        pymrt.geometry.rotatio_axes(),
        pymrt.geometry.auto_rotation(),
        pymrt.geometry.realigning()
    """
    # numpy.double to improve the accuracy of the norm and the weighted center
    arr = arr.astype(np.double)
    norm = sp.ndimage.sum(arr, labels, index)
    grid = np.ogrid[[slice(0, i) for i in arr.shape]]
    # numpy.double to improve the accuracy of the result
    center = np.zeros(arr.ndim).astype(np.double)
    for i in range(arr.ndim):
        center[i] = sp.ndimage.sum(arr * grid[i], labels, index) / norm
    return center


# ======================================================================
def weighted_covariance(
        arr,
        labels=None,
        index=None,
        origin=None):
    """
    Determine the weighted covariance matrix with respect to the origin.

    .. math::
        \\sum_i w_i (\\vec{x}_i - \\vec{o}) (\\vec{x}_i - \\vec{o})^T

    for i spanning through all support space, where:
    o is the origin vector,
    x_i is the coordinate vector of the point i,
    w_i is the weight, i.e. the value of the array at that coordinate.

    Args:
        arr (np.ndarray): The input array.
        labels (np.ndarray|None): Cumulative mask array for the objects.
            The output of `scipy.ndimage.label` is expected.
            The number of dimensions must be the same as `array`.
            Only uses the labels as indicated by `index`.
        index (int|Iterable[int]|None): Labels used for the calculation.
            If an int, uses all labels between 1 and the specified value.
            If a tuple of int, uses only the selected labels.
            If None, uses all positive labels.
        origin (np.ndarray|None): The origin for the covariance matrix.
            If None, the weighted center is used.

    Returns:
        cov (np.ndarray): The covariance weight matrix from the origin.

    See Also:
        pymrt.geometry.tensor_of_inertia,
        pymrt.geometry.rotation_axes,
        pymrt.geometry.auto_rotation,
        pymrt.geometry.realigning
    """
    # numpy.double to improve the accuracy of the norm and the weighted center
    arr = arr.astype(np.double)
    norm = sp.ndimage.sum(arr, labels, index)
    if origin is None:
        origin = np.array(sp.ndimage.center_of_mass(arr, labels, index))
    grid = np.ogrid[[slice(0, i) for i in arr.shape]] - origin
    # numpy.double to improve the accuracy of the result
    cov = np.zeros((arr.ndim, arr.ndim)).astype(np.double)
    for i in range(arr.ndim):
        for j in range(arr.ndim):
            if i <= j:
                cov[i, j] = sp.ndimage.sum(
                    arr * grid[i] * grid[j], labels, index) / norm
            else:
                # the covariance weight matrix is symmetric
                cov[i, j] = cov[j, i]
    return cov


# ======================================================================
def tensor_of_inertia(
        arr,
        labels=None,
        index=None,
        origin=None):
    """
    Determine the tensor of inertia with respect to the origin.

    I = Id * tr(C) - C

    where:
    C is the weighted covariance matrix,
    Id is the identity matrix.

    Args:
        arr (np.ndarray): The input array.
        labels (np.ndarray|None): Cumulative mask array for the objects.
            The output of `scipy.ndimage.label` is expected.
            The number of dimensions must be the same as `array`.
            Only uses the labels as indicated by `index`.
        index (int|Iterable[int]|None): Labels used for the calculation.
            If an int, uses all labels between 1 and the specified value.
            If a tuple of int, uses only the selected labels.
            If None, uses all positive labels.
        origin (np.ndarray|None): The origin for the covariance matrix.
            If None, the weighted center is used.

    Returns:
        inertia (np.ndarray): The tensor of inertia from the origin.

    See Also:
        pymrt.geometry.weighted_covariance(),
        pymrt.geometry.rotation_axes(),
        pymrt.geometry.auto_rotation(),
        pymrt.geometry.realigning()
    """
    cov = weighted_covariance(arr, labels, index, origin)
    inertia = np.eye(arr.ndim) * np.trace(cov) - cov
    return inertia


# ======================================================================
def rotation_axes(
        arr,
        labels=None,
        index=None,
        sort_by_shape=False):
    """
    Calculate the principal axes of rotation.

    These can be found as the eigenvectors of the tensor of inertia.

    Args:
        arr (np.ndarray): The input array.
        labels (np.ndarray|None): Cumulative mask array for the objects.
            The output of `scipy.ndimage.label` is expected.
            The number of dimensions must be the same as `array`.
            Only uses the labels as indicated by `index`.
        index (int|Iterable[int]|None): Labels used for the calculation.
            If an int, uses all labels between 1 and the specified value.
            If a tuple of int, uses only the selected labels.
            If None, uses all positive labels.
        sort_by_shape (bool): Sort the axes by the array shape.
            This is useful in order to obtain the optimal rotations to
            align the objects to the shape.
            Otherwise, it is sorted by increasing eigenvalues.

    Returns:
        axes (list[np.ndarray]): The principal axes of rotation.

    See Also:
        pymrt.geometry.weighted_covariance(),
        pymrt.geometry.tensor_of_inertia(),
        pymrt.geometry.auto_rotation(),
        pymrt.geometry.realigning()
    """
    # calculate the tensor of inertia with respect to the weighted center
    inertia = tensor_of_inertia(arr, labels, index, None).astype(np.double)
    # numpy.linalg only supports up to numpy.double
    eigenvalues, eigenvectors = np.linalg.eigh(inertia)
    if sort_by_shape:
        tmp = [
            (size, eigenvalue, eigenvector)
            for size, eigenvalue, eigenvector
            in zip(
                sorted(arr.shape, reverse=True),
                eigenvalues,
                tuple(eigenvectors.transpose()))]
        tmp = sorted(tmp, key=lambda x: arr.shape.index(x[0]))
        axes = []
        for size, eigenvalue, eigenvector in tmp:
            axes.append(eigenvector)
    else:
        axes = [axis for axis in eigenvectors.transpose()]
    return axes


# ======================================================================
def rotation_axes_to_matrix(axes):
    """
    Compute the rotation matrix from the principal axes of rotation.

    This matrix describes the linear transformation required to bring the
    principal axes of rotation along the axes of the canonical basis.

    Args:
        axes (Iterable[np.ndarray]): The principal axes of rotation.

    Returns:
        lin_mat (np.ndarray): The linear transformation matrix.
    """
    return np.array(axes).transpose()


# ======================================================================
def auto_rotation(
        arr,
        labels=None,
        index=None,
        origin=None):
    """
    Compute the linear transformation and shift for optimal rotation.

    The principal axis of rotation will be parallel to the cartesian axes.

    The result can be passed directly as `matrix` and `offset` parameters of
    `scipy.ndimage.affine_transform()`.

    Args:
        arr (np.ndarray): The input array.
        labels (np.ndarray|None): Cumulative mask array for the objects.
            The output of `scipy.ndimage.label` is expected.
            The number of dimensions must be the same as `array`.
            Only uses the labels as indicated by `index`.
        index (int|Iterable[int]|None): Labels used for the calculation.
            If an int, uses all labels between 1 and the specified value.
            If a tuple of int, uses only the selected labels.
            If None, uses all positive labels.
        origin (np.ndarray|None): The origin for the covariance matrix.
            If None, the weighted center is used.

    Returns:
        lin_mat (np.ndarray): The linear matrix for the rotation.
        offset (np.ndarray): The offset for the translation.

    See Also:
        scipy.ndimage.center_of_mass(),
        scipy.ndimage.affine_transform(),
        pymrt.geometry.weighted_covariance(),
        pymrt.geometry.tensor_of_inertia(),
        pymrt.geometry.rotation_axes(),
        pymrt.geometry.angles2linear(),
        pymrt.geometry.linear2angles(),
        pymrt.geometry.auto_rotation(),
        pymrt.geometry.realigning()
    """
    lin_mat = rotation_axes_to_matrix(rotation_axes(arr, labels, index, True))
    lin_mat, offset = prepare_affine(arr.shape, lin_mat, origin=origin)
    return lin_mat, offset


# ======================================================================
def auto_shifting(
        arr,
        labels=None,
        index=None,
        origin=None):
    """
    Compute the linear transformation and shift for optimal shifting.

    Weighted center will be at a given point (e.g. the middle of the support).

    The result can be passed directly as `matrix` and `offset` parameters of
    `scipy.ndimage.affine_transform()`.

    Args:
        arr (np.ndarray): The input array.
        labels (np.ndarray|None): Cumulative mask array for the objects.
            The output of `scipy.ndimage.label` is expected.
            The number of dimensions must be the same as `array`.
            Only uses the labels as indicated by `index`.
        index (int|Iterable[int]|None): Labels used for the calculation.
            If an int, uses all labels between 1 and the specified value.
            If a tuple of int, uses only the selected labels.
            If None, uses all positive labels.
        origin (np.ndarray|None): The origin for the covariance matrix.
            If None, the weighted center is used.

    Returns:
        lin_mat (np.ndarray): The linear matrix for the rotation.
        offset (np.ndarray): The offset for the translation.

    See Also:
        scipy.ndimage.center_of_mass(),
        scipy.ndimage.affine_transform(),
        pymrt.geometry.weighted_covariance(),
        pymrt.geometry.tensor_of_inertia(),
        pymrt.geometry.rotation_axes(),
        pymrt.geometry.angles2linear(),
        pymrt.geometry.linear2angles(),
        pymrt.geometry.auto_rotation(),
        pymrt.geometry.realigning()
    """
    lin_mat = np.eye(arr.ndim)
    com = np.array(sp.ndimage.center_of_mass(arr, labels, index))
    lin_mat, offset = prepare_affine(arr.shape, lin_mat, com, origin)
    return lin_mat, offset


# ======================================================================
def realigning(
        arr,
        labels=None,
        index=None,
        origin=None):
    """
    Compute the linear transformation and shift for optimal grid alignment.

    The principal axis of rotation will be parallel to the cartesian axes.
    Weighted center will be at a given point (e.g. the middle of the support).

    The result can be passed directly as `matrix` and `offset` parameters of
    `scipy.ndimage.affine_transform()`.

    Args:
        arr (np.ndarray): The input array.
        labels (np.ndarray|None): Cumulative mask array for the objects.
            The output of `scipy.ndimage.label` is expected.
            The number of dimensions must be the same as `array`.
            Only uses the labels as indicated by `index`.
        index (int|Iterable[int]|None): Labels used for the calculation.
            If an int, uses all labels between 1 and the specified value.
            If a tuple of int, uses only the selected labels.
            If None, uses all positive labels.
        origin (np.ndarray|None): The origin for the covariance matrix.
            If None, the weighted center is used.

    Returns:
        lin_mat (np.ndarray): The linear matrix for the rotation.
        offset (np.ndarray): The offset for the translation.

    See Also:
        scipy.ndimage.center_of_mass(),
        scipy.ndimage.affine_transform(),
        pymrt.geometry.weighted_covariance(),
        pymrt.geometry.tensor_of_inertia(),
        pymrt.geometry.rotation_axes(),
        pymrt.geometry.angles2linear(),
        pymrt.geometry.linear2angles(),
        pymrt.geometry.auto_rotation(),
        pymrt.geometry.auto_shift()
    """
    com = np.array(sp.ndimage.center_of_mass(arr, labels, index))
    lin_mat = rotation_axes_to_matrix(rotation_axes(arr, labels, index, True))
    lin_mat, offset = prepare_affine(arr.shape, lin_mat, com, origin)
    return lin_mat, offset


# ======================================================================
def rand_mask(
        arr,
        density=0.01):
    """
    Calculate a randomly distributed mask of specified density.

    Args:
        arr (np.ndarray): The target array.
        density (float): The density of the mask.
            Must be in the (0, 1) interval.

    Returns:
        mask
    """
    if not 0 < density < 1:
        raise ValueError('Density must be between 0 and 1')
    shape = arr.shape
    mask = np.zeros_like(arr).astype(np.bool).ravel()
    mask[random.sample(range(arr.size), int(arr.size * density))] = True
    return mask.reshape(shape)


# ======================================================================
def multi_render(
        shape,
        geom_shapes,
        n_dim=None,
        affine_kws=(('order', 0),),
        dtype=np.float):
    """
    Render multiple geometrical masks into a single array.

    Args:
        shape (int|Iterable[int]): The shape of the mask in px.
        geom_shapes (Iterable): The

        n_dim (int|None): The number of dimensions.
            If None, the number of dims is guessed from the shape parameter.
        affine_kws (dict|Iterable|None): Keyword parameters for the affine.
            These parameters are passed to `scipy.ndimage.affine_transform()`
            upon application of the (eventual) rotation.
            If Iterable, must be possible to cast to dict.
            If None, no parameter is set.
        dtype (np.dtype): The Numpy data type of the rendered array.
            Note that this will be used also for the internal interpolations.

    Returns:
        arr (np.ndarray): The
    """
    # check that ellipsoids parameters have the right size
    if n_dim is None:
        n_dim = len(shape)
    else:
        shape = mrt.utils.auto_repeat(shape, n_dim, False, True)
    inner_shape = [min(shape)] * n_dim
    affine_kws = \
        {} if affine_kws is None else dict(affine_kws)
    arr = np.zeros(shape, dtype=dtype)
    for scale, geom_shape, inner_pos, angles, outer_pos in geom_shapes:
        # generate the base geometric shape
        geom_name = geom_shape[0]
        geom_arr = None
        if geom_name in ('e', 'ellipsoid'):
            semisizes = geom_shape[1]
            geom_arr = nd_superellipsoid(inner_shape, semisizes)
        elif geom_name in ('s', 'superellipsoid'):
            indexes, semisizes = geom_shape[1:]
            geom_arr = nd_superellipsoid(inner_shape, semisizes, indexes)
        elif geom_name in ('c', 'cuboid'):
            indexes, semisizes = geom_shape[1:]
            geom_arr = nd_superellipsoid(inner_shape, semisizes)
        elif geom_name in ('p', 'prism'):
            axis, semisizes, indexes = geom_shape[1:]
            geom_arr = nd_superellipsoidal_prism(
                inner_shape, axis, semisizes, indexes)
        elif geom_name in ('g', 'gradient'):
            gen_ranges = geom_shape[1]
            geom_arr = nd_gradient(gen_ranges)
        else:
            text = ('unknown name `{geom_name}` while rendering with '
                    '`pymrt.geometry.multi_render`'.format(**locals()))
            warnings.warn(text)
        if outer_pos is None:
            outer_pos = 0.5
        if geom_arr is not None:
            # compute position and linear transformation matrix
            if angles is None:
                lin_mat = np.eye(n_dim)
            else:
                lin_mat = angles2linear(angles, n_dim)
            inner_pos = rel2abs(inner_shape, inner_pos)
            lin_mat, offset = prepare_affine(inner_shape, lin_mat, inner_pos)
            arr += reframe(scale * sp.ndimage.affine_transform(
                geom_arr.astype(dtype),
                lin_mat, offset, **affine_kws), shape, outer_pos)
    return arr


# ======================================================================
def _self_test_interactive():
    """
    Test the functions available in the package.

    Args:
        None

    Returns:
        None
    """
    import numex.gui_tk_mpl
    pos = 0.5
    dim = 128
    l1, l2, l3 = (16.0, 8.0, 32.0)
    # a1, a2, a3 = (math.pi / 3.0, math.pi / 2.0, math.pi / 4.0)

    # :: 2D Tests
    # :: - shape test
    numex.gui_tk_mpl.explore(square(dim, l1, pos))
    numex.gui_tk_mpl.explore(rectangle(dim, (l1, l2), pos))
    numex.gui_tk_mpl.explore(rhombus(dim, (l1, l2), pos))
    numex.gui_tk_mpl.explore(circle(dim, l1, pos))
    numex.gui_tk_mpl.explore(ellipsis(dim, (l1, l2), pos))
    # :: - Position test
    numex.gui_tk_mpl.explore(ellipsis(dim, (l1, l2), (0.2, 0.7)))

    # :: 3D Tests
    # :: - shape test
    numex.gui_tk_mpl.explore(cube(dim, l1, pos))
    numex.gui_tk_mpl.explore(cuboid(dim, (l1, l2, l3), pos))
    numex.gui_tk_mpl.explore(rhomboid(dim, (l1, l2, l3), pos))
    numex.gui_tk_mpl.explore(sphere(dim, l1, pos))
    numex.gui_tk_mpl.explore(ellipsoid(dim, (l1, l2, l3), pos))
    numex.gui_tk_mpl.explore(cylinder(dim, 2.0 * l1, l1, -1, pos))
    # :: - Position test
    numex.gui_tk_mpl.explore(ellipsoid(dim, (l1, l2, l3), (0.0, 1.0, 0.5)))


# ======================================================================
elapsed(__file__[len(DIRS['base']) + 1:])

# ======================================================================
if __name__ == '__main__':
    import doctest  # Test interactive Python examples

    msg(__doc__.strip())
    doctest.testmod()
    msg(report())
