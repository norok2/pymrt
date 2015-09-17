#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
mr_lib: package to create and manipulate 2D, 3D and N-D geometrical masks.

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

The N-D geometrical shapes currentyl available are:
- cuboid: sum[abs(x_n/a_n)^inf] < 1
- superellipsoid: sum[abs(x_n/a_n)^k] < 1
- prism: stack (N-1)-D mask on given axis
"""


# ======================================================================
# :: Future Imports
from __future__ import division
from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals


__version__ = '1.0.0.10'
# $Source$


# ======================================================================
# :: Custom Module Details
AUTHOR = 'Riccardo Metere'
CONTACT = 'metere@cbs.mpg.de'
DATE_INFO = {'day': 18, 'month': 'Sep', 'year': 2014}
DATE = ' '.join([str(v) for k, v in sorted(DATE_INFO.items())])
LICENSE = 'License GPLv3: GNU General Public License version 3'
COPYRIGHT = 'Copyright (C) ' + str(DATE_INFO['year'])
# first non-empty line of __doc__
DOC_FIRSTLINE = [line for line in __doc__.splitlines() if line][0]


# ======================================================================
# :: Python Standard Library Imports
# import os  # Miscellaneous operating system interfaces
# import shutil  # High-level file operations
import math  # Mathematical functions
import time  # Time access and conversions
import datetime  # Basic date and time types
# import operator  # Standard operators as functions
# import collections  # High-performance container datatypes
# import itertools  # Functions creating iterators for efficient looping
# import functools  # Higher-order functions and operations on callable objects
# import argparse  # Parser for command-line options, arguments and subcommands
# import subprocess  # Subprocess management
# import multiprocessing  # Process-based parallelism
# import csv  # CSV File Reading and Writing [CSV: Comma-Separated Values]
# import json  # JSON encoder and decoder [JSON: JavaScript Object Notation]

# :: External Imports
import numpy as np  # NumPy (multidimensional numerical arrays library)
# import scipy as sp  # SciPy (signal and image processing library)
# import matplotlib as mpl  # Matplotlib (2D/3D plotting library)
# import sympy as sym  # SymPy (symbolic CAS library)
# import PIL  # Python Image Library (image manipulation toolkit)
# import SimpleITK as sitk  # Image ToolKit Wrapper
# import nibabel as nib  # NiBabel (NeuroImaging I/O Library)
# import nipy  # NiPy (NeuroImaging in Python)
# import nipype  # NiPype (NiPy Pipelines and Interfaces)

# :: External Imports Submodules
import matplotlib.pyplot as plt  # Matplotlib's pyplot: MATLAB-like syntax
import mayavi.mlab as mlab  # Mayavi's mlab: MATLAB-like syntax
# import scipy.optimize  # SciPy: Optimization Algorithms
# import scipy.integrate  # SciPy: Integrations facilities
# import scipy.constants  # SciPy: Mathematal and Physical Constants
# import scipy.ndimage  # SciPy: ND-image Manipulation


# ======================================================================
# :: Custom defined constants
POS_MODE_ABS = 'abs'
POS_MODE_PROP = 'prop'

# ======================================================================
# :: Default values usable in functions.
D_POSITION = 0.5
D_SHAPE = 128
D_LENGTH_1 = 16.0
D_LENGTH_2 = (8.0, 16.0)
D_LENGTH_3 = (8.0, 16.0, 32.0)
D_ANGLE_1 = math.pi / 3.0
D_ANGLE_2 = (math.pi / 3.0, math.pi / 4.0)
D_ANGLE_3 = (math.pi / 2.0, math.pi / 3.0, math.pi / 4.0)


# ======================================================================
def relpos2coord(position, shape):
    """
    Calculate the absolute position with respect to a given shape.

    Parameters
    ==========
    position : float N-tuple
        Relative position (to the lowest edge). Values are in the range [0, 1].
    shape : int N-tuple
        Information on the shape.

    Returns
    =======
    coord : int N-tuple
        Coordinate of the specified position inside the shape.

    """
    if len(position) != len(shape):
        raise IndexError('length of tuples must match')
    return tuple([(s - 1.0) * p for p, s in zip(position, shape)])


# ======================================================================
def coord2relpos(coord, shape):
    """
    Calculate the proportional position with respect to a given shape.

    Parameters
    ==========
    coord : float N-tuple
        Coordinate of the specified position inside the shape.
    shape : int N-tuple
        Information on the shape.

    Returns
    =======
    position : float N-tuple
        Relative position (to the lowest edge) in the interval [0.0, 1.0]

    """
    if len(coord) != len(shape):
        raise IndexError
    return tuple([c / (s - 1.0) for c, s in zip(coord, shape)])


# ======================================================================
def render(
        mask,
        fill=(1, 0),
        dtype=float):
    """
    Render a mask as an image.

    Parameters
    ==========
    mask : nd-array
        Mask to be rendered.
    fill : (True value, False value)
        Values to render the mask with.
    dtype : data-type, optional
        Desired output data-type.

    Returns
    =======
    img : nd-array
        Image of the mask rendered with specified values and data type.

    """
    img = np.empty_like(mask, dtype)
    img[mask] = fill[0]
    img[~mask] = fill[1]
    return img


# ======================================================================
def square(
        shape,
        position,
        side):
    """
    Generate a mask whose shape is a square.

    Parameters
    ==========
    shape : int 2-tuple
        Shape of the generated mask.
    position : float 2-tuple
        Relative position (to the lowest edge). Values are in the range [0, 1].
    side : float 2-tuple
        Length of the side of the square.

    Returns
    =======
    mask : nd-array
        Array of boolean describing the geometrical object.

    """
    # set correct dimensions
    n_dim = 2
    # check parameters
    if not (len(shape) == len(position) == n_dim):
        raise IndexError('length of tuples must be {}'.format(n_dim))
    # adjust semisides
    semisides = tuple([side / 2.0]) * n_dim
    # use n-dim function
    return nd_cuboid(shape, position, semisides)


# ======================================================================
def rectangle(
        shape,
        position,
        semisides):
    """
    Generate a mask whose shape is a rectangle.

    Parameters
    ==========
    shape : int 2-tuple
        Shape of the generated mask.
    position : float 2-tuple
        Relative position (to the lowest edge). Values are in the range [0, 1].
    semisides : float 2-tuple
        Length of the semisides of the rectangle.

    Returns
    =======
    mask : nd-array
        Array of boolean describing the geometrical object.

    """
    # set correct dimensions
    n_dim = 2
    # check parameters
    if not (len(shape) == len(position) == len(semisides) == n_dim):
        raise IndexError('length of tuples must be {}'.format(n_dim))
    # use n-dim function
    return nd_cuboid(shape, position, semisides)


# ======================================================================
def rhombus(
        shape,
        position,
        semidiagonals):
    """
    Generate a mask whose shape is a rhombus.

    Parameters
    ==========
    shape : int 2-tuple
        Shape of the generated mask.
    position : float 2-tuple
        Relative position (to the lowest edge). Values are in the range [0, 1].
    semidiagonals : float 2-tuple
        Length of the semidiagonals of the rhombus.

    Returns
    =======
    mask : nd-array
        Array of boolean describing the geometrical object.

    """
    # set correct dimensions
    n_dim = 2
    # check parameters
    if not (len(shape) == len(position) == len(semidiagonals) == n_dim):
        raise IndexError('length of tuples must be {}'.format(n_dim))
    # adjust semiaxes
    semiaxes = semidiagonals
    # set index value
    rhombus_index = 1.0
    index = tuple([rhombus_index] * n_dim)
    # use n-dim function
    return nd_superellipsoid(shape, position, semiaxes, index)


# ======================================================================
def circle(
        shape,
        position,
        radius):
    """
    Generate a mask whose shape is a circle.

    Parameters
    ==========
    shape : int 2-tuple
        Shape of the generated mask.
    position : float 2-tuple
        Relative position (to the lowest edge). Values are in the range [0, 1].
    radius : float
        Length of the radius of the circle.

    Returns
    =======
    mask : nd-array
        Array of boolean describing the geometrical object.

    """
    # set correct dimensions
    n_dim = 2
    # check parameters
    if not (len(shape) == len(position) == n_dim):
        raise IndexError('length of tuples must be {}'.format(n_dim))
    # adjust semiaxes
    semiaxes = tuple([radius] * n_dim)
    # set index value
    circle_index = 2.0
    index = tuple([circle_index] * n_dim)
    # use n-dim function
    return nd_superellipsoid(shape, position, semiaxes, index)


# ======================================================================
def ellipsis(
        shape,
        position,
        semiaxes):
    """
    Generate a mask whose shape is an ellipsis.

    Parameters
    ==========
    shape : int 2-tuple
        Shape of the generated mask.
    position : float 2-tuple
        Relative position (to the lowest edge). Values are in the range [0, 1].
    semiaxes : float 2-tuple
        Length of the semiaxes of the ellipsis.

    Returns
    =======
    mask : nd-array
        Array of boolean describing the geometrical object.

    """
    # set correct dimensions
    n_dim = 2
    # check parameters
    if not (len(shape) == len(position) == len(semiaxes) == n_dim):
        raise IndexError('length of tuples must be {}'.format(n_dim))
    # set index value
    ellipsis_index = 2.0
    index = tuple([ellipsis_index] * n_dim)
    # use n-dim function
    return nd_superellipsoid(shape, position, semiaxes, index)


# ======================================================================
def cube(
        shape,
        position,
        side):
    """
    Generate a mask whose shape is a cube.

    Parameters
    ==========
    shape : int 3-tuple
        Shape of the generated mask.
    position : float 3-tuple
        Relative position (to the lowest edge). Values are in the range [0, 1].
    side : float
        Length of the side of the cube.

    Returns
    =======
    mask : nd-array
        Array of boolean describing the geometrical object.

    """
    # set correct dimensions
    n_dim = 3
    # check parameters
    if not (len(shape) == len(position) == n_dim):
        raise IndexError('length of tuples must be {}'.format(n_dim))
    # adjust semisides
    semisides = tuple([side / 2.0] * n_dim)
    # use n-dim function
    return nd_cuboid(shape, position, semisides)


# ======================================================================
def cuboid(
        shape,
        position,
        semisides):
    """
    Generate a mask whose shape is a cuboid.

    Parameters
    ==========
    shape : int 3-tuple
        Shape of the generated mask.
    position : float 3-tuple
        Relative position (to the lowest edge). Values are in the range [0, 1].
    semisides : float 3-tuple
        Length of the sides of the cuboid.

    Returns
    =======
    mask : nd-array
        Array of boolean describing the geometrical object.

    """
    # set correct dimensions
    n_dim = 3
    # check parameters
    if not (len(shape) == len(position) == len(semisides) == n_dim):
        raise IndexError('length of tuples must be {}'.format(n_dim))
    # use n-dim function
    return nd_cuboid(shape, position, semisides)


# ======================================================================
def rhomboid(
        shape,
        position,
        semidiagonals):
    """
    Generate a mask whose shape is a rhomboid.

    Parameters
    ==========
    shape : int 3-tuple
        Shape of the generated mask.
    position : float 3-tuple
        Relative position (to the lowest edge). Values are in the range [0, 1].
    semidiagonals : float 3-tuple
        Length of the semidiagonals of the rhomboid.

    Returns
    =======
    mask : nd-array
        Array of boolean describing the geometrical object.

    """
    # set correct dimensions
    n_dim = 3
    # check parameters
    if not (len(shape) == len(position) == len(semidiagonals) == n_dim):
        raise IndexError('length of tuples must be {}'.format(n_dim))
    # adjust semiaxes
    semiaxes = semidiagonals
    # set index value
    rhombus_index = 1.0
    index = tuple([rhombus_index] * n_dim)
    # use n-dim function
    return nd_superellipsoid(shape, position, semiaxes, index)


# ======================================================================
def sphere(
        shape,
        position,
        radius):
    """
    Generate a mask whose shape is a sphere.

    Parameters
    ==========
    shape : int 3-tuple
        Shape of the generated mask.
    position : float 3-tuple
        Relative position (to the lowest edge). Values are in the range [0, 1].
    radius : float
        Length of the radius of the sphere.

    Returns
    =======
    mask : nd-array
        Array of boolean describing the geometrical object.

    """
    # set correct dimensions
    n_dim = 3
    # check parameters
    if not (len(shape) == len(position) == n_dim):
        raise IndexError('length of tuples must be {}'.format(n_dim))
    # adjust semiaxes
    semiaxes = tuple([radius] * n_dim)
    # set index value
    circle_index = 2.0
    index = tuple([circle_index] * n_dim)
    # use n-dim function
    return nd_superellipsoid(shape, position, semiaxes, index)


# ======================================================================
def ellipsoid(
        shape,
        position,
        semiaxes):
    """
    Generate a mask whose shape is an ellipsoid.

    Parameters
    ==========
    shape : int 3-tuple
        Shape of the generated mask.
    position : float 3-tuple
        Relative position (to the lowest edge). Values are in the range [0, 1].
    semiaxes : float 3-tuple
        Length of the semiaxes of the ellipsoid.

    Returns
    =======
    mask : nd-array
        Array of boolean describing the geometrical object.

    """
    # set correct dimensions
    n_dim = 3
    # check parameters
    if not (len(shape) == len(position) == len(semiaxes) == n_dim):
        raise IndexError('length of tuples must be {}'.format(n_dim))
    # set index value
    ellipsis_index = 2.0
    index = tuple([ellipsis_index] * n_dim)
    # use n-dim function
    return nd_superellipsoid(shape, position, semiaxes, index)


# ======================================================================
def cylinder(
        shape,
        position,
        height,
        radius):
    """
    Generate a mask whose shape is a cylinder.

    Parameters
    ==========
    shape : int 3-tuple
        Shape of the generated mask.
    position : float 3-tuple
        Relative position (to the lowest edge). Values are in the range [0, 1].
    height : float
        Length of the height of the cylinder.
    radius : float
        Length of the radius of the cylinder.

    Returns
    =======
    mask : nd-array
        Array of boolean describing the geometrical object.

    """
    # set correct dimensions
    n_dim = 3
    # check parameters
    if not (len(shape) == len(position) == n_dim):
        raise IndexError('length of tuples must be {}'.format(n_dim))
    # calculate base mask
    axis = n_dim - 1
    base_mask = circle(shape[0:axis], position[0:axis], radius)
    # use n-dim function
    return nd_prism(base_mask, shape[axis], axis, position[axis], height / 2.0)


# ======================================================================
def nd_cuboid(
        shape,
        position,
        semisides):
    """
    Generate a mask whose shape is a cuboid: sum[abs(x_n/a_n)^inf] < 1.0

    Parameters
    ==========
    shape : int N-tuple
        Shape of the generated mask.
    position : float N-tuple
        Relative position (to the lowest edge). Values are in the range [0, 1].
    semisides : float N-tuple
        Length of the semisides of the cuboid

    Returns
    =======
    mask : nd-array
        Array of boolean describing the geometrical object.

    """
    # check compatibility of given parameters
    if not (len(shape) == len(position) == len(semisides)):
        raise IndexError('length of tuples must match')
    # calculate the position of the center of the solid inside the mask
    x_0 = relpos2coord(position, shape)
    # create the grid with origin in the middle
    grid = tuple([slice(-x0c, dim - x0c) for x0c, dim in zip(x_0, shape)])
    x_arr = np.ogrid[grid]
    # create the mask
    mask = np.ones(shape, dtype=bool)
    for x_i, semiside in zip(x_arr, semisides):
        mask *= (np.abs(x_i) < semiside)
    return mask


# ======================================================================
def nd_superellipsoid(
        shape,
        position,
        semiaxes,
        indexes):
    """
    Generate a mask whose shape is a superellipsoid: sum[abs(x_n/a_n)^k] < 1.0

    Parameters
    ==========
    shape : int N-tuple
        Shape of the generated mask.
    position : float N-tuple
        Relative position (to the lowest edge). Values are in the range [0, 1]..
    semiaxes : float N-tuple
        Length of the semiaxes of the cuboid a_n
    indexes : float N-tuple
        Exponent k to which summed terms are raised

    Returns
    =======
    mask : nd-array
        Array of boolean describing the geometrical object.

    """
    # check compatibility of given parameters
    if not (len(shape) == len(position) == len(semiaxes) == len(indexes)):
        raise IndexError('length of tuples must match')
    # calculate the position of the center of the solid inside the mask
    x_0 = relpos2coord(position, shape)
    # create the grid with origin in the middle
    grid = tuple([slice(-x0c, dim - x0c) for x0c, dim in zip(x_0, shape)])
    x_arr = np.ogrid[grid]
    # create the mask
    mask = np.zeros(shape, dtype=float)
    for x_i, semiaxis, index in zip(x_arr, semiaxes, indexes):
        mask += (np.abs(x_i / semiaxis) ** index)
    mask = mask < 1.0
    return mask


# ======================================================================
def nd_prism(
        base_mask,
        extra_shape,
        axis,
        position,
        semiheight):
    """
    Generate a mask whose shape is a N-dim prism.

    Parameters
    ==========
    base_mask : (N-1)-dim nd-array
        Base mask to stack in order to obtain the prism.
    extra_shape : int
        Size of the new dimension to be added.
    axis : int [0,N]
        Orientation of the prism in the N-dim space.
    position : float
        Position proportional to extra shape size relative to its center.
    semiheight : float
        Length of half of the height of the prism.

    Returns
    =======
    mask : nd-array
        Array of boolean describing the geometrical object.

    """
    n_dim = len(base_mask.shape) + 1
    if axis > n_dim:
        raise ValueError(
            'Axis of orientation must not exceed the number of dimensions')
    # calculate the position of the center of the solid inside the mask
    x_0 = (position + 1.0) * extra_shape / 2.0
    # create a grid with origin in the middle
    iii = slice(-x_0, extra_shape - x_0)
    xxx = np.ogrid[iii]
    # create the extra mask (height)
    extra_mask = np.abs(xxx) < semiheight
    # calculate mask shape
    shape = base_mask.shape[:axis] + tuple([extra_shape]) + \
        base_mask.shape[axis:]
    # create indefinite prism
    mask = np.zeros(shape, dtype=bool)
    for idx in range(extra_shape):
        if extra_mask[idx]:
            index = [slice(None)] * n_dim
            index[axis] = idx
            mask[tuple(index)] = base_mask
    return mask


# ======================================================================
def plot_2d(mask):
    """
    Plot a 2D mask using Matplotlib.

    Parameters
    ==========
    mask : nparray
        The mask to plot.

    Returns
    =======
    None

    """
    plt.figure()
    plt.imshow(mask.astype(np.int8))
    plt.show()


# ======================================================================
def plot_3d(mask):
    """
    Plot a 3D mask using Mayavi.

    Parameters
    ==========
    mask : nparray
        The mask to plot.

    Returns
    =======
    None

    """
    mlab.figure()
    mlab.contour3d(mask.astype(np.int8))
    mlab.show()


# ======================================================================
def _self_test():
    """
    Test all functions available in the package.

    Parameters
    ==========
    None

    Returns
    =======
    None

    """
    # :: 2D Tests
    shape_2d = tuple([D_SHAPE] * 2)
    position_2d = tuple([D_POSITION] * 2)
    # :: - shape test
    plot_2d(square(shape_2d, position_2d, D_LENGTH_1))
    plot_2d(rectangle(shape_2d, position_2d, D_LENGTH_2))
    plot_2d(rhombus(shape_2d, position_2d, D_LENGTH_2))
    plot_2d(circle(shape_2d, position_2d, D_LENGTH_1))
    plot_2d(ellipsis(shape_2d, position_2d, D_LENGTH_2))
    # :: - Position test
    plot_2d(ellipsis(shape_2d, (0.2, -0.2), D_LENGTH_2))
    # :: 3D Tests
    shape_3d = tuple([D_SHAPE] * 3)
    position_3d = tuple([D_POSITION] * 3)
    # :: - shape test
    plot_3d(cube(shape_3d, position_3d, D_LENGTH_1))
    plot_3d(cuboid(shape_3d, position_3d, D_LENGTH_3))
    plot_3d(rhomboid(shape_3d, position_3d, D_LENGTH_3))
    plot_3d(sphere(shape_3d, position_3d, D_LENGTH_1))
    plot_3d(ellipsoid(shape_3d, position_3d, D_LENGTH_3))
    plot_3d(cylinder(shape_3d, position_3d, 2.0 * D_LENGTH_1, D_LENGTH_1))
    # :: - Position test
    plot_3d(ellipsoid(shape_3d, (0.0, 1.0, 0.2), D_LENGTH_3))


# ======================================================================
if __name__ == '__main__':
    print(__doc__)
    begin_time = time.time()
    _self_test()
    end_time = time.time()
    print('ExecTime: ', datetime.timedelta(0, end_time - begin_time))
