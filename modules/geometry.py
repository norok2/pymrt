#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
mri_tools.geometry: create and manipulate 2D, 3D and N-D geometries.

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

Running this file directly will run some tests.
"""


# ======================================================================
# :: Future Imports
from __future__ import division
from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals


# ======================================================================
# :: Python Standard Library Imports
# import os  # Miscellaneous operating system interfaces
# import shutil  # High-level file operations
import math  # Mathematical functions
import time  # Time access and conversions
import datetime  # Basic date and time types
# import operator  # Standard operators as functions
# import collections  # High-performance container datatypes
import itertools  # Functions creating iterators for efficient looping
# import functools  # Higher-order functions and operations on callable objects
# import argparse  # Parser for command-line options, arguments and subcommands
# import subprocess  # Subprocess management
# import multiprocessing  # Process-based parallelism
# import csv  # CSV File Reading and Writing [CSV: Comma-Separated Values]
# import json  # JSON encoder and decoder [JSON: JavaScript Object Notation]
import warnings  # Warning control

# :: External Imports
import numpy as np  # NumPy (multidimensional numerical arrays library)
import scipy as sp  # SciPy (signal and image processing library)
# import matplotlib as mpl  # Matplotlib (2D/3D plotting library)
# import sympy as sym  # SymPy (symbolic CAS library)
# import PIL  # Python Image Library (image manipulation toolkit)
# import SimpleITK as sitk  # Image ToolKit Wrapper
# import nibabel as nib  # NiBabel (NeuroImaging I/O Library)
# import nipy  # NiPy (NeuroImaging in Python)
# import nipype  # NiPype (NiPy Pipelines and Interfaces)

# :: External Imports Submodules
# TODO: replace mlab by visvis
import mayavi.mlab as mlab  # Mayavi's mlab: MATLAB-like syntax
# import scipy.optimize  # SciPy: Optimization Algorithms
# import scipy.integrate  # SciPy: Integrations facilities
# import scipy.constants  # SciPy: Mathematical and Physical Constants
import scipy.ndimage  # SciPy: ND-image Manipulation


# :: Local Imports
import mri_tools.modules.base as mrb
import mri_tools.modules.plot as mrp
# from mri_tools import INFO
# from mri_tools import VERB_LVL
# from mri_tools import D_VERB_LVL
# from mri_tools import get_first_line


# ======================================================================
# :: Custom defined constants
# POS_MODE_ABS = 'abs'
# POS_MODE_PROP = 'prop'

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
def relative2coord(position, shape):
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
def coord2relative(coord, shape):
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
        raise IndexError('length of tuples must match')
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
        e_msg = 'length of tuples must be {}'.format(n_dim)
        raise IndexError(e_msg)
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
        e_msg = 'length of tuples must be {}'.format(n_dim)
        raise IndexError(e_msg)
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
        e_msg = 'length of tuples must be {}'.format(n_dim)
        raise IndexError(e_msg)
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
        e_msg = 'length of tuples must be {}'.format(n_dim)
        raise IndexError(e_msg)
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
        e_msg = 'length of tuples must be {}'.format(n_dim)
        raise IndexError(e_msg)
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
        e_msg = 'length of tuples must be {}'.format(n_dim)
        raise IndexError(e_msg)
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
        e_msg = 'length of tuples must be {}'.format(n_dim)
        raise IndexError(e_msg)
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
        e_msg = 'length of tuples must be {}'.format(n_dim)
        raise IndexError(e_msg)
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
        e_msg = 'length of tuples must be {}'.format(n_dim)
        raise IndexError(e_msg)
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
        e_msg = 'length of tuples must be {}'.format(n_dim)
        raise IndexError(e_msg)
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
    mask : ndarray
        Array of boolean describing the geometrical object.

    """
    # set correct dimensions
    n_dim = 3
    # check parameters
    if not (len(shape) == len(position) == n_dim):
        e_msg = 'length of tuples must be {}'.format(n_dim)
        raise IndexError(e_msg)
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
    mask : ndarray
        Array of boolean describing the geometrical object.

    """
    # check compatibility of given parameters
    if not (len(shape) == len(position) == len(semisides)):
        raise IndexError('length of tuples must match')
    # calculate the position of the center of the solid inside the mask
    x_0 = relative2coord(position, shape)
    # create the grid with origin in the specified position
    grid = [slice(-x0c, dim - x0c) for x0c, dim in zip(x_0, shape)]
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
        Relative position (to the lowest edge). Values are in the range [0, 1].
    semiaxes : float N-tuple
        Length of the semiaxes of the cuboid a_n
    indexes : float N-tuple
        Exponent k to which summed terms are raised

    Returns
    =======
    mask : ndarray
        Array of boolean describing the geometrical object.

    """
    # check compatibility of given parameters
    if not (len(shape) == len(position) == len(semiaxes) == len(indexes)):
        raise IndexError('length of tuples must match')
    # calculate the position of the center of the solid inside the mask
    x_0 = relative2coord(position, shape)
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
    base_mask : (N-1)-dim ndarray
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
    mask : ndarray
        Array of boolean describing the geometrical object.

    """
    n_dim = base_mask.ndim + 1
    if axis > n_dim:
        raise ValueError(
            'axis of orientation must not exceed the number of dimensions')
    # calculate the position of the center of the solid inside the mask
    x_0 = (position + 1.0) * extra_shape / 2.0
    # create a grid with origin in the middle
    iii = slice(-x_0, extra_shape - x_0)
    xxx = np.ogrid[iii]
    # create the extra mask (height)
    extra_mask = np.abs(xxx) < semiheight
    # calculate mask shape
    shape = (
        base_mask.shape[:axis] + tuple([extra_shape]) + base_mask.shape[axis:])
    # create indefinite prism
    mask = np.zeros(shape, dtype=bool)
    for idx in range(extra_shape):
        if extra_mask[idx]:
            index = [slice(None)] * n_dim
            index[axis] = idx
            mask[tuple(index)] = base_mask
    return mask


# ======================================================================
def frame(
        array,
        borders=0.05,
        background=0.0,
        use_longest=True):
    """
    Add a frame to an array.
    TODO: check with 'reframe'

    Parameters
    ==========
    array : ndarray
        The input array.
    border : int or int tuple (optional)
        The size of the border relative to the initial array shape.
    background : int or float (optional)
        The background value to be used for the frame.
    use_longest : bool (optional)
        Use longest dimension to get the border size.

    Returns
    =======
    result : ndarray
        The result array,

    """
    try:
        iter(borders)
    except TypeError:
        borders = [borders] * array.ndim
    if any(borders) < 0:
        raise ValueError('relative border cannot be negative')
    if use_longest:
        dim = max(array.shape)
        borders = [round(border * dim) for border in borders]
    else:
        borders = [
            round(border * dim) for dim, border in zip(array.shape, borders)]
    result = background * np.ones(
        [dim + 2 * border for dim, border in zip(array.shape, borders)])
    inner = [
        slice(border, border + dim, None) \
        for dim, border in zip(array.shape, borders)]
    result[inner] = array
    return result


# ======================================================================
def reframe(
        array,
        new_shape,
        background=0.0):
    """
    Add a frame to an array.
    TODO: check with 'frame'

    Parameters
    ==========
    array : ndarray
        The input array.
    border : int or int tuple (optional)
        The size of the border relative to the initial array shape.
    background : int or float (optional)
        The background value to be used for the frame.
    use_longest : bool (optional)
        Use longest dimension to get the border size.

    Returns
    =======
    result : ndarray
        The result array,

    """
    if array.ndim != len(new_shape):
        raise IndexError('number of dimensions must match')
    elif any([old > new for old, new in zip(array.shape, new_shape)]):
        raise ValueError('new shape cannot be smaller than the old one.')
    result = background * np.ones(new_shape)
    borders = [
        round((new - old) / 2.0) for old, new in zip(array.shape, new_shape)]
    inner = [
        slice(border, border + dim, None)
        for dim, border in zip(array.shape, borders)]
    result[inner] = array
    return result


# ======================================================================
def zoom_prepare(
        zoom,
        shape,
        extra_dim=True,
        fill_dim=True):
    """
    Prepare the zoom and shape tuples to allow for non homogeneous shapes.

    Parameters
    ==========
    zoom : float or float tuple
        The zoom factors for each directions.
    shape : int tuple
        The shape of the array to operate with.
    extra_dim : bool, optional
        Force extra dimensions in the zoom parameters.
    fill_dim : bool, optional
        Dimensions not specified are left untouched.

    Returns
    =======
    zoom : float or float tuple
        The zoom factors for each directions.
    shape : int tuple
        The shape of the array to operate with.

    """
    try:
        iter(zoom)
    except TypeError:
        zoom = [zoom] * len(shape)
    else:
        zoom = list(zoom)
    if extra_dim:
        shape = list(shape) + [1.0] * (len(zoom) - len(shape))
    else:
        zoom = zoom[:len(shape)]
    if fill_dim and len(zoom) < len(shape):
        zoom[len(zoom):] = [1.0] * (len(shape) - len(zoom))
    return zoom, shape


# ======================================================================
def shape2zoom(
        old_shape,
        new_shape,
        keep_ratio_method=None):
    """
    Calculate zoom (or conversion) factor between two shapes.

    Parameters
    ==========
    old_shape : int tuple
        The shape of the source array.
    new_shape : int tuple
        The target shape of the array.
    keep_ratio_method : callable or None
        | Function to be applied to zoom factors tuple to enforce aspect ratio.
        | None to avoid keeping aspect ratio.
        | Signature: keep_ratio_method(iterable) -> float
        | - 'min': image strictly contained into new shape
        | - 'max': new shape strictly contained into image

    Returns
    =======
    zoom : float or float tuple
        The zoom factors for each directions.

    """
    if len(old_shape) != len(new_shape):
        raise IndexError('length of tuples must match')
    zoom = [new / old for old, new in zip(old_shape, new_shape)]
    if keep_ratio_method:
        zoom = [keep_ratio_method(zoom)] * len(zoom)
    return zoom


# ======================================================================
def apply_to_complex(
        array,
        func,
        *args, **kwargs):
    """
    Apply the specified affine transformation to the array.

    Parameters
    ==========
    img : ndarray
        The n-dimensional image to be transformed.
    affine : ndarray
        The n+1 square matrix describing the affine transformation.
    opts : ...
        Additional options to be passed to: scipy.ndimage.affine_transform

    Returns
    =======
    img : ndarray
        The transformed image.

    """
    real = func(np.real(array), *args, **kwargs)
    imag = func(np.imag(array), *args, **kwargs)
    array = mrb.cartesian2complex(real, imag)
    return array


# ======================================================================
def decode_affine(
        affine):
    """
    Decompose the affine matrix into a linear transformation and a translation.

    Parameters
    ==========
    affine : ndarray
        The n+1 square matrix describing the affine transformation.

    Returns
    =======
    linear : ndarray
        The n square matrix describing the linear transformation.
    shift : array
        The array containing the shift along each axis.

    """
    num_dim = affine.shape
    linear = affine[:num_dim[0] - 1, :num_dim[1] - 1]
    shift = affine[:-1, -1]
    return linear, shift


# ======================================================================
def encode_affine(
        linear,
        shift):
    """
    Combine a linear transformation and a translation into the affine matrix.

    Parameters
    ==========
    linear : ndarray
        The n square matrix describing the linear transformation.
    shift : array
        The array containing the shift along each axis.

    Returns
    =======
    affine : ndarray
        The n+1 square matrix describing the affine transformation.

    """
    num_dim = linear.shape
    affine = np.eye(num_dim[0] + 1)
    affine[:num_dim[0], :num_dim[1]] = linear
    affine[:-1, -1] = shift
    return affine


# ======================================================================
def num_angles_from_dim(num_dim):
    """
    Calculate the complete number of angles given the dimension.

    Given the dimension of an array, calculate the number of all possible
    cartesian orthogonal planes of rotations, using the formula:

    N = n * (n - 1) / 2 [ = n! / 2! / (n - 2)! ]
    (N: num of angles, n: num of dim)

    Parameters
    ==========
    num_dim : int
        The number of dimensions

    Returns
    =======
    num_angles : int
        The corresponding number of angles.

    See Also
    ========
    mri_tools.geometry.angles2linear
    """
    return num_dim * (num_dim - 1) // 2


# ======================================================================
def angles2linear(
        angles,
        axes_list=None,
        use_degree=True):
    """
    Calculate the linear transformation relative to the specified rotations.

    Parameters
    ==========
    angles : float tuple
        The angles to be used for rotation.
    axes_list : list of int 2-tuple or None (optional)
        The axes couples defining the plane of rotation for each angle.
        If None, uses the output of `itertools.combinations(range(n_dim), 2)`.
    use_degree : bool (optional)
        If True, interpret angles in degree. Otherwise, use radians.

    Returns
    =======
    linear : ndarray
        The rotation matrix identified by the selected angles.

    See Also
    ========
    mri_tools.geometry.num_angles_from_dim,
    itertools.combinations

    """
    # solution to: n * (n - 1) / 2 = N  (N: num of angles, n: num of dim)
    num_dim = ((1 + np.sqrt(1 + 8 * len(angles))) / 2)
    if np.modf(num_dim)[0] != 0.0:
        raise ValueError('cannot get the dimension from the number of angles')
    else:
        num_dim = int(num_dim)
    if not axes_list:
        axes_list = list(itertools.combinations(range(num_dim), 2))
    linear = np.eye(num_dim).astype(np.double)  # longdouble?
    for angle, axes in zip(angles, axes_list):
        if use_degree:
            angle = np.deg2rad(angle)
        rotation = np.eye(num_dim)
        rotation[axes[0], axes[0]] = np.cos(angle)
        rotation[axes[1], axes[1]] = np.cos(angle)
        rotation[axes[0], axes[1]] = -np.sin(angle)
        rotation[axes[1], axes[0]] = np.sin(angle)
        linear = np.dot(linear, rotation)
    # :: check that this is a rotation matrix
    det = np.linalg.det(linear)
    tolerance = np.finfo(np.double).eps
    if np.abs(det) - 1.0 > tolerance:
        msg = 'rotation matrix may be inaccurate [det = {}]'.format(repr(det))
        warnings.warn(msg)
    return linear


# ======================================================================
def linear2angles():
    # todo: implement the inverse of angles2linear
    raise ValueError('function is not implemented yet')
    pass


# ======================================================================
def affine_transform(
        array,
        linear,
        shift,
        origin=None,
        *args, **kwargs):
    """
    Perform an affine transformation followed by a translation.

    Parameters
    ==========
    array : ndarray
        The array to operate with.
    linear : ndarray
        The linear transformation to apply.
    shift : ndarray
        The translation vector.
    origin : ndarray (optional)
        The offset at which the linear transformation is to be applied.
        If None, uses the center of the array.
    args : tuple (optional)
    kwargs : dict (optional)
        Additional arguments to be passed to the transformation function:
        `scipy.ndimage.affine_transform()`

    Returns
    =======
    array : ndarray
        The transformed array.

    """
    # other parameters accepted by `scipy.ndimage.affine_transform` are:
    #     output=None, order=3, mode='constant', cval=0.0, prefilter=True
    if origin is None:
        origin = np.array(relative2coord([0.5] * array.ndim, array.shape))
    offset = origin - np.dot(linear, origin + shift)
    array = sp.ndimage.affine_transform(
        array, linear, offset, *args, **kwargs)
    return array


# ======================================================================
def weighted_center(
        array,
        labels=None,
        index=None):
    """
    Determine the covariance mass matrix with respect to the origin.

    \latex{\sum_i (\vec{x}_i - \vec{o}_i) (\vec{x}_i - \vec{o}_i)^T}

    for i spanning through all support space.

    Parameters
    ==========
    array : ndarray
        The input array.
    labels : ndarray or None (optional)
        Labels for objects in `array`, as generated by `ndimage.label`.
        Only used with `index`.  Dimensions must be the same as `array`.
    index : int, int tuple or None (optional)
         Labels for which to calculate centers-of-mass. If not specified,
         all labels greater than zero are used.  Only used with `labels`.
    origin : ndarray or None (optional)
        The origin to be used. If None, the weighted center is used.

    Returns
    =======
    cov : ndarray
        The covariance matrix.

    See Also
    ========
    mri_tools.geometry.tensor_of_inertia,
    mri_tools.geometry.rotatio_axes,
    mri_tools.geometry.auto_rotate,
    mri_tools.geometry.realign

    """
    # numpy.double to improve the accuracy of the norm and the weighted center
    array = array.astype(np.double)
    norm = sp.ndimage.sum(array, labels, index)
    grid = np.ogrid[[slice(0, idx) for idx in array.shape]]
    # numpy.double to improve the accuracy of the result
    center = np.zeros(array.ndim).astype(np.double)
    for idx in range(array.ndim):
        center[idx] = sp.ndimage.sum(array * grid[idx], labels, index) / norm
    return center


# ======================================================================
def weighted_covariance(
        array,
        labels=None,
        index=None,
        origin=None):
    """
    Determine the weighted covariance matrix with respect to the origin.

    \latex{\sum_i w_i (\vec{x}_i - \vec{o}) (\vec{x}_i - \vec{o})^T}

    for i spanning through all support space, where:
    o is the origin vector (
    x_i is the coordinate vector of the point i
    w_i is the weight, i.e. the value of the array at that coordinate


    Parameters
    ==========
    array : ndarray
        The input array.
    labels : ndarray or None (optional)
        Labels for objects in `array`, as generated by `ndimage.label`.
        Only used with `index`.  Dimensions must be the same as `array`.
    index : int, int tuple or None (optional)
         Labels for which to calculate centers-of-mass. If not specified,
         all labels greater than zero are used.  Only used with `labels`.
    origin : ndarray or None (optional)
        The origin to be used. If None, the weighted center is used.

    Returns
    =======
    cov : ndarray
        The covariance matrix.

    See Also
    ========
    mri_tools.geometry.tensor_of_inertia,
    mri_tools.geometry.rotation_axes,
    mri_tools.geometry.auto_rotate,
    mri_tools.geometry.realign

    """
    # numpy.double to improve the accuracy of the norm and the weighted center
    array = array.astype(np.double)
    norm = sp.ndimage.sum(array, labels, index)
    if origin is None:
        origin = np.array(sp.ndimage.center_of_mass(array, labels, index))
    grid = np.ogrid[[slice(0, idx) for idx in array.shape]] - origin
    # numpy.double to improve the accuracy of the result
    cov = np.zeros((array.ndim, array.ndim)).astype(np.double)
    for idx in range(array.ndim):
        for jdx in range(array.ndim):
            if idx <= jdx:
                cov[idx, jdx] = sp.ndimage.sum(
                    array * grid[idx] * grid[jdx], labels, index) / norm
            else:
                # the covariance mass matrix is symmetric
                cov[idx, jdx] = cov[jdx, idx]
    return cov


# ======================================================================
def tensor_of_inertia(
        array,
        labels=None,
        index=None,
        origin=None):
    """
    Determine the tensor of inertia with respect to the origin.

    I = Id * tr(C) - C

    Parameters
    ==========
    array : ndarray
        The input array.
    labels : ndarray or None (optional)
        Labels for objects in `array`, as generated by `ndimage.label`.
        Only used with `index`.  Dimensions must be the same as `array`.
    index : int, int tuple or None (optional)
         Labels for which to calculate the weighted center. If not specified,
         all labels greater than zero are used.  Only used with `labels`.
    origin : ndarray or None (optional)
        The origin to be used. If None, the weighted center is used.

    Returns
    =======
    inertia : ndarray
        The tensor (rank 2, i.e. a matrix) of inertia.

    See Also
    ========
    mri_tools.geometry.weighted_covariance,
    mri_tools.geometry.rotatio_axes,
    mri_tools.geometry.auto_rotate,
    mri_tools.geometry.realign

    """
    cov = weighted_covariance(array, labels, index, origin)
    inertia = np.eye(array.ndim) * np.trace(cov) - cov
    return inertia


# ======================================================================
def rotation_axes(
        array,
        labels=None,
        index=None,
        sort_by_shape=False):
    """
    Calculate principal axes of rotation.

    Eigenvectors of the tensor of inertia.

    Parameters
    ==========
    array : ndarray
        The input array.
    labels : ndarray or None (optional)
        Labels for objects in `array`, as generated by `ndimage.label`.
        Only used with `index`.  Dimensions must be the same as `array`.
    index : int, int tuple or None (optional)
         Labels for which to calculate the weighted center. If not specified,
         all labels greater than zero are used.  Only used with `labels`.
    origin : ndarray or None (optional)
        The origin to be used. If None, the weighted center is used.
    sort_by_shape : bool
        If True, sort by the array shape (optimizes rotation to fit array).
        Otherwise it is sorted by increasing eigenvalue.

    Returns
    =======
    axes : ndarray
        A matrix containing the axes of rotation as columns.

    See Also
    ========
    mri_tools.geometry.weighted_covariance,
    mri_tools.geometry.tensor_of_inertia,
    mri_tools.geometry.auto_rotate,
    mri_tools.geometry.realign,

    """
    # calculate the tensor of inertia with respect to the weighted center
    inertia = tensor_of_inertia(array, labels, index, None).astype(np.double)
    # numpy.linalg only supports up to numpy.double
    eigenvalues, eigenvectors = np.linalg.eigh(inertia)
    if sort_by_shape:
        tmp = [
            (size, eigenvalue, eigenvector)
            for size, eigenvalue, eigenvector
            in zip(
                sorted(array.shape, reverse=True),
                eigenvalues,
                tuple(eigenvectors.transpose()))]
        tmp = sorted(tmp, key=lambda x: array.shape.index(x[0]))
        axes = []
        for size, eigenvalue, eigenvector in tmp:
            axes.append(eigenvector)
        axes = np.array(axes).transpose()
    else:
        axes = eigenvectors
    return axes


# ======================================================================
def auto_rotate(
        array,
        labels=None,
        index=None,
        origin=None,
        *args,
        **kwargs):
    """
    Rotate the array to have the principal axes of rotation along the axes.

    Parameters
    ==========
    array : ndarray
        The array to rotate along its principal axes.
    labels : ndarray or None (optional)
        Labels for objects in `array`, as generated by `ndimage.label`.
        Only used with `index`.  Dimensions must be the same as `array`.
    index : int, int tuple or None (optional)
         Labels for which to calculate the weighted center. If not specified,
         all labels greater than zero are used.  Only used with `labels`.
    origin : ndarray or None (optional)
        The origin to be used. If None, the weighted center is used.
    args : tuple (optional)
    kwargs : dict (optional)
        Additional arguments to be passed to the transformation function:
        `scipy.ndimage.affine_transform()`

    Returns
    =======
    rotated : ndarray
        The rotated array.
    rot_matrix : ndarray
        The rotation matrix used.
    offset : ndarray
        The offset used.

    See Also
    ========
    scipy.ndimage.center_of_mass,
    scipy.ndimage.affine_transform,
    mri_tools.weighted_covariance,
    mri_tools.tensor_of_inertia,
    mri_tools.rotation_axes,
    mri_tools.angles2linear,
    mri_tools.linear2angles,
    mri_tools.auto_rotate,
    mri_tools.realign

    """
    rot_matrix = rotation_axes(array, labels, index, True)
    if origin is None:
        origin = np.array(relative2coord([0.5] * array.ndim, array.shape))
    offset = origin - np.dot(rot_matrix, origin)
    rotated = sp.ndimage.affine_transform(
        array, rot_matrix, offset, *args, **kwargs)
    return rotated, rot_matrix, offset


# ======================================================================
def auto_shift(
        array,
        labels=None,
        index=None,
        origin=None,
        *args,
        **kwargs):
    """
    Shift the array to have the weighted center in a convenient location.

    Parameters
    ==========
    array : ndarray
        The array to autorotate along its principal axes.
    labels : ndarray or None (optional)
        Labels for objects in `array`, as generated by `ndimage.label`.
        Only used with `index`.  Dimensions must be the same as `array`.
    index : int, int tuple or None (optional)
         Labels for which to calculate the weighted center. If not specified,
         all labels greater than zero are used.  Only used with `labels`.
    origin : ndarray or None (optional)
        The origin to be used. If None, the weighted center is used.
    args : tuple (optional)
    kwargs : dict (optional)
        Additional arguments to be passed to the transformation function:
        `scipy.ndimage.affine_transform()`

    Returns
    =======
    rotated : ndarray
        The rotated array.
    offset : ndarray
        The offset used.

    See Also
    ========
    scipy.ndimage.center_of_mass,
    scipy.ndimage.affine_transform,
    mri_tools.weighted_covariance,
    mri_tools.tensor_of_inertia,
    mri_tools.rotation_axes,
    mri_tools.angles2linear,
    mri_tools.linear2angles,
    mri_tools.auto_rotate,
    mri_tools.realign

    """
    if origin is None:
        origin = relative2coord([0.5] * array.ndim, array.shape)
    com = np.array(sp.ndimage.center_of_mass(array, labels, index))
    offset = com - origin
    shifted = sp.ndimage.affine_transform(
        array, np.eye(array.ndim), offset, *args, **kwargs)
    return shifted, offset


# ======================================================================
def realign(
        array,
        labels=None,
        index=None,
        origin=None,
        *args,
        **kwargs):
    """
    Shift and rotate the array for optimal grid alignment.

    Principal axis of rotation will be parallel to cartesian axes.
    Weighted center will be at a given point (e.g. the middle of the support).

    Parameters
    ==========
    array : ndarray
        The array to autorotate along its principal axes.
    labels : ndarray or None (optional)
        Labels for objects in `array`, as generated by `ndimage.label`.
        Only used with `index`.  Dimensions must be the same as `array`.
    index : int, int tuple or None (optional)
         Labels for which to calculate the weighted center. If not specified,
         all labels greater than zero are used.  Only used with `labels`.
    origin : ndarray or None (optional)
        The final position of the weighted center. If None, set to the middle.
    args : tuple (optional)
    kwargs : dict (optional)
        Additional arguments to be passed to the transformation function:
        `scipy.ndimage.affine_transform()`

    Returns
    =======
    rotated : ndarray
        The rotated array.
    rot_matrix : ndarray
        The rotation matrix used.
    offset : ndarray
        The offset used.

    See Also
    ========
    scipy.ndimage.center_of_mass,
    scipy.ndimage.affine_transform,
    mri_tools.weighted_covariance,
    mri_tools.tensor_of_inertia,
    mri_tools.rotation_axes,
    mri_tools.angles2linear,
    mri_tools.linear2angles,
    mri_tools.auto_rotate,
    mri_tools.auto_shift

    """
    com = np.array(sp.ndimage.center_of_mass(array, labels, index))
    rot_matrix = rotation_axes(array, labels, index, True)
    if origin is None:
        origin = np.array(relative2coord([0.5] * array.ndim, array.shape))
    offset = com - np.dot(rot_matrix, origin)
    aligned = sp.ndimage.affine_transform(
        array, rot_matrix, offset, *args, **kwargs)
    return aligned, rot_matrix, offset


# ======================================================================
def _self_test_interactive():
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
    mrp.quick(square(shape_2d, position_2d, D_LENGTH_1))
    mrp.quick(rectangle(shape_2d, position_2d, D_LENGTH_2))
    mrp.quick(rhombus(shape_2d, position_2d, D_LENGTH_2))
    mrp.quick(circle(shape_2d, position_2d, D_LENGTH_1))
    mrp.quick(ellipsis(shape_2d, position_2d, D_LENGTH_2))
    # :: - Position test
    mrp.quick(ellipsis(shape_2d, (0.2, 0.7), D_LENGTH_2))

    # :: 3D Tests
    shape_3d = tuple([D_SHAPE] * 3)
    position_3d = tuple([D_POSITION] * 3)
    # :: - shape test
    mrp.quick(cube(shape_3d, position_3d, D_LENGTH_1))
    mrp.quick(cuboid(shape_3d, position_3d, D_LENGTH_3))
    mrp.quick(rhomboid(shape_3d, position_3d, D_LENGTH_3))
    mrp.quick(sphere(shape_3d, position_3d, D_LENGTH_1))
    mrp.quick(ellipsoid(shape_3d, position_3d, D_LENGTH_3))
    mrp.quick(cylinder(shape_3d, position_3d, 2.0 * D_LENGTH_1, D_LENGTH_1))
    # :: - Position test
    mrp.quick(ellipsoid(shape_3d, (0.0, 1.0, 0.5), D_LENGTH_3))

    mlab.show()


# ======================================================================
if __name__ == '__main__':
    print(__doc__)
    begin_time = time.time()
    # _self_test_interactive()
    end_time = time.time()
    print('ExecTime: ', datetime.timedelta(0, end_time - begin_time))
