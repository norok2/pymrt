#!python
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
# import unittest  # Unit testing framework
import doctest  # Test interactive Python examples

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
# import scipy.optimize  # SciPy: Optimization Algorithms
# import scipy.integrate  # SciPy: Integrations facilities
# import scipy.constants  # SciPy: Mathematical and Physical Constants
import scipy.ndimage  # SciPy: ND-image Manipulation

# :: Local Imports
import pymrt.base as mrb
import pymrt.plot as mrp

# from pymrt import INFO
# from pymrt import VERB_LVL
# from pymrt import D_VERB_LVL
# from pymrt import get_first_line


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
def relative2coord(shape, position):
    """
    Calculate the absolute position with respect to a given shape.

    Args:
        shape (tuple[int]): The shape of the mask in px.
        position (tuple[float]): Relative position (to the lowest edge).
            Values are in the range [0, 1].

    Returns:
        coord (tuple[int]): Coordinate of the position inside the shape.
    """
    if len(position) != len(shape):
        raise IndexError('length of tuples must match')
    return tuple((s - 1.0) * p for p, s in zip(position, shape))


# ======================================================================
def coord2relative(coord, shape):
    """
    Calculate the proportional position with respect to a given shape.

    Args:
        coord (tuple[int]): Coordinate of the position inside the shape.
        shape (tuple[int]): The shape of the mask in px.

    Returns:
        position (tuple[float]): Relative position (to the lowest edge).
            Values are in the range [0, 1].
    """
    if len(coord) != len(shape):
        raise IndexError('length of tuples must match')
    return tuple(c / (s - 1.0) for c, s in zip(coord, shape))


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
        dtype (np.dtype): Desired output data-type.

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
        position,
        side):
    """
    Generate a mask whose shape is a square.

    Args:
        shape (tuple[int]): The shape of the mask in px.
        position (tuple[float]): Relative position (to the lowest edge).
            Values are in the range [0, 1].
        side (float): The side of the square in px.

    Returns:
        mask (np.ndarray): Array of boolean describing the geometrical object.
    """
    # set correct dimensions
    n_dim = 2
    # check parameters
    if not (len(shape) == len(position) == n_dim):
        e_msg = 'length of tuples must be {}'.format(n_dim)
        raise IndexError(e_msg)
    # adjust semisides
    semisides = (side / 2.0,) * n_dim
    # use n-dim function
    return nd_cuboid(shape, position, semisides)


# ======================================================================
def rectangle(
        shape,
        position,
        semisides):
    """
    Generate a mask whose shape is a rectangle.

    Args:
        shape (tuple[int]): The shape of the mask in px.
        position (tuple[float]): Relative position (to the lowest edge).
            Values are in the range [0, 1].
        semisides (tuple[float]): The semisides of the rectangle in px.

    Returns:
        mask (np.ndarray): Array of boolean describing the geometrical object.
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

    Args:
        shape (tuple[int]): The shape of the mask in px.
        position (tuple[float]): Relative position (to the lowest edge).
            Values are in the range [0, 1].
        semidiagonals (tuple[float]): The semidiagonals of the rhombus in px.

    Returns:
        mask (np.ndarray): Array of boolean describing the geometrical object.
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
    index = (rhombus_index,) * n_dim
    # use n-dim function
    return nd_superellipsoid(shape, position, semiaxes, index)


# ======================================================================
def circle(
        shape,
        position,
        radius):
    """
    Generate a mask whose shape is a circle.

    Args:
        shape (tuple[int]): The shape of the mask in px.
        position (tuple[float]): Relative position (to the lowest edge).
            Values are in the range [0, 1].
        radius (float): The radius of the circle in px.

    Returns:
        mask (np.ndarray): Array of boolean describing the geometrical object.
    """
    # set correct dimensions
    n_dim = 2
    # check parameters
    if not (len(shape) == len(position) == n_dim):
        e_msg = 'length of tuples must be {}'.format(n_dim)
        raise IndexError(e_msg)
    # adjust semiaxes
    semiaxes = (radius,) * n_dim
    # set index value
    circle_index = 2.0
    index = (circle_index,) * n_dim
    # use n-dim function
    return nd_superellipsoid(shape, position, semiaxes, index)


# ======================================================================
def ellipsis(
        shape,
        position,
        semiaxes):
    """
    Generate a mask whose shape is an ellipsis.

    Args:
        shape (tuple[int]): The shape of the mask in px.
        position (tuple[float]): Relative position (to the lowest edge).
            Values are in the range [0, 1].
        semiaxes (tuple[float]): The semiaxes of the ellipsis in px.

    Returns:
        mask (np.ndarray): Array of boolean describing the geometrical object.
    """
    # set correct dimensions
    n_dim = 2
    # check parameters
    if not (len(shape) == len(position) == len(semiaxes) == n_dim):
        e_msg = 'length of tuples must be {}'.format(n_dim)
        raise IndexError(e_msg)
    # set index value
    ellipsis_index = 2.0
    index = (ellipsis_index,) * n_dim
    # use n-dim function
    return nd_superellipsoid(shape, position, semiaxes, index)


# ======================================================================
def cube(
        shape,
        position,
        side):
    """
    Generate a mask whose shape is a cube.

    Args:
        shape (tuple[int]): The shape of the mask in px.
        position (tuple[float]): Relative position (to the lowest edge).
            Values are in the range [0, 1].
        side (float): The side of the cube in px.

    Returns:
        mask (np.ndarray): Array of boolean describing the geometrical object.
    """
    # set correct dimensions
    n_dim = 3
    # check parameters
    if not (len(shape) == len(position) == n_dim):
        e_msg = 'length of tuples must be {}'.format(n_dim)
        raise IndexError(e_msg)
    # adjust semisides
    semisides = (side / 2.0,) * n_dim
    # use n-dim function
    return nd_cuboid(shape, position, semisides)


# ======================================================================
def cuboid(
        shape,
        position,
        semisides):
    """
    Generate a mask whose shape is a cuboid.

    Args:
        shape (tuple[int]): The shape of the mask in px.
        position (tuple[float]): Relative position (to the lowest edge).
            Values are in the range [0, 1].
        semisides (tuple[float]): The semisides of the cuboid in px.

    Returns:
        mask (np.ndarray): Array of boolean describing the geometrical object.
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

    Args:
        shape (tuple[int]): The shape of the mask in px.
        position (tuple[float]): Relative position (to the lowest edge).
            Values are in the range [0, 1].
        semidiagonals (tuple[float]): The semidiagonals of the rhomboid in px.

    Returns:
        mask (np.ndarray): Array of boolean describing the geometrical object.
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
    index = (rhombus_index,) * n_dim
    # use n-dim function
    return nd_superellipsoid(shape, position, semiaxes, index)


# ======================================================================
def sphere(
        shape,
        position,
        radius):
    """
    Generate a mask whose shape is a sphere.

    Args:
        shape (tuple[int]): The shape of the mask in px.
        position (tuple[float]): Relative position (to the lowest edge).
            Values are in the range [0, 1].
        radius (float): The radius of the sphere in px.

    Returns:
        mask (np.ndarray): Array of boolean describing the geometrical object.
    """
    # set correct dimensions
    n_dim = 3
    # check parameters
    if not (len(shape) == len(position) == n_dim):
        e_msg = 'length of tuples must be {}'.format(n_dim)
        raise IndexError(e_msg)
    # adjust semiaxes
    semiaxes = (radius,) * n_dim
    # set index value
    circle_index = 2.0
    index = (circle_index,) * n_dim
    # use n-dim function
    return nd_superellipsoid(shape, position, semiaxes, index)


# ======================================================================
def ellipsoid(
        shape,
        position,
        semiaxes):
    """
    Generate a mask whose shape is an ellipsoid.

    Args:
        shape (tuple[int]): The shape of the mask in px.
        position (tuple[float]): Relative position (to the lowest edge).
            Values are in the range [0, 1].
        semiaxes (tuple[float]): The semiaxes of the ellipsoid in px.

    Returns:
        mask (np.ndarray): Array of boolean describing the geometrical object.
    """
    # set correct dimensions
    n_dim = 3
    # check parameters
    if not (len(shape) == len(position) == len(semiaxes) == n_dim):
        e_msg = 'length of tuples must be {}'.format(n_dim)
        raise IndexError(e_msg)
    # set index value
    ellipsis_index = 2.0
    index = (ellipsis_index,) * n_dim
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

    Args:
        shape (tuple[int]): The shape of the mask in px.
        position (tuple[float]): Relative position (to the lowest edge).
            Values are in the range [0, 1].
        height (float): The height of the cylinder in px.
        radius (float): The radius of the cylinder in px.

    Returns:
        mask (np.ndarray): Array of boolean describing the geometrical object.
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
    Generate a mask whose shape is an ND cuboid.
    The cartesian equations are: sum[abs(x_n/a_n)^inf] < 1.0

    Args:
        shape (tuple[int]): The shape of the mask in px.
        position (tuple[float]): Relative position (to the lowest edge).
            Values are in the range [0, 1].
        semisides (tuple[float]): The semisides of the ND cuboid in px.

    Returns:
        mask (np.ndarray): Array of boolean describing the geometrical object.
    """
    # check compatibility of given parameters
    if not (len(shape) == len(position) == len(semisides)):
        raise IndexError('length of tuples must match')
    # calculate the position of the center of the solid inside the mask
    x_0 = relative2coord(shape, position)
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
    Generate a mask whose shape is an ND superellipsoid.
    The cartesian equations are: sum[abs(x_n/a_n)^k] < 1.0

    Args:
        shape (tuple[int]): The shape of the mask in px.
        position (tuple[float]): Relative position (to the lowest edge).
            Values are in the range [0, 1].
        semiaxes (tuple[float]): The semiaxes of the ND superellipsoid in px.
        indexes (tuple[float]): The exponent of the summed terms.

    Returns:
        mask (np.ndarray): Array of boolean describing the geometrical object.
    """
    # check compatibility of given parameters
    if not (len(shape) == len(position) == len(semiaxes) == len(indexes)):
        raise IndexError('length of tuples must match')
    # calculate the position of the center of the solid inside the mask
    x_0 = relative2coord(shape, position)
    # create the grid with origin in the middle
    grid = [slice(-x0c, dim - x0c) for x0c, dim in zip(x_0, shape)]
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
        height):
    """
    Generate a mask whose shape is a ND prism.

    Args:
        base_mask (np.ndarray): Base (N-1)-dim mask to stack as prism.
        extra_shape (int): Size of the new dimension to be added.
        axis (int): Orientation of the prism in the N-dim space.
        position (float): Relative position (to the lowest edge).
            This setting only affects the extra shape dimension.
            Values are in the range [0, 1].
        height (float): The height of the prism.

    Returns:
        mask (np.ndarray): Array of boolean describing the geometrical object.
    """
    n_dim = base_mask.ndim + 1
    if axis > n_dim:
        raise ValueError(
            'axis of orientation must not exceed the number of dimensions')
    # calculate the position of the center of the solid inside the mask
    x_0 = relative2coord((extra_shape,), (position,))[0]
    # create a grid with origin in the middle
    iii = slice(-x_0, extra_shape - x_0)
    xxx = np.ogrid[iii]
    # create the extra mask (height)
    extra_mask = np.abs(xxx) < (height / 2.0)
    # calculate mask shape
    shape = (
        base_mask.shape[:axis] + (extra_shape,) + base_mask.shape[axis:])
    # create indefinite prism
    mask = np.zeros(shape, dtype=bool)
    for i in range(extra_shape):
        if extra_mask[i]:
            index = [slice(None)] * n_dim
            index[axis] = i
            mask[tuple(index)] = base_mask
    return mask


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
        borders (float|tuple[float]): The relative border size.
            This is proportional to the initial array shape.
            If 'use_longest' is True, use the longest dimension for the
            calculations.
        background (int|float): The background value to be used for the frame.
        use_longest (bool): Use longest dimension to get the border size.

    Returns:
        result (np.ndarray): The result array with added borders.

    See Also:
        reframe
    """
    try:
        iter(borders)
    except TypeError:
        borders = [borders] * arr.ndim
    if any(borders) < 0:
        raise ValueError('relative border cannot be negative')
    if use_longest:
        dim = max(arr.shape)
        borders = [round(border * dim) for border in borders]
    else:
        borders = [
            round(border * dim) for dim, border in zip(arr.shape, borders)]
    result = background * np.ones(
        [dim + 2 * border for dim, border in zip(arr.shape, borders)])
    inner = [
        slice(border, border + dim, None) \
        for dim, border in zip(arr.shape, borders)]
    result[inner] = arr
    return result


# ======================================================================
def reframe(
        arr,
        new_shape,
        background=0.0):
    """
    Add a frame to an array by centering the input array into a new shape.

    Args:
        arr (np.ndarray): The input array.
        new_shape (tuple[int]): The shape of the output array.
            The number of dimensions between the input and the output array
            must match. Additionally, each dimensions of the new shape cannot
        background (int|float): The background value to be used for the frame.

    Returns:
        result (np.ndarray): The result array with added borders.

    Raises:
        IndexError: input and output shape sizes must match.
        ValueError: output shape cannot be smaller than the input shape.

    See Also:
        frame
    """
    if arr.ndim != len(new_shape):
        raise IndexError('number of dimensions must match')
    elif any([old > new for old, new in zip(arr.shape, new_shape)]):
        raise ValueError('new shape cannot be smaller than the old one.')
    result = background * np.ones(new_shape)
    borders = [
        round((new - old) / 2.0) for old, new in zip(arr.shape, new_shape)]
    inner = [
        slice(border, border + dim, None)
        for dim, border in zip(arr.shape, borders)]
    result[inner] = arr
    return result


# ======================================================================
def zoom_prepare(
        zoom,
        shape,
        extra_dim=True,
        fill_dim=True):
    """
    Prepare the zoom and shape tuples to allow for non-homogeneous shapes.

    Args:
        zoom (float|tuple[float]): The zoom factors for each directions.
        shape (tuple[int]): The shape of the array to operate with.
        extra_dim (bool): Force extra dimensions in the zoom parameters.
        fill_dim (bool): Dimensions not specified are left untouched.

    Returns:
        zoom (tuple[float]): The zoom factors for each directions.
        shape (tuple[int]): The shape of the array to operate with.
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
        aspect=None):
    """
    Calculate zoom (or conversion) factor between two shapes.

    Args:
        old_shape (tuple[int]): The shape of the source array.
        new_shape (tuple[int]): The target shape of the array.
        aspect (callable|None): Function for the manipulation of the zoom.
            Signature: aspect(tuple[float]) -> float.
            None to leave the zoom unmodified. It specified, the function is
            applied to zoom factors tuple for fine tuning of the aspect.
            Particularly, to obtain specific aspect ratio results:
             - 'min': image strictly contained into new shape
             - 'max': new shape strictly contained into image

    Returns:
        zoom (tuple[float]): The zoom factors for each directions.
    """
    if len(old_shape) != len(new_shape):
        raise IndexError('length of tuples must match')
    zoom = [new / old for old, new in zip(old_shape, new_shape)]
    if aspect:
        zoom = [aspect(zoom)] * len(zoom)
    return zoom


# ======================================================================
def apply_to_complex(
        arr,
        func,
        *args,
        **kwargs):
    """
    Apply a real-valued function to real and imaginary part of an array.

    This can be useful for geometric transformations, particularly affines,
    and generally for functions that are explicitly restricted to real values.

    Args:
        arr (np.ndarray): The N-dim array to be transformed.
        func (callable): Filtering function.
            func(arr, *args, **kwargs) -> arr
        args (tuple): Positional arguments passed to the filtering function.
        kwargs (dict): Keyword arguments passed to the filtering function.

    Returns:
        array (np.ndarray): The transformed N-dim array.
    """
    real = func(np.real(arr), *args, **kwargs)
    imag = func(np.imag(arr), *args, **kwargs)
    arr = mrb.cartesian2complex(real, imag)
    return arr


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
def num_angles_from_dim(num_dim):
    """
    Calculate the complete number of angles given the dimension.

    Given the dimension of an array, calculate the number of all possible
    cartesian orthogonal planes of rotations, using the formula:

    N = n * (n - 1) / 2 [ = n! / 2! / (n - 2)! ]
    (N: num of angles, n: num of dim)

    Args:
        num_dim (int): The number of dimensions.

    Returns:
        num_angles (int): The corresponding number of angles.

    See Also:
        angles2linear
    """
    return num_dim * (num_dim - 1) // 2


# ======================================================================
def angles2linear(
        angles,
        axes_list=None,
        use_degree=True,
        tol=2e-6):
    """
    Calculate the linear transformation relative to the specified rotations.

    Args:
        angles (tuple[float]): The angles to be used for rotation.
        axes_list (tuple[tuple[int]]|None): The axes of the rotation plane.
            If not None, for each rotation angle a pair of axes must be
            specified to define the associated plane of rotation.
            If None, uses output of `itertools.combinations(range(n_dim), 2)`.
        use_degree (bool): Interpret angles as expressed in degree.
            Otherwise, use radians.

    Returns:
        linear (np.ndarray): The rotation matrix as defined by the angles.

    See Also:
        num_angles_from_dim,
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
    if np.abs(det) - 1.0 > tol * np.finfo(float).eps:
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
        arr,
        linear,
        shift,
        origin=None,
        *args,
        **kwargs):
    """
    Perform an affine transformation followed by a translation on the array.

    Args:
        arr (np.ndarray): The N-dim array to operate with.
        linear (np.ndarray): The N-sized linear square matrix.
        shift (np.ndarray): The shift along each axis in px.
        origin (np.ndarray): The origin vector of the linear transformation.
            If None, uses the center of the array.
        args (tuple|None): Positional arguments for the transforming function.
            If not None, they are passed to `scipy.ndimage.affine_transform`.
        kwargs (dict|None): Keyword arguments for the transforming function.
            If not None, they are passed to `scipy.ndimage.affine_transform`.

    Returns:
        array (np.ndarray): The transformed array.

    See Also:
        scipy.ndimage.affine_transform
    """
    # other parameters accepted by `scipy.ndimage.affine_transform` are:
    #     output=None, order=3, mode='constant', cval=0.0, prefilter=True
    if args is None:
        args = ()
    if kwargs is None:
        kwargs = {}
    if origin is None:
        origin = np.array(relative2coord(arr.shape, (0.5,) * arr.ndim))
    offset = origin - np.dot(linear, origin + shift)
    arr = sp.ndimage.affine_transform(arr, linear, offset, *args, **kwargs)
    return arr


# ======================================================================
def weighted_center(
        arr,
        labels=None,
        index=None):
    """
    Determine the covariance mass matrix with respect to the origin.

    $$\sum_i (\vec{x}_i - \vec{o}_i) (\vec{x}_i - \vec{o}_i)^T$$

    for i spanning through all support space.

    Args:
        arr (np.ndarray): The input array.
        labels (np.ndarray|None): Cumulative mask array for the objects.
            The output of `scipy.ndimage.label` is expected.
            The number of dimensions must be the same as `array`.
            Only uses the labels as indicated by `index`.
        index (int|tuple[int]|None): Labels used for the calculation.
            If an int, uses all labels between 1 and the specified value.
            If a tuple of int, uses only the selected labels.
            If None, uses all positive labels.

    Returns:
        center (np.ndarray): The coordinates of the weighed center.

    See Also:
        pymrt.geometry.tensor_of_inertia,
        pymrt.geometry.rotatio_axes,
        pymrt.geometry.auto_rotate,
        pymrt.geometry.realign
    """
    # numpy.double to improve the accuracy of the norm and the weighted center
    arr = arr.astype(np.double)
    norm = sp.ndimage.sum(arr, labels, index)
    grid = np.ogrid[[slice(0, i) for i in array.shape]]
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

    \[\sum_i w_i (\vec{x}_i - \vec{o}) (\vec{x}_i - \vec{o})^T\]

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
        index (int|tuple[int]|None): Labels used for the calculation.
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
        pymrt.geometry.auto_rotate,
        pymrt.geometry.realign
    """
    # numpy.double to improve the accuracy of the norm and the weighted center
    arr = arr.astype(np.double)
    norm = sp.ndimage.sum(arr, labels, index)
    if origin is None:
        origin = np.array(sp.ndimage.center_of_mass(arr, labels, index))
    grid = np.ogrid[[slice(0, i) for i in array.shape]] - origin
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
        index (int|tuple[int]|None): Labels used for the calculation.
            If an int, uses all labels between 1 and the specified value.
            If a tuple of int, uses only the selected labels.
            If None, uses all positive labels.
        origin (np.ndarray|None): The origin for the covariance matrix.
            If None, the weighted center is used.

    Returns:
        inertia (np.ndarray): The tensor of inertia from the origin.

    See Also:
        pymrt.geometry.weighted_covariance,
        pymrt.geometry.rotation_axes,
        pymrt.geometry.auto_rotate,
        pymrt.geometry.realign
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
    Calculate principal axes of rotation.

    These can be found as the eigenvectors of the tensor of inertia.

    Args:
        arr (np.ndarray): The input array.
        labels (np.ndarray|None): Cumulative mask array for the objects.
            The output of `scipy.ndimage.label` is expected.
            The number of dimensions must be the same as `array`.
            Only uses the labels as indicated by `index`.
        index (int|tuple[int]|None): Labels used for the calculation.
            If an int, uses all labels between 1 and the specified value.
            If a tuple of int, uses only the selected labels.
            If None, uses all positive labels.
        sort_by_shape (bool): Sort the axes by the array shape.
            This is useful in order to obtain the optimal rotations to
            align the objects to the shape.
            Otherwise, it is sorted by increasing eigenvalues.

    Returns:
        axes (ndarray): A matrix containing the axes of rotation as columns.

    See Also:
        pymrt.geometry.weighted_covariance,
        pymrt.geometry.tensor_of_inertia,
        pymrt.geometry.auto_rotate,
        pymrt.geometry.realign
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
        axes = np.array(axes).transpose()
    else:
        axes = eigenvectors
    return axes


# ======================================================================
def auto_rotate(
        arr,
        labels=None,
        index=None,
        origin=None,
        *args,
        **kwargs):
    """
    Rotate the array to have the principal axes of rotation along the axes.

    Args:
        arr (np.ndarray): The input array.
        labels (np.ndarray|None): Cumulative mask array for the objects.
            The output of `scipy.ndimage.label` is expected.
            The number of dimensions must be the same as `array`.
            Only uses the labels as indicated by `index`.
        index (int|tuple[int]|None): Labels used for the calculation.
            If an int, uses all labels between 1 and the specified value.
            If a tuple of int, uses only the selected labels.
            If None, uses all positive labels.
        origin (np.ndarray|None): The origin for the covariance matrix.
            If None, the weighted center is used.
        args (tuple|None): Positional arguments for the transforming function.
            If not None, they are passed to `scipy.ndimage.affine_transform`.
        kwargs (dict|None): Keyword arguments for the transforming function.
            If not None, they are passed to `scipy.ndimage.affine_transform`.

    Returns:
        rotated (np.ndarray): The rotated array.
        rot_matrix (np.ndarray): The rotation matrix used.
        offset (np.ndarray): The offset used.

    See Also:
        scipy.ndimage.center_of_mass,
        scipy.ndimage.affine_transform,
        pymrt.weighted_covariance,
        pymrt.tensor_of_inertia,
        pymrt.rotation_axes,
        pymrt.angles2linear,
        pymrt.linear2angles,
        pymrt.auto_rotate,
        pymrt.realign
    """
    if args is None:
        args = ()
    if kwargs is None:
        kwargs = {}
    rot_matrix = rotation_axes(arr, labels, index, True)
    if origin is None:
        origin = np.array(relative2coord(arr.shape, (0.5,) * arr.ndim))
    offset = origin - np.dot(rot_matrix, origin)
    rotated = sp.ndimage.affine_transform(
        arr, rot_matrix, offset, *args, **kwargs)
    return rotated, rot_matrix, offset


# ======================================================================
def auto_shift(
        arr,
        labels=None,
        index=None,
        origin=None,
        *args,
        **kwargs):
    """
    Shift the array to have the weighted center in a convenient location.

    Args:
        arr (np.ndarray): The input array.
        labels (np.ndarray|None): Cumulative mask array for the objects.
            The output of `scipy.ndimage.label` is expected.
            The number of dimensions must be the same as `array`.
            Only uses the labels as indicated by `index`.
        index (int|tuple[int]|None): Labels used for the calculation.
            If an int, uses all labels between 1 and the specified value.
            If a tuple of int, uses only the selected labels.
            If None, uses all positive labels.
        origin (np.ndarray|None): The origin for the covariance matrix.
            If None, the weighted center is used.
        args (tuple|None): Positional arguments for the transforming function.
            If not None, they are passed to `scipy.ndimage.affine_transform`.
        kwargs (dict|None): Keyword arguments for the transforming function.
            If not None, they are passed to `scipy.ndimage.affine_transform`.

    Returns:
        rotated (np.ndarray): The rotated array.
        offset (np.ndarray): The offset used.

    See Also:
        scipy.ndimage.center_of_mass,
        scipy.ndimage.affine_transform,
        pymrt.weighted_covariance,
        pymrt.tensor_of_inertia,
        pymrt.rotation_axes,
        pymrt.angles2linear,
        pymrt.linear2angles,
        pymrt.auto_rotate,
        pymrt.realign
    """
    if args is None:
        args = ()
    if kwargs is None:
        kwargs = {}
    if origin is None:
        origin = relative2coord(arr.shape, (0.5,) * arr.ndim)
    com = np.array(sp.ndimage.center_of_mass(arr, labels, index))
    offset = com - origin
    shifted = sp.ndimage.affine_transform(
        arr, np.eye(arr.ndim), offset, *args, **kwargs)
    return shifted, offset


# ======================================================================
def realign(
        arr,
        labels=None,
        index=None,
        origin=None,
        *args,
        **kwargs):
    """
    Shift and rotate the array for optimal grid alignment.

    Principal axis of rotation will be parallel to cartesian axes.
    Weighted center will be at a given point (e.g. the middle of the support).

    Args:
        arr (np.ndarray): The input array.
        labels (np.ndarray|None): Cumulative mask array for the objects.
            The output of `scipy.ndimage.label` is expected.
            The number of dimensions must be the same as `array`.
            Only uses the labels as indicated by `index`.
        index (int|tuple[int]|None): Labels used for the calculation.
            If an int, uses all labels between 1 and the specified value.
            If a tuple of int, uses only the selected labels.
            If None, uses all positive labels.
        origin (np.ndarray|None): The origin for the covariance matrix.
            If None, the weighted center is used.
        args (tuple|None): Positional arguments for the transforming function.
            If not None, they are passed to `scipy.ndimage.affine_transform`.
        kwargs (dict|None): Keyword arguments for the transforming function.
            If not None, they are passed to `scipy.ndimage.affine_transform`.

    Returns:
        rotated (np.ndarray): The rotated array.
        rot_matrix (np.ndarray): The rotation matrix used.
        offset (np.ndarray): The offset used.

    See Also:
        scipy.ndimage.center_of_mass,
        scipy.ndimage.affine_transform,
        pymrt.weighted_covariance,
        pymrt.tensor_of_inertia,
        pymrt.rotation_axes,
        pymrt.angles2linear,
        pymrt.linear2angles,
        pymrt.auto_rotate,
        pymrt.auto_shift
    """
    if args is None:
        args = ()
    if kwargs is None:
        kwargs = {}
    com = np.array(sp.ndimage.center_of_mass(arr, labels, index))
    rot_matrix = rotation_axes(arr, labels, index, True)
    if origin is None:
        origin = np.array(relative2coord(arr.shape, (0.5,) * arr.ndim))
    offset = com - np.dot(rot_matrix, origin)
    aligned = sp.ndimage.affine_transform(
        arr, rot_matrix, offset, *args, **kwargs)
    return aligned, rot_matrix, offset


# ======================================================================
def _self_test_interactive():
    """
    Test the functions available in the package.

    Args:
        None

    Returns:
        None
    """
    # :: 2D Tests
    shape_2d = (D_SHAPE,) * 2
    position_2d = (D_POSITION,) * 2
    # :: - shape test
    mrp.quick(square(shape_2d, position_2d, D_LENGTH_1))
    mrp.quick(rectangle(shape_2d, position_2d, D_LENGTH_2))
    mrp.quick(rhombus(shape_2d, position_2d, D_LENGTH_2))
    mrp.quick(circle(shape_2d, position_2d, D_LENGTH_1))
    mrp.quick(ellipsis(shape_2d, position_2d, D_LENGTH_2))
    # :: - Position test
    mrp.quick(ellipsis(shape_2d, (0.2, 0.7), D_LENGTH_2))

    # :: 3D Tests
    shape_3d = (D_SHAPE,) * 3
    position_3d = (D_POSITION,) * 3
    # :: - shape test
    mrp.quick(cube(shape_3d, position_3d, D_LENGTH_1))
    mrp.quick(cuboid(shape_3d, position_3d, D_LENGTH_3))
    mrp.quick(rhomboid(shape_3d, position_3d, D_LENGTH_3))
    mrp.quick(sphere(shape_3d, position_3d, D_LENGTH_1))
    mrp.quick(ellipsoid(shape_3d, position_3d, D_LENGTH_3))
    mrp.quick(cylinder(shape_3d, position_3d, 2.0 * D_LENGTH_1, D_LENGTH_1))
    # :: - Position test
    mrp.quick(ellipsoid(shape_3d, (0.0, 1.0, 0.5), D_LENGTH_3))


# ======================================================================
if __name__ == '__main__':
    print(__doc__)
    # doctest.testmod()
    _self_test_interactive()
    mrb.elapsed('self_test_interactive')
    mrb.print_elapsed()
