#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PyMRT: generate numerical electro-magnetic fields
"""

# ======================================================================
# :: Future Imports
from __future__ import (
    division, absolute_import, print_function, unicode_literals, )

# ======================================================================
# :: Python Standard Library Imports
# import os  # Miscellaneous operating system interfaces
# import shutil  # High-level file operations
# import math  # Mathematical functions
# import time  # Time access and conversions
# import datetime  # Basic date and time types
# import operator  # Standard operators as functions
# import collections  # Container datatypes
# import argparse  # Parser for command-line options, arguments and subcommands
# import itertools  # Functions creating iterators for efficient looping
# import subprocess  # Subprocess management
# import multiprocessing  # Process-based parallelism
# import csv  # CSV File Reading and Writing [CSV: Comma-Separated Values]
# import json  # JSON encoder and decoder [JSON: JavaScript Object Notation]
# import struct  # Interpret strings as packed binary data
# import doctest  # Test interactive Python examples
# import glob  # Unix style pathname pattern expansion
# import warnings  # Warning control

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
import flyingcircus as fc

# :: External Imports Submodules
# import matplotlib.pyplot as plt  # Matplotlib's pyplot: MATLAB-like syntax
# import scipy.optimize  # SciPy: Optimization Algorithms
# import scipy.integrate  # SciPy: Integrations facilities
import scipy.constants  # SciPy: Mathematal and Physical Constants
import scipy.ndimage  # SciPy: ND-image Manipulation
import scipy.special  # SciPy: Special functions
import flyingcircus.util
import flyingcircus.num

# :: Local Imports
import pymrt as mrt
import pymrt.utils
import pymrt.geometry

from pymrt import INFO, PATH
from pymrt import VERB_LVL, D_VERB_LVL, VERB_LVL_NAMES
from pymrt import elapsed, report
from pymrt import msg, dbg


def _to_3d(
        vect,
        extra_dim=0):
    """
    Ensure that the input is 3D.

    Args:
        vect (Iterable[int|float]): The input object.
        extra_dim (int|float): The fill value for the extra dimension.
            This value is used to fill the third dimension in case of 2D input.

    Returns:
        vect (np.ndarray): A 3D array.
    """
    if vect.size == 2:
        vect_3d = np.zeros(3, dtype=vect.dtype)
        vect_3d[:2] = vect
        if extra_dim:
            vect_3d[2] = extra_dim
        return vect_3d
    elif vect.size == 3:
        return vect
    else:  # vect.size > 3 or vect.size < 2
        raise ValueError('Vector must be 2D or 3D')


# ======================================================================
class CircularLoop(object):
    """
    Circular loop object for electro-magnetic simulations.
    """

    def __init__(
            self,
            radius=0.25,
            center=(0.5, 0.5, 0.5),
            normal=(0., 0., 1.),
            current=1.):
        """
        Define the circular loop.

        Args:
            radius (int|float): The radius of the circle.
                Units are not specified.
            center (Iterable[int|float]): The center of the loop.
                This can be a 2D or 3D vector.
                Units are not specified.
            normal (Iterable[int|float]): The orientation (normal) of the loop.
                This is a 2D or 3D unit vector.
                The any magnitude information will be lost.
            current (int|float): The current circulating in the loop.
                Units are not specified.
        """
        self.center = _to_3d(np.array(center))
        self.normal = _to_3d(fc.num.normalize(normal))
        self.radius = radius
        self.current = current


# ======================================================================
class RectLoop(object):
    """
    Rectangular loop object for electro-magnetic simulations.
    """

    def __init__(
            self,
            size=0.5,
            center=(0.5, 0.5, 0.5),
            normal=(0., 0., 1.),
            current=1.):
        """
        Define the rectangular loop.

        Args:
            size (int|float|Iterable[int|float]): The size(s) of the rectangle.
                If int or float, the two sides of the rectangle are equal.
                If Iterable, its size must be 2.
                Units are not specified.
            center (Iterable[int|float]): The center of the loop.
                This can be a 2D or 3D vector.
                Units are not specified.
            normal (Iterable[int|float]): The orientation (normal) of the loop.
                This is a 2D or 3D unit vector.
                The any magnitude information will be lost.
            current (int|float): The current circulating in the loop.
                Units are not specified.
        """
        self.center = _to_3d(np.array(center))
        self.normal = _to_3d(fc.num.normalize(normal))
        self.radius = fc.util.auto_repeat(size, 2, check=True)
        self.current = current


# ======================================================================
def is_vector_field(
        shape,
        index=0):
    """
    Check if a given array qualifies as a vector field.

    A N-dim array can be a vector field if for a given index, the number
    of dims is at least as the size of the array for that specific index.
    For example, a ND array can represent a 3D vector field if the dimension
    identified by has size at most N.

    Args:
        shape (Iterable[int]): The shape of the array.
        index (int): The dimension satisfying the vector field relationship.
            When the index is 0, this is consistent with `np.mgrid()`.

    Returns:
        result (bool): True if `arr` can be a vector field, False otherwise.
    """
    return len(shape[:index]) <= shape[index]


# ======================================================================
def b_circular_loop(
        shape,
        circ_loop,
        n_dim=3,
        rel_position=True,
        rel_sizes=max,
        zero_cutoff=np.spacing(1.0)):
    """
    Conpute the magnetic field generated by a single circular loop.

    For 2D inputs, the normal of the circular loop is assumed to be in the
    2D plane and only the field in that plane is computed.

    Args:
        shape (int|Iterable[int]): The shape of the container in px.
        circ_loop (CircularLoop): The circular loop.
        n_dim (int|None): The number of dimensions of the input.
            If None, the number of dims is guessed from the other parameters,
            but `shape` must be iterable.
        rel_position (bool|callable): Interpret positions as relative values.
            Determine the interpretation of `circ_loop.center` using `shape`.
            Uses `fc.num.grid_coord()` internally, see its `is_relative`
            parameter for more details.
        rel_sizes (bool|callable): Interpret sizes as relative values.
            Determine the interpretation of `circ_loop.radius` using `shape`.
            Uses `fc.num.coord()` internally, see its `is_relative`
            parameter for more details.
        zero_cutoff (float|None): The threshold value for masking zero values.
            If None, no cut-off is performed.

    Returns:
        b_arr (np.ndarray): The B 3D-vector field.
            The first dim contains the cartesian components of the field:
            B_x = b_arr[0, ...], B_y = b_arr[1, ...], etc.
            Even if the input is 2D, the result is always a 3D vector field
            The 3D vector field is represented as a 4D array
            (with the 1st dim of size 3).

    References:
        - Bergeman, T., Gidon Erez, and Harold J. Metcalf. “Magnetostatic
          Trapping Fields for Neutral Atoms.” Physical Review A 35, no. 4
          (February 1, 1987): 1535–46.
          https://doi.org/10.1103/PhysRevA.35.1535.
        - Simpson, James C. Lane. “Simple Analytic Expressions for the
          Magnetic Field of a Circular Current Loop,” January 1,
          2001. https://ntrs.nasa.gov/search.jsp?R=20010038494.
    """
    # : extend 2D to 3D
    if n_dim is None:
        n_dim = fc.util.combine_iter_len((shape,))
    if n_dim == 2:
        shape = fc.util.auto_repeat(shape, n_dim, check=True) + (1,)
    elif n_dim == 3:
        shape = fc.util.auto_repeat(shape, n_dim, check=True)
    else:
        raise ValueError('The number of dimensions must be either 2 or 3.')
    # : generate coordinates
    normal = np.array([0., 0., 1.])
    # : rotate coordinates ([0, 0, 1] is the standard loop normal)
    xx = fc.num.grid_coord(
        shape, circ_loop.center, is_relative=rel_position, use_int=False)
    rot_matrix = fc.num.rotation_3d_from_vectors(normal, circ_loop.normal)
    irot_matrix = fc.num.rotation_3d_from_vectors(circ_loop.normal, normal)
    if not np.all(normal == circ_loop.normal):
        xx = fc.num.grid_transform(xx, rot_matrix)
    # : remove zeros
    if zero_cutoff is not None:
        for i in range(n_dim):
            xx[i][np.abs(xx[i]) < zero_cutoff] = zero_cutoff
    # inline `rr2` for lower memory footprint (but execution will be slower)
    rr2 = (xx[0] ** 2 + xx[1] ** 2 + xx[2] ** 2)
    aa = fc.num.coord(
        shape, circ_loop.radius, is_relative=rel_sizes, use_int=False)[0]
    cc = circ_loop.current * sp.constants.mu_0 / np.pi
    rho2 = (xx[0] ** 2 + xx[1] ** 2)
    ah2 = aa ** 2 + rr2 - 2 * aa * np.sqrt(rho2)
    bh2 = aa ** 2 + rr2 + 2 * aa * np.sqrt(rho2)
    ekk2 = sp.special.ellipe(1 - ah2 / bh2)
    kkk2 = sp.special.ellipkm1(ah2 / bh2)
    # gh = xx[0] ** 2 - xx[1] ** 2  # not used for the field formulae
    with np.errstate(divide='ignore', invalid='ignore'):
        b_x = xx[0] * xx[2] / (2 * ah2 * np.sqrt(bh2) * rho2) * (
                (aa ** 2 + rr2) * ekk2 - ah2 * kkk2)
        b_y = xx[1] * xx[2] / (2 * ah2 * np.sqrt(bh2) * rho2) * (
                (aa ** 2 + rr2) * ekk2 - ah2 * kkk2)
        b_z = 1 / (2 * ah2 * np.sqrt(bh2)) * (
                (aa ** 2 - rr2) * ekk2 + ah2 * kkk2)
    # : clean up some memory
    del xx, rho2, ah2, bh2, ekk2, kkk2
    b_arr = np.stack((b_x, b_y, b_z), 0)
    del b_x, b_y, b_z
    # : handle singularities
    for masker, setter in zip(
            (np.isnan, np.isposinf, np.isneginf), (np.max, np.max, np.min)):
        mask = masker(b_arr)
        b_arr[mask] = setter(b_arr[~mask])
        del mask
    if not np.all(normal == circ_loop.normal):
        b_arr = fc.num.grid_transform(b_arr, irot_matrix)
    return cc * b_arr


# ======================================================================
def b_circular_loops(
        shape,
        circ_loops,
        n_dim=3,
        rel_position=True,
        rel_sizes=True):
    """
    Compute the magnetic field generated by a set of circular loops.

    Args:
        shape (int|Iterable[int]): The shape of the container in px.
        circ_loops (Iterable[CircularLoop]): The circular loops.
        n_dim (int|None): The number of dimensions of the input.
            If None, the number of dims is guessed from the other parameters,
            but `shape` must be iterable.
        rel_position (bool|callable): Interpret positions as relative values.
            Determine the interpretation of `circ_loop.center` using `shape`.
            Uses `fc.num.grid_coord()` internally, see its `is_relative`
            parameter for more details.
        rel_sizes (bool|callable): Interpret sizes as relative values.
            Determine the interpretation of `circ_loop.radius` using `shape`.
            Uses `fc.num.coord()` internally, see its `is_relative`
            parameter for more details.

    Returns:
        b_arr (np.ndarray): The B 3D-vector field.
            The last dim contains the cartesian components of the field:
            B_x = b_arr[..., 0], B_y = b_arr[..., 1], etc.
            Even if the input is 2D, the result is always a 3D-vector field
            (with the 3rd dim of size 1).
            The 3D vector field is represented as a 4D array
            (with the 4th dim of size 3).
    """
    assert (len(circ_loops) > 0)
    b_arr = b_circular_loop(
        shape, circ_loops[0], n_dim, rel_position, rel_sizes)
    for circ_loop in circ_loops[1:]:
        b_arr += b_circular_loop(
            shape, circ_loop, n_dim, rel_position, rel_sizes)
    return b_arr


# ======================================================================
def field_magnitude(
        arr,
        index=0):
    """
    Compute the magnitude of the vector field.

    Args:
        arr (np.ndarray): The input vector field.
        index (int): The dimension satisfying the vector field relationship.
            When the index is 0, this is consistent with `np.mgrid()`.

    Returns:
        arr (np.ndarray): The vector field magnitude.
    """
    assert (is_vector_field(arr.shape))
    return np.sqrt(np.sum(arr ** 2, axis=index))


# ======================================================================
def field_phase(
        arr,
        index=0,
        axes=(0, 1)):
    """
    Compute the orthogonal phase of the vector field.

    This assumes that `arr` is 3D-vector field represented as a 4D array
    (with the 4th dim of size 3).

    Args:
        arr (np.ndarray): The input vector field.
        index (int): The dimension satisfying the vector field relationship.
            When the index is 0, this is consistent with `np.mgrid()`.
        axes (Iterable[int]): The vector field components to use.
            Only the first 2 values are used.

    Returns:
        arr (np.ndarray): The vector field phase for the specified axes.
    """
    assert (is_vector_field(arr.shape))
    masks = [
        tuple(
            slice(None) if i != index else axis
            for i, d in enumerate(arr.shape))
        for axis in axes]
    return np.arctan2(arr[masks[0]], arr[masks[1]])


# ======================================================================
def stacked_circular_loops(
        radiuses,
        positions,
        currents,
        position=0.5,
        normal=(0., 1., 0.),
        n_loops=None):
    """
    Generate parallel circular loops.

    Args:
        radiuses (int|float|Iterable[int|float]): The radiusies of the loops.
        positions (int|float|Iterable[int|float]): The positions of the loops.
            These are the positions of the loops relative to `position` and
            along the `normal` direction.
        currents (int|float|Iterable[int|float]): The currents in the loops.
        position (float|Iterable[float]): The position of the center.
            Values are relative to the lowest edge.
        normal (Iterable[int|float]): The orientation (normal) of the loop.
            This is a 2D or 3D unit vector.
            The any magnitude information will be lost.
        n_loops (int|None): The number of loops.
            If None, this is inferred from the other parameters, but at least
            one of `radiuses`, `currents` must be iterable.

    Returns:
        circ_loops (list[CircularLoop]): The circular loops.
    """
    if n_loops is None:
        n_loops = fc.util.combine_iter_len(radiuses, positions, currents)
    radiuses = fc.util.auto_repeat(radiuses, n_loops, check=True)
    positions = fc.util.auto_repeat(positions, n_loops, check=True)
    currents = fc.util.auto_repeat(currents, n_loops, check=True)
    # : compute circular loop centers
    n_dim = 3
    position = np.array(fc.util.auto_repeat(position, n_dim, check=True))
    normal = np.array(fc.num.normalize(normal))
    centers = [position + x * normal for x in positions]
    circ_loops = [
        CircularLoop(radius, center, normal, current)
        for center, radius, current in zip(centers, radiuses, currents)]
    return circ_loops


# ======================================================================
def stacked_circular_loops_alt(
        radius_factors=1,
        distance_factors=None,
        current_factors=1,
        position=0.5,
        normal=(0., 1., 0.),
        radius=0.25,
        current=1,
        n_loops=None):
    """
    Generate parallel circular loops (using alternate input).

    This is equivalent to `pymrt.extras.em_fields.stacked_circular_loops()`
    except that the inputs are formulated differently
    (but are otherwise equivalent).

    Args:
        radius_factors (Iterable[int|float]): The factors for the radiuses.
        distance_factors (Iterable[int|float]): The factors for the distances.
        current_factors (Iterable[int|float]): The factors for the currents.
        position (float|Iterable[float]): The position of the center.
            Values are relative to the lowest edge.
        normal (Iterable[int|float]): The orientation (normal) of the loop.
            This is a 2D or 3D unit vector.
            The any magnitude information will be lost.
        radius (int|float): The base value for the radius.
            This is also used to compute the distances.
        current (int|float): The base value for the current.
        n_loops (int|None): The number of loops.
            If None, this is inferred from the other parameters, but at least
            one of `radiuses`, `currents` must be iterable.

    Returns:
        circ_loops (list[CircularLoop]): The circular loops.
    """
    if n_loops is None:
        n_loops = fc.util.combine_iter_len((radius_factors, current_factors))
    radius_factors = fc.util.auto_repeat(
        radius_factors, n_loops, check=True)
    if distance_factors is None:
        distance_factors = 2 / n_loops
    distance_factors = fc.util.auto_repeat(
        distance_factors, n_loops - 1, check=True)
    current_factors = fc.util.auto_repeat(
        current_factors, n_loops, check=True)
    distances = [k * radius for k in distance_factors]
    positions = fc.num.distances2displacements(distances)
    return stacked_circular_loops(
        [k * radius for k in radius_factors],
        positions,
        [k * current for k in current_factors],
        position, normal,
        n_loops)


# ======================================================================
def crossing_circular_loops(
        position=0.5,
        direction=(0., 0., 1.),
        radiuses=0.4,
        angles=None,
        currents=1,
        n_loops=None):
    """
    Generate circular loops sharing the same diameter.

    Args:
        position (float|Iterable[float]): The position of the center.
            Values are relative to the lowest edge.
        direction (Iterable[int|float]: The direction of the shared diameter.
            Must have size 3.
        radiuses (int|float|Iterable[int|float]): The loop radiuses.
            If int or float, the same value is used for all loops.
            If Iterable, its size must be `n_loops`.
        angles (int|float|Iterable[int|float]|None): The loop angles in deg.
            This is the tilting of the circular loop around `direction`.
            If int or float, a single loop is assumed.
            If Iterable, its size must be `n_loops`.
            If None, the angles are linearly distributed in the [0, 180) range,
            resulting in equally angularly spaced loops.
        currents (int|float|Iterable[int|float]): The currents in the loops.
        n_loops (int|None): The number of loops.
            If None, this is inferred from the other parameters, but at least
            one of `radiuses`, `angles`, `currents` must be iterable.

    Returns:
        circ_loops (list[CircularLoop]): The circular loops.
    """
    n_dim = 3
    position = np.array(fc.util.auto_repeat(position, n_dim, check=True))
    if not n_loops:
        n_loops = fc.util.combine_iter_len((angles, radiuses, currents))
    angles = np.linspace(0.0, 180.0, n_loops, False)
    radiuses = fc.util.auto_repeat(radiuses, n_loops, check=True)
    currents = fc.util.auto_repeat(currents, n_loops, check=True)
    orientation = (0., 0., 1.)
    rot_matrix = fc.num.rotation_3d_from_vectors(orientation, direction)
    normal = np.dot(rot_matrix, (0., 1., 0.))
    normals = [
        np.dot(fc.num.rotation_3d_from_vector(orientation, angle), normal)
        for angle in angles]
    circ_loops = [
        CircularLoop(radius, position, normal, current)
        for radius, normal, current in zip(radiuses, normals, currents)]
    return circ_loops


# ======================================================================
def cylinder_with_circular_loops(
        position=0.5,
        angle=0.0,
        diameter=0.6,
        height=0.8,
        direction=(0., 0., 1.),
        radiuses=None,
        distance_factors=1,
        currents=1,
        n_series=6,
        loops_per_series=4,
        n_loops=None):
    """
    Generate circular loops along the lateral surface of a cylinder.

    Args:
        position (float|Iterable[float]): The position of the center.
            Values are relative to the lowest edge.
        angle (int|float): The rotation of the cylinder in deg.
        diameter (int|float): The diameter of the cylinder.
        height (int|float): The height of the cylinder in deg.
        direction (Iterable[int|float]: The direction of the cylinder.
            Must have size 3.
        radiuses (int|float|Iterable[int|float]|None): The loop radiuses.
            If int or float, the same value is used for all loops.
            If Iterable, its size must be `n_loops`.
            If None, the radius is the same for all loops and is computed
            so that the loops in each series cover the all cylinder height
            without overlapping.
            The overlapping behavior can be tweaked using `distance_factors`
            smaller than 1.
        distance_factors (int|float|Iterable[int|float]): The distance factors.
            These determine the distance of the circular loops within
            each series.
        currents (int|float|Iterable[int|float]): The currents in the loops.
        n_series (int|None): The number of loop series.
            The series are equally distributed along the lateral surface
            of the cylinder.
            If None, this is computed from `n_loops` and `loops_per_series`,
            and they cannot be None.
        loops_per_series (int|None): The number of loop per series.
            If None, this is computed from `n_loops` and `n_series`,
            and they cannot be None.
        n_loops (int|None): The total number of loops.
            If None, this is computed from `n_series` and `loops_per_series`,
            and they cannot be None.

    Returns:
        circ_loops (list[CircularLoop]): The circular loops.
    """
    n_dim = 3
    if not n_loops and n_series and loops_per_series:
        n_loops = n_series * loops_per_series
    elif n_loops and not n_series and loops_per_series:
        n_series = n_loops // loops_per_series
        if n_loops % loops_per_series:
            text = 'Values of `n_loops={}` and `loops_per_serie={}` ' \
                   'do not match.'.format(n_loops, loops_per_series)
            raise ValueError(text)
    elif n_loops and n_series and not loops_per_series:
        loops_per_series = n_loops // n_series
        if n_loops % n_series:
            text = 'Values of `n_loops={}` and `n_series={}` ' \
                   'do not match.'.format(n_loops, n_series)
            raise ValueError(text)
    else:
        text = 'At least two of `n_loops`, `n_series` and `loops_per_serie` ' \
               'must be larger than 0'
        raise ValueError(text)
    position = np.array(fc.util.auto_repeat(position, n_dim, check=True))
    currents = fc.util.auto_repeat(currents, n_loops, check=True)
    if not radiuses:
        radiuses = height / loops_per_series / 2
    radiuses = fc.util.auto_repeat(radiuses, n_loops, check=True)
    distance_factors = fc.util.auto_repeat(
        distance_factors, loops_per_series - 1)
    distances = tuple(
        k * (r_m1 + r_p1) for k, r_m1, r_p1
        in zip(distance_factors, radiuses[:-1], radiuses[1:]))
    orientation = np.array((0., 0., 1.))
    rot_matrix = np.dot(
        fc.num.rotation_3d_from_vector(orientation, angle),
        fc.num.rotation_3d_from_vectors(orientation, direction))
    centers, normals = [], []
    for i in range(n_series):
        r = diameter / 2
        phi = 2. * np.pi * i / n_series
        for k in fc.num.distances2displacements(distances):
            center = np.array([r * np.cos(phi), r * np.sin(phi), k])
            centers.append(np.dot(rot_matrix, center) + position)
            normal = np.array([-np.cos(phi), -np.sin(phi), 0])
            normals.append(np.dot(rot_matrix, normal))
    circ_loops = [
        CircularLoop(radius, center, normal, current)
        for radius, center, normal, current
        in zip(radiuses, centers, normals, currents)]
    return circ_loops


# ======================================================================
def sphere_with_circular_loops(
        position=0.5,
        angles=0.0,
        diameter=0.8,
        n_loops=24,
        radiuses=None,
        currents=1,
        coord_gen=lambda x: fc.num.fibonacci_sphere(x).transpose()):
    """
    Generate circular loops along the surface of a sphere.

    Args:
        position (float|Iterable[float]): The position of the center.
            Values are relative to the lowest edge.
        angles (int|float|Iterable[int|float]): The angles of rotation in deg.
            These are used to rotate the positions of the circular loops
            in the sphere surface.
            Each angle specifies the rotation using the canonical basis
            as axes of rotation.
            If float, the value is repeated for all axes.
            Otherwise, it must have size 3 (for more info on the number of
            angles in 3 dimensions, see `fc.num.square_size_to_num_tria()`).
            The rotation is computed using `fc.num.angles2linear(angles)`.
            See that for more info.
        diameter (int|float): The diameter of the sphere.
        n_loops (int|None): The total number of loops.
        radiuses (int|float|Iterable[int|float]|None): The loop radiuses.
            If int or float, the same value is used for all loops.
            If Iterable, its size must be `n_loops`.
            If None, the radius is the same for all loops and is computed
            to be 1/2 of the minimum distance between any two centers.
            This ensures that the loops do not overlap.
        currents (int|float|Iterable[int|float]): The currents in the loops.
        coord_gen (callable): The generator for loop centers.
            This is a function that computes the cartesian coordinates of
            points on the surface of a unitary sphere.
            Must have the following signature: coord_gen(int) -> Iterable.
            Each element of the iterable must have size 3.

    Returns:
        circ_loops (list[CircularLoop]): The circular loops.
    """
    n_dim = 3
    position = np.array(fc.util.auto_repeat(position, n_dim, check=True))
    currents = fc.util.auto_repeat(currents, n_loops, check=True)
    angles = fc.util.auto_repeat(
        angles, fc.num.square_size_to_num_tria(n_dim), check=True)
    rot_matrix = fc.num.angles2linear(angles)
    centers = [
        np.dot(rot_matrix, center * diameter / 2) + position
        for center in coord_gen(n_loops)]
    normals = [
        fc.num.vectors2direction(center, position)
        for center in centers]
    if not radiuses:
        radiuses = min(fc.num.pairwise_distances(centers)) / 2
    radiuses = fc.util.auto_repeat(radiuses, n_loops, check=True)
    circ_loops = [
        CircularLoop(radius, center, normal, current)
        for radius, center, normal, current
        in zip(radiuses, centers, normals, currents)]
    return circ_loops


# ======================================================================
def helmholtz_uniform(
        position=0.5,
        normal=(0., 1., 0.),
        radius=0.25,
        current=1):
    """
    Generate the Helmholtz uniform (2 circular loops) configuration.

    This is just a short-hand (with the correct pre-settings) for
    `pymrt.extras.em_fields.stacked_circular_loops_alt()`.


    Args:
        position (float|Iterable[float]): The position of the center.
            Values are relative to the lowest edge.
        normal (Iterable[int|float]): The orientation (normal) of the loop.
            This is a 2D or 3D unit vector.
            The any magnitude information will be lost.
        radius (int|float): The base value for the radius.
            This is also used to compute the distances.
        current (int|float): The base value for the current.

    Returns:
        circ_loops (list[CircularLoop]): The circular loops.

    See Also:
        - https://en.wikipedia.org/wiki/Helmholtz_coil
    """
    radius_factors = (1., 1.)
    distance_factors = 1.
    current_factors = (1., 1.)
    return stacked_circular_loops_alt(
        radius_factors, distance_factors, current_factors,
        position, normal, radius, current)


# ======================================================================
def maxwell_gradient(
        position=0.5,
        normal=(0., 1., 0.),
        radius=0.25,
        current=1):
    """
    Generate the Maxwell gradient (2 circular loops) configuration.

    This is just a short-hand (with the correct pre-settings) for
    `pymrt.extras.em_fields.stacked_circular_loops_alt()`.

    Args:
        position (float|Iterable[float]): The position of the center.
            Values are relative to the lowest edge.
        normal (Iterable[int|float]): The orientation (normal) of the loop.
            This is a 2D or 3D unit vector.
            The any magnitude information will be lost.
        radius (int|float): The base value for the radius.
            This is also used to compute the distances.
        current (int|float): The base value for the current.

    Returns:
        circ_loops (list[CircularLoop]): The circular loops.

    See Also:
        - https://en.wikipedia.org/wiki/Maxwell_coil
    """
    radius_factors = (1., 1.)
    distance_factors = (3. ** 0.5)
    current_factors = (1., -1.)
    return stacked_circular_loops_alt(
        radius_factors, distance_factors, current_factors,
        position, normal, radius, current)


# ======================================================================
def maxwell_uniform(
        position=0.5,
        normal=(0., 1., 0.),
        radius=0.25,
        current=1):
    """
    Generate the Maxwell uniform (3 circular loops) configuration.

    This is just a short-hand (with the correct pre-settings) for
    `pymrt.extras.em_fields.stacked_circular_loops_alt()`.

    Args:
        position (float|Iterable[float]): The position of the center.
            Values are relative to the lowest edge.
        normal (Iterable[int|float]): The orientation (normal) of the loop.
            This is a 2D or 3D unit vector.
            The any magnitude information will be lost.
        radius (int|float): The base value for the radius.
            This is also used to compute the distances.
        current (int|float): The base value for the current.

    Returns:
        circ_loops (list[CircularLoop]): The circular loops.

    See Also:
        - https://en.wikipedia.org/wiki/Maxwell_coil
    """
    radius_factors = ((3. / 7.) ** 0.5, 1., (3. / 7.) ** 0.5)
    distance_factors = ((3. / 7.) ** 0.5, (3. / 7.) ** 0.5)
    current_factors = (49., 64., 49.)
    return stacked_circular_loops_alt(
        radius_factors, distance_factors, current_factors,
        position, normal, radius, current)


# circ_loops = (CircularLoop(0.4, (0.5, 0.5, 0.5), (0, 1, 0), 1),)
# circ_loops = stacked_circular_loops_alt(radius=0.4, normal=(0, 1, 0))
circ_loops = crossing_circular_loops(n_loops=4)
# circ_loops = cylinder_with_circular_loops(angle=0.0)
# circ_loops = sphere_with_circular_loops(n_loops=24)
# circ_loops = helmoltz_uniform(normal=(0, 1, 0), radius=0.4)

# from mpl_toolkits.mplot3d import axes3d
# import matplotlib.pyplot as plt
# import numpy as np
#
# fig = plt.figure()
# ax = fig.gca(projection='3d')
#
# num = 12
# xx = fc.num.grid_coord((num, num, num))
# arr = b_circular_loops(num, circ_loops)
#
# x, y, z = xx
# u, v, w = arr
#
# # values = field_magnitude(arr)
# values = field_phase(arr, axes=(1, 0))
# values = (values.ravel() - values.min()) / values.ptp()
# values = np.concatenate((values, np.repeat(values, 2)))
# colors = plt.cm.jet(values)
#
# ax.quiver(x, y, z, u, v, w, length=0.55, normalize=True, colors=colors)
#
# plt.show()


from numex.gui_tk_mpl import explore

num = 200
arr = b_circular_loops(num, circ_loops)
mag_arr = field_magnitude(arr)
# mag_arr = arr[2]
max_arr = np.max(arr)
# threshold = mag_arr[num // 2, num // 2, num // 2] * 2
threshold = np.quantile(mag_arr, 0.95)
mask = mag_arr > threshold
mag_arr[mask] = threshold
explore(mag_arr / np.max(mag_arr))

from pymrt.recipes import phs

phs_arr = field_phase(arr, axes=(0, 1))
# explore(phs.unwrap_sorting_path_2d_3d(phs_arr))
explore(phs_arr)

# ======================================================================
elapsed(__file__[len(PATH['base']) + 1:])

# ======================================================================
if __name__ == '__main__':
    import doctest  # Test interactive Python examples

    msg(__doc__.strip())
    doctest.testmod()
    msg(report())
