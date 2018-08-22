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


# ======================================================================
class CircularLoop(object):
    def __init__(
            self,
            radius=0.25,
            center=(0.5, 0.5, 0.5),
            normal=(0., 0., 1.),
            current=1.):
        """

        Args:
            center (Iterable[int|float]): The center of the loop.
                This can be a 2D or 3D vector.
                Units are not specified.
            normal (Iterable[int|float]): The orientation (normal) of the loop.
                This is a 2D or 3D unit vector.
                The any magnitude information will be lost.
            radius (int|float): The radius of the circle.
                Units are not specified.
            current (int|float): The current circulating in the loop.
        """
        self.center = self._to_3d(np.array(center))
        self.normal = self._to_3d(fc.num.normalize(normal))
        self.radius = radius
        self.current = current

    # --------------------------------
    @staticmethod
    def _to_3d(vect, extra_dim=0):
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
    Conpute the magnetic field generated by a set of circular loops.

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
        normal=(0, 0, 1),
        n_loops=None):
    """
    Generate an arbitrary number of parallel circular loops.

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
        circ_loops (list[CircularLoop]): The parallel circular loops.
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
        radius_factors,
        distance_factors,
        current_factors,
        position=0.5,
        normal=(0., 0., 1.),
        radius=0.25,
        current=1,
        n_loops=None):
    """
    Generate an arbitrary number of parallel circular loops (alternate input).

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
        circ_loops (list[CircularLoop]): The parallel circular loops.
    """
    if n_loops is None:
        n_loops = fc.util.combine_iter_len((radius_factors, current_factors))
    radius_factors = fc.util.auto_repeat(
        radius_factors, n_loops, check=True)
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
def cylinder_with_circular_loops():
    raise NotImplementedError


# ======================================================================
def sphere_with_circular_loops():
    raise NotImplementedError


# ======================================================================
def helmholtz_uniform(
        position=0.5,
        normal=(0., 0., 1.),
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
        circ_loops (list[CircularLoop]): The parallel circular loops.

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
        normal=(0., 0., 1.),
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
        circ_loops (list[CircularLoop]): The parallel circular loops.

    See Also:
        - https://en.wikipedia.org/wiki/Maxwell_coil
    """
    radius_factors = (1., 1.)
    distance_factors = np.sqrt(3.)
    current_factors = (1., -1.)
    return stacked_circular_loops_alt(
        radius_factors, distance_factors, current_factors,
        position, normal, radius, current)


# ======================================================================
def maxwell_uniform(
        position=0.5,
        normal=(0, 0, 1),
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
        circ_loops (list[CircularLoop]): The parallel circular loops.

    See Also:
        - https://en.wikipedia.org/wiki/Maxwell_coil
    """
    radius_factors = (np.sqrt(3. / 7.), 1., np.sqrt(3. / 7.))
    distance_factors = (np.sqrt(3. / 7.), np.sqrt(3. / 7.))
    current_factors = (49., 64., 49.)
    return stacked_circular_loops_alt(
        radius_factors, distance_factors, current_factors,
        position, normal, radius, current)


# circ_loops = helmoltz_uniform(normal=(0, 0, 1), radius=0.4)
circ_loops = (CircularLoop(0.5, (0.5, 0.5, 0.5), (0, 0, 1), 1),)

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
# max_arr = np.max(arr)
threshold = mag_arr[num // 2, num // 2, num // 2] * 2
mask = mag_arr > threshold
mag_arr[mask] = threshold
explore(mag_arr / np.max(mag_arr))

from pymrt.recipes import phs

phs_arr = field_phase(arr, axes=(0, 1))
# explore(phs.unwrap_sorting_path_2d_3d(phs_arr))


# ======================================================================
elapsed(__file__[len(PATH['base']) + 1:])

# ======================================================================
if __name__ == '__main__':
    import doctest  # Test interactive Python examples

    msg(__doc__.strip())
    doctest.testmod()
    msg(report())
