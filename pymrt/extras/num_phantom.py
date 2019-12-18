#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PyMRT: generate numerical phantoms
"""

# ======================================================================
# :: Future Imports
from __future__ import (
    division, absolute_import, print_function, unicode_literals, )

# ======================================================================
# :: Python Standard Library Imports
# import shutil  # High-level file operations
# import time  # Time access and conversions
# import operator  # Standard operators as functions
# import collections  # Container datatypes
# import argparse  # Parser for command-line options, arguments and subcommands
# import subprocess  # Subprocess management
# import multiprocessing  # Process-based parallelism
# import csv  # CSV File Reading and Writing [CSV: Comma-Separated Values]
# import json  # JSON encoder and decoder [JSON: JavaScript Object Notation]
import random  # Generate pseudo-random numbers

# :: External Imports
# import matplotlib as mpl  # Matplotlib (2D/3D plotting library)
# import sympy as sym  # SymPy (symbolic CAS library)
# import PIL  # Python Image Library (image manipulation toolkit)
# import SimpleITK as sitk  # Image ToolKit Wrapper
# import nibabel as nib  # NiBabel (NeuroImaging I/O Library)
# import nipy  # NiPy (NeuroImaging in Python)
# import nipype  # NiPype (NiPy Pipelines and Interfaces)
import flyingcircus as fc  # Everything you always wanted to have in Python*

# :: External Imports Submodules
# import matplotlib.pyplot as plt  # Matplotlib's pyplot: MATLAB-like syntax
# import scipy.optimize  # SciPy: Optimization Algorithms
# import scipy.integrate  # SciPy: Integrations facilities
# import scipy.constants  # SciPy: Mathematal and Physical Constants

# :: Local Imports
from pymrt import INFO, PATH
from pymrt import VERB_LVL, D_VERB_LVL, VERB_LVL_NAMES
from pymrt import elapsed, report, run_doctests
from pymrt import msg, dbg, fmt, fmtm

from raster_geometry import (
    ellipsoid_specs, superellipsoid_specs, cuboid_specs, prism_specs,
    gradient_specs)


# ======================================================================
def shepp_logan_like(
        values=1.0):
    """
    The Shepp-Logan phantom with custom intensity values.

    Args:
        values (int|float|Iterable[int|float]): Intensities of the ellipses.
            If Iterable, must have a length of 10.

    Returns:
        geom_shapes (Iterable[Iterable]): The geometric specifications.
            These are of the form:
            (intensity, [name, *_args], shift, angles, position).
            See the `geom_shapes` parameter of `raster_geometry.multi_render()`
            for more details.

    References:
        - Shepp, L. A., and B. F. Logan. “The Fourier Reconstruction of a
          Head Section.” IEEE Transactions on Nuclear Science 21, no. 3 (June
          1974): 21–43. https://doi.org/10.1109/TNS.1974.6499235.

    Notes:
        - This is implemented based on `raster_geometry.nd_superellipsoid()`
          and therefore the sizes and the inner positions are halved, while
          angular values are defined differently compared to the reference
          paper.
    """
    geom_shapes = [
        [['ellipsoid', [0.3450, 0.4600]], [+0.0000, +0.0000], [+000.], 0.5],
        [['ellipsoid', [0.3312, 0.4370]], [+0.0000, -0.0092], [+000.], 0.5],
        # angle: 72 = 90 - 18
        [['ellipsoid', [0.1550, 0.0550]], [+0.1100, +0.0000], [+072.], 0.5],
        # angle: 108 = 90 + 18
        [['ellipsoid', [0.2050, 0.0800]], [-0.1100, +0.0000], [+108.], 0.5],
        [['ellipsoid', [0.1250, 0.1050]], [+0.0000, +0.1750], [+000.], 0.5],
        [['ellipsoid', [0.0230, 0.0230]], [+0.0000, +0.0500], [+000.], 0.5],
        [['ellipsoid', [0.0230, 0.0230]], [+0.0000, -0.0500], [+000.], 0.5],
        [['ellipsoid', [0.0230, 0.0115]], [-0.0400, -0.3025], [+000.], 0.5],
        [['ellipsoid', [0.0115, 0.0115]], [+0.0000, -0.3025], [+000.], 0.5],
        [['ellipsoid', [0.0115, 0.0230]], [+0.0300, -0.3025], [+000.], 0.5], ]
    fc.base.auto_repeat(values, len(geom_shapes), False, True)
    geom_shapes = [
        ([value] + geom_shape)
        for value, geom_shape in zip(values, geom_shapes)]
    return geom_shapes


# ======================================================================
def shepp_logan():
    """
    The classical Shepp-Logan phantom.

    Returns:
        geom_shapes (Iterable[Iterable]): The geometric specifications.
            These are of the form:
            (intensity, [name, *_args], shift, angles, position).
            See the `geom_shapes` parameter of `raster_geometry.multi_render()`
            for more details.

    References:
        - Shepp, L. A., and B. F. Logan. “The Fourier Reconstruction of a
          Head Section.” IEEE Transactions on Nuclear Science 21, no. 3 (June
          1974): 21–43. https://doi.org/10.1109/TNS.1974.6499235.

    Notes:
        - This is implemented based on `raster_geometry.nd_superellipsoid()`
          and therefore the sizes and the inner positions are halved, while
          angular values are defined differently compared to the reference
          paper.
    """
    return shepp_logan_like(
        [2.00, -0.98, -0.02, -0.02, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01])


# ======================================================================
def shepp_logan_toft():
    """
    Toft modification of Shepp-Logan phantom.

    Returns:
        geom_shapes (Iterable[Iterable]): The geometric specifications.
            These are of the form:
            (intensity, [name, *_args], shift, angles, position).
            See the `geom_shapes` parameter of `raster_geometry.multi_render()`
            for more details.

    References:
        - Shepp, L. A., and B. F. Logan. “The Fourier Reconstruction of a
          Head Section.” IEEE Transactions on Nuclear Science 21, no. 3 (June
          1974): 21–43. https://doi.org/10.1109/TNS.1974.6499235.

    Notes:
        - This is implemented based on `raster_geometry.nd_superellipsoid()`
          and therefore the sizes and the inner positions are halved, while
          angular values are defined differently compared to the reference
          paper.
    """
    return shepp_logan_like(
        [1.0, -0.8, -0.2, -0.2, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])


# ======================================================================
def kak_roberts():
    """
    Kak and Roberts modification of Shepp-Logan phantom with 3D extension.

    Returns:
        geom_shapes (Iterable[Iterable]): The geometric specifications.
            These are of the form:
            (intensity, [name, *_args], shift, angles, position).
            See the `geom_shapes` parameter of `raster_geometry.multi_render()`
            for more details.

    References:
        - Young, Tzay Y., and K. S. Fu, eds. Handbook of Pattern Recognition
          and Image Processing. Handbooks in Science and Technology. Orlando:
          Academic Press, 1986.
        - Kak, Aninash C., and Malcolm Slaney. Principles of Computerized
          Tomographic Imaging. Philadelphia: Society for Industrial and
          Applied Mathematics, 2001.
        - Koay, Cheng Guan, Joelle E. Sarlls, and Evren Özarslan.
          “Three-Dimensional Analytical Magnetic Resonance Imaging Phantom in
          the Fourier Domain.” Magnetic Resonance in Medicine 58, no. 2 (
          August 1, 2007): 430–36. https://doi.org/10.1002/mrm.21292.

    Notes:
        - This is implemented based on `raster_geometry.nd_superellipsoid()`
          and therefore the sizes and the inner positions are halved, while
          angular values are defined differently compared to the reference
          paper.
        - In Fig.2 of doi:10.1002/mrm.21292, the rotation of last and 3rd last
          geom_shapes are swapped, contrarily to the parametric definition
          given (which is also implemented here).
    """
    geom_shapes = [
        [+2.0, ['ellipsoid', [0.3450, 0.4600, 0.4500]],
         [+0.0000, +0.0000, +0.0000], [+000.], 0.5],
        [-0.8, ['ellipsoid', [0.3312, 0.4370, 0.4400]],
         [+0.0000, +0.0000, +0.0000], [+000.], 0.5],
        [-0.2, ['ellipsoid', [0.2050, 0.0800, 0.1050]],
         [-0.1100, +0.0000, -0.1250], [-072.], 0.5],
        [-0.2, ['ellipsoid', [0.1550, 0.0550, 0.1100]],
         [+0.1100, +0.0000, -0.1250], [-108.], 0.5],
        [+0.2, ['ellipsoid', [0.1050, 0.1250, 0.2500]],
         [+0.0000, +0.1750, -0.1250], [+000.], 0.5],
        [+0.2, ['ellipsoid', [0.0230, 0.0230, 0.0230]],
         [+0.0000, +0.0500, -0.1250], [+000.], 0.5],
        [+0.1, ['ellipsoid', [0.0230, 0.0115, 0.0100]],
         [-0.0400, -0.3250, -0.1250], [+000.], 0.5],
        [+0.1, ['ellipsoid', [0.0230, 0.0115, 0.0100]],
         [+0.0300, -0.3250, -0.1250], [-090.], 0.5],
        [+0.2, ['ellipsoid', [0.0280, 0.0200, 0.0500]],
         [+0.0300, -0.0525, +0.3125], [-090.], 0.5],
        [-0.2, ['ellipsoid', [0.0280, 0.0280, 0.0500]],
         [+0.0000, +0.0500, +0.3125], [+000.], 0.5],
    ]
    return geom_shapes


# ======================================================================
shepp_logan_3d = kak_roberts


# ======================================================================
def metere_brain_transverse_2d():
    raise NotImplementedError


# ======================================================================
def metere_brain_sagittal_2d():
    raise NotImplementedError


# ======================================================================
def metere_brain_coronal_2d():
    raise NotImplementedError


# ======================================================================
def metere_brain_3d():
    raise NotImplementedError


# ======================================================================
def metere_brain_fmri_4d():
    raise NotImplementedError


# ======================================================================
def metere_brain_motion_4d():
    raise NotImplementedError


# ======================================================================
def scaled_random(
        geom_shapes,
        interval=0.1j,
        fallback_interval=5j):
    """
    Randomize the numerical parameters of the geometric elements.

    Args:
        geom_shapes (Iterable): The geometric specifications.
            These are of the form:
            (intensity, [name, *_args], shift, angles, position).
            See the `geom_shapes` parameter of `raster_geometry.multi_render()`
            for more details.
        interval (Any):
        fallback_interval (Any):

    Returns:
        rand_geom_shapes (Iterable): The randomized geometric specifications.
    """
    rand_geom_shapes = fc.base.deep_filter_map(
        geom_shapes,
        lambda x: fc.extra.scaled_randomizer(x, interval, fallback_interval),
        lambda x: isinstance(x, (int, float)))
    return rand_geom_shapes


# ======================================================================
def auto_random(geom_shape):
    """
    Generate a random goemetric shape from its specifications.

    The goemetric shape specification are passed through
    `flyingcircus.extra.auto_random()`.
    For a random value to be produced, the corresponding parameter must
    produce a random value when passed through this function.

    Args:
        geom_shape (tuple): The geometric specification.
            This is of the form:
            (intensity, [name, *_args], shift, angles, position).
            See the `geom_shapes` parameter of `raster_geometry.multi_render()`
            for more details.

    Returns:
        geom_shape (tuple): The randomized geometric specification.
            (with the same structure as the input).
    """
    return fc.base.deep_map(fc.extra.auto_random, geom_shape)


# ======================================================================
def random_geom_shapes(
        n_dim,
        nums,
        geom_shapes_specs=(
                (ellipsoid_specs, (), (
                        ('semisizes', slice(0.0, 0.25)),)),
                (superellipsoid_specs, (), (
                        ('semisizes', slice(0.0, 0.25)),
                        ('indexes', slice(0.0, 16.0)))),
                (cuboid_specs, (), (
                        ('semisizes', slice(0.0, 0.25)),)),
                (prism_specs, (), (
                        ('semisizes', slice(0.0, 0.25)),
                        ('indexes', slice(0.0, 16.0))),
                 ('axis', slice(0, 100))),
                (gradient_specs, (), (
                        ('gen_ranges', None),))),
        kws=(
                ('intensity', slice(-1.0, 1.0)),
                ('shift', slice(-0.5, 0.5)),
                ('angles', slice(0.0, 360.0)),
                ('position', slice(0.0, 1.0)))):
    """
    Generate random goemetric shapes.

    Args:
        n_dim (int): The number of dimensions.
        nums (int|Iterable[int]): The number of geometric shapes to generate.
            If int, this is the total number of shapes generated.
            If Iterable[int], indicates the number of geometric shape to
            generate associated with the corresponding `geom_shapes_specs`.
        geom_shapes_specs (Iterable[Iterable]): The geometric shapes specs.
            Each element must be of the form: (func, args, kws)
            where:
             - func (callable): The function to generate the geometric shape.
             - args (tuple|None): Positional arguments of `func`.
             - kws (Mappable|None): Keyword arguments of `func`.
                If tuple, must be castable to dict.
        kws (Mappable): Common keyword arguments.
            They must exist for any callable specified in `geom_shapes_specs`.
            The keyword arguments passed in `geom_shapes_specs` may override
            those specified here.

    Returns:
        geom_shapes (Iterable[Iterable]): The geometric specifications.
            These are of the form:
            (intensity, [name, *_args], shift, angles, position).
            See the `geom_shapes` parameter of `raster_geometry.multi_render()`
            for more details.
    """

    def use_specs(spec):
        args_ = tuple(spec[1]) if spec[1] is not None else ()
        kws_ = dict(kws) if kws is not None else {}
        kws_.update(dict(spec[2]) if spec[2] is not None else {})
        return spec[0](n_dim, *args_, **kws_)

    if isinstance(nums, int):
        return [
            auto_random(
                use_specs(
                    geom_shapes_specs[
                        random.randint(0, len(geom_shapes_specs) - 1)]))
            for _ in range(nums)]
    else:
        assert (len(nums) <= len(geom_shapes_specs))
        result = []
        for i, num in enumerate(nums):
            result.extend(
                random_geom_shapes(
                    n_dim, num, geom_shapes_specs[i:i + 1], kws))
        return result


# ======================================================================
elapsed(__file__[len(PATH['base']) + 1:])

# ======================================================================
if __name__ == '__main__':
    run_doctests(__doc__)
