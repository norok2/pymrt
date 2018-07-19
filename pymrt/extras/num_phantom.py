#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PyMRT: parse Bruker raw data.

EXPERIMENTAL!
"""

# ======================================================================
# :: Future Imports
from __future__ import (
    division, absolute_import, print_function, unicode_literals, )

# ======================================================================
# :: Python Standard Library Imports
import os  # Miscellaneous operating system interfaces
# import shutil  # High-level file operations
import math  # Mathematical functions
# import time  # Time access and conversions
import datetime  # Basic date and time types
# import operator  # Standard operators as functions
# import collections  # Container datatypes
# import argparse  # Parser for command-line options, arguments and subcommands
import itertools  # Functions creating iterators for efficient looping
# import subprocess  # Subprocess management
# import multiprocessing  # Process-based parallelism
# import csv  # CSV File Reading and Writing [CSV: Comma-Separated Values]
# import json  # JSON encoder and decoder [JSON: JavaScript Object Notation]
import struct  # Interpret strings as packed binary data
import doctest  # Test interactive Python examples
import glob  # Unix style pathname pattern expansion
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
# import matplotlib.pyplot as plt  # Matplotlib's pyplot: MATLAB-like syntax
# import scipy.optimize  # SciPy: Optimization Algorithms
# import scipy.integrate  # SciPy: Integrations facilities
# import scipy.constants  # SciPy: Mathematal and Physical Constants
import scipy.ndimage  # SciPy: ND-image Manipulation

# :: Local Imports
import pymrt as mrt
import pymrt.utils
import pymrt.geometry

from pymrt import INFO, DIRS
from pymrt import VERB_LVL, D_VERB_LVL, VERB_LVL_NAMES
from pymrt import elapsed, report
from pymrt import msg, dbg


# ======================================================================
def shepp_logan():
    """
    The classical Shepp-Logan phantom.

    Returns:
        geom_shapes (Iterable[Iterable]): The geometric specifications.
            These are of the form: [val, semizes, position, angles].
            They can be passed directly to
            `pymrt.extras.num_phantom.render_geom_shapes()`.
            See this function for more details.


    References:
        - Shepp, L. A., and B. F. Logan. “The Fourier Reconstruction of a
          Head Section.” IEEE Transactions on Nuclear Science 21, no. 3 (June
          1974): 21–43. https://doi.org/10.1109/TNS.1974.6499235.

    Notes:
        - This is implemented based on `pymrt.geometry.nd_superellipsoid()`
          and therefore the sizes and the inner positions are halved, while
          angular values are defined differently compared to the reference
          paper.
    """
    geom_shapes = [
        [+2.00, ['ellipsoid', [0.3450, 0.4600]], [+0.0000, +0.0000], [+000.],
         None],
        [-0.98, ['ellipsoid', [0.3312, 0.4370]], [+0.0000, -0.0092], [+000.],
         None],
        [-0.02, ['ellipsoid', [0.1550, 0.0550]], [+0.1100, +0.0000], [+108.],
         None],  # 90 + 18
        [-0.02, ['ellipsoid', [0.2050, 0.0800]], [-0.1100, +0.0000], [+072.],
         None],  # 90 - 18
        [+0.01, ['ellipsoid', [0.1250, 0.1050]], [+0.0000, +0.1750], [+000.],
         None],
        [+0.01, ['ellipsoid', [0.0230, 0.0230]], [+0.0000, +0.0500], [+000.],
         None],
        [+0.01, ['ellipsoid', [0.0230, 0.0230]], [+0.0000, -0.0500], [+000.],
         None],
        [+0.01, ['ellipsoid', [0.0230, 0.0115]], [-0.0400, -0.3025], [+000.],
         None],
        [+0.01, ['ellipsoid', [0.0115, 0.0115]], [+0.0000, -0.3025], [+000.],
         None],
        [+0.01, ['ellipsoid', [0.0115, 0.0230]], [+0.0300, -0.3025], [+000.],
         None],
    ]
    return geom_shapes


# ======================================================================
def shepp_logan_toft():
    """
    Toft modification of Shepp-Logan phantom.

    Returns:
        geom_shapes (Iterable[Iterable]): The geometric specifications.
            These are of the form: [val, semizes, position, angles].
            They can be passed directly to
            `pymrt.extras.num_phantom.render_geom_shapes()`.
            See this function for more details.

    References:
        - Shepp, L. A., and B. F. Logan. “The Fourier Reconstruction of a
          Head Section.” IEEE Transactions on Nuclear Science 21, no. 3 (June
          1974): 21–43. https://doi.org/10.1109/TNS.1974.6499235.

    Notes:
        - This is implemented based on `pymrt.geometry.nd_superellipsoid()`
          and therefore the sizes and the inner positions are halved, while
          angular values are defined differently compared to the reference
          paper.
    """
    new_vals = [1.0, -0.8, -0.2, -0.2, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
    geom_shapes = [
        ([new_val] + geom_shape[1:])
        for new_val, geom_shape in zip(new_vals, shepp_logan())]
    return geom_shapes


# ======================================================================
def kak_roberts():
    """
    Kak and Roberts modification of Shepp-Logan phantom with 3D extension.

    Returns:
        geom_shapes (Iterable[Iterable]): The geometric specifications.
            These are of the form: [val, semizes, position, angles].
            They can be passed directly to
            `pymrt.extras.num_phantom.render_geom_shapes()`.
            See this function for more details.

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
        - This is implemented based on `pymrt.geometry.nd_superellipsoid()`
          and therefore the sizes and the inner positions are halved, while
          angular values are defined differently compared to the reference
          paper.
        - In Fig.2 of doi:10.1002/mrm.21292, the rotation of last and 3rd last
          geom_shapes are swapped, contrarily to the parametric definition
          given (which is also implemented here).
    """
    geom_shapes = [
        [+2.0, ['ellipsoid', [0.3450, 0.4600, 0.4500]],
         [+0.0000, +0.0000, +0.0000], [+000.], None],
        [-0.8, ['ellipsoid', [0.3312, 0.4370, 0.4400]],
         [+0.0000, +0.0000, +0.0000], [+000.], None],
        [-0.2, ['ellipsoid', [0.2050, 0.0800, 0.1050]],
         [-0.1100, +0.0000, -0.1250], [-108.], None],
        [-0.2, ['ellipsoid', [0.1550, 0.0550, 0.1100]],
         [+0.1100, +0.0000, -0.1250], [-072.], None],
        [+0.2, ['ellipsoid', [0.1050, 0.1250, 0.2500]],
         [+0.0000, +0.1750, -0.1250], [+000.], None],
        [+0.2, ['ellipsoid', [0.0230, 0.0230, 0.0230]],
         [+0.0000, +0.0500, -0.1250], [+000.], None],
        [+0.1, ['ellipsoid', [0.0230, 0.0115, 0.0100]],
         [-0.0400, -0.3250, -0.1250], [+000.], None],
        [+0.1, ['ellipsoid', [0.0230, 0.0115, 0.0100]],
         [+0.0300, -0.3250, -0.1250], [-090.], None],
        [+0.2, ['ellipsoid', [0.0280, 0.0200, 0.0500]],
         [+0.0300, -0.0525, +0.3125], [-090.], None],
        [-0.2, ['ellipsoid', [0.0280, 0.0280, 0.0500]],
         [+0.0000, +0.0500, +0.3125], [+000.], None],
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
def render(
        shape,
        method,
        method_kws=None,
        n_dim=None):
    """
    Generate a numerical phantom.
    
    Args:
        shape (int|Iterable[int]):  
        method: 
        method_kws: 
        n_dim: 

    Returns:

    """
    methods = {
        'geom_shapes': None,
        'shepp_logan': 2, 'shepp_logan_toft': 2,
        'kak_roberts': 3, 'shepp_logan_3d': 3,
    }
    if isinstance(method, str):
        method = method.lower()
    method_kws = {} if method_kws is None else dict(method_kws)
    if method in methods:
        method_kws['shape'] = shape
        method_kws['geom_shapes'] = eval(method)()
        method_kws['n_dim'] = methods[method]
        method = mrt.geometry.multi_render
    arr = method(**method_kws) if callable(method) else None
    return arr


import numex.gui_tk_mpl

# arr = render(256, 'shepp_logan_toft')
# mrt.plot.quick(arr.T, origin='lower')

arr = render(256, 'shepp_logan_3d')
numex.gui_tk_mpl.explore(arr)


# ======================================================================
elapsed(__file__[len(DIRS['base']) + 1:])

# ======================================================================
if __name__ == '__main__':
    import doctest  # Test interactive Python examples

    msg(__doc__.strip())
    doctest.testmod()
    msg(report())
