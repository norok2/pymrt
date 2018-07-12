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
def render_superellipsoids(
        shape,
        superellipsoids,
        num_dim=None,
        order=0):
    # check that superellipsoids parameters have the right size
    if num_dim is None:
        num_dim = len(shape)
    else:
        shape = mrt.utils.auto_repeat(shape, num_dim, False, True)
    arr = np.zeros(shape)
    for val, semisizes, position, angles in superellipsoids:
        if angles is None:
            lin_mat = np.eye(num_dim)
        else:
            lin_mat = mrt.geometry.angles2linear(angles)
        position = mrt.geometry.rel2abs(shape, position)
        lin_mat, offset = mrt.geometry.prepare_affine(
            shape, lin_mat, position)
        arr += val * sp.ndimage.affine_transform(
            mrt.geometry.nd_superellipsoid(shape, semisizes).astype(np.float),
            lin_mat, offset, order=order)
    return arr


# ======================================================================
def shepp_logan():
    """


    Returns:


    References:
        - Shepp, L. A., and B. F. Logan. “The Fourier Reconstruction of a
          Head Section.” IEEE Transactions on Nuclear Science 21, no. 3 (June
          1974): 21–43. https://doi.org/10.1109/TNS.1974.6499235.
    """
    superellipsoids = [
        [+2.00, [0.4600, 0.3450], [+0.0000, +0.0000], None],
        [-0.98, [0.4370, 0.3312], [-0.0092, +0.0000], None],
        [-0.02, [0.1550, 0.0550], [+0.0000, +0.1100], [-018.0]],
        [-0.02, [0.2050, 0.0800], [+0.0000, -0.1100], [+018.0]],
        [+0.01, [0.1250, 0.1050], [+0.1750, +0.0000], None],
        [+0.01, [0.0230, 0.0230], [+0.0500, +0.0000], None],
        [+0.01, [0.0230, 0.0230], [-0.0500, +0.0000], None],
        [+0.01, [0.0115, 0.0230], [-0.3025, -0.0400], None],
        [+0.01, [0.0115, 0.0115], [-0.3025, +0.0000], None],
        [+0.01, [0.0230, 0.0115], [-0.3025, +0.0300], None],
    ]
    return superellipsoids


# ======================================================================
def shepp_logan_toft():
    """
    Toft modification of Shepp-Logan phantom.

    Returns:
        superellipsoids (Iterable[Iterable]): The geometric specifications.
            These are of the form: [val, semizes, position, angles].
            They can be passed directly to
            `pymrt.extras.num_phantom.render_superellipsoids()`.
            See this function for more details.

    References:
        - Shepp, L. A., and B. F. Logan. “The Fourier Reconstruction of a
          Head Section.” IEEE Transactions on Nuclear Science 21, no. 3 (June
          1974): 21–43. https://doi.org/10.1109/TNS.1974.6499235.
    """
    new_vals = [1.0, -0.8, -0.2, -0.2, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
    superellipsoids = [
        [new_val, semisizes, position, angles]
        for new_val, (val, semisizes, position, angles) in zip(
            new_vals, shepp_logan())]
    return superellipsoids


# ======================================================================
def kak_roberts():
    """
    Kak and Roberts modification of Shepp-Logan phantom with 3D extension.

    Returns:
        superellipsoids (Iterable[Iterable]): The geometric specifications.
            These are of the form: [val, semizes, position, angles].
            They can be passed directly to
            `pymrt.extras.num_phantom.render_superellipsoids()`.
            See this function for more details.

    References:
        - Young, Tzay Y., and K. S. Fu, eds. Handbook of Pattern Recognition
          and Image Processing. Handbooks in Science and Technology. Orlando:
          Academic Press, 1986.
        - Koay, Cheng Guan, Joelle E. Sarlls, and Evren Özarslan.
          “Three-Dimensional Analytical Magnetic Resonance Imaging Phantom in
          the Fourier Domain.” Magnetic Resonance in Medicine 58, no. 2 (
          August 1, 2007): 430–36. https://doi.org/10.1002/mrm.21292.
    """
    superellipsoids = [
        [+2.0, [0.4600, 0.3450, 0.4500], [+0.0000, +0.0000, +0.0000], None],
        [-0.8, [0.4370, 0.3312, 0.4400], [-0.0000, +0.0000, +0.0000], None],
        [-0.2, [0.2050, 0.0800, 0.1050], [+0.0000, -0.1100, -0.2500],
         [+018.0, +000.0, +000.0]],
        [-0.2, [0.1550, 0.0550, 0.1100], [+0.0000, +0.1100],
         [+018.0, +000.0, +000.0]],
        [+0.2, [0.1250, 0.1050], [+0.1750, +0.0000], None],
        [+0.2, [0.0230, 0.0230], [+0.0500, +0.0000], None],
        [+0.1, [0.0230, 0.0230], [-0.0500, +0.0000], None],
        [+0.1, [0.0115, 0.0230], [-0.3025, -0.0400], None],
        [+0.2, [0.0115, 0.0115], [-0.3025, +0.0000], None],
        [-0.2, [0.0230, 0.0115], [-0.3025, +0.0300], None],
    ]
    return superellipsoids


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
def generate(shape, method, method_kws, n_dim=None):
    if isinstance(method, str):
        method = method.lower()
    if method in methods:
        pass
    if callable(method):
        arr = method()
    return arr


import matplotlib.pyplot as plt
import pymrt.plot

arr = render_superellipsoids(256, kak_roberts(), 3)

mrt.plot.quick_3d(arr)
plt.show()
