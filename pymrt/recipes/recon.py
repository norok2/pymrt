#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pymrt.recipes.recon: Reconstruction techniques.

EXPERIMENTAL!
"""

# ======================================================================
# :: Future Imports
from __future__ import (
    division, absolute_import, print_function, unicode_literals, )

# ======================================================================
# :: Python Standard Library Imports
import itertools  # Functions creating iterators for efficient looping
import warnings  # Warning control
import collections  # Container datatypes

# :: External Imports
import numpy as np  # NumPy (multidimensional numerical arrays library)
import scipy as sp  # SciPy (signal and image processing library)
# import matplotlib as mpl  # Matplotlib (2D/3D plotting library)
# import sympy as sym  # SymPy (symbolic CAS library)

# :: External Imports Submodules
# import matplotlib.pyplot as plt  # Matplotlib's pyplot: MATLAB-like syntax
# import scipy.optimize  # SciPy: Optimization Algorithms
# import scipy.integrate  # SciPy: Integrations facilities
# import scipy.constants  # SciPy: Mathematal and Physical Constants
import scipy.ndimage  # SciPy: ND-image Manipulation

# :: Local Imports
import pymrt as mrt
import pymrt.utils


# import pymrt.utils
# import pymrt.computation as pmc

# from pymrt import VERB_LVL, D_VERB_LVL, VERB_LVL_NAMES
# from pymrt import elapsed, report
# from pymrt import msg, dbg


# ======================================================================
def k2r_space_cartesian(
        arr,
        axes=None):
    """
    Transform data from k-space to r-space.

    The k-space is the spatial frequency (or raw) domain.
    The r-space is the spatial (or image) domain.

    EXPERIMENTAL!

    Args:
        arr (np.ndarray): The input array.
            Data can be either in k-space or in image space.

    Returns:
        arr (np.ndarray): The output array.
    """
    if not axes:
        axes = tuple(range(arr.ndim))
    raise NotImplementedError


# ======================================================================
def r2k_space_cartesian(
        arr,
        axes=None):
    """
    Transform data from r-space to k-space.

    The k-space is the spatial frequency (or raw) domain.
    The r-space is the spatial (or image) domain.

    EXPERIMENTAL!

    Args:
        arr (np.ndarray): The input array.
            Data can be either in k-space or in image space.

    Returns:
        arr (np.ndarray): The output array.
    """
    if not axes:
        axes = tuple(range(arr.ndim))
    raise NotImplementedError


# ======================================================================
def grappa(
        arr,
        sens,
        coil_index=-1):
    """
    EXPERIMENTAL!

    Args:
        arr (np.ndarray): The input array.
            Data can be either in k-space or in image space.

    Returns:
        arr (np.ndarray): The output array.
    """
    raise NotImplementedError


# ======================================================================
def sense(
        arr,
        sens,
        coil_index=-1):
    """
    EXPERIMENTAL!

    Args:
        arr (np.ndarray): The input array.
            Data can be either in k-space or in image space.

    Returns:
        arr (np.ndarray): The output array.
    """
    raise NotImplementedError


# ======================================================================
def espirit(
        arr):
    """
    EXPERIMENTAL!

    Args:
        arr (np.ndarray): The input array.
            Data can be either in k-space or in image space.

    Returns:
        arr (np.ndarray): The output array.
    """
    raise NotImplementedError


# ======================================================================
def compressed_sensing(
        arr):
    """
    EXPERIMENTAL!

    Args:
        arr (np.ndarray): The input array.
            Data can be either in k-space or in image space.

    Returns:
        arr (np.ndarray): The estimated coil sensitivity.
    """
    raise NotImplementedError


# ======================================================================
def g_noise():
    raise NotImplementedError
