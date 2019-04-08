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
import flyingcircus as fc  # Everything you always wanted to have in Python.*

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
        axes (Iterable|None): The spatial dimensions.
            If None, the spatial dimensions are assumed to be:
             - (0, 1, 2) if arr.ndim is at least 3
             - (0, 1) otherwise

    Returns:
        arr (np.ndarray): The output array.
    """
    if axes is None:
        axes = tuple(range(min(arr.ndim, 3)))
    return fc.num.ft


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
        axes (Iterable|None): The spatial dimensions.
            If None, the spatial dimensions are assumed to be:
             - (0, 1, 2) if arr.ndim is at least 3
             - (0, 1) otherwise

    Returns:
        arr (np.ndarray): The output array.
    """
    if axes is None:
        axes = tuple(range(min(arr.ndim, 3)))
    raise NotImplementedError


# ======================================================================
def grappa(
        arr,
        sens,
        kernel=None,
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
def g_factor_multi_replica(
        arr,
        reco_func,
        reco_args=None,
        reco_kwargs=None,
        noise_level=0.01,
        num=100):
    """
    Estimate geometric noise amplification (g-factor) with multi-replica.

    This a Monte Carlo method, effectively consisting of computing the
    standard deviation for multiple instances of the difference between
    the images reconstructed with and without additional Gaussian noise
    in the complex raw time-domain data.

    Args:
        arr (np.ndarray): The input raw data.
        reco_func (callable): The reconstruction function.
        reco_args (Iterable|None): Positional arguments for `reco_func`.
        reco_kwargs (tuple|dict|None): Keyword arguments for `reco_func`.
        noise_level (int|float): The noise level.
            This is used to determine the st.dev of the Gaussian noise
            being added at each iteration.
            The st.dev is computed as the peak-to-peak value multiplied by
            `noise_level` (the peak-to-peak value is the maximum of the
            peak-to-peak value for real and imaginary data separately).
        num (int): The number of repetitions.

    Returns:
        arr (np.ndarray): The estimated g-factor map.
    """
    reco_args = tuple(reco_args) if reco_args else ()
    reco_kwargs = dict(reco_kwargs) if reco_kwargs else {}

    # noise-less reco
    img_arr = reco_func(arr, *reco_args, **reco_kwargs)

    mean_arr = np.zeros_like(img_arr, dtype=float)
    mvar_arr = np.zeros_like(img_arr, dtype=float)

    # compute desired noise std
    # re_min, re_max = fc.num.minmax(np.real(arr))
    # im_min, im_max = fc.num.minmax(np.imag(arr))
    # cx_min, cx_max = min(re_min, im_min), max(re_max, im_max)
    cx_ptp = max(np.ptp(np.real(arr)), np.ptp(np.imag(arr)))
    noise_std = cx_ptp * noise_level

    for i in range(num):
        noise_arr = np.random.normal(0, noise_std, arr.shape)
        new_img_arr = reco_func(arr + noise_arr, *reco_args, **reco_kwargs)
        del noise_arr
        err_arr = np.abs(img_arr) - np.abs(new_img_arr)
        del new_img_arr

        mean_arr, mvar_arr = fc.util.next_mean_mvar(
            err_arr, mean_arr, mvar_arr, i)

    return np.sqrt(mvar_arr / (num - 1))
