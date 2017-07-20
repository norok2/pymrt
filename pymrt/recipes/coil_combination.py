#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
pymrt.recipes.coil_combination: Coil combination of phased array.
"""

# ======================================================================
# :: Future Imports
from __future__ import (
    division, absolute_import, print_function, unicode_literals)

# ======================================================================
# :: Python Standard Library Imports
# import itertools  # Functions creating iterators for efficient looping
import warnings  # Warning control
import collections  # Container datatypes

# :: External Imports
import numpy as np  # NumPy (multidimensional numerical arrays library)

# :: Local Imports
import pymrt as mrt


# import pymrt.utils
# import pymrt.computation as pmc

# from pymrt import VERB_LVL, D_VERB_LVL, VERB_LVL_NAMES
# from pymrt import elapsed, print_elapsed
# from pymrt import msg, dbg


# ======================================================================
def sum_of_squares(
        arr,
        coil_index=-1):
    """
    Coil sensitivity for the 'sum_of_squares' combination method.

    Note: the input itself is used as sensitivity. Therefore, this function
    actually returns the same array used for input, and the `coil_index`
    parameter is left unused.

    Args:
        arr (np.ndarray): The input array.
        coil_index (int): The coil dimension.
            The dimension of `arr` along which single coil elements are stored.

    Returns:
        arr (np.ndarray): The estimated coil sensitivity.
    """
    return arr


# ======================================================================
def conjugate_hermitian(
        arr,
        coil_index=-1):
    """
    Coil sensitivity for the 'conjugate_hermitian' combination method.

    Note: the input itself is used as sensitivity. Therefore, this function
    actually returns the same array used for input, and the `coil_index`
    parameter is left unused.

    Args:
        arr (np.ndarray): The input array.
        coil_index (int): The coil dimension.
            The dimension of `arr` along which single coil elements are stored.

    Returns:
        arr (np.ndarray): The estimated coil sensitivity.
    """
    return sensitivity


# ======================================================================
def adaptive(
        arr,
        sigma=5,
        max_iter=8,
        coil_index=-1):
    """
    Coil sensitivity for the 'adaptive' combination method.

    Args:
        arr (np.ndarray): The input array.
        sigma:
        max_iter:
        coil_index (int): The coil dimension.
            The dimension of `arr` along which single coil elements are stored.

    Returns:
        arr (np.ndarray): The estimated coil sensitivity.

    See Also:
        - Walsh, D.O., Gmitro, A.F., Marcellin, M.W., 2000. Adaptive
          reconstruction of phased array MR imagery. Magn. Reson. Med. 43,
          682–690. doi:10.1002/(SICI)1522-2594(
          200005)43:5<682::AID-MRM10>3.0.CO;2-G
    """
    coil_index = coil_index % arr.ndim
    shape = arr.shape
    num_coils = shape[coil_index]
    base_shape = shape[:coil_index] + shape[coil_index:]
    base_mask = [slice(None)] * arr.ndim

    # calculate the coil covariance
    coil_cov = np.zeros(base_shape + (num_coils,) * 2)
    mask_i = base_mask.copy()
    mask_j = base_mask.copy()
    for i in range(num_coils):
        for j in range(num_coils):
            mask_i[coil_index] = i
            mask_j[coil_index] = j
            coil_cov[..., i, j] = arr[mask_i] * arr[mask_j].conj()

    return sensitivity


# ======================================================================
def block_adaptive(
        arr,
        sigma=5,
        max_iter=8,
        threshold=1e-4,
        coil_index=-1):
    """
    Coil sensitivity for the 'block_adaptive' combination method.

    Args:
        arr (np.ndarray): The input array.
        sigma (float|iterable[float]:
        max_iter (int):
        threshold (float):
        coil_index (int): The coil dimension.
            The dimension of `arr` along which single coil elements are stored.

    Returns:
        arr (np.ndarray): The estimated coil sensitivity.

    See Also:
        - Walsh, D.O., Gmitro, A.F., Marcellin, M.W., 2000. Adaptive
          reconstruction of phased array MR imagery. Magn. Reson. Med. 43,
          682–690. doi:10.1002/(SICI)1522-2594(
          200005)43:5<682::AID-MRM10>3.0.CO;2-G
        - Inati, S.J., Hansen, M.S., Kellman, P., 2013. A Solution to the
          Phase Problem in Adaptive Coil Combination, in: Proceedings of the
          ISMRM 21st Annual Meeting & Exhibition. Presented at the 21st Annual
          Meeting & Exhibition of the International Society for Magnetic
          Resonance in Medicine, ISMRM, Salt Lake City, Utah, USA.
        - Inati, S.J., Hansen, M.S., Kellman, P., 2014. A Fast Optimal
          Method for Coil Sensitivity Estimation and Adaptive Coil
          Combination for Complex Images, in: Proceedings of the ISMRM 22nd
          Annual Meeting & Exhibition. Presented at the 22nd Annual Meeting
          & Exhibition of the International Society for Magnetic Resonance
          in Medicine, ISMRM, Milan, Italy.
    """
    num_coils = arr.shape[coil_index]
    return sensitivity


# ======================================================================
def virtual_reference(
        arr,
        coil_index=-1):
    """
    Coil sensitivity for the 'virtual_reference' combination method.

    Args:
        arr:
        coil_index:

    Returns:

    """
    return sensitivity


# ======================================================================
def coil_sensitivity(
        arr,
        method='block_adaptive',
        coil_index=-1):
    if method:
        method = method.lower()
    methods = (
        'sum_of_squares', 'conjugate_hermitian', 'adaptive', 'block_adaptive',
        'virtual_reference')
    if method in methods:
        sensitivity = exec(method)(arr, coil_index=coil_index)
    else:
        warnings.warn(
            'Sensitivity estimation method `{}` not known'.format(method) +
            ' Using default.')
        sensitivity = coil_sensitivity(arr, coil_index=coil_index)
    return sensitivity


# ======================================================================
def coil_combine(
        arr,
        sensitivity,
        coil_index=-1):
    """
    Calculate the coil combination array from multiple coil elements.

    The coil sensitivity is specified as parameter.

    Args:
        arr (np.ndarray): The input array.
        sensitivity (np.ndarray|str): The coil sensitivity.
            If `np.ndarray`, its shape must match with `arr`.
            If `str`, the sensitivity is calculated using `coil_sensitivity()`.
        coil_index (int): The coil dimension.
            The dimension of `arr` along which single coil elements are stored.

    Returns:
        arr (np.ndarray): The combined array.
    """
    if isinstance(sensitivity, str):
        sensitivity = coil_sensitivity(arr, sensitivity)
    assert (arr.shape == sensitivity.shape)
    arr = (
        np.sum(arr * sensitivity.conj(), coil_index) /
        (np.sum(np.abs(sensitivity) * np.abs(sensitivity), coil_index) +
         np.finfo(np.float).eps)
    return arr
