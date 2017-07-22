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
# from pymrt import elapsed, print_elapsed
# from pymrt import msg, dbg


# ======================================================================
def sum_of_squares(
        arr,
        coil_axis=-1):
    """
    Coil sensitivity for the 'sum_of_squares' combination method.

    Note: the input itself is used as sensitivity. Therefore, this function
    actually returns the same array used for input, and the `coil_axis`
    parameter is left unused.

    Args:
        arr (np.ndarray): The input array.
        coil_axis (int): The coil dimension.
            The dimension of `arr` along which single coil elements are stored.

    Returns:
        arr (np.ndarray): The estimated coil sensitivity.
    """
    return arr


# ======================================================================
def adaptive(
        arr,
        max_iter=8,
        threshold=1e-6,
        coil_axis=-1):
    """
    Coil sensitivity for the 'adaptive' combination method.

    Effectively calls `block_adaptive()` with block size of 1.

    Args:
        arr (np.ndarray): The input array.
        max_iter (int): Maximum iterations in power algorithm.
            This is the maximum number of iterations used for determining the
            principal component (eigenvalue/vector) using the power algorithm.
        threshold (float): Threshold in power algorithm.
            If the next iteration modifies the eigenvalue (in absolute terms)
            less than the threshold, the power algorithm stops.
        coil_axis (int): The coil dimension.
            The dimension of `arr` along which single coil elements are stored.

    Returns:
        arr (np.ndarray): The estimated coil sensitivity.

    References:
        - Walsh, D.O., Gmitro, A.F., Marcellin, M.W., 2000. Adaptive
          reconstruction of phased array MR imagery. Magn. Reson. Med. 43,
          682–690. doi:10.1002/(SICI)1522-2594(
          200005)43:5<682::AID-MRM10>3.0.CO;2-G
    """
    return block_adaptive(arr, 1, max_iter, threshold, coil_axis)


# ======================================================================
def block_adaptive(
        arr,
        block=5,
        max_iter=8,
        threshold=1e-6,
        coil_axis=-1):
    """
    Coil sensitivity for the 'block_adaptive' combination method.

    Args:
        arr (np.ndarray): The input array.
        block (int|float|iterable[int|float]): The size of the block in px.
            Used for smoothing the coil covariance.
            If int or float, the block is isotropic all non-coil dimensions.
            If iterable, each size is applied to the corresponding dimension
            and its size must match the number of non-coil dimensions.
            If set to 0, no smoothing is performed and the algorithm
            reduces to non-block adaptive.
        max_iter (int): Maximum iterations in power algorithm.
            This is the maximum number of iterations used for determining the
            principal component (eigenvalue/vector) using the power algorithm.
        threshold (float): Threshold in power algorithm.
            If the next iteration modifies the eigenvalue (in absolute terms)
            less than the threshold, the power algorithm stops.
        coil_axis (int): The coil dimension.
            The dimension of `arr` along which single coil elements are stored.

    Returns:
        sens (np.ndarray): The estimated coil sensitivity.

    References:
        - Walsh, D.O., Gmitro, A.F., Marcellin, M.W., 2000. Adaptive
          reconstruction of phased array MR imagery. Magn. Reson. Med. 43,
          682–690. doi:10.1002/(SICI)1522-2594(
          200005)43:5<682::AID-MRM10>3.0.CO;2-G
        - Inati, S.J., Hansen, M.S., Kellman, P., 2013. A Solution to the
          Phase Problem in Adaptive Coil Combination, in: Proceedings of the
          ISMRM 21st Annual Meeting & Exhibition. Presented at the 21st Annual
          Meeting & Exhibition of the International Society for Magnetic
          Resonance in Medicine, ISMRM, Salt Lake City, Utah, USA.
    """
    coil_axis = coil_axis % arr.ndim
    shape = arr.shape
    num_coils = shape[coil_axis]
    arr = np.swapaxes(arr, coil_axis, -1)
    base_shape = arr.shape[:-1]

    # calculate the coil covariance
    coil_cov = np.zeros(base_shape + (num_coils,) * 2, dtype=np.complex)
    for i in range(num_coils):
        for j in range(num_coils):
            coil_cov[..., i, j] = arr[..., i] * arr[..., j].conj()

    # if block is not 0, smooth the coil covariance
    if block > 0:
        for i in range(num_coils):
            for j in range(num_coils):
                coil_cov[..., i, j] = mrt.utils.filter_cx(
                    coil_cov[..., i, j], sp.ndimage.uniform_filter,
                    (), dict(size=block))

    # calculate the principal eigenvector of the coil covariance
    # using the power method (pointwise through all spatial dimensions)
    sens = np.zeros(base_shape + (num_coils,), dtype=np.complex)
    for i in itertools.product(*[range(k) for k in base_shape]):
        ii = tuple(j for j in i if i != slice(None)) + (slice(None),)
        iii = tuple(j for j in i if i != slice(None)) + (slice(None),) * 2
        sensitivity_i = np.sum(coil_cov[iii], axis=-1)
        power_i = np.linalg.norm(sensitivity_i)
        if power_i:
            sensitivity_i = sensitivity_i / power_i
        else:
            sensitivity_i *= 0
            continue
        for _ in range(max_iter):
            sensitivity_i = np.dot(coil_cov[iii], sensitivity_i)
            last_power_i = power_i
            power_i = np.linalg.norm(sensitivity_i)
            if power_i:
                sensitivity_i = sensitivity_i / power_i
            else:
                sensitivity_i *= 0
                break
            if np.abs(last_power_i - power_i) < threshold:
                break
        sens[ii] = sensitivity_i
    sens = np.swapaxes(sens, -1, coil_axis)
    return sens


# ======================================================================
def block_adaptive_iter(
        arr,
        block=5,
        max_iter=16,
        threshold=1e-7,
        coil_axis=-1):
    """
    Coil sensitivity for the 'block_adaptive_iter' combination method.

    This is an iterative and faster implementation of the algorithm for
    computing 'block_adaptive' sensitivity, with improved phase accuracy.

    Args:
        arr (np.ndarray): The input array.
        block (int|float|iterable[int|float]): The size of the block in px.
            Used for smoothing the coil covariance.
            If int or float, the block is isotropic all non-coil dimensions.
            If iterable, each size is applied to the corresponding dimension
            and its size must match the number of non-coil dimensions.
            If set to 0, no smoothing is performed and the algorithm
            reduces to non-block adaptive.
        max_iter (int): Maximum number of iterations.
            If `threshold` > 0, the algorithm may stop earlier.
        threshold (float): Threshold for next iteration.
            If the next iteration globally modifies the sensitivity by less
            than `threshold`, the algorithm stops.
        coil_axis (int): The coil dimension.
            The dimension of `arr` along which single coil elements are stored.

    Returns:
        arr (np.ndarray): The estimated coil sensitivity.

    References:
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
    coil_axis = coil_axis % arr.ndim
    shape = arr.shape
    num_coils = shape[coil_axis]
    arr = np.swapaxes(arr, coil_axis, -1)
    base_shape = arr.shape[:-1]
    if isinstance(block, int):
        block = (block,) * (arr.ndim - 1) + (1,)
    else:
        assert (len(block) + 1 == arr.ndim)

    epsilon = np.finfo(np.float).eps
    no_coil_axes = tuple(range(0, arr.ndim - 1))

    sens = np.zeros_like(arr, dtype=np.complex)

    mean_coil = np.sum(arr, no_coil_axes)
    mean_coil /= np.linalg.norm(mean_coil)
    rho = np.einsum('...i,i', arr, mean_coil.conj())

    for i in range(max_iter):
        last_rho = rho.copy() if threshold > 0 else rho
        sens_b = mrt.utils.filter_cx(
            arr * rho[..., None].conj(),
            sp.ndimage.uniform_filter, (), dict(size=block))
        sens = sens_b / (
            np.sqrt(np.sum(sens_b * sens_b.conj(), -1)) + epsilon)[..., None]
        rho = np.sum(arr * np.conj(sens), -1)
        mean_coil = np.sum(sens * rho[..., None], no_coil_axes)
        mean_coil /= np.linalg.norm(mean_coil)

        extra_phase = np.einsum('...i,i', sens, mean_coil.conj())
        extra_phase /= (np.abs(extra_phase) + epsilon)
        rho *= extra_phase
        sens *= extra_phase[..., None].conj()
        if threshold > 0:
            delta = (np.linalg.norm(rho - last_rho) / np.linalg.norm(rho))
            if delta < threshold:
                break
    sens = np.swapaxes(sens, -1, coil_axis)
    return sens


# ======================================================================
def block_subspace_fourier(
        arr,
        coil_axis=-1):
    """
    Coil sensitivity for the 'block_subspace_fourier' combination method.

    Args:
        arr:
        coil_axis:

    Returns:

    References:
        - Gol Gungor, D., Potter, L.C., 2016. A subspace-based coil
          combination method for phased-array magnetic resonance imaging.
          Magn. Reson. Med. 75, 762–774. doi:10.1002/mrm.25664
    """
    raise NotImplementedError


# ======================================================================
def virtual_reference(
        arr,
        coil_axis=-1):
    """
    Coil sensitivity for the 'virtual_reference' combination method.

    Args:
        arr:
        coil_axis:

    Returns:

    References:
        - Parker, D.L., Payne, A., Todd, N., Hadley, J.R., 2014. Phase
          reconstruction from multiple coil data using a virtual reference
          coil. Magn. Reson. Med. 72, 563–569. doi:10.1002/mrm.24932
    """
    raise NotImplementedError


# ======================================================================
def me_conjugate_hermitian(
        arr,
        echo_axis=-2,
        coil_axis=-1):
    """
    Coil sensitivity for the 'conjugate_hermitian' combination method.

    Note: the input itself is used as sensitivity. Therefore, this function
    actually returns the same array used for input, and the `coil_axis`
    parameter is left unused.

    Args:
        arr (np.ndarray): The input array.
        echo_axis (int): The echo dimension.
            The dimension of `arr` along which different echoes are stored.
        coil_axis (int): The coil dimension.
            The dimension of `arr` along which single coil elements are stored.

    Returns:
        arr (np.ndarray): The estimated coil sensitivity.
    """
    raise NotImplementedError


# ======================================================================
def me_svd(
        arr,
        echo_axis=-2,
        coil_axis=-1):
    """
    Coil sensitivity for the 'conjugate_hermitian' combination method.

    Note: the input itself is used as sensitivity. Therefore, this function
    actually returns the same array used for input, and the `coil_axis`
    parameter is left unused.

    Args:
        arr (np.ndarray): The input array.
        echo_axis (int): The echo dimension.
            The dimension of `arr` along which different echoes are stored.
        coil_axis (int): The coil dimension.
            The dimension of `arr` along which single coil elements are stored.

    Returns:
        arr (np.ndarray): The estimated coil sensitivity.
    """
    raise NotImplementedError


# ======================================================================
def me_composer(
        arr,
        echo_axis=-2,
        coil_axis=-1):
    """
    Coil sensitivity for the 'conjugate_hermitian' combination method.

    Note: the input itself is used as sensitivity. Therefore, this function
    actually returns the same array used for input, and the `coil_axis`
    parameter is left unused.

    Args:
        arr (np.ndarray): The input array.
        echo_axis (int): The echo dimension.
            The dimension of `arr` along which different echoes are stored.
        coil_axis (int): The coil dimension.
            The dimension of `arr` along which single coil elements are stored.

    Returns:
        arr (np.ndarray): The estimated coil sensitivity.
    """
    raise NotImplementedError


# ======================================================================
def ref_snr_optimal(
        arr,
        ref,
        coil_axis=-1):
    """
    Coil sensitivity for the 'conjugate_hermitian' combination method.

    Note: the input itself is used as sensitivity. Therefore, this function
    actually returns the same array used for input, and the `coil_axis`
    parameter is left unused.

    Args:
        arr (np.ndarray): The input array.
        echo_axis (int): The echo dimension.
            The dimension of `arr` along which different echoes are stored.
        coil_axis (int): The coil dimension.
            The dimension of `arr` along which single coil elements are stored.

    Returns:
        arr (np.ndarray): The estimated coil sensitivity.

    References:
        - Roemer, P.B., Edelstein, W.A., Hayes, C.E., Souza, S.P., Mueller,
          O.M., 1990. The NMR phased array. Magn Reson Med 16, 192–225.
          doi:10.1002/mrm.1910160203
    """
    raise NotImplementedError


# ======================================================================
def ref_adaptive(
        arr,
        ref,
        coil_axis=-1):
    """
    Coil sensitivity for the 'conjugate_hermitian' combination method.

    Note: the input itself is used as sensitivity. Therefore, this function
    actually returns the same array used for input, and the `coil_axis`
    parameter is left unused.

    Args:
        arr (np.ndarray): The input array.
        echo_axis (int): The echo dimension.
            The dimension of `arr` along which different echoes are stored.
        coil_axis (int): The coil dimension.
            The dimension of `arr` along which single coil elements are stored.

    Returns:
        arr (np.ndarray): The estimated coil sensitivity.

    References:
        - Jellúš, V., Kannengiesser, S.A.R., 2014. Adaptive Coil Combination
          Using a Body Coil Scan as Phase Reference, in: Proceedings of the
          ISMRM 22nd Annual Meeting & Exhibition. Presented at the 22nd Annual
          Meeting & Exhibition of the International Society for Magnetic
          Resonance in Medicine, Milan, Italy.
    """
    raise NotImplementedError


# ======================================================================
def sensitivity(
        arr,
        method='block_adaptive_iter',
        coil_axis=-1):
    if method:
        method = method.lower()
    methods = (
        'sum_of_squares', 'block_adaptive_iter', 'adaptive', 'block_adaptive',
        'virtual_reference')
    if method in methods:
        sens = exec(method)(arr, coil_axis=coil_axis)
    else:
        warnings.warn(
            'Sensitivity estimation method `{}` not known'.format(method) +
            ' Using default.')
        sens = sensitivity(arr, coil_axis=coil_axis)
    return sens


# ======================================================================
def combine(
        arr,
        sens,
        norm=False,
        coil_axis=-1):
    """
    Calculate the coil combination array from multiple coil elements.

    The coil sensitivity is specified as parameter.

    Args:
        arr (np.ndarray): The input array.
        sens (np.ndarray|str): The coil sensitivity.
            If `np.ndarray`, its shape must match with `arr`.
            If `str`, the sensitivity is calculated using `sensitivity()`.
        norm (bool): Normalize using coil sensitivity magnitude.
        coil_axis (int): The coil dimension.
            The dimension of `arr` along which single coil elements are stored.

    Returns:
        arr (np.ndarray): The combined array.
    """
    if isinstance(sens, str):
        sens = sensitivity(arr, sens)
    assert (arr.shape == sens.shape)
    arr = np.sum(sens.conj() * arr, coil_axis)
    if norm:
        epsilon = np.finfo(np.float).eps
        arr /= (np.sum(np.abs(sens) ** 2, coil_axis) + epsilon)
    return arr


# ======================================================================
def combine_ref(
        arr,
        sens,
        norm=False,
        coil_axis=-1):
    """
    Calculate the coil combination array from multiple coil elements.

    The coil sensitivity is specified as parameter.

    Args:
        arr (np.ndarray): The input array.
        sens (np.ndarray|str): The coil sensitivity.
            If `np.ndarray`, its shape must match with `arr`.
            If `str`, the sensitivity is calculated using `sensitivity()`.
        norm (bool): Normalize using coil sensitivity magnitude.
        coil_axis (int): The coil dimension.
            The dimension of `arr` along which single coil elements are stored.

    Returns:
        arr (np.ndarray): The combined array.
    """
    if isinstance(sens, str):
        sens = sensitivity(arr, sens)
    assert (arr.shape == sens.shape)
    arr = np.sum(sens.conj() * arr, coil_axis)
    if norm:
        epsilon = np.finfo(np.float).eps
        arr /= (np.sum(np.abs(sens) ** 2, coil_axis) + epsilon)
    return arr


# ======================================================================
def combine_me(
        arr,
        sens,
        norm=False,
        coil_axis=-1):
    """
    Calculate the coil combination array from multiple coil elements.

    The coil sensitivity is specified as parameter.

    Args:
        arr (np.ndarray): The input array.
        sens (np.ndarray|str): The coil sensitivity.
            If `np.ndarray`, its shape must match with `arr`.
            If `str`, the sensitivity is calculated using `sensitivity()`.
        norm (bool): Normalize using coil sensitivity magnitude.
        coil_axis (int): The coil dimension.
            The dimension of `arr` along which single coil elements are stored.

    Returns:
        arr (np.ndarray): The combined array.
    """
    if isinstance(sens, str):
        sens = sensitivity(arr, sens)
    assert (arr.shape == sens.shape)
    arr = np.sum(sens.conj() * arr, coil_axis)
    if norm:
        epsilon = np.finfo(np.float).eps
        arr /= (np.sum(np.abs(sens) ** 2, coil_axis) + epsilon)
    return arr
