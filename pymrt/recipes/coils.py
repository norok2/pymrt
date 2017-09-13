#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pymrt.recipes.coils: Coil sensitivity and combination for phased array.
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
# import scipy.sparse  # SciPy: Sparse Matrices
import scipy.linalg  # Scipy: Linear Algebra

# :: Local Imports
import pymrt as mrt
import pymrt.utils
import pymrt.segmentation

from pymrt import VERB_LVL, D_VERB_LVL, VERB_LVL_NAMES
from pymrt import elapsed, report
from pymrt import msg, dbg


# ======================================================================
def complex_sum(
        arr,
        coil_axis=-1,
        verbose=D_VERB_LVL):
    """
    Coil sensitivity for the 'complex_sum' combination method.

    Note: this function returns a constant number, therefore the `coil_axis`
    parameter is left unused.

    Args:
        arr (np.ndarray): The input array.
        coil_axis (int): The coil dimension.
            The dimension of `arr` along which single coil elements are stored.
        verbose (int): Set level of verbosity.

    Returns:
        arr (np.ndarray): The estimated coil sensitivity.

    References:
        - Roemer, P.B., Edelstein, W.A., Hayes, C.E., Souza, S.P.,
          Mueller, O.M., 1990. The NMR phased array. Magn Reson Med 16,
          192–225. doi:10.1002/mrm.1910160203
    """
    return 1.0


# ======================================================================
def sum_of_squares(
        arr,
        coil_axis=-1,
        verbose=D_VERB_LVL):
    """
    Coil sensitivity for the 'sum_of_squares' combination method.

    Note: this function returns the same array used for input except for the
    normalization, therefore the `coil_axis` parameter is left unused.

    Args:
        arr (np.ndarray): The input array.
        coil_axis (int): The coil dimension.
            The dimension of `arr` along which single coil elements are stored.
        verbose (int): Set level of verbosity.

    Returns:
        arr (np.ndarray): The estimated coil sensitivity.

    References:
        - Roemer, P.B., Edelstein, W.A., Hayes, C.E., Souza, S.P.,
          Mueller, O.M., 1990. The NMR phased array. Magn Reson Med 16,
          192–225. doi:10.1002/mrm.1910160203
    """
    sens = arr / np.abs(arr)
    return sens


# ======================================================================
def smooth_sum_of_squares(
        arr,
        block=3,
        coil_axis=-1,
        verbose=D_VERB_LVL):
    """
    Coil sensitivity for the 'smooth_sum_of_squares' combination method.

    Use the smoothed input as coil sensitivity.

    Args:
        arr (np.ndarray): The input array.
        block (int|float|iterable[int|float]): The size of the block in px.
            Used for smoothing the coil covariance.
            If int or float, the block is isotropic in all non-coil dimensions.
            If iterable, each size is applied to the corresponding dimension
            and its size must match the number of non-coil dimensions.
            If set to 0, no smoothing is performed and the algorithm
            reduces to non-block adaptive.
        coil_axis (int): The coil dimension.
            The dimension of `arr` along which single coil elements are stored.
        verbose (int): Set level of verbosity.

    Returns:
        arr (np.ndarray): The estimated coil sensitivity.
    """
    coil_axis = coil_axis % arr.ndim
    arr = np.swapaxes(arr, coil_axis, -1)
    if isinstance(block, int):
        block = (block,) * (arr.ndim - 1) + (0,)
    else:
        assert (len(block) + 1 == arr.ndim)
    sens = mrt.utils.filter_cx(
        arr / np.abs(arr), sp.ndimage.uniform_filter, (), dict(size=block))
    sens = np.swapaxes(sens, -1, coil_axis)
    return sens


# ======================================================================
def adaptive(
        arr,
        max_iter=16,
        threshold=1e-7,
        coil_axis=-1,
        verbose=D_VERB_LVL):
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
        verbose (int): Set level of verbosity.

    Returns:
        arr (np.ndarray): The estimated coil sensitivity.

    References:
        - Walsh, D.O., Gmitro, A.F., Marcellin, M.W., 2000. Adaptive
          reconstruction of phased array MR imagery. Magn. Reson. Med. 43,
          682–690. doi:10.1002/(SICI)1522-2594(
          200005)43:5<682::AID-MRM10>3.0.CO;2-G
    """
    return block_adaptive(arr, 0, max_iter, threshold, coil_axis, verbose)


# ======================================================================
def block_adaptive(
        arr,
        block=5,
        max_iter=16,
        threshold=1e-7,
        coil_axis=-1,
        verbose=D_VERB_LVL):
    """
    Coil sensitivity for the 'block_adaptive' combination method.

    Args:
        arr (np.ndarray): The input array.
        block (int|float|iterable[int|float]): The size of the block in px.
            Used for smoothing the coil covariance.
            If int or float, the block is isotropic in all non-coil dimensions.
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
        verbose (int): Set level of verbosity.

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
    coil_cov = np.zeros(base_shape + (num_coils,) * 2, dtype=complex)
    for i in range(num_coils):
        for j in range(num_coils):
            coil_cov[..., i, j] = arr[..., i] * arr[..., j].conj()

    # if block is larger than 1, smooth the coil covariance
    if block > 0:
        for i in range(num_coils):
            for j in range(num_coils):
                coil_cov[..., i, j] = mrt.utils.filter_cx(
                    coil_cov[..., i, j], sp.ndimage.uniform_filter,
                    (), dict(size=block))

    # calculate the principal eigenvector of the coil covariance
    # using the power method (pointwise through all spatial dimensions)
    sens = np.zeros(base_shape + (num_coils,), dtype=complex)
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
        threshold=1e-8,
        coil_axis=-1,
        verbose=D_VERB_LVL):
    """
    Coil sensitivity for the 'block_adaptive_iter' combination method.

    This is an iterative and faster implementation of the algorithm for
    computing 'block_adaptive' sensitivity, with improved phase accuracy.

    Args:
        arr (np.ndarray): The input array.
        block (int|float|iterable[int|float]): The size of the block in px.
            Used for smoothing the coil covariance.
            If int or float, the block is isotropic in all non-coil dimensions.
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
        verbose (int): Set level of verbosity.

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
    if isinstance(block, (int, float)):
        block = (block,) * (arr.ndim - 1) + (0,)
    else:
        assert (len(block) + 1 == arr.ndim)

    msg('arr.shape={}'.format(arr.shape), verbose, VERB_LVL['debug'])
    msg('block={}'.format(block), verbose, VERB_LVL['debug'])
    epsilon = np.finfo(np.float).eps
    no_coil_axes = tuple(range(0, arr.ndim - 1))
    msg('threshold={}'.format(threshold), verbose, VERB_LVL['debug'])
    msg('max_iter={}'.format(max_iter), verbose, VERB_LVL['debug'])
    sens = np.zeros_like(arr, dtype=complex)

    with np.errstate(divide='ignore', invalid='ignore'):
        mean_coil = np.sum(arr, no_coil_axes)
        mean_coil /= np.linalg.norm(mean_coil)
        rho = np.einsum('...i,i', arr, mean_coil.conj())

        for i in range(max_iter):
            last_rho = rho.copy() if threshold > 0 else rho
            sens = mrt.utils.filter_cx(
                arr * rho[..., None].conj(),
                sp.ndimage.uniform_filter, (), dict(size=block))
            sens /= (
                np.sqrt(np.sum(sens * sens.conj(), -1))
                + epsilon)[..., None]
            rho = np.sum(arr * np.conj(sens), -1)
            mean_coil = np.sum(sens * rho[..., None], no_coil_axes)
            mean_coil /= np.linalg.norm(mean_coil)

            extra_phase = np.einsum('...i,i', sens, mean_coil.conj())
            extra_phase /= (np.abs(extra_phase) + epsilon)
            rho *= extra_phase
            sens *= extra_phase[..., None].conj()
            msg('{}'.format(i + 1),
                verbose, VERB_LVL['debug'], end=' ' if threshold else ', ',
                flush=True)
            if threshold > 0:
                delta = (np.linalg.norm(rho - last_rho) / np.linalg.norm(rho))
                msg('delta={}'.format(delta), verbose, VERB_LVL['debug'],
                    end=', ' if i + 1 < max_iter else '.\n', flush=True)
                if delta < threshold:
                    break
    sens = np.swapaxes(sens, -1, coil_axis)
    return sens


# ======================================================================
def wavelet_adaptive_iter(
        arr,
        block=5,
        max_iter=16,
        threshold=1e-8,
        coil_axis=-1,
        verbose=D_VERB_LVL):
    """
    Coil sensitivity for the 'block_adaptive_iter' combination method.

    This is an iterative and faster implementation of the algorithm for
    computing 'block_adaptive' sensitivity, with improved phase accuracy.

    Args:
        arr (np.ndarray): The input array.
        block (int|float|iterable[int|float]): The size of the block in px.
            Used for smoothing the coil covariance.
            If int or float, the block is isotropic in all non-coil dimensions.
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
        verbose (int): Set level of verbosity.

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
    raise NotImplementedError
    coil_axis = coil_axis % arr.ndim
    shape = arr.shape
    num_coils = shape[coil_axis]
    arr = np.swapaxes(arr, coil_axis, -1)
    base_shape = arr.shape[:-1]
    if isinstance(block, (int, float)):
        block = (block,) * (arr.ndim - 1) + (0,)
    else:
        assert (len(block) + 1 == arr.ndim)

    msg('arr.shape={}'.format(arr.shape), verbose, VERB_LVL['debug'])
    msg('block={}'.format(block), verbose, VERB_LVL['debug'])
    epsilon = np.finfo(np.float).eps
    no_coil_axes = tuple(range(0, arr.ndim - 1))
    msg('threshold={}'.format(threshold), verbose, VERB_LVL['debug'])
    msg('max_iter={}'.format(max_iter), verbose, VERB_LVL['debug'])
    sens = np.zeros_like(arr, dtype=complex)

    with np.errstate(divide='ignore', invalid='ignore'):
        mean_coil = np.sum(arr, no_coil_axes)
        mean_coil /= np.linalg.norm(mean_coil)
        rho = np.einsum('...i,i', arr, mean_coil.conj())

        for i in range(max_iter):
            last_rho = rho.copy() if threshold > 0 else rho
            sens = mrt.utils.filter_cx(
                arr * rho[..., None].conj(),
                sp.ndimage.uniform_filter, (), dict(size=block))
            sens /= (
                np.sqrt(np.sum(sens * sens.conj(), -1))
                + epsilon)[..., None]
            rho = np.sum(arr * np.conj(sens), -1)
            mean_coil = np.sum(sens * rho[..., None], no_coil_axes)
            mean_coil /= np.linalg.norm(mean_coil)

            extra_phase = np.einsum('...i,i', sens, mean_coil.conj())
            extra_phase /= (np.abs(extra_phase) + epsilon)
            rho *= extra_phase
            sens *= extra_phase[..., None].conj()
            msg('{}'.format(i + 1),
                verbose, VERB_LVL['debug'], end=' ' if threshold else ', ',
                flush=True)
            if threshold > 0:
                delta = (np.linalg.norm(rho - last_rho) / np.linalg.norm(rho))
                msg('delta={}'.format(delta), verbose, VERB_LVL['debug'],
                    end=', ' if i + 1 < max_iter else '.\n', flush=True)
                if delta < threshold:
                    break
    sens = np.swapaxes(sens, -1, coil_axis)
    return sens


# ======================================================================
def compress_svd(
        arr,
        k_svd='quad_weight',
        coil_axis=-1,
        verbose=D_VERB_LVL):
    """
    Compress the coil data to the SVD principal components.

    Rearranges (diagonalize) the acquired single-channel data into virtual
    single-channel data sorted by eigenvalue magnitude.
    If the number of SVD components `k_svd` is smaller than the number of
    coils, this is useful both as a denoise method and for reducing the
    complexity of the problem and the memory usage.

    Args:
        arr (np.ndarray): The input array.
        k_svd (int|float|str): The number of SVD principal components.
            If int, the exact number is given. It must not exceed the size
            of the `coil_axis` dimension.
            If float, the number is interpreted as relative to the size of
            the `coil_axis` dimension, and values must be in the
            [0.1, 1] interval.
            If str, the number is automatically estimated from the magnitude
            of the eigenvalues using a specific method.
            Available methods include:
             - 'all': use all components.
             - 'full': same as 'all'.
             - 'elbow': use `utils.marginal_sep_elbow()`.
             - 'quad': use `utils.marginal_sep_quad()`.
             - 'quad_weight': use `utils.marginal_sep_quad_weight()`.
             - 'quad_inv_weight': use `utils.marginal_sep_quad_inv_weight()`.
             - 'otsu': use `segmentation.threshold_otsu()`.
             - 'X%': set the threshold at 'X' percent of the largest eigenval.
        coil_axis (int): The coil dimension.
            The dimension of `arr` along which single coil elements are stored.
        verbose (int): Set level of verbosity.

    Returns:
        arr (np.ndarray): The estimated coil sensitivity.

    References:
        - Buehrer, M., Pruessmann, K.P., Boesiger, P., Kozerke, S.,
          2007. Array compression for MRI with large coil arrays. Magn. Reson.
          Med. 57, 1131–1139. doi:10.1002/mrm.21237
    """
    coil_axis = coil_axis % arr.ndim
    shape = arr.shape
    num_coils = shape[coil_axis]
    arr = np.swapaxes(arr, coil_axis, -1)
    base_shape = arr.shape[:-1]

    arr = arr.reshape((-1, num_coils))

    # left_eigvects, eigvals, right_eigvects = sp.linalg.svd(
    #         arr, full_matrices=False, compute_uv=True)

    # : memory-friendly version of:
    #   square_arr = np.dot(arr.conj().transpose(), arr)
    square_arr = np.zeros((num_coils, num_coils), dtype=complex)
    for i in range(num_coils):
        square_arr[i, :] = np.dot(arr[:, i].conj(), arr)

    eigvals, right_eigvects = sp.linalg.eig(square_arr)
    eig_sort = np.argsort(eigvals)[::-1]

    msg('k_svd={}'.format(k_svd), verbose, VERB_LVL['debug'])
    if isinstance(k_svd, int):
        assert (0 < k_svd <= num_coils)
    elif isinstance(k_svd, float):
        k_svd = int(num_coils * min(max(k_svd, 0.1), 1.0))
    elif isinstance(k_svd, str):
        eig_sorted = eigvals[eig_sort] / np.max(eigvals)
        k_svd = k_svd.lower()
        if k_svd == 'elbow':
            k_svd = mrt.utils.marginal_sep_elbow(
                np.abs(eig_sorted / eig_sorted[0]))
        elif k_svd == 'quad':
            k_svd = mrt.utils.marginal_sep_quad(
                np.abs(eig_sorted / eig_sorted[0]))
        elif k_svd == 'quad_weight':
            k_svd = mrt.utils.marginal_sep_quad_weight(
                np.abs(eig_sorted / eig_sorted[0]))
        elif k_svd == 'quad_inv_weight':
            k_svd = mrt.utils.marginal_sep_quad_inv_weight(
                np.abs(eig_sorted / eig_sorted[0]))
        elif k_svd == 'otsu':
            k_svd = mrt.segmentation.threshold_otsu(eigvals)
            k_svd = np.where(eig_sorted < k_svd)[0]
            k_svd = k_svd[0] if len(k_svd) else num_coils
        elif k_svd.endswith('%') and (100.0 > float(k_svd[:-1]) >= 0.0):
            k_svd = np.abs(eig_sorted[0]) * float(k_svd[:-1]) / 100.0
            k_svd = np.where(np.abs(eig_sorted) < k_svd)[0]
            k_svd = k_svd[0] if len(k_svd) else num_coils
        else:
            warnings.warn(
                '`{}`: unknown method for `k_svd` determination'.format(k_svd))
            k_svd = num_coils
        if not 0 < k_svd <= num_coils:
            k_svd = num_coils
    msg('k_svd={}'.format(k_svd), verbose, VERB_LVL['medium'])

    # arr = np.dot(
    #     left_eigvects[eig_sort, :],
    #     np.dot(
    #         np.diag(eigvals[eig_sort]),
    #         right_eigvects[:, eig_sort][:, :k_svd]))
    arr = np.dot(arr, right_eigvects[:, eig_sort][:, :k_svd])

    arr = arr.reshape(base_shape + (k_svd,))
    arr = np.swapaxes(arr, -1, coil_axis)
    return arr


# ======================================================================
def block_subspace_fourier(
        arr,
        coil_axis=-1,
        verbose=D_VERB_LVL):
    """
    Coil sensitivity for the 'block_subspace_fourier' combination method.

    Args:
        arr:
        coil_axis:
        verbose (int): Set level of verbosity.

    Returns:

    References:
        - Gol Gungor, D., Potter, L.C., 2016. A subspace-based coil
          combination method for phased-array magnetic resonance imaging.
          Magn. Reson. Med. 75, 762–774. doi:10.1002/mrm.25664
    """
    raise NotImplementedError


# ======================================================================
def virtual_ref(
        arr,
        coil_axis=-1,
        verbose=D_VERB_LVL):
    """
    Coil sensitivity for the 'virtual_ref' combination method.

    Args:
        arr:
        coil_axis:
        verbose (int): Set level of verbosity.

    Returns:

    References:
        - Parker, D.L., Payne, A., Todd, N., Hadley, J.R., 2014. Phase
          reconstruction from multiple coil data using a virtual reference
          coil. Magn. Reson. Med. 72, 563–569. doi:10.1002/mrm.24932
    """
    raise NotImplementedError


# ======================================================================
def multi_conjugate_hermitian(
        arr,
        echo_axis=-2,
        coil_axis=-1,
        verbose=D_VERB_LVL):
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
        verbose (int): Set level of verbosity.

    Returns:
        arr (np.ndarray): The estimated coil sensitivity.
    """
    raise NotImplementedError


# ======================================================================
def multi_svd(
        arr,
        echo_axis=-2,
        coil_axis=-1,
        verbose=D_VERB_LVL):
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
    shape = arr.shape
    num_coils = shape[coil_axis]
    arr = np.swapaxes(arr, coil_axis, -1)
    base_shape = arr.shape[:-1]

    raise NotImplementedError


# ======================================================================
def me_composer(
        arr,
        echo_axis=-2,
        coil_axis=-1,
        verbose=D_VERB_LVL):
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
        coil_axis=-1,
        verbose=D_VERB_LVL):
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
        verbose (int): Set level of verbosity.

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
        coil_axis=-1,
        verbose=D_VERB_LVL):
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
        verbose (int): Set level of verbosity.

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
        method_kws=None,
        coil_axis=-1,
        split_axis=-2,
        verbose=D_VERB_LVL):
    """
    Estimate the coil sensitivity.

    Args:
        arr:
        method (str|None): The coil sensitivity method.
            If str, uses the specified method.
            Available options are:
             - 'complex_sum': use `complex_sum()`;
             - 'sum_of_squares': use `sum_of_squares()`;
             - 'smooth_sum_of_squares': use `smooth_sum_of_squares()`;
             - 'adaptive': use `adaptive()`;
             - 'block_adaptive': use `block_adaptive()`;
             - 'block_adaptive_iter': use `block_adaptive_iter()`;
             - 'virtual_ref': use `virtual_ref()`.
        method_kws (dict|None): Keyword arguments to pass to `method`.
            If None, only `coil_axis` is passed to method.
        coil_axis (int): The coil dimension.
            The dimension of `arr` along which single coil elements are stored.
            This is passed to `method`.
        split_axis (int|None): The split dimension.
            If int, indicates the dimension of `arr` along which the
            algorithm is sequentially applied to reduce memory usage,
            but at the cost of accuracy.
            If None, the algorithm is applied to the whole `arr` at once.
        verbose (int): Set level of verbosity.

    Returns:
        sens (arr): The coil sensitivity.
    """
    methods = (
        'complex_sum', 'sum_of_squares', 'smooth_sum_of_squares',
        'adaptive', 'block_adaptive', 'block_adaptive_iter',
        'wavelet_adaptive_iter', 'virtual_ref', 'block_virtual_ref')
    method = method.lower() if method else 'block_adaptive_iter'
    if method_kws is None:
        method_kws = dict()
    if method in methods:
        sens_method = None
        if method == 'complex_sum':
            sens_method = complex_sum
        elif method == 'sum_of_squares':
            sens_method = sum_of_squares
        elif method == 'smooth_sum_of_squares':
            sens_method = smooth_sum_of_squares
        elif method == 'adaptive':
            sens_method = adaptive
        elif method == 'block_adaptive':
            sens_method = block_adaptive
        elif method == 'block_adaptive_iter':
            sens_method = block_adaptive_iter
        elif method == 'virtual_ref':
            sens_method = virtual_ref
        msg(method, verbose, VERB_LVL['medium'], end='', flush=True)
        if split_axis is not None:
            shape = arr.shape
            sens = np.zeros(shape, dtype=complex)
            arr = np.swapaxes(arr, split_axis, 0)
            sens = np.swapaxes(sens, split_axis, 0)
            msg(': split={}'.format(shape[split_axis]),
                verbose, VERB_LVL['medium'], end='\n', flush=True)
            for i in range(shape[split_axis]):
                msg('{}'.format(i + 1), verbose, VERB_LVL['high'],
                    end=' ' if i + 1 < shape[split_axis] else '\n', flush=True)
                sens[i, ...] = sens_method(
                    arr[i, ...], coil_axis=coil_axis, verbose=verbose,
                    **method_kws)
            sens = np.swapaxes(sens, 0, split_axis)
        else:
            msg('', verbose, VERB_LVL['medium'])
            sens = sens_method(
                arr, coil_axis=coil_axis, verbose=verbose, **method_kws)
    else:
        warnings.warn(
            'Unknown sensitivity estimation method `{}`'.format(method) +
            ' Using default.')
        sens = sensitivity(
            arr, coil_axis=coil_axis, verbose=verbose, **method_kws)
    return sens


# ======================================================================
def combine(
        arr,
        sens=None,
        k_svd='quad_weight',
        norm=False,
        coil_axis=-1,
        split_axis=None,
        verbose=D_VERB_LVL):
    """
    Calculate the combination of multiple coil elements using coil sensitivity.

    The coil sensitivity is specified as a parameter.
    An optional SVD preprocessing step can be used to reduce computational
    complexity and eventually reduce the noise.

    Args:
        arr (np.ndarray): The input array.
        sens (np.ndarray|str|None): The coil sensitivity.
            If `np.ndarray`, its shape must match with `arr`.
            If `str`, the sensitivity is calculated using `sensitivity()`.
            If None, the default method of `sensitivity()` is used.
        k_svd (int|float|str|None): The number of SVD principal components.
            If int, float or str, see `compress_svd()` for more information.
            If None, no SVD preprocessing is performed.
        norm (bool): Normalize using the coil sensitivity magnitude.
        coil_axis (int): The coil dimension.
            The dimension of `arr` along which single coil elements are stored.
        split_axis (int|None): The split dimension.
            If int, indicates the dimension of `arr` along which the
            algorithm is sequentially applied to reduce memory usage,
            but at the cost of accuracy.
            If None, the algorithm is applied to the whole `arr` at once.
        verbose (int): Set level of verbosity.

    Returns:
        arr (np.ndarray): The combined array.
    """
    if k_svd is not None:
        arr = compress_svd(arr, k_svd, coil_axis, verbose)

    if sens is None:
        sens = 'block_adaptive_iter'
    if isinstance(sens, str):
        sens = sensitivity(arr, sens, None, coil_axis, split_axis, verbose)
    assert (arr.shape == sens.shape)

    if split_axis is not None:
        shape = arr.shape
        cx_arr = np.zeros(
            tuple(d for i, d in enumerate(shape) if i != coil_axis % arr.ndim),
            dtype=complex)
        split_axis = split_axis % arr.ndim
        cx_arr = np.swapaxes(cx_arr, split_axis, 0)
        arr = np.swapaxes(arr, split_axis, 0)
        sens = np.swapaxes(sens, split_axis, 0)

        split_coil_axis = coil_axis % arr.ndim - 1
        for i in range(shape[split_axis]):
            cx_arr[i, ...] = np.sum(
                sens[i, ...].conj() * arr[i, ...], split_coil_axis)
        cx_arr = np.swapaxes(cx_arr, 0, split_axis)
        arr = np.swapaxes(arr, 0, split_axis)
        sens = np.swapaxes(sens, 0, split_axis)
    else:
        cx_arr = np.sum(sens.conj() * arr, axis=coil_axis)

    if norm:
        msg('Normalizing.', verbose, VERB_LVL['medium'])
        epsilon = np.finfo(np.float).eps
        cx_arr /= (np.sum(np.abs(sens) ** 2, axis=coil_axis) + epsilon)
    del sens

    if np.isclose(np.mean(np.abs(np.angle(cx_arr))), 0.0, equal_nan=True):
        msg('Adding sum-of-squares phase.', verbose, VERB_LVL['medium'])
        arr = cx_arr * np.exp(1j * np.angle(np.sum(arr, axis=coil_axis)))
    else:
        arr = cx_arr
    elapsed(combine.__name__)
    msg(report(only_last=True))
    return arr


# ======================================================================
def combine_ref(
        arr,
        ref,
        norm=False,
        coil_axis=-1,
        verbose=D_VERB_LVL):
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
        verbose (int): Set level of verbosity.

    Returns:
        arr (np.ndarray): The combined array.

    References:
        - Roemer, P.B., Edelstein, W.A., Hayes, C.E., Souza, S.P.,
          Mueller, O.M., 1990. The NMR phased array. Magn Reson Med 16,
          192–225. doi:10.1002/mrm.1910160203
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
def combine_multi(
        arr,
        method,
        multi_axis=-2,
        coil_axis=-1,
        split_axis=None,
        verbose=D_VERB_LVL):
    """
    Calculate the combination of multiple coil elements from different images.

    Args:
        arr (np.ndarray): The input array.
        sens (np.ndarray|str): The coil sensitivity.
            If `np.ndarray`, its shape must match with `arr`.
            If `str`, the sensitivity is calculated using `sensitivity()`.
        norm (bool): Normalize using the coil sensitivity magnitude.
        echo_axis: The
        coil_axis (int): The coil dimension.
            The dimension of `arr` along which single coil elements are stored.
        split_axis (int|None): The split dimension.
            If int, indicates the dimension of `arr` along which the
            algorithm is sequentially applied to reduce memory usage,
            but at the cost of accuracy.
            If None, the algorithm is applied to the whole `arr` at once.
        verbose (int): Set level of verbosity.

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
def qq(
        coils_arr,
        combined_arr,
        factor=100,
        coil_axis=-1,
        verbose=D_VERB_LVL):
    """
    Calculate the voxel-wise quality metric Q for coil combination.

    This is defined as:

    .. math::
        Q_i = k \\frac{|y_i|}{\\sum_j |x_{ij}|}

    where :math:`x` is the uncombined coil array, :math:`y` is the combined
    coil array, :math:`i` indicate the voxel index, :math:`j` indicate the coil
    index and :math:`k` is a proportionality constant (traditionally set to
    100 for percentage units).

    Args:
        coils_arr (np.ndarray): The uncombined coil data array.
        combined_arr (np.ndarray): The combined coil data array.
        factor (int|float): The multiplicative scale factor.
        coil_axis (int): The coil dimension.
            The dimension of `arr` along which single coil elements are stored.
        verbose (int): Set level of verbosity.

    Returns:
        qq_arr (np.ndarray): The voxel-wise quality metric array.
    """
    qq_arr = np.abs(combined_arr) / np.sum(np.abs(coils_arr), axis=coil_axis)
    return factor * qq_arr
