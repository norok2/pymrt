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

# :: Local Imports
import pymrt as mrt
import pymrt.utils
import pymrt.segmentation


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
def smooth_sum_of_squares(
        arr,
        block=1,
        coil_axis=-1):
    """
    Coil sensitivity for the 'smooth_sum_of_squares' combination method.

    Note: the input itself is used as sensitivity. Therefore, this function
    actually returns the same array used for input, and the `coil_axis`
    parameter is left unused.

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

    Returns:
        arr (np.ndarray): The estimated coil sensitivity.
    """
    return mrt.utils.filter_cx(
        arr, sp.ndimage.gaussian_filter, (), dict(sigma=block))


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
    return block_adaptive(arr, 0, max_iter, threshold, coil_axis)


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

    sens = np.zeros_like(arr, dtype=complex)

    with np.errstate(divide='ignore', invalid='ignore'):
        mean_coil = np.sum(arr, no_coil_axes)
        mean_coil /= np.linalg.norm(mean_coil)
        rho = np.einsum('...i,i', arr, mean_coil.conj())

        for i in range(max_iter):
            last_rho = rho.copy() if threshold > 0 else rho
            sens_b = mrt.utils.filter_cx(
                arr * rho[..., None].conj(),
                sp.ndimage.uniform_filter, (), dict(size=block))
            sens = sens_b / (
                np.sqrt(np.sum(sens_b * sens_b.conj(), -1))
                + epsilon)[..., None]
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
def compress_svd(
        arr,
        k_svd='quad_weight',
        coil_axis=-1):
    """
    Compress the coil data to the SVD principal components.

    This is useful both as a denoise method and for reducing the complexity
    of the problem and the memory usage.

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
            Avaliable methods include:
             - 'elbow': use `utils.marginal_sep_elbow()`.
             - 'quad': use `utils.marginal_sep_quad()`.
             - 'quad_weight': use `utils.marginal_sep_quad_weight()`.
             - 'quad_inv_weight': use `utils.marginal_sep_quad_inv_weight()`.
             - 'otsu': use `segmentation.threshold_otsu()`.
             - 'X%': set the threshold at 'X' percent of the largest eigenval.
        coil_axis (int): The coil dimension.
            The dimension of `arr` along which single coil elements are stored.

    Returns:
        arr (np.ndarray): The estimated coil sensitivity.
    """
    coil_axis = coil_axis % arr.ndim
    shape = arr.shape
    num_coils = shape[coil_axis]
    arr = np.swapaxes(arr, coil_axis, -1)
    base_shape = arr.shape[:-1]

    arr = arr.reshape((-1, num_coils))
    eigvals, right_eigvects = sp.linalg.eig(
        np.dot(arr.conj().transpose(), arr))
    eig_sort = np.argsort(eigvals)[::-1]

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

    arr = np.dot(arr[:, :], right_eigvects[:, eig_sort][:, :k_svd])

    arr = arr.reshape(base_shape + (k_svd,))
    arr = np.swapaxes(arr, -1, coil_axis)
    return arr


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
        coil_axis=-1,
        split_axis=0):
    """

    Args:
        arr:
        method (str|None): The coil sensitivity method.
            It None, uses
            Available options are:
             - 'sum_of_squares': use `sum_of_squares()`;
             - 'smooth_sum_of_squares': use `smooth_sum_of_squares()`
             - 'adaptive': use `adaptive()`;
             - 'block_adaptive': use `block_adaptive()`;
             - 'block_adaptive_iter': use `block_adaptive_iter()`;
             - 'virtual_reference': use `virtual_reference()`.
        coil_axis (int): The coil dimension.
            The dimension of `arr` along which single coil elements are stored.
        split_axis (int|None): The split dimension.
            If int, indicates the dimension of `arr` along which the
            algorithm is sequentially applied, to reduce memory usage.
            If None, the algorithm is applied to the whole `arr` at once.

    Returns:

    """
    methods = (
        'sum_of_squares', 'smooth_sum_of_squares',
        'adaptive', 'block_adaptive', 'block_adaptive_iter',
        'virtual_reference')
    if method:
        method = method.lower()
    else:
        method = 'sum_of_squares'
    if method in methods:
        if method == 'sum_of_squares':
            method = sum_of_squares
        elif method == 'smooth_sum_of_squares':
            method = smooth_sum_of_squares
        elif method == 'adaptive':
            method = adaptive
        elif method == 'block_adaptive':
            method = block_adaptive
        elif method == 'block_adaptive_iter':
            method = block_adaptive_iter
        elif method == 'virtual_reference':
            method = virtual_reference
        if split_axis is not None:
            shape = arr.shape
            sens = np.zeros(shape, dtype=complex)
            arr = np.swapaxes(arr, split_axis, 0)
            sens = np.swapaxes(sens, split_axis, 0)
            for i in range(shape[split_axis]):
                sens[i, ...] = method(arr[i, ...], coil_axis=coil_axis)
            arr = np.swapaxes(arr, 0, split_axis)
            sens = np.swapaxes(sens, 0, split_axis)
        else:
            sens = method(arr, coil_axis=coil_axis)
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
        k_svd='quad_weight',
        norm=False,
        coil_axis=-1,
        split_axis=0):
    """
    Calculate the coil combination array from multiple coil elements.

    The coil sensitivity is specified as a parameter.
    An optional SVD preprocessing step can be used to reduce computational
    complexity and improve the SNR.

    Args:
        arr (np.ndarray): The input array.
        sens (np.ndarray|str|None): The coil sensitivity.
            If `np.ndarray`, its shape must match with `arr`.
            If `str`, the sensitivity is calculated using `sensitivity()`.
            If None, the default method of `sensitivity()` is used.
        k_svd (int|float|str|None): The number of SVD principal components.
            If int, float or str, see `compress_svd()` for more information.
            If None, no SVD preprocessing is performed.
        norm (bool): Normalize using coil sensitivity magnitude.
        coil_axis (int): The coil dimension.
            The dimension of `arr` along which single coil elements are stored.
        split_axis (int|None): The split dimension.
            If int, indicates the dimension of `arr` along which the
            algorithm is sequentially applied, to reduce memory usage.
            If None, the algorithm is applied to the whole `arr` at once.

    Returns:
        arr (np.ndarray): The combined array.
    """
    if k_svd is not None:
        arr = compress_svd(arr, k_svd, coil_axis)

    if sens is None:
        sens = 'sum_of_squares'
    if isinstance(sens, str):
        sens = sensitivity(arr, sens, coil_axis, split_axis)
    assert (arr.shape == sens.shape)

    if split_axis is not None:
        shape = arr.shape
        cx_arr = np.zeros(
            tuple(d for i, d in enumerate(shape) if i != coil_axis % arr.ndim),
            dtype=complex)
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
        cx_arr = np.sum(sens.conj() * arr, coil_axis)

    if norm:
        epsilon = np.finfo(np.float).eps
        cx_arr /= (np.sum(np.abs(sens) ** 2, coil_axis) + epsilon)
    del sens

    if not np.iscomplex(cx_arr.all()):
        arr = cx_arr * np.exp(1j * np.angle(np.sum(arr, coil_axis)))
    else:
        arr = cx_arr
    return arr


# ======================================================================
def combine_ref(
        arr,
        ref,
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
        echo_axis=-2,
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
