#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pymrt.recipes.coils: Coil sensitivity and combination for phased array.

Note: there may be some overlap with: `pymrt.recipes.b1r`.
"""

# ======================================================================
# :: Future Imports
from __future__ import (
    division, absolute_import, print_function, unicode_literals, )

# ======================================================================
# :: Python Standard Library Imports
import itertools  # Functions creating iterators for efficient looping
import warnings  # Warning control
# import collections  # Container datatypes
import datetime  # Basic date and time types
import multiprocessing  # Process-based parallelism

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
def combine_sens(
        arr,
        sens,
        norm=True,
        coil_axis=-1,
        verbose=D_VERB_LVL):
    """
    Calculate the coil combination using the coil sensitivities.

    Args:
        arr (np.ndarray): The input array.
        sens (np.ndarray): The coil sensitivity.
        norm (bool): Normalize using the coil sensitivity magnitude.
        coil_axis (int): The coil dimension.
            The dimension of `arr` along which single coil elements are stored.
        verbose (int): Set level of verbosity.

    Returns:
        combined (np.ndarray): The combined data.
    """
    assert (arr.shape == sens.shape)
    combined = np.sum(sens.conj() * arr, axis=coil_axis)
    if norm:
        epsilon = np.finfo(np.float).eps
        combined /= (np.sum(np.abs(sens) ** 2, axis=coil_axis) + epsilon)
    return combined


# ======================================================================
def compress_svde(
        arr,
        k_svd='quad_weight',
        coil_axis=-1,
        verbose=D_VERB_LVL):
    """
    Compress the coil data using the SVD extended principal components.

    EXPERIMENTAL!

    This computes the SVD on the matrix obtained by putting side by side
    the flattened single coil images.

    Args:
        arr (np.ndarray): The input array.
        k_svd (int|float|str): The number of principal components.
            See `pymrt.utils.optimal_num_components()` for more details.
        coil_axis (int): The coil dimension.
            The dimension of `arr` along which single coil elements are stored.
        verbose (int): Set level of verbosity.

    Returns:
        arr (np.ndarray): The compressed coil array.
    """
    raise NotImplementedError


# ======================================================================
def compress_svd(
        arr,
        k_svd='quad_weight',
        orthonormal=False,
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
        k_svd (int|float|str): The number of principal components.
            See `pymrt.utils.optimal_num_components()` for more details.
        orthonormal: Uses the orthonormal approximation.
            Uses the pseudo-inverse, instead of the hermitian conjugate,
            to form the signal compression matrix.
        coil_axis (int): The coil dimension.
            The dimension of `arr` along which single coil elements are stored.
        verbose (int): Set level of verbosity.

    Returns:
        arr (np.ndarray): The compressed coil array.

    References:
        - Buehrer, M., Pruessmann, K.P., Boesiger, P., Kozerke, S., 2007.
          Array compression for MRI with large coil arrays. Magn. Reson. Med.
          57, 1131–1139. doi:10.1002/mrm.21237
    """
    coil_axis = coil_axis % arr.ndim
    shape = arr.shape
    num_coils = shape[coil_axis]
    arr = np.swapaxes(arr, coil_axis, -1)
    base_shape = arr.shape[:-1]

    arr = arr.reshape((-1, num_coils))
    if orthonormal:
        inv_arr = np.linalg.pinv(arr)

    square_arr = np.zeros((num_coils, num_coils), dtype=complex)
    for i in range(num_coils):
        if orthonormal:
            square_arr[i, :] = np.dot(inv_arr[i, :], arr)
        else:
            square_arr[i, :] = np.dot(arr[:, i].conj(), arr)

    if orthonormal:
        del inv_arr

    eigvals, right_eigvects = sp.linalg.eig(square_arr)
    eig_sort = np.argsort(np.abs(eigvals))[::-1]

    k_svd = mrt.utils.auto_num_components(
        k_svd, np.abs(eigvals[eig_sort]) / np.max(np.abs(eigvals)),
        verbose=verbose)

    arr = np.dot(arr, right_eigvects[:, eig_sort][:, :k_svd])

    arr = arr.reshape(base_shape + (k_svd,))
    arr = np.swapaxes(arr, -1, coil_axis)
    return arr


# ======================================================================
def compress(
        arr,
        method='compress_svd',
        method_kws=None,
        coil_axis=-1,
        verbose=D_VERB_LVL):
    """
    Compress multiple coil elements into fewer coil elements.

    Args:
        arr (np.ndarray): The input array.
        method (str|None): The compression method.
            If str, uses the specified method as found in this module.
            Accepted values are:
             - 'compress_svd': use `pymrt.recipes.coils.compress_svd()`.
            If None, no coil compression is performed.
        method_kws (dict|tuple|None): Keyword arguments to pass to `method`.
            If None, only `coil_axis`, `verbose` are passed.
        coil_axis (int): The coil dimension.
            The dimension of `arr` along which single coil elements are stored.
        verbose (int): Set level of verbosity.

    Returns:
        arr (np.ndarray): The compressed coil array.
    """
    begin_time = datetime.datetime.now()

    methods = ('compress_svd',)

    msg('compression', verbose, VERB_LVL['debug'])

    method = method.lower()
    msg('method={}'.format(method), verbose, VERB_LVL['debug'])

    if method in methods:
        method = eval(method)
    method_kws = {} if method_kws is None else dict(method_kws)

    if callable(method):
        arr = method(
            arr, coil_axis=coil_axis, verbose=verbose, **dict(method_kws))
    else:
        text = 'Unknown compression method. None performed.'
        warnings.warn(text)

    end_time = datetime.datetime.now()
    msg('ExecTime({}): {}'.format('coils.compress', end_time - begin_time),
        verbose, D_VERB_LVL)

    return arr


# ======================================================================
def complex_sum(
        arr,
        coil_axis=-1,
        verbose=D_VERB_LVL):
    """
    Complex Sum coil combination method.

    This is equivalent to setting the sensitivity to 1.0.

    Args:
        arr (np.ndarray): The input array.
        coil_axis (int): The coil dimension.
            The dimension of `arr` along which single coil elements are stored.
        verbose (int): Set level of verbosity.

    Returns:
        result (tuple): The tuple
            contains:
             - combined (np.ndarray): The combined data.
             - sens (np.ndarray): The coil sensitivity.

    References:
        - Roemer, P.B., Edelstein, W.A., Hayes, C.E., Souza, S.P.,
          Mueller, O.M., 1990. The NMR phased array. Magn Reson Med 16,
          192–225. doi:10.1002/mrm.1910160203
    """
    return np.sum(arr, axis=coil_axis), 1.0


# ======================================================================
def sum_of_squares(
        arr,
        coil_axis=-1,
        verbose=D_VERB_LVL):
    """
    Sum-of-Squares coil combination method.

    Note: this function returns the same array used for input except for the
    normalization, therefore the `coil_axis` parameter is left unused.

    Args:
        arr (np.ndarray): The input array.
        coil_axis (int): The coil dimension.
            The dimension of `arr` along which single coil elements are stored.
        verbose (int): Set level of verbosity.

    Returns:
        result (tuple): The tuple
            contains:
             - combined (np.ndarray): The combined data.
             - sens (np.ndarray): The coil sensitivity.

    References:
        - Roemer, P.B., Edelstein, W.A., Hayes, C.E., Souza, S.P.,
          Mueller, O.M., 1990. The NMR phased array. Magn Reson Med 16,
          192–225. doi:10.1002/mrm.1910160203
    """
    return np.sum(np.abs(arr), axis=coil_axis), arr / np.abs(arr)


# ======================================================================
def adaptive(
        arr,
        filtering=None,
        filtering_kws=None,
        max_iter=16,
        threshold=1e-7,
        coil_axis=-1,
        verbose=D_VERB_LVL):
    """
    Block Adaptive coil combination method.

    Args:
        arr (np.ndarray): The input array.
        filtering (callable|None): The filtering function.
            If callable, it is used to separate the sensitivity from the input.
            Typically, a low-pass filter is used, under the assumption that
            the coil sensitivity is smooth compared to the sources.
            If None, no separation is performed.
        filtering_kws (dict|None): Keyword arguments to pass to `filtering`.
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
        result (tuple): The tuple
            contains:
             - combined (np.ndarray): The combined data.
             - sens (np.ndarray): The coil sensitivity.

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

    if filtering:
        for i in range(num_coils):
            for j in range(num_coils):
                coil_cov[..., i, j] = mrt.utils.filter_cx(
                    coil_cov[..., i, j], filtering, (), filtering_kws)

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

    combined = combine_sens(arr, sens, coil_axis=coil_axis)
    return combined, sens


# ======================================================================
def block_adaptive(
        arr,
        block=5,
        max_iter=16,
        threshold=1e-7,
        coil_axis=-1,
        verbose=D_VERB_LVL):
    """
    Block Adaptive coil combination method.

    Args:
        arr (np.ndarray): The input array.
        block (int|float|Iterable[int|float]): The size of the block in px.
            Smooth the coil covariance using a uniform filter with the
            specified block size.
            If int or float, the block is isotropic in all non-coil dimensions.
            If Iterable, each size is applied to the corresponding dimension
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
        result (tuple): The tuple
            contains:
             - combined (np.ndarray): The combined data.
             - sens (np.ndarray): The coil sensitivity.

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
    if isinstance(block, (int, float)):
        block = (block,) * (arr.ndim - 1) + (0,)
    else:
        assert (len(block) + 1 == arr.ndim)
    msg('block={}'.format(block), verbose, VERB_LVL['debug'])

    return adaptive_iter(
        arr,
        filtering=sp.ndimage.uniform_filter, filtering_kws=dict(size=block),
        max_iter=max_iter, threshold=threshold,
        coil_axis=coil_axis, verbose=verbose)


# ======================================================================
def adaptive_iter(
        arr,
        filtering=None,
        filtering_kws=None,
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
        filtering (callable|None): The filtering function.
            If callable, it is used to separate the sensitivity from the input.
            Typically, a low-pass filter is used, under the assumption that
            the coil sensitivity is smooth compared to the sources.
            If None, no separation is performed.
        filtering_kws (dict|None): Keyword arguments to pass to `filtering`.
        max_iter (int): Maximum number of iterations.
            If `threshold` > 0, the algorithm may stop earlier.
        threshold (float): Threshold for next iteration.
            If the next iteration globally modifies the sensitivity by less
            than `threshold`, the algorithm stops.
        coil_axis (int): The coil dimension.
            The dimension of `arr` along which single coil elements are stored.
        verbose (int): Set level of verbosity.

    Returns:
        result (tuple): The tuple
            contains:
             - combined (np.ndarray): The combined data.
             - sens (np.ndarray): The coil sensitivity.

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
    arr = np.swapaxes(arr, coil_axis, -1)

    msg('arr.shape={}'.format(arr.shape), verbose, VERB_LVL['debug'])
    msg('threshold={}'.format(threshold), verbose, VERB_LVL['debug'])
    msg('max_iter={}'.format(max_iter), verbose, VERB_LVL['debug'])

    epsilon = np.finfo(np.float).eps
    other_axes = tuple(range(0, arr.ndim - 1))

    with np.errstate(divide='ignore', invalid='ignore'):
        weights = np.sum(arr, other_axes)
        weights /= np.linalg.norm(weights)
        # combined == weighted
        combined = np.einsum('...i,i', arr, weights.conj())
        sens = np.zeros_like(arr, dtype=complex)
        delta = 1.0
        for i in range(max_iter):
            last_combined = combined.copy() if threshold > 0 else combined
            sens = arr * combined[..., None].conj()
            if filtering:
                sens = mrt.utils.filter_cx(sens, filtering, (), filtering_kws)
            sens /= (
                np.sqrt(np.sum(sens * sens.conj(), -1))
                + epsilon)[..., None]
            combined = np.sum(sens.conj() * arr, -1)
            # include the additional phase
            weights = np.sum(sens * combined[..., None], other_axes)
            weights /= np.linalg.norm(weights)
            weighted = np.einsum('...i,i', sens, weights.conj())
            weighted /= (np.abs(weighted) + epsilon)
            combined *= weighted
            sens *= weighted[..., None].conj()
            msg('{}'.format(i + 1),
                verbose, VERB_LVL['debug'], end=' ' if threshold else ', ',
                flush=True)
            if threshold > 0:
                last_delta = delta
                delta = (
                    np.linalg.norm(combined - last_combined) /
                    np.linalg.norm(combined))
                msg('delta={}'.format(delta), verbose, VERB_LVL['debug'],
                    end=', ' if i + 1 < max_iter else '.\n', flush=True)
                if delta < threshold or last_delta < delta:
                    break

    sens = np.swapaxes(sens, -1, coil_axis)
    return combined, sens


# ======================================================================
def block_adaptive_iter(
        arr,
        block=5,
        max_iter=16,
        threshold=1e-8,
        coil_axis=-1,
        verbose=D_VERB_LVL):
    """
    Block Adaptive Iterative coil combination method.

    This is an iterative and faster implementation of the algorithm for
    computing the Block Adaptive coil combination method.

    Args:
        arr (np.ndarray): The input array.
        block (int|float|Iterable[int|float]): The size of the block in px.
            Smooth the coil covariance using a uniform filter with the
            specified block size.
            If int or float, the block is isotropic in all non-coil dimensions.
            If Iterable, each size is applied to the corresponding dimension
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
        result (tuple): The tuple
            contains:
             - combined (np.ndarray): The combined data.
             - sens (np.ndarray): The coil sensitivity.

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
    if isinstance(block, (int, float)):
        block = (block,) * (arr.ndim - 1) + (0,)
    else:
        assert (len(block) + 1 == arr.ndim)
    msg('block={}'.format(block), verbose, VERB_LVL['debug'])

    return adaptive_iter(
        arr,
        filtering=sp.ndimage.uniform_filter, filtering_kws=dict(size=block),
        max_iter=max_iter, threshold=threshold,
        coil_axis=coil_axis, verbose=verbose)


# ======================================================================
def block_subspace_fourier(
        arr,
        coil_axis=-1,
        verbose=D_VERB_LVL):
    """
    Coil sensitivity for the 'block_subspace_fourier' combination method.

    EXPERIMENTAL!

    Args:
        arr (np.ndarray): The input array.
        coil_axis (int): The coil dimension.
            The dimension of `arr` along which single coil elements are stored.
        verbose (int): Set level of verbosity.

    Returns:
        combined (np.ndarray): The combined data.

    References:
        - Gol Gungor, D., Potter, L.C., 2016. A subspace-based coil
          combination method for phased-array magnetic resonance imaging.
          Magn. Reson. Med. 75, 762–774. doi:10.1002/mrm.25664
    """
    raise NotImplementedError


# ======================================================================
def virtual_ref(
        arr,
        method='svd',
        coil_axis=-1,
        verbose=D_VERB_LVL):
    """
    Coil sensitivity for the 'virtual_ref' combination method.

    EXPERIMENTAL!

    Args:
        arr (np.ndarray): The input array.
        method (str): The method for determining the virtual reference.
            Accepted values are:
             - 'svd': Use the first component of SVD compression;
             - 'svdo': Use the first component of SVD orthonormal compression.
        coil_axis (int): The coil dimension.
            The dimension of `arr` along which single coil elements are stored.
        verbose (int): Set level of verbosity.

    Returns:
        combined (np.ndarray): The combined data.

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

    EXPERIMENTAL!

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
        num_proc=None,
        multi_axis=-2,
        coil_axis=-1,
        verbose=D_VERB_LVL):
    """
    Coil sensitivity for the 'conjugate_hermitian' combination method.

    Note: the input itself is used as sensitivity. Therefore, this function
    actually returns the same array used for input, and the `coil_axis`
    parameter is left unused.

    Args:
        arr (np.ndarray): The input array.
        num_proc (int|None): The number of parallel processes.
            If 1, the execution is sequential.
            If 0 or None, the number of workers is determined automatically.
            Otherwise, uses the specified number of workers.
            If the number of workers is > 1, the execution is in parallel.
        multi_axis (int): The echo dimension.
            The dimension of `arr` along which different echoes are stored.
        coil_axis (int): The coil dimension.
            The dimension of `arr` along which single coil elements are stored.
        verbose (int): Set level of verbosity.

    Returns:
        arr (np.ndarray): The estimated coil sensitivity.
    """
    shape = arr.shape
    num_coils = shape[coil_axis]
    num_multi = shape[multi_axis]
    arr = np.swapaxes(arr, coil_axis, -1)
    arr = np.swapaxes(arr, multi_axis, -2)
    base_shape = arr.shape[:-2]

    arr = arr.reshape((-1, num_multi, num_coils))
    num_points = arr.shape[0]

    combined = np.zeros((num_points, num_multi), dtype=complex)
    sens = np.zeros((num_points, num_coils), dtype=complex)
    if not num_proc:
        num_proc = multiprocessing.cpu_count() + 1

    msg('num_proc={}'.format(num_proc), verbose, VERB_LVL['debug'])
    if num_proc == 1:
        for i in range(num_points):
            u, s, v = np.linalg.svd(arr[i, ...])
            combined[i, :] = u[:, 0] * s[0]
            sens[i, :] = v[0, :]
    else:
        chunksize = 2 * multiprocessing.cpu_count()
        pool = multiprocessing.Pool(num_proc)
        for i, res in enumerate(pool.map(np.linalg.svd, arr, chunksize)):
            u, s, v = res
            combined[i, :] = u[:, 0] * s[0]
            sens[i, :] = v[0, :]

    combined = combined.reshape(base_shape + (num_multi,))
    return combined, sens


# ======================================================================
def composer(
        arr,
        ref,
        multi_axis=-2,
        coil_axis=-1,
        verbose=D_VERB_LVL):
    """
    Coil sensitivity for the 'conjugate_hermitian' combination method.

    EXPERIMENTAL!

    Args:
        arr (np.ndarray): The input array.
        ref (np.ndarray|str|None):
            If np.ndarray, must be a complex reference data array.
            The shape must match that of `arr`, except for `coil_axis` and,
            optionally, `multi_axis` is that is set to None.
        multi_axis (int|None): The multiple images dimension.
            This can be the axis spanning, e.g.: repetitions, echoes, etc.
        coil_axis (int): The coil dimension.
            The dimension of `arr` along which single coil elements are stored.
        verbose (int): Set level of verbosity.

    Returns:
        arr (np.ndarray): The estimated coil sensitivity.
    """
    raise NotImplementedError


# ======================================================================
def aspire(
        arr,
        ref,
        multi_axis=-2,
        coil_axis=-1,
        verbose=D_VERB_LVL):
    """
    Coil sensitivity for the 'conjugate_hermitian' combination method.

    EXPERIMENTAL!

    Args:
        arr (np.ndarray): The input array.
        ref (np.ndarray|str|None):
            If np.ndarray, must be a complex reference data array.
            The shape must match that of `arr`, except for `coil_axis` and,
            optionally, `multi_axis` is that is set to None.
        multi_axis (int|None): The multiple images dimension.
            This can be the axis spanning, e.g.: repetitions, echoes, etc.
        coil_axis (int): The coil dimension.
            The dimension of `arr` along which single coil elements are stored.
        verbose (int): Set level of verbosity.

    Returns:
        arr (np.ndarray): The estimated coil sensitivity.
    """
    raise NotImplementedError


# ======================================================================
def snr_optimal_ref(
        arr,
        ref,
        coil_axis=-1,
        verbose=D_VERB_LVL):
    """
    Coil sensitivity for the SNR optimal with reference combination method.

    EXPERIMENTAL!

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
def adaptive_ref(
        arr,
        ref,
        coil_axis=-1,
        verbose=D_VERB_LVL):
    """
    Coil sensitivity for the 'conjugate_hermitian' combination method.

    EXPERIMENTAL!

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
        split_axis=None,
        verbose=D_VERB_LVL):
    """
    Estimate the coil sensitivity.

    Args:
        arr: The input array.
        method (str): The coil sensitivity method.
            If str, uses the specified method as found in this module.
            Accepted values are:
             - 'complex_sum';
             - 'sum_of_squares';
             - 'adaptive';
             - 'block_adaptive';
             - 'adaptive_iter';
             - 'block_adaptive_iter';
        method_kws (dict|tuple|None): Keyword arguments to pass to `method`.
            If None, only `coil_axis` and `split_axis` are passed to `method`.
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
        'complex_sum', 'sum_of_squares',
        'adaptive', 'block_adaptive', 'adaptive_iter', 'block_adaptive_iter',
        'multi_svd')

    msg('compression', verbose, VERB_LVL['debug'])

    method = method.lower()
    msg('method={}'.format(method), verbose, VERB_LVL['medium'])
    method_kws = {} if method_kws is None else dict(method_kws)

    if method in methods:
        method = eval(method)
    if not callable(method):
        text = (
            'Unknown method `{}` in `recipes.coils.sensitivity(). ' +
            'Using fallback `{}`.'.format(method, methods[0]))
        warnings.warn(text)
        method = eval(methods[0])

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
            _, sens[i, ...] = method(
                arr[i, ...], coil_axis=coil_axis, verbose=verbose,
                **dict(method_kws))
            del _
        sens = np.swapaxes(sens, 0, split_axis)
    else:
        msg('', verbose, VERB_LVL['medium'])
        sens = method(
            arr, coil_axis=coil_axis, verbose=verbose, **dict(method_kws))
    return sens


# ======================================================================
def combine(
        arr,
        method='block_adaptive_iter',
        method_kws=None,
        compression='compress_svd',
        compression_kws=None,
        coil_axis=-1,
        split_axis=None,
        verbose=D_VERB_LVL):
    """
    Calculate the combination of multiple coil elements.

    An optional coil compression preprocessing step can be used to reduce both
    the computational complexity and (eventually) the noise.

    Note coil combination can be seen as a particular case of coil compression
    where the coils are compressed to a single one.
    If this is the desired behavior, `complex_sum` should be used as `method`.
    However, coil compression methods are typically not suitable for coil
    combination.

    Args:
        arr (np.ndarray): The input array.
        method (str): The combination method.
            If str, uses the specified method as found in this module.
            Some methods require `ref` and/or `multi_axis` to be set in
            `method_kws`.
            Accepted values not requiring `ref` or `multi_axis` are:
             - 'complex_sum': use `pymrt.recipes.coils.complex_sum()`;
             - 'sum_of_squares': use `pymrt.recipes.coils.sum_of_squares()`;
             - 'adaptive': use `pymrt.recipes.coils.adaptive()`;
             - 'block_adaptive': use `pymrt.recipes.coils.block_adaptive()`;
             - 'adaptive_iter': use `pymrt.recipes.coils.adaptive_iter()`;
             - 'block_adaptive_iter': use
               `pymrt.recipes.coils.block_adaptive_iter()`;
            Accepted values requiring `ref` but not `multi_axis` are:
             Not implemented yet.
            Accepted values requiring `multi_axis` but not `ref` are:
             - 'multi_svd': use `pymrt.recipes.coils.mult_svd()`
            Accepted values requiring both `ref` and `multi_axis` are:
             Not implemented yet.

        method_kws (dict|tuple|None): Keyword arguments to pass to `method`.
            If None, only `coil_axis`, `split_axis`, `verbose` are passed.
        compression (callable|str|None): The compression method.
            This is passed as `method` to `compress`.
        compression_kws (dict|None): Keyword arguments to pass to
        `compression`.
            This is passed as `method_kwd` to `compress`.
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
    begin_time = datetime.datetime.now()

    sens_methods = (
        'complex_sum', 'sum_of_squares',
        'adaptive', 'block_adaptive', 'adaptive_iter', 'block_adaptive_iter',
        'multi_svd')
    methods = sens_methods + (
        'virtual_ref', 'multi_svd')

    if compression:
        arr = compress(
            arr, method=compression, method_kws=compression_kws,
            coil_axis=coil_axis, verbose=verbose)

    method = method.lower()
    msg('method={}'.format(method), verbose, VERB_LVL['medium'])
    method_kws = {} if method_kws is None else dict(method_kws)

    has_sens = method in sens_methods

    if method in methods:
        method = eval(method)
    if not callable(method):
        text = (
            'Unknown method `{}` in `recipes.coils.combine(). ' +
            'Using fallback `{}`.'.format(method, methods[0]))
        warnings.warn(text)
        method = eval(methods[0])
        has_sens = True

    if split_axis is not None:
        shape = arr.shape
        combined = np.zeros(
            tuple(d for i, d in enumerate(shape) if i != coil_axis % arr.ndim),
            dtype=complex)
        split_axis = split_axis % arr.ndim
        combined = np.swapaxes(combined, split_axis, 0)
        arr = np.swapaxes(arr, split_axis, 0)
        msg(': split={}'.format(shape[split_axis]),
            verbose, VERB_LVL['medium'], end='\n', flush=True)
        for i in range(shape[split_axis]):
            msg('{}'.format(i + 1), verbose, VERB_LVL['high'],
                end=' ' if i + 1 < shape[split_axis] else '\n', flush=True)
            if has_sens:
                combined[i, ...], _ = method(
                    arr[i, ...],
                    coil_axis=coil_axis, verbose=verbose, **dict(method_kws))
                del _
            else:
                combined[i, ...] = method(
                    arr[i, ...],
                    coil_axis=coil_axis, verbose=verbose, **dict(method_kws))
        combined = np.swapaxes(combined, 0, split_axis)
        arr = np.swapaxes(arr, 0, split_axis)
    else:
        if has_sens:
            combined, _ = method(
                arr, coil_axis=coil_axis, verbose=verbose, **dict(method_kws))
            del _
        else:
            combined = method(
                arr, coil_axis=coil_axis, verbose=verbose, **dict(method_kws))

    if np.isclose(np.mean(np.abs(np.angle(combined))), 0.0, equal_nan=True):
        combined = combined.astype(complex)
        msg('Adding summed phase.', verbose, VERB_LVL['medium'])
        combined *= np.exp(1j * np.angle(np.sum(arr, axis=coil_axis)))

    end_time = datetime.datetime.now()
    msg('ExecTime({}): {}'.format('coils.combine', end_time - begin_time),
        verbose, D_VERB_LVL)

    return combined


# ======================================================================
def quality(
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
    sum_arr = np.sum(np.abs(coils_arr), axis=coil_axis)
    abs_arr = np.abs(combined_arr)
    return factor * (
        abs_arr / sum_arr)  # * (np.max(sum_arr) / np.max(abs_arr))
