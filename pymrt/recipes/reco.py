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
import pymrt.correction

from pymrt import VERB_LVL, D_VERB_LVL, VERB_LVL_NAMES
from pymrt import elapsed, report
from pymrt import msg, dbg


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
        acceleration,
        autocalib,
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
    autocalib_mask = slice(None)
    acceleration_mask = slice(None)
    acq_arr = arr[acceleration_mask]
    calib_arr = arr[autocalib_mask]
    raise NotImplementedError

    def kernel_target(sizes, factors):
        slicing = tuple(
            fc.util.complement_slice(tuple(range(d)),
                                     slice(None, None, r)) if r > 1 else d // 2
            for i, (d, r) in enumerate(
                itertools.zip_longest(sizes, factors, fillvalue=0)))
        return slicing

    kernel_size = (3, 5)
    kernel_size = tuple(max(a + 1, k) if a > 1 else k for a, k in
                        zip(acceleration_factors, kernel_size))
    if verbose: print(f'Kernel: {kernel_size}')

    calib_arr = extract_autocalib(phantom_coils_k_arr, n_autocalib, False)

    calib_padded_arr = fc.num.nd_windowing(calib_arr, kernel_size)
    if verbose: print(calib_padded_arr.shape)

    # define target and calibration matrices
    target_slicing = (slice(None),) * calib_arr.ndim + kernel_target(
        kernel_size, acceleration_factors)
    calib_mat_slicing = (slice(None),) * calib_arr.ndim + acceleration_sampled(
        kernel_size, acceleration_factors)
    if verbose: print(f'Target slicing: {target_slicing}')
    if verbose: print(f'Calib Mat slicing: {calib_mat_slicing}')
    num_targets = 2
    num_calibs = 2 * kernel_size[1]
    target_arr = calib_padded_arr[target_slicing].reshape(-1,
                                                          n_coils *
                                                          num_targets)  # FIXME
    calib_mat_arr = calib_padded_arr[calib_mat_slicing].reshape(-1,
                                                                n_coils *
                                                                num_calibs)
    # FIXME
    if verbose: print(target_arr.shape)
    if verbose: print(calib_mat_arr.shape)

    # compute calibration weights
    weights_arr, _, _, _ = np.linalg.lstsq(calib_mat_arr, target_arr,
                                           rcond=None)
    if verbose: print('Weights: ', weights_arr.shape)

    # use weights to compute missing k-space values
    source_padded_arr = fc.num.nd_windowing(phantom_coils_undersampled_k_arr,
                                            kernel_size)
    if verbose: print(f'Source Padded: {source_padded_arr.shape}')
    source_mat_slicing = (
                         slice(None),) * calib_arr.ndim + acceleration_sampled(
        kernel_size, acceleration_factors)
    if verbose: print(f'Source Mat slicing: {calib_mat_slicing}')
    source_mat_arr = source_padded_arr[source_mat_slicing].reshape(-1,
                                                                   n_coils *
                                                                   num_calibs)
    if verbose: print(f'Source: {source_mat_arr.shape}')
    unknown_arr = np.dot(source_mat_arr, weights_arr)
    if verbose: print(f'Unknown: {unknown_arr.shape}')

    # fill in GRAPPA reconstructed missing points
    unsampled = acceleration_unsampled(phantom_coils_acq_k_arr.shape,
                                       acceleration_factors)
    if verbose: print('Unsampled: ', unsampled)
    phantom_coils_grappa_k_arr = phantom_coils_acq_k_arr.copy()
    if verbose: print(phantom_coils_grappa_k_arr[unsampled].shape)
    if verbose: print(unknown_arr.shape)
    phantom_coils_grappa_k_arr[unsampled] = unknown_arr.reshape(
        shape[0] - kernel_size[0] + 1, shape[1] - kernel_size[1] + 1, n_coils)[
                                            ::2, :, :]

    phantom_coils_grappa_arr = fc.num.idftn(phantom_coils_grappa_k_arr,
                                            ft_axes)
    phantom_grappa_arr, grappa_coil_sens_arr = \
        mrt.recipes.coils.sum_of_squares(phantom_coils_grappa_arr, -1)


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
def pics(
        arr):
    """
    Parallel Imaging and Compressed Sensing
    EXPERIMENTAL!

    Args:
        arr (np.ndarray): The input array.
            Data can be either in k-space or in image space.

    Returns:
        arr (np.ndarray): The estimated coil sensitivity.
    """
    raise NotImplementedError


# ======================================================================
def noise(
        raw_arr,
        reco_func,
        reco_args=None,
        reco_kwargs=None,
        noise_level=1,
        num=64,
        verbose=D_VERB_LVL):
    """
    Estimate the noise for reco using a pseudo-multi-replica approach.

    This a Monte Carlo method, effectively consisting of computing the
    standard deviation for multiple instances of the image reconstructed after
    the addition of white noise in the complex raw data.

    This is can be used to compute the signal-to-noise (SNR) and the
    geometric noise amplification factor (g-factor).

    The SNR can be computed by:
    snr = reco_arr / noise_arr

    with:
    reco_arr = reco_func(raw_arr, *reco_args, **reco_kwargs)

    Args:
        raw_arr (np.ndarray): The input raw data as acquired (k-space).
        reco_func (callable): The reconstruction function.
            Must accept the raw data array as first argument.
        reco_args (Iterable|None): Positional arguments for `reco_func`.
        reco_kwargs (tuple|dict|None): Keyword arguments for `reco_func`.
        noise_level (int|float|str|None): The noise level.
            This is used to determine the st.dev of the Gaussian noise
            being added at each iteration.
            If int or float, the value is the st.dev of the Gaussian noise.
            If str and the value is a percentage, the st.dev is computed as
            the peak-to-peak value multiplied by `noise_level` percentage
            (the peak-to-peak value is the maximum of the peak-to-peak value
            for real and imaginary data separately).
            If None, the noise level is estimated using
             `pymrt.correction.estimate_noise_sigma()`.
        num (int): The number of repetitions.
        verbose (int): Set level of verbosity.

    Returns:
        result (tuple): The tuple
            contains:
            - noise_arr (np.ndarray): The st.dev. of noised reconstructions.
            - reco_arr (np.ndarray): The reco from data without extra noise.

    """
    reco_args = tuple(reco_args) if reco_args else ()
    reco_kwargs = dict(reco_kwargs) if reco_kwargs else {}

    # noise-less reco
    reco_arr = reco_func(raw_arr, *reco_args, **reco_kwargs)

    # compute desired noise std
    if noise_level is None:
        noise_std_val = mrt.correction.estimate_noise_sigma(raw_arr)
    elif isinstance(noise_level, (int, float)):
        noise_std_val = noise_level
    else:
        noise_std_val = fc.util.to_percent(noise_level)
        if noise_std_val:
            cx_ptp = max(np.ptp(np.real(raw_arr)), np.ptp(np.imag(raw_arr)))
            noise_std_val = cx_ptp * noise_level
        else:
            noise_std_val = 1
    msg('Noise St.Dev.: {} (Level: {:.0%})'.format(noise_std_val, noise_level),
        verbose, VERB_LVL['debug'])

    mean_noised_arr = np.zeros_like(reco_arr, dtype=float)
    mvar_noised_arr = np.zeros_like(reco_arr, dtype=float)
    for i in range(num):
        msg('Replica #{}'.format(i), verbose, VERB_LVL['debug'])
        noise_raw_arr = np.random.normal(0, noise_std_val, raw_arr.shape)
        noised_arr = reco_func(
            raw_arr + noise_raw_arr, *reco_args, **reco_kwargs)
        mean_noised_arr, mvar_noised_arr = fc.util.next_mean_mvar(
            np.real(noised_arr), mean_noised_arr, mvar_noised_arr, i)
    noise_arr = np.sqrt(mvar_noised_arr / num)
    return noise_arr, reco_arr


# ======================================================================
def g_factor(
        test_snr_arr,
        ref_snr_arr,
        sampling_ratio):
    """
    Compute the geometric noise amplification factor (g-factor).

    If the signal level is assumed to be constant, the following substitutions
    can be used:

    - test_snr_arr -> ref_noise_arr
    - ref_snr_arr -> test_noise_arr

    or, alternatively:

    - test_snr_arr -> 1 / test_noise_arr
    - ref_snr_arr -> 1 / ref_noise_arr

    Args:
        test_snr_arr (np.ndarray): The test signal.
        ref_snr_arr (np.ndarray): The reference signal.
        sampling_ratio (int|float): The sampling ratio.
            This is the ratio between the number of samples used to compute
            the test signal and the number of samples used to compute
            reference signal.

    Returns:
        g_factor_arr (np.ndarray): The g-factor map.
    """
    return ref_snr_arr / test_snr_arr / np.sqrt(sampling_ratio)


# ======================================================================
def gen_pseudo_multi_replica(
        raw_arr,
        reco_func,
        reco_args=None,
        reco_kwargs=None,
        optim_func=None,
        optim_args=None,
        optim_kwargs=None,
        noise_level=1,
        num=64,
        verbose=D_VERB_LVL):
    """
    Estimate the SNR and g-factor using the generalized pseudo-multi-replica.

    This a Monte Carlo method, effectively consisting of computing the
    standard deviation for multiple instances of the image reconstructed after
    the addition of white noise in the complex raw data (`sd_noised_arr`), and
    the same reconstruction applied to noise-only data (`sd_noise_arr`).

    This is then used to compute the signal-to-noise (SNR) map and the
    geometric noise amplification factor (g-factor).

    R*: effective undersampling or acceleration factor

    reco = reconstructed image without noise addition
    R = reco.size / raw.size

    SNR = reco / sd_noised_reco
    g_factor = sd_noised_reco / sd_noised_optim / sqrt(R)

    Args:
        raw_arr (np.ndarray): The input raw data as acquired (k-space).
        reco_func (callable): The reconstruction function.
            Must accept the raw data array as first argument.
        reco_args (Iterable|None): Positional arguments for `reco_func`.
        reco_kwargs (tuple|dict|None): Keyword arguments for `reco_func`.
        optim_args (Iterable|None): Positional arguments for `reco_func`.
            This are used to generate the optimal reconstruction.
        optim_kwargs (tuple|dict|None): Keyword arguments for `reco_func`.
            This are used to generate the optimal reconstruction.
        noise_level (int|float|str|None): The noise level.
            This is used to determine the st.dev of the Gaussian noise
            being added at each iteration.
            If int or float, the value is the st.dev of the Gaussian noise.
            If str and the value is a percentage, the st.dev is computed as
            the peak-to-peak value multiplied by `noise_level` percentage
            (the peak-to-peak value is the maximum of the peak-to-peak value
            for real and imaginary data separately).
            If None, the noise level is estimated using
             `pymrt.correction.estimate_noise_sigma()`.
        num (int): The number of repetitions.
        verbose (int): Set level of verbosity.

    Returns:
        result (tuple): The tuple
            contains:
            - snr_arr (np.ndarray): The estimated SNR map.
            - g_factor_arr (np.ndarray): The estimated g-factor map.
    """
    reco_args = tuple(reco_args) if reco_args else ()
    reco_kwargs = dict(reco_kwargs) if reco_kwargs else {}
    if not optim_func:
        optim_func = reco_func
    optim_args = tuple(optim_args) if optim_args else ()
    optim_kwargs = dict(optim_kwargs) if optim_kwargs else {}

    # "noiseless" reco
    reco_arr = reco_func(raw_arr, *reco_args, **reco_kwargs)

    mean_noised_reco_arr = np.zeros_like(reco_arr, dtype=float)
    mvar_noised_reco_arr = np.zeros_like(reco_arr, dtype=float)
    mean_noised_optim_arr = np.zeros_like(reco_arr, dtype=float)
    mvar_noised_optim_arr = np.zeros_like(reco_arr, dtype=float)

    # compute desired noise std
    if noise_level is None:
        noise_std_val = mrt.correction.estimate_noise_sigma(raw_arr)
    elif isinstance(noise_level, (int, float)):
        noise_std_val = noise_level
    else:
        noise_std_val = fc.util.to_percent(noise_level)
        if noise_std_val:
            cx_ptp = max(np.ptp(np.real(raw_arr)), np.ptp(np.imag(raw_arr)))
            noise_std_val = cx_ptp * noise_level
        else:
            noise_std_val = 1
    msg('Noise St.Dev.: {} (Level: {:.0%})'.format(noise_std_val, noise_level),
        verbose, VERB_LVL['debug'])

    # compute the effective acceleration factor
    sampling_ratio = reco_arr.size / raw_arr.size

    for i in range(num):
        msg('Replica #{}'.format(i), verbose, VERB_LVL['debug'])
        noise_arr = np.random.normal(0, noise_std_val, raw_arr.shape)
        noised_reco_arr = reco_func(
            raw_arr + noise_arr, *reco_args, **reco_kwargs)
        # new uncorrelated noise
        noise_arr = np.random.normal(0, noise_std_val, reco_arr.shape)
        noised_optim_arr = optim_func(noise_arr, *optim_args, **optim_kwargs)

        mean_noised_reco_arr, mvar_noised_reco_arr = fc.util.next_mean_mvar(
            np.real(noised_reco_arr),
            mean_noised_reco_arr, mvar_noised_reco_arr, i)
        mean_noised_optim_arr, mvar_noised_optim_arr = fc.util.next_mean_mvar(
            np.real(noised_optim_arr),
            mean_noised_optim_arr, mvar_noised_optim_arr, i)

    noise_reco_arr = np.sqrt(mvar_noised_reco_arr / (num - 1))
    noise_optim_arr = np.sqrt(mvar_noised_optim_arr / (num - 1))

    snr_arr = np.abs(reco_arr) / noise_reco_arr
    g_factor_arr = g_factor(noise_optim_arr, noise_reco_arr, sampling_ratio)
    return snr_arr, g_factor_arr


# ======================================================================
def pseudo_multi_replica(
        raw_arr,
        mask,
        reco_func,
        reco_args=None,
        reco_kwargs=None,
        noise_level=1,
        num=64,
        verbose=D_VERB_LVL):
    """
    Estimate SNR and g-factor with the multi-replica method.

    This a Monte Carlo method, effectively consisting of computing the
    standard deviation for multiple instances of the difference between
    the images reconstructed with and without additional Gaussian noise
    in the complex raw time-domain data.

    This is then used to compute the signal-to-noise (SNR) map and the
    geometric noise amplification factor (g-factor).

    SNR = img / sd_noise
    g_factor = sd_noised_img

    Args:
        raw_arr (np.ndarray): The input raw data.
            This does not need to be fully sampled, but it must have the
            correct size for fully sampled data.
            The values that will be masked can be zero-ed.
        mask (np.ndarray[bool]|slice|Iterable[slice]): The undersampling mask.
        reco_func (callable): The reconstruction function.
            Must accept:
             - the raw data array as first argument;
             - the mask/undersampling scheme as second argument.
        reco_args (Iterable|None): Positional arguments for `reco_func`.
        reco_kwargs (tuple|dict|None): Keyword arguments for `reco_func`.
        noise_level (int|float|str|None): The noise level.
            This is used to determine the st.dev of the Gaussian noise
            being added at each iteration.
            If int or float, the value is the st.dev of the Gaussian noise.
            If str and the value is a percentage, the st.dev is computed as
            the peak-to-peak value multiplied by `noise_level` percentage
            (the peak-to-peak value is the maximum of the peak-to-peak value
            for real and imaginary data separately).
            If None, the noise level is estimated using
             `pymrt.correction.estimate_noise_sigma()`.
        num (int): The number of repetitions.
        verbose (int): Set level of verbosity.

    Returns:
        result (tuple): The tuple
            contains:
            - snr_arr (np.ndarray): The estimated SNR map.
            - g_factor_arr (np.ndarray): The estimated g-factor map.

    References:
        - Robson, Philip M., Aaron K. Grant, Ananth J. Madhuranthakam,
          Riccardo Lattanzi, Daniel K. Sodickson, and Charles A. McKenzie.
          “Comprehensive Quantification of Signal-to-Noise Ratio and g-Factor
          for Image-Based and k-Space-Based Parallel Imaging Reconstructions.”
          Magnetic Resonance in Medicine 60, no. 4 (2008): 895–907.
          https://doi.org/10.1002/mrm.21728.
    """
    reco_args = tuple(reco_args) if reco_args else ()
    reco_kwargs = dict(reco_kwargs) if reco_kwargs else {}

    # noise-less reco
    reco_arr = reco_func(raw_arr, *reco_args, **reco_kwargs)

    mean_noised_reco_arr = np.zeros_like(reco_arr, dtype=float)
    mvar_noised_reco_arr = np.zeros_like(reco_arr, dtype=float)
    mean_noised_optim_arr = np.zeros_like(reco_arr, dtype=float)
    mvar_noised_optim_arr = np.zeros_like(reco_arr, dtype=float)

    # compute desired noise std
    if noise_level is None:
        noise_std_val = mrt.correction.estimate_noise_sigma(raw_arr)
    elif isinstance(noise_level, (int, float)):
        noise_std_val = noise_level
    else:
        noise_std_val = fc.util.to_percent(noise_level)
        if noise_std_val:
            cx_ptp = max(np.ptp(np.real(raw_arr)), np.ptp(np.imag(raw_arr)))
            noise_std_val = cx_ptp * noise_level
        else:
            noise_std_val = 1
    msg('Noise St.Dev.: {} (Level: {:.0%})'.format(noise_std_val, noise_level),
        verbose, VERB_LVL['debug'])

    # compute the effective acceleration factor
    sampling_ratio = raw_arr[mask].size / raw_arr.size

    for i in range(num):
        msg('Replica #{}'.format(i), verbose, VERB_LVL['debug'])
        noise_arr = np.random.normal(0, noise_std_val, raw_arr.shape)
        noised_reco_arr = reco_func(
            raw_arr + noise_arr, mask, *reco_args, **reco_kwargs)
        # new uncorrelated noise
        noise_arr = np.random.normal(0, noise_std_val, reco_arr.shape)
        noised_optim_arr = reco_func(
            noise_arr, None, *reco_args, **reco_kwargs)

        mean_noised_reco_arr, mvar_noised_reco_arr = fc.util.next_mean_mvar(
            np.real(noised_reco_arr),
            mean_noised_reco_arr, mvar_noised_reco_arr, i)
        mean_noised_optim_arr, mvar_noised_optim_arr = fc.util.next_mean_mvar(
            np.real(noised_optim_arr),
            mean_noised_optim_arr, mvar_noised_optim_arr, i)

    noise_reco_arr = np.sqrt(mvar_noised_reco_arr / (num - 1))
    noise_optim_arr = np.sqrt(mvar_noised_optim_arr / (num - 1))

    snr_arr = np.abs(reco_arr) / noise_reco_arr
    g_factor_arr = g_factor(noise_optim_arr, noise_reco_arr, sampling_ratio)
    return snr_arr, g_factor_arr
