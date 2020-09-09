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
import flyingcircus as fc  # Everything you always wanted to have in Python*
import flyingcircus_numeric as fcn  # FlyingCircus with NumPy/SciPy

# :: External Imports Submodules
# import matplotlib.pyplot as plt  # Matplotlib's pyplot: MATLAB-like syntax
# import scipy.optimize  # SciPy: Optimization Algorithms
# import scipy.integrate  # SciPy: Integrations facilities
# import scipy.constants  # SciPy: Mathematal and Physical Constants
import scipy.ndimage  # SciPy: ND-image Manipulation

# :: Local Imports
import pymrt as mrt  # Python Magnetic Resonance Tools: the multi-tool of MRI
import pymrt.util
import pymrt.correction

from pymrt import INFO, PATH
from pymrt import VERB_LVL, D_VERB_LVL, VERB_LVL_NAMES
from pymrt import elapsed, report, run_doctests
from pymrt import msg, dbg, fmt, fmtm


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
    return fcn.ft


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
def acceleration_slices(
        shape,
        factors):
    """
    Compute the slicing for acceleration.

    Args:
        shape (Iterable[int]): The input shape.
        factors (Iterable[int|None]: The acceleration factors.
            If

    Returns:

    """
    return tuple(
        slice(None, None, factor) if factor is not None and factor > 1 else
        slice(None)
        for i, (d, factor) in enumerate(
            itertools.zip_longest(shape, factors, fillvalue=0)))


# ======================================================================
def autocalib_slices(
        shape,
        size,
        factors):
    return tuple(
        slice((d - size) // 2, (d + size) // 2)
        if factor is not None and factor > 1 else
        slice(None)
        for i, (d, factor) in enumerate(
            itertools.zip_longest(shape, factors, fillvalue=None)))


# ======================================================================
def grappa_1d(
        arr,
        acceleration=2,
        autocalib=16,
        kernel_span=1,
        acceleration_axis=0,
        coil_axis=-1):
    """
    Perform GRAPPA interpolation with 1D-accelerated cartesian k-data.

    Args:
        arr (np.ndarray): The input array.
            Data is in k-space and missing k-space lines are zero-filled.
        acceleration (int): The acceleration factor (along 1 dimension).
        autocalib (int): The number of central k-space lines acquired.
        kernel_span (int): The half-size of the kernel.
            The kernel window size in the non-accelerated dimension is given
            by: `kernel_size = kernel_span * 2 + 1`.
            Kernel span must be non-negative.
            The kernel window size in the accelerated dimension is equal to
            the `acceleration + 1`.
        acceleration_axis (int): The accelerated dimension.
        coil_axis (int): The coil dimension.
            The dimension of `arr` along which single coil elements are stored.

    Returns:
        arr (np.ndarray): The output array in k-space.

    See Also:
        - Griswold, Mark A., Peter M. Jakob, Robin M. Heidemann,
          Mathias Nittka, Vladimir Jellus, Jianmin Wang, Berthold Kiefer,
          and Axel Haase. “Generalized Autocalibrating Partially Parallel
          Acquisitions (GRAPPA).” Magnetic Resonance in Medicine 47, no. 6
          (June 1, 2002): 1202–10. https://doi.org/10.1002/mrm.10171.
    """
    # : ensure coil axis is the last
    assert(-arr.ndim <= coil_axis < arr.ndim)
    coil_axis = fc.valid_index(coil_axis, arr.ndim)
    last_axis = -1 % arr.ndim
    if coil_axis != last_axis:
        arr = np.swapaxes(arr, coil_axis, last_axis)

    # : prepare parameters
    acc_factors = tuple(
        acceleration
        if i == acceleration_axis else 1 if i != coil_axis else None
        for i in range(arr.ndim))
    acc_slicing = acceleration_slices(arr.shape, acc_factors)
    autocalib_slicing = autocalib_slices(arr.shape, autocalib, acc_factors)
    acc_arr = arr[acc_slicing]
    calib_arr = arr[autocalib_slicing]
    kernel_size = kernel_span * 2 + 1
    kernel_window = tuple(
        1 if factor is None else kernel_size if factor == 1 else factor + 1
        for factor in acc_factors)
    kernel_calib_size = 2 * kernel_size
    # number of target points within a kernel window
    n_targets = acceleration - 1

    # : define target and calibration matrices
    calib_padded_arr = fcn.nd_windowing(calib_arr, kernel_window)
    target_slicing = \
        tuple(slice(None) for _ in calib_arr.shape) \
        + tuple((slice(None) if factor is None else
                 slice(kernel_size // 2,
                       kernel_size // 2 + 1) if factor == 1 else
                 slice(1, factor))
                for factor in acc_factors)
    target_arr = calib_padded_arr[target_slicing] \
        .reshape(-1, calib_arr.shape[-1] * n_targets)
    calib_mat_slicing = \
        tuple(slice(None) for _ in calib_arr.shape) \
        + tuple(
            (slice(None) if factor is None or factor == 1 else (0, factor))
            for factor in acc_factors)
    calib_mat_arr = calib_padded_arr[calib_mat_slicing] \
        .reshape(-1, calib_arr.shape[-1] * kernel_calib_size)

    # : compute calibration weights
    weights_arr, _, _, _ = np.linalg.lstsq(
        calib_mat_arr, target_arr, rcond=None)

    # : use weights to compute missing k-space values
    # todo: avoid computing useless lines instead of selecting missing lines
    source_padded_arr = fcn.rolling_window_nd(
        arr, kernel_window, 1, out_mode='same')
    source_mat_arr = source_padded_arr[calib_mat_slicing] \
        .reshape(-1, calib_arr.shape[-1] * kernel_calib_size)
    unknown_arr = np.dot(source_mat_arr, weights_arr)

    # : fill-in GRAPPA-reconstructed missing points
    result = np.zeros_like(arr)
    unknown_shape = source_padded_arr.shape[:arr.ndim] + (n_targets,)
    unknown_arr = unknown_arr.reshape(unknown_shape)
    for i in range(n_targets):
        target_missing_slicing = tuple(
            slice(None) if factor is None else
            slice(kernel_span, -kernel_span) if factor == 1 else
            slice(i + 1, None, factor)
            for dim, factor in zip(arr.shape, acc_factors))
        source_missing_slicing = tuple(
            slice(None) if factor is None else
            slice(kernel_span, -kernel_span) if factor == 1 else
            slice(n_targets, dim - factor + n_targets + 1, factor)
            for dim, factor in zip(arr.shape, acc_factors))
        result[target_missing_slicing] = \
            unknown_arr[..., i][source_missing_slicing]
    result[autocalib_slicing] = calib_arr
    result[acc_slicing] = acc_arr

    if coil_axis != last_axis:
        result = np.swapaxes(result, last_axis, coil_axis)
    return result


# ======================================================================
def msense_1d(
        arr,
        acceleration=2,
        autocalib=16,
        acceleration_axis=0,
        coil_axis=-1):
    """
    Perform modified SENSE reconstruction with 1D-accelerated cartesian k-data.

    The coil sensitivity is estimated from the autocalibration lines.

    Args:
        arr (np.ndarray): The input array.
            Data is in k-space and missing k-space lines are zero-filled.
        acceleration (int): The acceleration factor (along 1 dimension).
        autocalib (int): The number of central k-space lines acquired.
        acceleration_axis (int): The accelerated dimension.
        coil_axis (int): The coil dimension.
            The dimension of `arr` along which single coil elements are stored.

    Returns:
        arr (np.ndarray): The output array.

    See Also:
        - Pruessmann, Klaas P., Markus Weiger, Markus B. Scheidegger, and
          Peter Boesiger. 1999. “SENSE: Sensitivity Encoding for Fast MRI.”
          Magnetic Resonance in Medicine 42 (5): 952–62.
          https://doi.org/10.1002/
          (SICI)1522-2594(199911)42:5<952::AID-MRM16>3.0.CO;2-S.
        - Griswold, Mark A., Felix Breuer, Martin Blaimer, Stephan
          Kannengiesser, Robin M. Heidemann, Matthias Mueller, Mathias Nittka,
          Vladimir Jellus, Berthold Kiefer, and Peter M. Jakob. 2006.
          “Autocalibrated Coil Sensitivity Estimation for Parallel Imaging.”
          NMR in Biomedicine 19 (3): 316–24. https://doi.org/10.1002/nbm.1048.
    """
    # : ensure coil axis is the last
    assert(-arr.ndim <= coil_axis < arr.ndim)
    coil_axis = fc.valid_index(coil_axis, arr.ndim)
    last_axis = -1 % arr.ndim
    if coil_axis != last_axis:
        arr = np.swapaxes(arr, coil_axis, last_axis)

    # : prepare parameters
    acc_factors = tuple(
        acceleration
        if i == acceleration_axis else 1 if i != coil_axis else None
        for i in range(arr.ndim))
    acc_slicing = acceleration_slices(arr.shape, acc_factors)
    autocalib_slicing = autocalib_slices(arr.shape, autocalib, acc_factors)
    acc_arr = arr[acc_slicing]
    calib_arr = arr[autocalib_slicing]



    if coil_axis != last_axis:
        result = np.swapaxes(result, last_axis, coil_axis)
    return result


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
        reco_kws=None,
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
    reco_arr = reco_func(raw_arr, *reco_args, **reco_kws)

    Args:
        raw_arr (np.ndarray): The input raw data as acquired (k-space).
        reco_func (callable): The reconstruction function.
            Must accept the raw data array as first argument.
        reco_args (Iterable|None): Positional arguments for `reco_func`.
        reco_kws (Mappable|None): Keyword arguments for `reco_func`.
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
    reco_kws = dict(reco_kws) if reco_kws else {}

    # noise-less reco
    reco_arr = reco_func(raw_arr, *reco_args, **reco_kws)

    # compute desired noise std
    if noise_level is None:
        noise_std_val = mrt.correction.estimate_noise_sigma(raw_arr)
    elif isinstance(noise_level, (int, float)):
        noise_std_val = noise_level
    else:
        noise_std_val = fc.to_percent(noise_level)
        if noise_std_val:
            cx_ptp = max(np.ptp(np.real(raw_arr)), np.ptp(np.imag(raw_arr)))
            noise_std_val = cx_ptp * noise_std_val
        else:
            noise_std_val = 1
    msg(fmtm('Noise St.Dev.: {noise_std_val} (Level: {noise_level})'),
        verbose, VERB_LVL['debug'])

    mean_noised_arr = np.zeros_like(reco_arr, dtype=float)
    sosd_noised_arr = np.zeros_like(reco_arr, dtype=float)
    for i in range(num):
        msg(fmtm('Replica #{i}'), verbose, VERB_LVL['debug'])
        noise_raw_arr = np.random.normal(0, noise_std_val, raw_arr.shape)
        noised_arr = reco_func(
            raw_arr + noise_raw_arr, *reco_args, **reco_kws)
        mean_noised_arr, sosd_noised_arr, _ = fc.next_mean_and_sosd(
            np.real(noised_arr), mean_noised_arr, sosd_noised_arr, i)
    noise_arr = fc.sosd2stdev(sosd_noised_arr, num)
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
        reco_kws=None,
        optim_func=None,
        optim_args=None,
        optim_kws=None,
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
        reco_kws (Mappable|None): Keyword arguments for `reco_func`.
        optim_func (callable): The reconstruction function.
            Must accept the raw data array as first argument.
        optim_args (Iterable|None): Positional arguments for `reco_func`.
            This are used to generate the optimal reconstruction.
        optim_kws (Mappable|None): Keyword arguments for `reco_func`.
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

    References:
        - Robson, Philip M., Aaron K. Grant, Ananth J. Madhuranthakam,
          Riccardo Lattanzi, Daniel K. Sodickson, and Charles A. McKenzie.
          “Comprehensive Quantification of Signal-to-Noise Ratio and g-Factor
          for Image-Based and k-Space-Based Parallel Imaging Reconstructions.”
          Magnetic Resonance in Medicine 60, no. 4 (2008): 895–907.
          https://doi.org/10.1002/mrm.21728.
    """
    #TODO: difference with `pseudo_multi_replica()`?
    reco_args = tuple(reco_args) if reco_args else ()
    reco_kws = dict(reco_kws) if reco_kws else {}
    if not optim_func:
        optim_func = reco_func
    optim_args = tuple(optim_args) if optim_args else ()
    optim_kws = dict(optim_kws) if optim_kws else {}

    # "noiseless" reco
    reco_arr = reco_func(raw_arr, *reco_args, **reco_kws)

    mean_noised_reco_arr = np.zeros_like(reco_arr, dtype=float)
    sosd_noised_reco_arr = np.zeros_like(reco_arr, dtype=float)
    mean_noised_optim_arr = np.zeros_like(reco_arr, dtype=float)
    sosd_noised_optim_arr = np.zeros_like(reco_arr, dtype=float)

    # compute desired noise std
    if noise_level is None:
        noise_std_val = mrt.correction.estimate_noise_sigma(raw_arr)
    elif isinstance(noise_level, (int, float)):
        noise_std_val = noise_level
    else:
        noise_std_val = fc.to_percent(noise_level)
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
            raw_arr + noise_arr, *reco_args, **reco_kws)
        # new uncorrelated noise
        noise_arr = np.random.normal(0, noise_std_val, reco_arr.shape)
        noised_optim_arr = optim_func(noise_arr, *optim_args, **optim_kws)

        mean_noised_reco_arr, sosd_noised_reco_arr, _ = \
            fc.next_mean_and_sosd(
                np.real(noised_reco_arr),
                mean_noised_reco_arr, sosd_noised_reco_arr, i)
        mean_noised_optim_arr, sosd_noised_optim_arr, _ = \
            fc.next_mean_and_sosd(
                np.real(noised_optim_arr),
                mean_noised_optim_arr, sosd_noised_optim_arr, i)

    noise_reco_arr = fc.sosd2stdev(sosd_noised_reco_arr, num)
    noise_optim_arr = fc.sosd2stdev(sosd_noised_optim_arr, num)

    snr_arr = np.abs(reco_arr) / noise_reco_arr
    g_factor_arr = g_factor(noise_optim_arr, noise_reco_arr, sampling_ratio)
    return snr_arr, g_factor_arr


# ======================================================================
def pseudo_multi_replica(
        raw_arr,
        mask,
        reco_func,
        reco_args=None,
        reco_kws=None,
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

    Uses a specific `mask` to describe the undersampling pattern.

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
        reco_kws (Mappable|None): Keyword arguments for `reco_func`.
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
    reco_kws = dict(reco_kws) if reco_kws else {}

    # noise-less reco
    reco_arr = reco_func(raw_arr, *reco_args, **reco_kws)

    mean_noised_reco_arr = np.zeros_like(reco_arr, dtype=float)
    sosd_noised_reco_arr = np.zeros_like(reco_arr, dtype=float)
    mean_noised_optim_arr = np.zeros_like(reco_arr, dtype=float)
    sosd_noised_optim_arr = np.zeros_like(reco_arr, dtype=float)

    # compute desired noise std
    if noise_level is None:
        noise_std_val = mrt.correction.estimate_noise_sigma(raw_arr)
    elif isinstance(noise_level, (int, float)):
        noise_std_val = noise_level
    else:
        noise_std_val = fc.to_percent(noise_level)
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
            raw_arr + noise_arr, mask, *reco_args, **reco_kws)
        # new uncorrelated noise
        noise_arr = np.random.normal(0, noise_std_val, reco_arr.shape)
        noised_optim_arr = reco_func(
            noise_arr, None, *reco_args, **reco_kws)

        mean_noised_reco_arr, sosd_noised_reco_arr, _ = \
            fc.next_mean_and_sosd(
                np.real(noised_reco_arr),
                mean_noised_reco_arr, sosd_noised_reco_arr, i)
        mean_noised_optim_arr, sosd_noised_optim_arr, _ = \
            fc.next_mean_and_sosd(
                np.real(noised_optim_arr),
                mean_noised_optim_arr, sosd_noised_optim_arr, i)

    noise_reco_arr = fc.sosd2stdev(sosd_noised_reco_arr, num)
    noise_optim_arr = fc.sosd2stdev(sosd_noised_optim_arr, num)

    snr_arr = np.abs(reco_arr) / noise_reco_arr
    g_factor_arr = g_factor(noise_optim_arr, noise_reco_arr, sampling_ratio)
    return snr_arr, g_factor_arr


# ======================================================================
elapsed(__file__[len(PATH['base']) + 1:])

# ======================================================================
if __name__ == '__main__':
    run_doctests(__doc__)
