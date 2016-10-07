#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Participation to the QSM 2016 challenge."""

# ======================================================================
# :: Future Imports
from __future__ import(
    division, absolute_import, print_function, unicode_literals)

# ======================================================================
# :: Python Standard Library Imports
import os  # Miscellaneous operating system interfaces
import random  # Generate pseudo-random numbers

# :: External Imports
import numpy as np  # NumPy (multidimensional numerical arrays library)
import scipy as sp  # SciPy (signal and image processing library)
import neurolab as nnl  # NeuroLab (neural network with MATLAB-like interface)
import skimage as ski

# :: External Imports Submodules
import scipy.linalg
import scipy.ndimage
import scipy.signal

import matplotlib.pyplot as plt  # Matplotlib's MATLAB-like interface

# :: Local Imports
from pymrt.base import check_redo, scale, auto_repeat, calc_stats, subst
from pymrt.base import realpath, change_ext, EXT
from pymrt.base import dft, idft, coord, gaussian_nd, laplacian, inv_laplacian
from pymrt.base import unwrap_phase_laplacian as unwrap_phase
from pymrt.recipes.chi import dipole_kernel

from pymrt.geometry import rand_mask
from pymrt.input_output import load, save

from pymrt import msg, dbg
from pymrt import elapsed, print_elapsed
from pymrt import VERB_LVL, D_VERB_LVL


# ======================================================================
def my_qsm(cx_arr, b0z=3.0, theta=0.0, phi=0.0):
    # susceptibility in Fourier space (zero-centered)

    assert (len(cx_arr.shape) == 3)

    from skimage.restoration import unwrap_phase, denoise_tv_chambolle
    mag_arr = np.abs(cx_arr)
    phs_arr = unwrap_phase(np.angle(cx_arr))
    re_arr = np.real(cx_arr)
    im_arr = np.imag(cx_arr)
    dn_re_arr = denoise_tv_chambolle(re_arr, weight=0.2)
    dn_im_arr = denoise_tv_chambolle(im_arr, weight=0.2)
    dn_phs_arr = np.angle((re_arr - dn_re_arr) + 1j * (im_arr - dn_im_arr))

    # threshold = 0.0  # np.percentile(mag_arr, 0)
    # mask = mag_arr >= threshold
    # msg('Mask Size: {:%}'.format(np.sum(mask) / np.size(mask)))
    # phs_arr[~mask] = 0.0

    chi_arr = unwrap_phase(dn_phs_arr)

    # arr_phs_k = dft(arr_phs)
    # arr_dk_k = dipole_kernel(arr_cx)
    # # field shift along z in Tesla in Fourier space
    # chi_k = arr_phs_k / arr_dk_k
    # arr_chi = np.real(idft(arr_chi_k))
    return chi_arr


# ======================================================================
def rdif(
        arr1,
        arr2,
        arr_interval=None):
    """
    Calculate the mean of the abs. difference, scaled to the values interval.

    RDIF = |arr1 - arr2| / (max(arr1, arr2) - min(arr1, arr2))

    Args:
        arr1 (np.ndarray): The first input array.
        arr2 (np.ndarray): The second input array.
        scaling (float): The scaling factor.
            Useful to express the results in percent.

    Returns:
        rmse (float): The root mean squared error
    """
    assert (arr1.shape == arr2.shape)
    if not arr_interval:
        arr_interval = (
            min(np.min(arr1), np.min(arr2)), max(np.max(arr1), np.max(arr2)))
    rdif = np.mean(np.abs(arr1 - arr2)) / np.ptp(arr_interval)
    return rdif


# ======================================================================
def rmse(
        arr1,
        arr2):
    """
    Calculate the root mean squared error of the first vs the second array.

    RMSE = A * ||arr1 - arr2|| / ||arr2||

    Args:
        arr1 (np.ndarray): The first input array.
        arr2 (np.ndarray): The second input array.

    Returns:
        rmse (float): The root mean squared error.
    """
    assert (arr1.shape == arr2.shape)
    norm = scipy.linalg.norm
    rmse_val = norm(arr1 - arr2) / norm(arr2)
    return rmse_val


# ======================================================================
def hfen(
        arr1,
        arr2,
        filter_sizes=15,
        sigmas=1.5):
    """
    Compute the high-frequency error norm.

    The Laplacian of a Gaussian filter is used to get high frequency
    information.

    Args:
        arr1 (np.ndarray): The first input array.
        arr2 (np.ndarray): The second input array.
        filter_sizes (tuple[int]|int): The size of the filter in px.
            If a single value is given, is is assumed to be equal in all dims.
        sigmas (tuple[float]|float): The sigma of the gaussian kernel in px.
            If a single value is given, it is assumed to be equal in all dims.

    Returns:
        hfen (float): The high-frequency error norm.
    """
    assert (arr1.shape == arr2.shape)
    ndim = arr1.ndim

    filter_sizes = auto_repeat(filter_sizes, ndim)
    sigmas = auto_repeat(sigmas, ndim)
    assert (len(sigmas) == len(filter_sizes))

    grid = [slice(-filter_size // 2 + 1, filter_size // 2 + 1)
            for filter_size in filter_sizes]
    coord = np.ogrid[grid]
    gaussian_filter = gaussian_nd(filter_sizes, sigmas)
    hfen_factor = \
        sum([x ** 2 / sigma ** 4 for x, sigma in zip(coord, sigmas)]) + \
        - sum([1 / sigma ** 2 for sigma in sigmas])
    arr_filter = gaussian_filter * hfen_factor
    arr_filter = arr_filter - np.sum(arr_filter) / np.prod(arr_filter.shape)

    # the filter should be symmetric, therefore: correlate == convolve
    # additionally, fftconvolve much faster than direct convolve or correlate

    # arr1_corr = scipy.ndimage.filters.correlate(arr1, arr_filter)
    arr1_corr = scipy.signal.fftconvolve(arr1, arr_filter, 'same')
    # arr2_corr = scipy.ndimage.filters.correlate(arr2, arr_filter)
    arr2_corr = scipy.signal.fftconvolve(arr2, arr_filter, 'same')

    hfen_val = rmse(arr1_corr, arr2_corr)
    return hfen_val


# ======================================================================
def ssim(
        arr1,
        arr2,
        vals_interval=None,
        aa=(1, 1, 1),
        kk=(0.010, 0.030, 0.015)):
    """
    Calculate the structure similarity index, SSIM.

    This is defined as: SSIM = (lum ** alpha) * (con ** beta) * (sti ** gamma)
     - lum is a measure of the luminosity, with exp. weight alpha
     - con is a measure of the contrast, with exp. weight beta
     - sti is a measure of the structural information, with exp. weight gamma

    Args:
        arr1 (np.ndarray): The first input array.
        arr2 (np.ndarray): The second input array.
        arr_interval (tuple[float]): Minimum and maximum allowed values.
            The values of both arr1 and arr2 should be within this interval.
        aa (tuple[float]): The exponentiation weight factors. Must be 3.
            Modulate the relative weight of the three SSIM components
            (luminosity, contrast and structural information).
            If they are all equal to 1, the computation can be simplified.
        kk (tuple[float]): The ratio regularization constant factors. Must be 3.
            Determine the regularization constants as a factors of the total
            interval size (squared) for the three SSIM components
            (luminosity, contrast and structural information).
            Must be numbers much smaller than 1.

    Returns:
        ssim (float): The structure similarity index.

    See Also:
        Wang, Zhou, A. C. Bovik, H. R. Sheikh, and E. P. Simoncelli. “Image
        Quality Assessment: From Error Visibility to Structural Similarity.”
        IEEE Transactions on Image Processing 13, no. 4 (April 2004):
        600–612. doi:10.1109/TIP.2003.819861.
    """
    assert (arr1.shape == arr2.shape)
    arr_interval = (
        min(np.min(arr1), np.min(arr2)), max(np.max(arr1), np.max(arr2)))
    if vals_interval:
        msk1 = arr1 != 0.0
        msk2 = arr2 != 0.0
        arr1[msk1] = scale(arr1[msk1], arr_interval)
        arr2[msk2] = scale(arr2[msk2], arr_interval)
        arr_interval = vals_interval
    interval_size = np.ptp(arr_interval)
    cc = [(k * interval_size) ** 2 for k in kk]
    mu1 = np.mean(arr1)
    mu2 = np.mean(arr2)
    sigma1 = np.std(arr1)
    sigma2 = np.std(arr2)
    sigma12 = np.sum((arr1 - mu1) * (arr2 - mu2)) / (arr1.size - 1)
    ff = [
        (2 * mu1 * mu2 + cc[0]) / (mu1 ** 2 + mu2 ** 2 + cc[0]),
        (2 * sigma1 * sigma2 + cc[1]) / (sigma1 ** 2 + sigma2 ** 2 + cc[1]),
        (sigma12 + cc[2]) / (sigma1 * sigma2 + cc[2])
    ]
    ssim_val = np.prod(np.array([f ** a for (f, a) in zip(ff, aa)]), 0)
    return ssim_val


# ======================================================================
def check_performance(test, ref, name):
    rdif_val = rdif(test, ref)
    rmse_val = rmse(test, ref)
    hfen_val = hfen(test, ref)
    ssim_val = ssim(test, ref)
    msg('RDIF to {}: {:.2f}'.format(name, rdif_val))
    msg('RMSE to {}: {:.2%}'.format(name, rmse_val))
    msg('HFEN to {}: {:.2%}'.format(name, hfen_val))
    msg('SSIM to {}: {:.2%}'.format(name, ssim_val))


# ======================================================================
def meta_to_str(metas, pre='__', post='__'):
    def preprocess(val):
        val = str(val)
        if '_' in val:
            val = val.replace('_', '-')
        if ' ' in val:
            val = ''.join(val.split())  # remove all whitespaces
        return val

    text = '_'.join(
        ['{}={:s}'.format(key, preprocess(val))
         for key, val in sorted(metas.items())])
    if text:
        text = pre + text + post
    return text


def test_my_qsm(
        base=realpath('~/hd2/cache/qsm_2016_challenge/challenge'),
        target='chi_cosmos.nii.gz',
        force=True,
        verbose=D_VERB_LVL):
    """

    Args:
        base ():
        target():
        force ():
        verbose ():

    Returns:

    """
    msg('Dirpath: {}'.format(base))
    filepaths = {
        'mag': os.path.join(base, 'mag.nii.gz'),
        'phs': os.path.join(base, 'phs.nii.gz'),
        'tgt': os.path.join(base, target),
    }
    arr_mag = load(filepaths['mag']).astype(np.float64)
    arr_phs = load(filepaths['phs']).astype(np.float64)
    arr_tgt = load(filepaths['tgt']).astype(np.float64)

    metas = {}

    chi_filename = 'chi_myqsm' + meta_to_str(metas, post='') + '.' + EXT['niz']
    chi_filepath = os.path.join(base, chi_filename)
    msg('chi_myqsm: {}'.format(chi_filepath), verbose)
    if check_redo(list(filepaths.values()), [chi_filepath], force):
        msg('Calculating ...')
        arr_cx = arr_mag / np.max(arr_mag) * np.exp(-1j * arr_phs)
        arr_chi = my_qsm(arr_cx)

        save(chi_filepath, arr_chi)
    else:
        msg('Loading ...')
        arr_chi = load(chi_filepath)

    if verbose >= VERB_LVL['low']:
        check_performance(
            arr_chi, arr_tgt, change_ext(target, '', EXT['niz']))


# ======================================================================
def test_ann_1(
        base=os.path.expanduser('~/hd2/cache/qsm_2016_challenge/challenge'),
        target='chi_cosmos.nii.gz',
        density=0.20,
        force=False,
        verbose=D_VERB_LVL):
    """

    Args:
        base ():
        target ():
        density ():
        force ():
        verbose ():

    Returns:

    """
    msg('Dirpath: {}'.format(base))
    filepaths = {
        'mag': os.path.join(base, 'mag.nii.gz'),
        'phs': os.path.join(base, 'phs.nii.gz'),
        'msk': os.path.join(base, 'mask.nii.gz'),
        'tgt': os.path.join(base, target),
    }
    arr_mag = load(filepaths['mag']).astype(np.float64)
    arr_phs = load(filepaths['phs']).astype(np.float64)
    arr_msk = load(filepaths['msk']).astype(np.bool)
    arr_tgt = load(filepaths['tgt']).astype(np.float64)

    arr_cx = arr_mag / np.max(arr_mag) * np.exp(-1j * arr_phs)
    arr_cx_k = dft(arr_cx)
    arr_cxm = arr_cx * arr_msk
    arr_cxm_k = dft(arr_cxm)

    arr_dk_k = dipole_kernel(arr_cx.shape)

    # calculate ANN parameters
    np.random.seed(0)
    mask = rand_mask(arr_cxm_k, density=density)
    # mask *= arr_msk
    arr_input = np.stack(
        (np.real(arr_cxm_k[mask]), np.imag(arr_cxm_k[mask]), arr_dk_k[mask]),
        axis=-1)
    input_dims = arr_input.shape[-1]

    arr_target = arr_tgt[mask].reshape(-1, 1)
    hidden_layers = [24, 20, 16]

    metas = {
        'a': change_ext(target, '', EXT['niz']),
        'dens': density,
        'hidden-layers': hidden_layers,
        'regular': 1e-3,
        'z-stop': 1e-6,
    }
    for k, v in sorted(metas.items()):
        msg('{}: {:s}'.format(k, str(v)), verbose)
    ann_filename = 'chi_ann__' + meta_to_str(metas) + '.neurolab.ann'
    ann_filepath = os.path.join(base, ann_filename)
    msg('ANN: {}'.format(ann_filename), verbose)
    if check_redo(tuple(filepaths.values()), [ann_filepath], force):
        msg('Calculating ...')
        ann = nnl.net.newff([[-1, 1]] * input_dims, hidden_layers + [1])
        err = ann.train(
            arr_input, arr_target,
            epochs=4096, show=16,
            goal=metas['z-stop'], rr=metas['regular'])

        ann.save(ann_filepath)
    else:
        msg('Loading ...')
        ann = nnl.load(ann_filepath)

    chi_filename = 'chi_ann__' + meta_to_str(metas) + '.' + EXT['niz']
    chi_filepath = os.path.join(base, chi_filename)
    msg('chi_ann: {}'.format(ann_filepath), verbose)
    if check_redo(list(filepaths.values()) + [ann_filepath],
                  [chi_filepath], force):
        msg('Calculating ...')
        arr_input_all = np.stack(
            (np.real(arr_cx_k), np.imag(arr_cx_k), arr_dk_k), axis=-1)
        arr_chiann = np.real(idft(ann.sim(
            arr_input_all.reshape((-1, input_dims))).reshape(arr_cx.shape)))

        save(chi_filepath, arr_chiann)
    else:
        msg('Loading ...')
        arr_chiann = load(chi_filepath)

    if verbose >= VERB_LVL['low']:
        check_performance(
            arr_chiann, arr_tgt, change_ext(target, '', EXT['niz']))


# ======================================================================
def test_metrics():
    base_path = os.path.expanduser('~/hd2/cache/qsm_2016_challenge/backup')
    a1 = load(
        os.path.join(base_path, 'data', 'chi_cosmos.nii.gz')).astype(np.float64)
    a2 = load(
        os.path.join(base_path, 'data', 'chi_33.nii.gz')).astype(np.float64)

    # import profile
    # profile.run('compute_hfen(arr1, arr2)', sort=1)

    print(rdif(a1, a2))
    elapsed('std_diff')

    print(rmse(a1, a2))
    elapsed('rmse')

    print(hfen(a1, a2))
    elapsed('hfen')

    print(ssim(a1, a2))
    elapsed('ssim')
    print_elapsed()


# ======================================================================
if __name__ == '__main__':
    msg(__doc__.strip())
    # test_metrics()
    # test_ann_1()
    test_my_qsm()
