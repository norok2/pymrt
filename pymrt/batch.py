#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pymrt.computation: generic computation utilities for MRI data analysis.

See Also:
    pymrt.recipes
"""

# ======================================================================
# :: Future Imports
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

# todo: use kwargs instead of opts
# todo: get rid of tty colorify

# ======================================================================
# :: Python Standard Library Imports
import os  # Miscellaneous operating system interfaces
import shutil  # High-level file operations
# import math  # Mathematical functions
# import time  # Time access and conversions
# import datetime  # Basic date and time types
# import operator  # Standard operators as functions
# import collections  # Container datatypes
import itertools  # Functions creating iterators for efficient looping
# import functools  # Higher-order functions and operations on callable objects
import re  # Regular expression operations
# import subprocess  # Subprocess management
import multiprocessing  # Process-based parallelism
# import inspect  # Inspect live objects
# import csv  # CSV File Reading and Writing [CSV: Comma-Separated Values]
import json  # JSON encoder and decoder [JSON: JavaScript Object Notation]
import hashlib  # Secure hashes and message digests

# :: External Imports
import numpy as np  # NumPy (multidimensional numerical arrays library)
# import matplotlib as mpl  # Matplotlib (2D/3D plotting library)
# import sympy as sym  # SymPy (symbolic CAS library)
# import PIL  # Python Image Library (image manipulation toolkit)
# import SimpleITK as sitk  # Image ToolKit Wrapper
# import nibabel as nib  # NiBabel (NeuroImaging I/O Library)
# import nipy  # NiPy (NeuroImaging in Python)
# import nipype  # NiPype (NiPy Pipelines and Interfaces)

# :: External Imports Submodules
# import matplotlib.pyplot as plt  # Matplotlib's pyplot: MATLAB-like syntax
# import scipy.optimize  # SciPy: Optimization Algorithms
# import scipy.integrate  # SciPy: Integrations facilities
# import scipy.constants  # SciPy: Mathematal and Physical Constants
# import scipy.stats  # SciPy: Statistical functions

# :: Local Imports
import pymrt as mrt
import pymrt.utils
import pymrt.naming
import pymrt.input_output

# from dcmpi.lib.common import ID

# from pymrt import INFO
from pymrt import VERB_LVL, D_VERB_LVL
from pymrt import msg, dbg

# ======================================================================
META_EXT = 'info'  # ID['info']

D_OPTS = {
    # sources
    'data_ext': mrt.utils.EXT['niz'],
    'meta_ext': META_EXT,
    'multi_acq': False,
    'use_meta': True,
    'param_select': [None],
    'match': None,
    'pattern': [None],
    'groups': None,
    # compute
    'types': [None],
    'mask': [None],
    'adapt_mask': True,
}


# ======================================================================
def _simple_affines(affines):
    return tuple(affines[0] for affine in affines)


# ======================================================================
def preset_t1_mp2rage_builtin():
    """
    Preset to get built-in T1 maps from the MP2RAGE sequence.
    """
    new_opts = {
        'types': ['T1', 'INV2M'],
        'param_select': ['ProtocolName', '_series'],
        'match': '(?i).*mp2rage.*',
        'dtype': 'float',
        'mask': [[None], [None], [None], [1]],
    }
    new_opts.update({
        'compute_func': 'match_series',
        'compute_kwargs': {
            'matches': (
                ('.*_T1_Images.*', new_opts['types'][0]),
                ('.*_INV2(?!_PHS).*', new_opts['types'][1]),
            ),
        }
    })
    return new_opts


# ======================================================================
def preset_t2s_memp2rage_loglin2():
    """
    Preset to get built-in T2* maps from the ME-MP2RAGE sequence.
    """
    new_opts = {
        'types': ['T2S', 'T1w'],
        'param_select': ['ProtocolName', 'EchoTime::ms', '_series'],
        'match': '(?i).*me-mp2rage.*_INV2(?!_PHS).*',
        'dtype': 'float',
        'multi_acq': False,
        'compute_func': 'fit_monoexp_decay_loglin2',
        'compute_kwargs': {
            'ti_label': 'EchoTime::ms',
            'img_types': {'tau': 'T2S', 's_0': 'T1w'}}
    }
    return new_opts


# ======================================================================
def preset_t2s_flash_loglin2():
    """
    Preset to get T2* maps from multi-echo data using a log-linear fit.
    """
    new_opts = {
        'types': ['T2S', 'T1w'],
        'param_select': ['ProtocolName', 'EchoTime::ms', '_series'],
        'match': '(?i).*(gre|flash).*',
        'dtype': 'float',
        'multi_acq': False,
        'compute_func': 'fit_monoexp_decay_loglin',
        'compute_kwargs': {
            'ti_label': 'EchoTime::ms',
            'img_types': {'tau': 'T2S', 's_0': 'T1w'}}
    }
    return new_opts


# ======================================================================
def preset_t2s_flash_builtin():
    """
    Preset to get built-in T2* maps from the FLASH sequence.
    """
    new_opts = {
        'types': ['T2S', 'T1w'],
        'param_select': ['ProtocolName', '_series'],
        'match': '.*T2Star_Images.*',
        'dtype': 'float',
    }
    return new_opts


# ======================================================================
def preset_t2s_multiecho_loglin():
    """
    Preset to get T2* maps from multi-echo squared data using a log-linear fit.
    """
    new_opts = {
        'types': ['T2S', 'T1w'],
        'param_select': ['ProtocolName', 'EchoTime::ms', '_series'],
        'match': '(?i).*(gre|flash|me).*',
        'dtype': 'float',
        'multi_acq': False,
        'compute_func': 'fit_monoexp_decay_loglin2',
        'compute_kwargs': {
            'ti_label': 'EchoTime::ms',
            'img_types': {'tau': 'T2S', 's_0': 'T1w'}}
    }
    return new_opts


# ======================================================================
def preset_t2s_multiecho_loglin2():
    """
    Preset to get T2* maps from multi-echo squared data using a log-linear fit.
    """
    new_opts = {
        'types': ['T2S', 'T1w'],
        'param_select': ['ProtocolName', 'EchoTime::ms', '_series'],
        'match': '(?i).*(gre|flash|me).*',
        'dtype': 'float',
        'multi_acq': False,
        'compute_func': 'fit_monoexp_decay_loglin2',
        'compute_kwargs': {
            'ti_label': 'EchoTime::ms',
            'img_types': {'tau': 'T2S', 's_0': 'T1w'}}
    }
    return new_opts


# ======================================================================
def preset_t2s_multiecho_leasq():
    """
    Preset to get T2* maps from multi-echo data using a least-squares fit.
    """
    new_opts = {
        'types': ['T2S', 'T1w'],
        'param_select': ['ProtocolName', 'EchoTime::ms', '_series'],
        'match': '.*(FLASH|ME-MP2RAGE).*',
        'dtype': 'float',
        'multi_acq': False,
        'compute_func': 'fit_monoexp_decay_leasq',
        'compute_kwargs': {
            'ti_label': 'EchoTime::ms',
            'img_types': {'tau': 'T2S', 's_0': 'T1w'}}
    }
    return new_opts


# ======================================================================
def preset_b1t_afi():
    """
    Preset to get B1+ maps from the AFI sequence.
    """
    new_opts = {
        'types': ['B1T'],
        'param_select': [
            'ProtocolName', 'RepetitionTime::ms', 'FlipAngle::deg',
            '_series'],
        'match': '.*(afi|b1).*',
        'dtype': 'float',
        'multi_acq': False,
        'compute_func': 'calc_afi',
        'compute_kwargs': {
            'ti_label': 'RepetitionTime::ms',
            'fa_label': 'FlipAngle::deg',
            'img_types': {'eff': 'B1T'}}
    }
    return new_opts


# ======================================================================
def preset_qsm_as_legacy():
    """
    Preset to get CHI maps from a multi-echo sequence.
    """
    new_opts = {
        'types': ['CHI', 'MSK'],
        'param_select': [
            'ProtocolName', 'EchoTime::ms', 'ImagingFrequency', '_series'],
        # 'match': '.*((FLASH)|(ME-MP2RAGE.*INV2)).*',
        'match': '.*(ME-MP2RAGE.*INV2).*',
        'dtype': 'float',
        'multi_acq': False,
        'compute_func': 'ext_qsm_as_legacy',
        'compute_kwargs': {
            'te_label': 'EchoTime::ms',
            'img_types': {'qsm': 'CHI', 'mask': 'MSK'}}
    }
    return new_opts


# ======================================================================
def ext_qsm_as_legacy(
        images,
        affines,
        params,
        te_label,
        # b0_label,
        # th_label,
        img_types):
    """

    Args:
        images ():
        affines ():
        params ():
        te_label ():
        img_types ():

    Returns:

    """
    # determine correct TE
    max_te = 25.0  # ms
    selected = len(params[te_label])
    for i, te in enumerate(params[te_label]):
        if te < max_te:
            selected = i
    tmp_dirpath = '/tmp/{}'.format(hashlib.md5(str(params)).hexdigest())
    if not os.path.isdir(tmp_dirpath):
        os.makedirs(tmp_dirpath)
    tmp_filenames = ('magnitude.nii.gz', 'phase.nii.gz',
                     'qsm.nii.gz', 'mask.nii.gz')
    tmp_filepaths = tuple(os.path.join(tmp_dirpath, tmp_filename)
                          for tmp_filename in tmp_filenames)
    # export temp input
    if len(images) > 2:
        images = images[-2:]
        affines = affines[-2:]
    for image, affine, tmp_filepath in zip(images, affines, tmp_filepaths):
        mrt.input_output.save(tmp_filepath, image[..., selected], affine)
    # execute script on temp input
    cmd = [
        'qsm_as_legacy.py',
        '--magnitude_input', tmp_filepaths[0],
        '--phase_input', tmp_filepaths[1],
        '--qsm_output', tmp_filepaths[2],
        '--mask_output', tmp_filepaths[3],
        '--echo_time', str(params[te_label][selected]),
        # '--field_strength', str(params[b0_label][selected]),
        # '--angles', str(params[th_label][selected]),
        '--units', 'ppb']
    mrt.utils.execute(str(' '.join(cmd)))
    # import temp output
    arr_list, meta_list = [], []
    for tmp_filepath in tmp_filepaths[2:]:
        img, meta = mrt.input_output.load(tmp_filepath, meta=True)
        arr_list.append(img)
        meta_list.append(meta)
    # clean up tmp files
    if os.path.isdir(tmp_dirpath):
        shutil.rmtree(tmp_dirpath)
    # prepare output
    type_list = ('qsm', 'mask')
    params_list = ({'te': params[te_label][selected]}, {})
    img_type_list = tuple(img_types[key] for key in type_list)
    return arr_list, meta_list, img_type_list, params_list


# ======================================================================
def qsm_sdi(
        images,
        affines,
        params,
        img_types):
    pass


# ======================================================================
def calc_afi(
        images,
        affines,
        params,
        ti_label,
        fa_label,
        img_types):
    """
    Fit monoexponential decay to images using the log-linear method.
    """
    y_arr = np.stack(images, -1).astype(float)

    s_arr = mrt.utils.polar2complex(y_arr[..., 0], fix_phase_interval(y_arr[..., 1]))
    # s_arr = images[0]
    t_r = params[ti_label]
    nominal_fa = params[fa_label]

    mask = s_arr[..., 0] != 0.0
    r = np.zeros_like(s_arr[..., 1])
    r[mask] = s_arr[..., 0][mask] / s_arr[..., 1][mask]
    n = t_r[1] / t_r[0]  # usually: t_r[1] > t_r[0]
    fa = np.rad2deg(np.real(np.arccos((r * n - 1) / (n - r))))

    img_list = [fa / nominal_fa]
    aff_list = _simple_affines(affines)
    type_list = ['eff']
    img_type_list = tuple(img_types[key] for key in type_list)
    params_list = ({},) * len(img_list)
    return img_list, aff_list, img_type_list, params_list


# ======================================================================
def time_to_rate(
        array,
        in_units='ms',
        out_units='Hz'):
    k = 1.0
    if in_units == 'ms':
        k *= 1.0e3
    if out_units == 'kHz':
        k *= 1.0e-3
    array[array != 0.0] = k / array[array != 0.0]
    return array


# ======================================================================
def rate_to_time(
        array,
        in_units='Hz',
        out_units='ms'):
    k = 1.0
    if in_units == 'kHz':
        k *= 1.0e3
    if out_units == 'ms':
        k *= 1.0e-3
    array[array != 0.0] = k / array[array != 0.0]
    return array


# ======================================================================
def fix_phase_interval(arr):
    """
    Ensure that the range of values is interpreted as valid phase information.

    This is useful for DICOM-converted images (without post-processing).

    Args:
        arr (np.ndarray): Array to be processed.

    Returns:
        array (np.ndarray): An array scaled to (-pi,pi).

    Examples:
        >>> fix_phase_interval(np.arange(8))
        array([-3.14159265, -2.24399475, -1.34639685, -0.44879895,  0.44879895,
                1.34639685,  2.24399475,  3.14159265])
        >>> fix_phase_interval(np.array([-10, -5, 0, 5, 10]))
        array([-3.14159265, -1.57079633,  0.        ,  1.57079633,  3.14159265])
        >>> fix_phase_interval(np.array([-10, 10, 1, -3]))
        array([-3.14159265,  3.14159265,  0.31415927, -0.9424778 ])
    """
    if not mrt.utils.is_in_range(arr, (-np.pi, np.pi)):
        arr = mrt.utils.scale(arr.astype(float), (-np.pi, np.pi))
    return arr


# ======================================================================
def func_exp_recovery(t_arr, tau, s_0, eff=1.0, const=0.0):
    """
    s(t)= s_0 * (1 - 2 * eff * exp(-t/tau)) + const

    [s_0 > 0, tau > 0, eff > 0]
    """
    if s_0 > 0.0 and tau > 0.0 and eff > 0.0:
        s_t_arr = s_0 * (1.0 - 2.0 * eff * np.exp(-t_arr / tau)) + const
    else:
        s_t_arr = np.tile(np.inf, len(t_arr))
    return s_t_arr


# ======================================================================
def func_exp_decay(t_arr, tau, s_0, const=0.0):
    """
    s(t)= s_0 * exp(-t/tau) + const

    [s_0 > 0, tau > 0]
    """
    s_t_arr = s_0 * np.exp(-t_arr / tau) + const
    #    if s_0 > 0.0 and tau > 0.0:
    #        s_t_arr = s_0 * np.exp(-t_arr / tau) + const
    #    else:
    #        s_t_arr = np.tile(np.inf, len((t_arr)))
    return s_t_arr


# ======================================================================
def func_flash(m0, fa, tr, t1, te, t2s):
    """
    The FLASH (a.k.a. GRE, TFL, SPGR) signal expression:
    S = M0 sin(fa) exp(-TE/T2*) (1 - exp(-TR/T1)) / (1 - cos(fa) exp(-TR/T1))
    """
    return m0 * np.sin(fa) * np.exp(-te / t2s) * \
           (1.0 - np.exp(-tr / t1)) / (1.0 - np.cos(fa) * np.exp(-tr / t1))


# ======================================================================
def rho_mp2rage(
        inv1m_arr,
        inv1p_arr,
        inv2m_arr,
        inv2p_arr,
        regularization=np.spacing(1),
        values_interval=None):
    """
    Calculate the rho image from an MP2RAGE acquisition.

    Args:
        inv1m_arr (float|np.ndarray): Magnitude of the first inversion image.
        inv1p_arr (float|np.ndarray): Phase of the first inversion image.
        inv2m_arr (float|np.ndarray): Magnitude of the second inversion image.
        inv2p_arr (float|np.ndarray): Phase of the second inversion image.
        regularization (float|int): Parameter for the regularization.
            This parameter is added to the denominator of the rho expression
            for normalization purposes, therefore should be much smaller than
            the average of the magnitude images.
            Larger values of this parameter will have the side effect of
            denoising the background.
        values_interval (tuple[float|int]|None): The output values interval.
            The standard values are linearly converted to this range.
            If None, the natural [-0.5, 0.5] interval will be used.

    Returns:
        rho_arr (float|np.ndarray): The calculated rho image from
        MP2RAGE.
    """
    if not regularization:
        regularization = 0
    inv1m_arr = inv1m_arr.astype(float)
    inv2m_arr = inv2m_arr.astype(float)
    inv1p_arr = fix_phase_interval(inv1p_arr)
    inv2p_arr = fix_phase_interval(inv2p_arr)
    inv1_arr = mrt.utils.polar2complex(inv1m_arr, inv1p_arr)
    inv2_arr = mrt.utils.polar2complex(inv2m_arr, inv2p_arr)
    rho_arr = np.real(inv1_arr.conj() * inv2_arr /
                          (inv1m_arr ** 2 + inv2m_arr ** 2 + regularization))
    if values_interval:
        rho_arr = mrt.utils.scale(rho_arr, values_interval, (-0.5, 0.5))
    return rho_arr


# ======================================================================
def rho_to_t1_mp2rage(
        rho_arr,
        eff_arr=None,
        t1_values_range=(100, 5000),
        t1_num=512,
        eff_num=32,
        **acq_params_kws):
    """
    Calculate the T1 map from an MP2RAGE acquisition.

    Args:
        rho_arr (float|np.ndarray): Magnitude of the first inversion image.
        eff_arr (float|np.array|None): Efficiency of the RF pulse excitation.
            This is equivalent to the normalized B1T field.
            Note that this must have the same spatial dimensions as the images
            acquired with MP2RAGE.
            If None, no correction for the RF efficiency is performed.
        t1_values_range (tuple[float]): The T1 value range to consider.
            The format is (min, max) where min < max.
            Values should be positive.
        t1_num (int): The base number of sampling points of T1.
            The actual number of sampling points is usually smaller, because of
            the removal of non-bijective branches.
            This affects the precision of the MP2RAGE estimation.
        eff_num (int): The base number of sampling points for the RF efficiency.
            This affects the precision of the RF efficiency correction.
        **acq_params_kws (dict): The acquisition parameters.
            This should match the signature of:  `mp2rage.acq_to_seq_params`.

    Returns:
        t1_arr (float|np.ndarray): The calculated T1 map for MP2RAGE.
    """
    from pymrt.sequences import mp2rage
    if eff_arr:
        # todo: implement B1T correction
        raise NotImplementedError('B1T correction is not yet implemented')
    else:
        # determine the rho expression
        t1 = np.linspace(t1_values_range[0], t1_values_range[1], t1_num)
        rho = mp2rage.rho(
            t1, **mp2rage.acq_to_seq_params(**acq_params_kws)[0])
        # remove non-bijective branches
        bijective_slice = mrt.utils.bijective_part(rho)
        t1 = t1[bijective_slice]
        rho = rho[bijective_slice]
        if rho[0] > rho[-1]:
            rho = rho[::-1]
            t1 = t1[::-1]
        # check that rho values are strictly increasing
        if not np.all(np.diff(rho) > 0):
            raise ValueError('MP2RAGE look-up table was not properly prepared.')

        # fix values range for rho
        if not mrt.utils.is_in_range(rho_arr, mp2rage.RHO_INTERVAL):
            rho_arr = mrt.utils.scale(rho_arr, mp2rage.RHO_INTERVAL)

        t1_arr = np.interp(rho_arr, rho, t1)
    return t1_arr


# ======================================================================
def t1_mp2rage(
        inv1m_arr,
        inv1p_arr,
        inv2m_arr,
        inv2p_arr,
        regularization=np.spacing(1),
        eff_arr=None,
        t1_values_range=(100, 5000),
        t1_num=512,
        eff_num=32,
        **acq_param_kws):
    """
    Calculate the T1 map from an MP2RAGE acquisition.

    Args:
        inv1m_arr (float|np.ndarray): Magnitude of the first inversion image.
        inv1p_arr (float|np.ndarray): Phase of the first inversion image.
        inv2m_arr (float|np.ndarray): Magnitude of the second inversion image.
        inv2p_arr (float|np.ndarray): Phase of the second inversion image.
        regularization (float|int): Parameter for the regularization.
            This parameter is added to the denominator of the rho expression
            for normalization purposes, therefore should be much smaller than
            the average of the magnitude images.
            Larger values of this parameter will have the side effect of
            denoising the background.
        eff_arr (float|np.array|None): Efficiency of the RF pulse excitation.
            This is equivalent to the normalized B1T field.
            Note that this must have the same spatial dimensions as the images
            acquired with MP2RAGE.
            If None, no correction for the RF efficiency is performed.
        t1_values_range (tuple[float]): The T1 value range to consider.
            The format is (min, max) where min < max.
            Values should be positive.
        t1_num (int): The base number of sampling points of T1.
            The actual number of sampling points is usually smaller, because of
            the removal of non-bijective branches.
            This affects the precision of the MP2RAGE estimation.
        eff_num (int): The base number of sampling points for the RF efficiency.
            This affects the precision of the RF efficiency correction.
        **acq_param_kws (dict): The acquisition parameters.
            This should match the signature of:  `mp2rage.acq_to_seq_params`.

    Returns:
        t1_arr (float|np.ndarray): The calculated T1 map for MP2RAGE.
    """
    rho_arr = rho_mp2rage(
        inv1m_arr, inv1p_arr, inv2m_arr, inv2p_arr, regularization,
        values_interval=None)
    t1_arr = rho_to_t1_mp2rage(
        rho_arr, eff_arr, t1_values_range, t1_num, eff_num, **acq_param_kws)
    return t1_arr


# ======================================================================
def fit_monoexp_decay_leasq(
        images,
        affines,
        params,
        ti_label,
        img_types):
    """
    Fit monoexponential decay to images using the least-squares method.
    """
    norm_factor = 1e4
    y_arr = np.stack(images, -1).astype(float)
    y_arr = y_arr[..., 0]  # use only the modulus
    y_arr = y_arr / np.max(y_arr) * norm_factor
    x_arr = np.array(params[ti_label]).astype(float)
    p_arr = voxel_curve_fit(
        y_arr, x_arr, func_exp_decay,
        [np.median(x_arr), np.median(y_arr)], method='curve_fit')
    img_list = np.split(p_arr, 2, -1)
    type_list = ('tau', 's_0')
    img_type_list = tuple(img_types[key] for key in type_list)
    aff_list = _simple_affines(affines)
    params_list = ({},) * len(img_list)
    np.mean()
    return img_list, aff_list, img_type_list, params_list


# ======================================================================
def fit_monoexp_decay_loglin(
        images,
        affines,
        params,
        ti_label,
        img_types):
    """
    Fit monoexponential decay to images using the log-linear method.
    """

    def prepare(arr, factor=0):
        log_arr = np.zeros_like(arr)
        # calculate logarithm only of strictly positive values
        log_arr[arr > 0.0] = np.log(arr[arr > 0.0] * np.e ** factor)
        return log_arr

    def fix(arr, factor=0):
        # tau = p_arr[..., 0]
        # s_0 = p_arr[..., 1]
        mask = arr[..., 0] != 0.0
        arr[..., 0][mask] = - 1.0 / arr[..., 0][mask]
        arr[..., 1] = np.exp(arr[..., 1] - factor)
        return arr

    exp_factor = 12  # 0: untouched, other values might improve results
    y_arr = np.stack(images, -1).astype(float)
    y_arr = y_arr[..., 0]  # use only the modulus
    x_arr = np.array(params[ti_label]).astype(float)
    p_arr = voxel_curve_fit(
        y_arr, x_arr,
        None, (np.mean(x_arr), np.mean(y_arr)),
        prepare, [exp_factor], {},
        fix, [exp_factor], {},
        method='poly')
    img_list = np.split(p_arr, 2, -1)
    aff_list = _simple_affines(affines)
    type_list = ('tau', 's_0')
    img_type_list = tuple(img_types[key] for key in type_list)
    params_list = ({},) * len(img_list)
    return img_list, aff_list, img_type_list, params_list


# ======================================================================
def fit_monoexp_decay_loglin2(
        images,
        affines,
        params,
        ti_label,
        img_types):
    """
    Fit monoexponential decay to squared images using the log-linear method.
    """

    def prepare(arr, factor=0, noise=0):
        log_arr = np.zeros_like(arr)
        # calculate logarithm only of strictly positive values
        arr -= noise
        mask = arr > 0.0
        log_arr[mask] = np.log(arr[mask] ** 2.0 * np.e ** factor)
        return log_arr

    def fix(arr, factor=0):
        # tau = p_arr[..., 0]
        # s_0 = p_arr[..., 1]
        mask = arr[..., 0] != 0.0
        arr[..., 0][mask] = - 2.0 / arr[..., 0][mask]
        arr[..., 1] = np.exp(arr[..., 1] - factor)
        return arr

    exp_factor = 12  # 0: untouched, other values might improve results
    y_arr = np.stack(images, -1).astype(float)
    y_arr = y_arr[..., 0]  # use only the modulus
    x_arr = np.array(params[ti_label]).astype(float)
    noise_level = np.percentile(y_arr, 3)
    p_arr = voxel_curve_fit(
        y_arr, x_arr,
        None, (np.mean(x_arr), np.mean(y_arr)),
        prepare, [exp_factor, noise_level], {},
        fix, [exp_factor], {},
        method='poly')

    img_list = np.split(p_arr, 2, -1)
    aff_list = _simple_affines(affines)
    type_list = ('tau', 's_0')
    img_type_list = tuple(img_types[key] for key in type_list)
    params_list = ({},) * len(img_list)
    return img_list, aff_list, img_type_list, params_list


# ======================================================================
def voxel_curve_fit(
        y_arr,
        x_arr,
        fit_func=None,
        fit_params=None,
        pre_func=None,
        pre_args=None,
        pre_kwargs=None,
        post_func=None,
        post_args=None,
        post_kwargs=None,
        method='curve_fit'):
    """
    Curve fitting for y = F(x, p)

    Args:
        y_arr (np.ndarray): Dependent variable with x dependence in the n-th dim
        x_arr (np.ndarray): Independent variable with same size as n-th dim of y
        fit_func (func):
        fit_params (list[float]):
        pre_func (func):
        pre_args (list):
        pre_kwargs (dict):
        post_func (func):
        post_args (list):
        post_kwargs (dict):
        method (str): Method to use for the curve fitting procedure.

    Returns:
        p_arr (np.ndarray) :
    """
    # TODO: finish documentation

    # y_arr : ndarray ???
    #    Dependent variable (x dependence in the n-th dimension).
    # x_arr : ndarray ???
    #    Independent variable (same number of elements as the n-th dimension).

    # reshape to linearize the independent dimensions of the array
    support_axis = -1
    shape = y_arr.shape
    support_size = shape[support_axis]
    y_arr = y_arr.reshape((-1, support_size))
    num_voxels = y_arr.shape[0]
    p_arr = np.zeros((num_voxels, len(fit_params)))
    # preprocessing
    if pre_func is not None:
        if pre_args is None:
            pre_args = []
        if pre_kwargs is None:
            pre_kwargs = {}
        y_arr = pre_func(y_arr, *pre_args, **pre_kwargs)

    if method == 'curve_fit':
        print(y_arr.shape, support_size)
        iter_param_list = [
            (fit_func, x_arr, y_i_arr, fit_params)
            for y_i_arr in np.split(y_arr, support_size, support_axis)]
        pool = multiprocessing.Pool(multiprocessing.cpu_count())
        for i, (par_opt, par_cov) in \
                enumerate(pool.imap(mrt.utils.curve_fit, iter_param_list)):
            p_arr[i] = par_opt

    elif method == 'poly':
        # polyfit requires to change matrix orientation using transpose
        p_arr = np.polyfit(x_arr, y_arr.transpose(), len(fit_params) - 1)
        # transpose the results back
        p_arr = p_arr.transpose()

    else:
        try:
            p_arr = fit_func(y_arr, x_arr, fit_params)
        except Exception as ex:
            print('WW: Exception "{}" in ndarray_fit() method "{}"'.format(
                ex, method))

    # revert to original shape
    p_arr = p_arr.reshape(list(shape[:support_axis]) + [len(fit_params)])
    # post process
    if post_func is not None:
        if post_args is None:
            post_args = []
        if post_kwargs is None:
            post_kwargs = {}
        p_arr = post_func(p_arr, *post_args, **post_kwargs)
    return p_arr


# ======================================================================
def match_series(images, affines, params, matches):
    """
    TODO: finish documentation
    """
    img_list, aff_list, img_type_list = [], [], []
    for match, img_type in matches:
        for i, series in enumerate(params['_series']):
            if re.match(match, series):
                # print(match, series, img_type, images[i].shape)  # DEBUG
                img_list.append(images[i])
                aff_list.append(affines[i])
                img_type_list.append(img_type)
                break
    params_list = ({},) * len(img_list)
    return img_list, aff_list, img_type_list, params_list


# ======================================================================
def sources_generic(
        data_dirpath,
        meta_dirpath=None,
        opts=None,
        force=False,
        verbose=D_VERB_LVL):
    """
    Get source files (both data and metadata) from specified directories

    Args:
        data_dirpath (str): Directory containing data files
        meta_dirpath (str|None): Directory containing metadata files
        opts (dict):
            Accepted options:
                - data_ext (str): File extension of the data files
                - meta_ext (str): File extension of the metadata files
                - multi_acq (bool): Use multiple acquisitions for computation
                - use_meta (bool): Use metadata, instead of filenames, to get
                  parameters
                - param_select (list[str]): Parameters to select from metadata
                - match (str): regular expression used to select data filenames
                - pattern (tuple[int]): Slicing applied to data list
                - groups (list[int]|None): Split results into groups
                  (cyclically)
        force (bool): Force calculation of output.
        verbose (int): Set level of verbosity.

    Returns:
        sources_list (list[list[str]]): List of lists of filenames to be used
            for computation
        params_list : (list[list[str|float|int]]): List of lists of parameters
            associated with the specified sources

    See Also:
        pymrt.computation.compute_generic,
        pymrt.computation.compute,
        pymrt.computation.D_OPTS
    """
    sources_list = []
    params_list = []
    opts = mrt.utils.merge_dicts(D_OPTS, opts)
    if verbose >= VERB_LVL['medium']:
        print('Opts:\t{}'.format(json.dumps(opts)))
    if os.path.isdir(data_dirpath):
        pattern = slice(*opts['pattern'])
        sources, params = [], {}
        last_acq, new_acq = None, None
        data_filepath_list = mrt.utils.listdir(
            data_dirpath, opts['data_ext'])[pattern]
        for data_filepath in data_filepath_list:
            info = mrt.naming.parse_filename(
                mrt.utils.change_ext(mrt.utils.os.path.basename(data_filepath), '',
                               mrt.utils.EXT['niz']))
            if opts['use_meta']:
                # import parameters from metadata
                info['seq'] = None
                series_meta_filepath = os.path.join(
                    meta_dirpath,
                    mrt.naming.to_filename(info, ext=opts['meta_ext']))
                if os.path.isfile(series_meta_filepath):
                    with open(series_meta_filepath, 'r') as meta_file:
                        series_meta = json.load(meta_file)
                    acq_meta_filepath = os.path.join(
                        meta_dirpath, series_meta['_acquisition'] +
                                      mrt.utils.add_extsep(opts['meta_ext']))
                    if os.path.isfile(acq_meta_filepath):
                        with open(acq_meta_filepath, 'r') as meta_file:
                            acq_meta = json.load(meta_file)
                    data_params = {}
                    if opts['param_select']:
                        for item in opts['param_select']:
                            data_params[item] = acq_meta[item] \
                                if item in acq_meta else None
                    else:
                        data_params = acq_meta
                    new_acq = (last_acq and acq_meta['_series'] != last_acq)
                    last_acq = acq_meta['_series']
            else:
                # import parameters from filename
                base, data_params = mrt.naming.parse_series_name(info['name'])
                new_acq = (last_acq and base != last_acq)
                last_acq = base
            if not opts['multi_acq'] and new_acq and sources:
                sources_list.append(sources)
                params_list.append(params)
                sources, params = [], {}
            if not opts['match'] or \
                    re.match(opts['match'], os.path.basename(data_filepath)):
                sources.append(data_filepath)
                if opts['use_meta']:
                    params.update(data_params)
                else:
                    for key, val in data_params.items():
                        params[key] = (params[key] if key in params else []) \
                                      + [val]
        if sources:
            sources_list.append(sources)
            params_list.append(params)

        if opts['groups']:
            grouped_sources_list, grouped_params_list = [], []
            grouped_sources, grouped_params = [], []
            for sources, params in zip(sources_list, params_list):
                grouping = list(opts['groups']) * \
                           int((len(sources) / sum(opts['groups'])) + 1)
                seps = mrt.utils.accumulate(grouping) if grouping else []
                for i, source in enumerate(sources):
                    grouped_sources.append(source)
                    grouped_params.append(params)
                    if i + 1 in seps or i + 1 == len(sources):
                        grouped_sources_list.append(grouped_sources)
                        grouped_params_list.append(grouped_params)
                        grouped_sources, grouped_params = [], []
            sources_list = grouped_sources_list
            params_list = grouped_params_list

        if verbose >= VERB_LVL['debug']:
            for sources, params in zip(sources_list, params_list):
                print(mrt.utils.tty_colorify('DEBUG', 'r'))
                print(sources, params)
    elif verbose >= VERB_LVL['medium']:
        print("WW: no data directory '{}'. Skipping.".format(data_dirpath))
    return sources_list, params_list


# ======================================================================
def compute_generic(
        sources,
        out_dirpath,
        params=None,
        opts=None,
        force=False,
        verbose=D_VERB_LVL):
    """
    Perform the specified computation on source files.

    Args:
        sources (list[str]): Directory containing data files.
        out_dirpath (str): Directory containing metadata files.
        params (dict): Parameters associated with the sources.
        opts (dict):
            Accepted options:
                - types (list[str]): List of image types to use for results.
                - mask: (tuple[tuple[int]): Slicing for each dimension.
                - adapt_mask (bool): adapt over- or under-sized mask.
                - dtype (str): data type to be used for the target images.
                - compute_func (str): function used for the computation.

                  compute_func(images, params, compute_args, compute_kwargs)
                  -> img_list, img_type_list
                - compute_args (list): additional positional parameters for
                  compute_func
                - compute_kwargs (dict): additional keyword parameters for
                  compute_func
                - affine_func (str): name of the function for affine
                  computation: affine_func(affines, affine_args...) -> affine
                - affine_args (list): additional parameters for affine_func
        force (bool): Force calculation of output
        verbose (int): Set level of verbosity.

    Returns:
        targets ():

    See Also:
        pymrt.computation.sources_generic,
        pymrt.computation.compute,
        pymrt.computation.D_OPTS
    """
    # TODO: implement affine_func, affine_args, affine_kwargs?
    # get the num, name and seq from first source file
    opts = mrt.utils.merge_dicts(D_OPTS, opts)

    if params is None:
        params = {}
    if opts is None:
        opts = {}

    targets = []
    info = mrt.naming.parse_filename(sources[0])
    if 'ProtocolName' in params:
        info['name'] = params['ProtocolName']
    for image_type in opts['types']:
        info['type'] = image_type
        targets.append(os.path.join(out_dirpath, mrt.naming.to_filename(info)))

    # perform the calculation
    if mrt.utils.check_redo(sources, targets, force):
        if verbose > VERB_LVL['none']:
            print('{}:\t{}'.format('Object', os.path.basename(info['name'])))
        if verbose >= VERB_LVL['medium']:
            print('Opts:\t{}'.format(json.dumps(opts)))
        images, affines = [], []
        mask = [
            (slice(*dim) if dim is not None else slice(None))
            for dim in opts['mask']]
        for source in sources:
            if verbose > VERB_LVL['none']:
                print('Source:\t{}'.format(os.path.basename(source)))
            if verbose > VERB_LVL['none']:
                print('Params:\t{}'.format(params))
            image, affine, header = mrt.input_output.load(source, meta=True)
            # fix mask if shapes are different
            if opts['adapt_mask']:
                mask = [
                    (mask[i] if i < len(mask) else slice(None))
                    for i in range(len(image.shape))]
            images.append(image[mask])
            affines.append(affine)
        if 'compute_func' in opts:
            compute_func = eval(opts['compute_func'])
            if 'compute_args' not in opts:
                opts['compute_args'] = []
            if 'compute_kwargs' not in opts:
                opts['compute_kwargs'] = {}
            img_list, aff_list, img_type_list, params_list = compute_func(
                images, affines, params,
                *opts['compute_args'], **opts['compute_kwargs'])
        else:
            img_list, aff_list, img_type_list = zip(
                *[(img, aff, img_type) for img, aff, img_type
                  in zip(images, affines, itertools.cycle(opts['types']))])
            params_list = ({},) * len(img_list)

        for target, target_type in zip(targets, opts['types']):
            for img, aff, img_type, params in \
                    zip(img_list, aff_list, img_type_list, params_list):
                if img_type == target_type:
                    if 'dtype' in opts:
                        img = img.astype(opts['dtype'])
                    if params:
                        for key, val in params.items():
                            target = mrt.naming.change_param_val(target, key, val)
                    if verbose > VERB_LVL['none']:
                        print('Target:\t{}'.format(os.path.basename(target)))
                    mrt.input_output.save(target, img, affine=aff)
                    break
    return targets


# ======================================================================
def compute(
        sources_func,
        sources_args,
        sources_kwargs,
        compute_func,
        compute_args,
        compute_kwargs,
        in_dirpath,
        out_dirpath,
        recursive=False,
        meta_subpath=None,
        data_subpath=None,
        verbose=D_VERB_LVL):
    """
    Interface to perform calculation from all input files within a path.

    If recursive flag is set or if input directory contains no suitable file,
    it tries to descend into subdirectories.
    If meta_subpath is set, it will look there for metadata files.
    If data_subpath is set, it will look there for data files.

    Args:
        sources_func (func): Returns a list of list of filepaths used as input.
            Each list of filepaths should contain the exhaustive input for the
            computation to be performed. Function expected signature:
            sources_func(data_path, meta_path, sources_args...) ->
            ((string, dict) list) list.
        sources_args (list): Positional parameters passed to get_sources_func.
        sources_kwargs (dict): Keyword parameters passed to get_sources_func.
        compute_func (func): Calculation to perform on each list of filepaths.
            Function expected signature:
            compute_func(source_list, out_dirpath, compute_args...) ->
            out_filepath.
        compute_args (list): Positional parameters passed to compute_func.
        compute_kwargs (dict): Keyword parameters passed to compute_func.
        in_dirpath (str): Path to input directory path.
        out_dirpath (str): Path to output directory path.
            The input directory structure is preserved during the recursion.
        recursive (bool): Process subdirectories recursively.
        meta_subpath (str): Subpath appended when searching for metadata.
            Appending is performed (non-cumulatively) at each iteration
            recursion.
        data_subpath (str): Subpath appended when searching for data.
            Appending is performed (non-cumulatively) at each iteration
            recursion.
        verbose (int): Set level of verbosity.

    Returns:
        None

    See Also:
        pymrt.computation.compute_generic,
        pymrt.computation.source_generic,
        pymrt.computation.D_OPTS

    """
    # handle extra subdirectories in input path
    data_dirpath = os.path.join(in_dirpath, data_subpath) \
        if data_subpath is not None else in_dirpath
    meta_dirpath = os.path.join(in_dirpath, meta_subpath) \
        if meta_subpath is not None else None

    # extract input files from directory
    sources_list, params_list = sources_func(
        data_dirpath, meta_dirpath, *sources_args, **sources_kwargs)
    if sources_list and params_list:
        if not out_dirpath:
            out_dirpath = in_dirpath
        elif not os.path.exists(out_dirpath):
            os.makedirs(out_dirpath)
        if verbose > VERB_LVL['none']:
            print('Input:\t{}'.format(in_dirpath))
            print('Output:\t{}'.format(out_dirpath))
        if verbose >= VERB_LVL['medium']:
            print('Data subpath:\t{}'.format(data_subpath))
        if meta_dirpath and verbose >= VERB_LVL['medium']:
            print('Meta subpath:\t{}'.format(meta_subpath))
        for sources, params in zip(sources_list, params_list):
            compute_func(
                sources, out_dirpath, params,
                *compute_args, **compute_kwargs)
            mrt.utils.elapsed('Time: ')
            if verbose >= VERB_LVL['medium']:
                mrt.utils.print_elapsed(only_last=True)
    else:
        recursive = True

    # descend into subdirectories
    if recursive:
        recursive = recursive or bool(sources_list)
        subdirs = [subdir for subdir in os.listdir(in_dirpath)
                   if os.path.isdir(os.path.join(in_dirpath, subdir))]
        for subdir in subdirs:
            new_in_dirpath = os.path.join(in_dirpath, subdir)
            new_out_dirpath = os.path.join(out_dirpath, subdir)
            compute(
                sources_func, sources_args, sources_kwargs,
                compute_func, compute_args, compute_kwargs,
                new_in_dirpath, new_out_dirpath, recursive,
                meta_subpath, data_subpath, verbose)


# ======================================================================
if __name__ == '__main__':
    msg(__doc__.strip())

mrt.utils.elapsed('pymrt.computation')
