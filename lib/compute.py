#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
mr_lib: basic and advanced generic computations for MRI data analysis.
"""


# ======================================================================
# :: Future Imports
from __future__ import division
from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals


# ======================================================================
# :: Python Standard Library Imports
import os  # Miscellaneous operating system interfaces
# import shutil  # High-level file operations
# import math  # Mathematical functions
import time  # Time access and conversions
import datetime  # Basic date and time types
# import operator  # Standard operators as functions
# import collections  # High-performance container datatypes
import itertools  # Functions creating iterators for efficient looping
# import functools  # Higher-order functions and operations on callable objects
import re  # Regular expression operations
# import subprocess  # Subprocess management
import multiprocessing  # Process-based parallelism
# import inspect  # Inspect live objects
# import csv  # CSV File Reading and Writing [CSV: Comma-Separated Values]
import json  # JSON encoder and decoder [JSON: JavaScript Object Notation]


# :: External Imports
import numpy as np  # NumPy (multidimensional numerical arrays library)
# import scipy as sp  # SciPy (signal and image processing library)
# import matplotlib as mpl  # Matplotlib (2D/3D plotting library)
# import sympy as sym  # SymPy (symbolic CAS library)
# import PIL  # Python Image Library (image manipulation toolkit)
# import SimpleITK as sitk  # Image ToolKit Wrapper
import nibabel as nib  # NiBabel (NeuroImaging I/O Library)
# import nipy  # NiPy (NeuroImaging in Python)
# import nipype  # NiPype (NiPy Pipelines and Interfaces)

# :: External Imports Submodules
# import matplotlib.pyplot as plt  # Matplotlib's pyplot: MATLAB-like syntax
# import mayavi.mlab as mlab  # Mayavi's mlab: MATLAB-like syntax
# import scipy.optimize  # SciPy: Optimization Algorithms
# import scipy.integrate  # SciPy: Integrations facilities
# import scipy.constants  # SciPy: Mathematal and Physical Constants
# import scipy.ndimage  # SciPy: ND-image Manipulation
# import scipy.stats  # SciPy: Statistical functions

# :: Local Imports
import mri_tools.lib.base as mrb
import mri_tools.lib.utils as mru
import mri_tools.lib.nifti as mrn
# from dcmpi.lib.common import ID
# from mri_tools import INFO
from mri_tools import VERB_LVL
from mri_tools import D_VERB_LVL
# from mri_tools import _firstline


# ======================================================================
META_EXT = 'info'  # ID['info']

D_OPTS = {
    # sources
    'data_ext': mrn.D_EXT,
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
def preset_t1_mp2rage_builtin():
    """
    Preset to get built-in T1 maps from the MP2RAGE sequence.
    """
    new_opts = {
        'types': ['T1', 'INV2M'],
        'param_select': ['ProtocolName', '_series'],
        'match': '.*MP2RAGE.*',
        'dtype': 'float',
        'mask': [[None], [None], [None], [0]],
        }
    new_opts.update({
        'img_func': 'match_series',
        'img_params': (
            (
                ('.*T1_Images.*', new_opts['types'][0]),
                ('.*INV2(?!_PHS).*', new_opts['types'][1]),
                ),
            )
        })
    return new_opts


# ======================================================================
def preset_t2s_flash_builtin():
    """
    Preset to get built-in T2* maps from the FLASH sequence.
    """
    new_opts = {
        'types': ['T2S', 'PD'],
        'param_select': ['ProtocolName', '_series'],
        'match': '.*T2Star_Images.*',
        'dtype': 'float',
        }
    return new_opts


# ======================================================================
def preset_t2s_multiecho_loglin():
    """
    Preset to get T2* maps from multi-echo data using a log-linear fit.
    """
    new_opts = {
        'types': ['T2S', 'PD'],
        'param_select': ['ProtocolName', 'EchoTime::ms', '_series'],
        'match': '.*FLASH.*',
        'dtype': 'float',
        'multi_acq': False,
        'img_func': 'fit_monoexp_decay_loglin',
        'img_params': ('EchoTime::ms', {'tau': 'T2S', 's_0': 'PD'})
        }
    return new_opts


# ======================================================================
def preset_t2s_multiecho_leasq():
    """
    Preset to get T2* maps from multi-echo data using a least-squares fit.
    """
    new_opts = {
        'types': ['T2S', 'PD'],
        'param_select': ['ProtocolName', 'EchoTime::ms', '_series'],
        'match': '.*FLASH.*',
        'dtype': 'float',
        'multi_acq': False,
        'img_func': 'fit_monoexp_decay_leasq',
        'img_params': ('EchoTime::ms', {'tau': 'T2S', 's_0': 'PD'})
        }
    return new_opts


# ======================================================================
def ensure_phase_range(array):
    """
    Ensure that the range of values is interpreted as valid phase information.

    This is useful for DICOM-converted images (without post-processing).

    Parameters
    ==========
    array : ndarray
        Array to be processed.

    Returns
    =======
    array : ndarray
        An array scaled to (-pi,pi).

    """
    # correct phase value range (useful for DICOM-converted images)
    if mrb.range_size(mrb.range_array(array)) > 2.0 * np.pi:
        array = mrb.to_range(array, mrb.range_array(array), (-np.pi, np.pi))
    return array


# ======================================================================
def func_exp_recovery(t_arr, tau, s_0, eff=1.0, const=0.0):
    """
    s(t)= s_0 * (1 - 2 * eff * exp(-t/tau)) + const

    [s_0 > 0, tau > 0, eff > 0]
    """
    if s_0 > 0.0 and tau > 0.0 and eff > 0.0:
        s_t_arr = s_0 * (1.0 - 2.0 * eff * np.exp(-t_arr / tau)) + const
    else:
        s_t_arr = np.tile(np.inf, len((t_arr)))
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
def fit_monoexp_decay_leasq(
        images,
        params,
        ti_label,
        img_types):
    """
    Fit monoexponential decay to images using the least-squares method.
    """
    norm_factor = 1e4
    y_arr = mrb.ndstack(images, -1).astype(float)
    y_arr = y_arr[..., 0]  # use only the modulus
    y_arr = y_arr / np.max(y_arr) * norm_factor
    x_arr = np.array(params[ti_label]).astype(float)
    p_arr = fit_ndarray(
        y_arr, x_arr, func_exp_decay,
        (np.mean(x_arr), np.mean(y_arr)), method='non-linear')
    img_list = mrb.ndsplit(p_arr, -1)
    type_list = ('tau', 's_0')
    img_type_list = tuple([img_types[key] for key in type_list])
    return img_list, img_type_list


# ======================================================================
def fit_monoexp_decay_loglin(
        images,
        params,
        ti_label,
        img_types):
    """
    Fit monoexponential decay to images using the log-linear method.
    """
    def prepare(y_arr, factor=0):
        log_arr = np.zeros_like(y_arr)
        # calculate logarithm only of strictly positive values
        log_arr[y_arr > 0.0] = np.log(y_arr[y_arr > 0.0] * np.e ** factor)
        return log_arr

    def fix(p_arr, factor=0):
        # tau = p_arr[..., 0]
        # s_0 = p_arr[..., 1]
        p_arr[..., 0][p_arr[..., 0] != 0.0] = \
            - 1.0 / p_arr[..., 0][p_arr[..., 0] != 0.0]
        p_arr[..., 1] = np.exp(p_arr[..., 1] - factor)
        return p_arr

    factor = 12  # 0: untouched, other values might improve results
    y_arr = mrb.ndstack(images, -1).astype(float)
    y_arr = y_arr[..., 0]  # use only the modulus
    x_arr = np.array(params[ti_label]).astype(float)
    p_arr = fit_ndarray(
        y_arr, x_arr, None, (np.mean(x_arr), np.mean(y_arr)),
        prepare, [factor], fix, [factor],
        method='linear')
    img_list = mrb.ndsplit(p_arr, -1)
    type_list = ('tau', 's_0')
    img_type_list = tuple([img_types[key] for key in type_list])
    return img_list, img_type_list


# ======================================================================
def fit_ndarray(
        y_arr,
        x_arr,
        fit_func=None,
        fit_params=None,
        pre_func=None,
        pre_params=None,
        post_func=None,
        post_params=None,
        method='non-linear'):
    """
    Curve fitting for y = F(x, p)
    TODO: finish documentation

    Parameters
    ==========
    y_arr : ndarray
        Dependent variable (x dependence in the n-th dimension).
    x_arr : ndarray
        Independent variable (same number of elements as the n-th dimension).

    Returns
    =======
    p_arr : ndarray
        Parameters fit (must be)

    """
    # reshape to linearize the independent dimensions of the array
    support_axis = -1
    shape = y_arr.shape
    support_size = shape[support_axis]
    y_arr = y_arr.reshape((-1, support_size))
    num_voxels = y_arr.shape[0]
    p_arr = np.zeros((num_voxels, len(fit_params)))
    # preprocessing
    if pre_func is not None and pre_params is not None:
        y_arr = pre_func(y_arr, *pre_params)

    if method == 'non-linear':
        iter_param_list = [
            (fit_func, x_arr, y_i_arr, fit_params)
            for y_i_arr in mrb.ndsplit(y_arr, 0)]
        pool = multiprocessing.Pool(multiprocessing.cpu_count())
        for idx, (par_opt, par_cov) in \
                enumerate(pool.imap(mrb.curve_fit, iter_param_list)):
            p_arr[idx] = par_opt

    elif method == 'linear':
        # polifit requires to change matrix orientation using transpose
        p_arr = np.polyfit(x_arr, y_arr.transpose(), len(fit_params) - 1)
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
    if post_func is not None and post_params is not None:
        p_arr = post_func(p_arr, *post_params)
    return p_arr


# ======================================================================
def match_series(images, params, matchings):
    """
    TODO: finish documentation
    """
    img_list, img_type_list = [], []
    for idx, series in enumerate(params['_series']):
        for match, img_type in matchings:
            if re.match(match, series):
                img_list.append(images[idx])
                img_type_list.append(img_type)
                break
    return img_list, img_type_list


# ======================================================================
def sources_generic(
        data_dirpath,
        meta_dirpath=None,
        opts=None,
        force=False,
        verbose=D_VERB_LVL):
    """
    Get source files (both data and metadata) from specified directories.

    Parameters
    ==========
    data_dirpath : str
        Directory containing data files.
    meta_dirpath : str or None
        Directory containing metadata files.
    opts : dict
        | Accepted options:
        | data_ext: str: File extension of the data files
        | meta_ext: str: File extension of the metadata files
        | multi_acq: bool: Use multiple acquisitions for computation
        | use_meta: bool: Use metadata, instead of filenames, to get parameters
        | param_select: str or None list: Parameters to select from metadata
        | match: str: REGEX used to select data filenames
        | pattern: int (1|2|3)-tuple: Slicing applied to data list.
        | groups: int list or None: Results are split into groups (cyclically)
    force : boolean (optional)
        Force calculation of output.
    verbose : int (optional)
        Set level of verbosity.

    Returns
    =======
    sources_list : (str list) list
        List of lists of filenames to be used for computation.
    params_list : (str or num list) list
        List of lists of parameters associated with the specified sources.

    See Also
    ========
    compute.compute_generic, compute.auto_compute, compute.D_OPTS

    """
    sources_list = []
    params_list = []
    if verbose >= VERB_LVL['medium']:
        print('Opts:\t{}'.format(json.dumps(opts)))
    if os.path.isdir(data_dirpath):
        opts = mrb.merge_dicts(D_OPTS, opts)
        pattern = slice(*opts['pattern'])
        sources, params = [], {}
        last_acq = None
        data_filepath_list = mrb.listdir(
            data_dirpath, opts['data_ext'], pattern)
        for data_filepath in data_filepath_list:
            info = mru.parse_filename(mrn.filename_noext(
                mrb.os.path.basename(data_filepath)))
            if opts['use_meta']:
                # import parameters from metadata
                info['seq'] = None
                series_meta_filepath = os.path.join(
                    meta_dirpath, mru.to_filename(info, ext=opts['meta_ext']))
                if os.path.isfile(series_meta_filepath):
                    with open(series_meta_filepath, 'r') as meta_file:
                        series_meta = json.load(meta_file)
                    acq_meta_filepath = os.path.join(
                        meta_dirpath, series_meta['_acquisition'] +
                        mrb.add_extsep(opts['meta_ext']))
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
                base, data_params = mru.parse_series_name(info['name'])
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
                    for key, val in data_params.items:
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
                seps = mrb.accumulate(grouping) if grouping else []
                for idx, source in enumerate(sources):
                    grouped_sources.append(source)
                    grouped_params.append(params)
                    if idx + 1 in seps or idx + 1 == len(sources):
                        grouped_sources_list.append(grouped_sources)
                        grouped_params_list.append(grouped_params)
                        grouped_sources, grouped_params = [], []
            sources_list = grouped_sources_list
            params_list = grouped_params_list

        if verbose >= VERB_LVL['debug']:
            for sources, params in zip(sources_list, params_list):
                print(mrb.tty_colorify('DEBUG', 'r'))
                print(sources, params)
    elif verbose >= VERB_LVL['medium']:
        print("WW: no data directory '{}'. Skipping.".format(data_dirpath))
    return sources_list, params_list


# ======================================================================
def compute_generic(
        sources,
        out_dirpath,
        params={},
        opts={},
        force=False,
        verbose=D_VERB_LVL):
    """
    Perform the speficified computation on source files.

    Parameters
    ==========
    sources : str list
        Directory containing data files.
    out_dirpath : str
        Directory containing metadata files.
    params : dict (optional)
        Parameters associated with the sources.
    opts : dict (optional)
        | Accepted options:
        * types: str list: List of image types to use for results
        * mask: (int (1|2|3)-tuple) tuple: Slicing for each dimension
        * adapt_mask: bool: adapt over- or under-sized mask
        * dtype: str: data type to be used for the target images
        * | img_func: str: name of the function used for computation
          | img_func(images, params, img_params...) -> img_list, img_type_list
        * img_params: list: additional parameters for img_func
        * | aff_func: str: name of the function for affine computation
          | aff_func(affines, aff_params...) -> affine
        * aff_params: list: additional parameters for aff_func
    force : boolean (optional)
        Force calculation of output.
    verbose : int (optional)
        Set level of verbosity.

    Returns
    =======
    sources_list : (str list) list
        List of lists of filenames to be used for computation.
    params_list : (str or num list) list
        List of lists of parameters associated with the specified sources.

    See Also
    ========
    compute.sources_generic, compute.auto_compute, compute.D_OPTS

    """
    # get the num, name and seq from first source file
    opts = mrb.merge_dicts(D_OPTS, opts)

    targets = []
    info = mru.parse_filename(sources[0])
    if 'ProtocolName' in params:
        info['name'] = params['ProtocolName']
    for image_type in opts['types']:
        info['type'] = image_type
        targets.append(os.path.join(out_dirpath, mru.to_filename(info)))

    # perform the calculation
    if mrb.check_redo(sources, targets, force):
        if verbose > VERB_LVL['none']:
            print(
                '{}:\t{}'.format(mrb.tty_colorify('Object', 'g'),
                 os.path.basename(info['name'])))
        if verbose >= VERB_LVL['medium']:
            print('Opts:\t{}'.format(json.dumps(opts)))
        images, affines = [], []
        mask = [(slice(*dim) if dim is not None else slice(None))
            for dim in opts['mask']]
        for source in sources:
            if verbose > VERB_LVL['none']:
                print('Source:\t{}'.format(os.path.basename(source)))
            if verbose > VERB_LVL['none']:
                print('Params:\t{}'.format(params))
            nifti = nib.load(source)
            image = nifti.get_data()
            affine = nifti.get_affine()
            # fix mask if shape is to
            if opts['adapt_mask']:
                mask = [
                    (mask[i] if i < len(mask) else slice(None))
                    for i in range(len(image.shape))]
            images.append(image[mask])
            affines.append(affine)
        if 'img_func' in opts:
            img_func = globals()[opts['img_func']]
            img_params = opts['img_params'] if 'img_params' in opts else []
            img_list, img_type_list = img_func(images, params, *img_params)
        else:
            img_list, img_type_list = zip(*[(img, img_type)
                for img, img_type
                in zip(images, itertools.cycle(opts['types']))])
        if 'aff_func' in opts:
            aff_func = globals()[opts['aff_func']]
            aff_params = opts['aff_params'] if 'aff_params' in opts else []
            aff = aff_func(affines, *aff_params)
        else:
            aff = affines[0]
        for target, target_type in zip(targets, opts['types']):
            for img, img_type in zip(img_list, img_type_list):
                if img_type == target_type:
                    if 'dtype' in opts:
                        img = img.astype(opts['dtype'])
                    if verbose > VERB_LVL['none']:
                        print('Target:\t{}'.format(os.path.basename(target)))
                    mrn.img_maker(target, img, aff)
                    break
    return targets


# ======================================================================
def auto_compute(
        sources_func,
        sources_params,
        calc_func,
        calc_params,
        in_dirpath,
        out_dirpath,
        recursive=False,
        meta_subpath=None,
        data_subpath=None,
        verbose=D_VERB_LVL):
    """
    Interface to perform calculation from all input files in a directory.
    If recursive flag is set or if input directory contains no suitable file,
    it tries to descend into subdirectories.
    If meta_subpath is set, it will look there for metadata files.
    If data_subpath is set, it will look there for data files.

    Parameters
    ==========
    get_sources_func : func
        | Function returning a list of list of filepaths.
        | sources_func(data_path, meta_path, sources_params...) ->
        | ((string, dict) list) list
    get_sources_params : list
        Parameters to be passed to get_sources_func.
    calc_func : func
        | Function performing calculation on each list of filepaths.
        | calc_func(source_list, out_dirpath, calc_params...) -> out_filepath
    calc_params : list
        Parameters to be passed to calc_func.
    in_dirpath : string
        Path to input directory.
    out_dirpath : string
        Path to output directory (updated at each iteration).
    recursive : boolean (optional)
        Force descending into subdirectories.
    meta_subpath : string
        Subdirectory appended (at each iteration) when searching for metadata.
    data_subpath : string
        Subdirectory appended (at each iteration) when searching for data.
    verbose : int (optional)
        Set level of verbosity.

    Returns
    =======
    None.

    See Also
    ========
    compute.sources_generic, compute_generic, compute.D_OPTS

    """
    # handle extra subdirectories in input path
    data_dirpath = os.path.join(in_dirpath, data_subpath) \
        if data_subpath is not None else in_dirpath
    meta_dirpath = os.path.join(in_dirpath, meta_subpath) \
        if meta_subpath is not None else None

    # extract input files from directory
    sources_list, params_list = sources_func(
        data_dirpath, meta_dirpath, *sources_params)
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
            begin_time = time.time()
            calc_func(sources, out_dirpath, params, *calc_params)
            end_time = time.time()
            if verbose >= VERB_LVL['medium']:
                print('Time:\t', datetime.timedelta(0, end_time - begin_time))
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
            auto_compute(
                sources_func, sources_params,
                calc_func, calc_params,
                new_in_dirpath, new_out_dirpath, recursive,
                meta_subpath, data_subpath, verbose)


# ======================================================================
if __name__ == '__main__':
    print(__doc__)
