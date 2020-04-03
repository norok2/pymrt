#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pymrt.computation: generic computation utilities for MRI data analysis.

See Also:
    pymrt.recipes
"""

# ======================================================================
# :: Future Imports
from __future__ import (
    absolute_import, division, print_function, unicode_literals)

# todo: clarify difference between computation and batch (get rid of both?)
# todo: use kws instead of opts
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
# import multiprocessing  # Process-based parallelism
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
import flyingcircus as fc  # Everything you always wanted to have in Python*


# :: External Imports Submodules
# import matplotlib.pyplot as plt  # Matplotlib's pyplot: MATLAB-like syntax
# import scipy.optimize  # SciPy: Optimization Algorithms
# import scipy.integrate  # SciPy: Integrations facilities
# import scipy.constants  # SciPy: Mathematal and Physical Constants
# import scipy.stats  # SciPy: Statistical functions

# :: Local Imports
import pymrt as mrt  # Python Magnetic Resonance Tools: the multi-tool of MRI.

import pymrt.util
import pymrt.naming
import pymrt.input_output
from pymrt.recipes.generic import voxel_curve_fit

# from dcmpi.lib.common import ID

# from pymrt import INFO
from pymrt import VERB_LVL, D_VERB_LVL
from pymrt import msg, dbg, fmt, fmtm

# ======================================================================
META_EXT = 'info'  # ID['info']

D_OPTS = {
    # sources
    'data_ext': mrt.util.EXT['niz'],
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
    fc.execute(str(' '.join(cmd)))
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
def func_exp_recovery(t_arr, tau, s_0, eff=1.0, const=0.0):
    """
    s(t)= s_0 * (1 - 2 * eff * exp(-t/tau)) + const

    [s_0 > 0, tau > 0, eff > 0]
    """
    s_t_arr = s_0 * (1.0 - 2.0 * eff * np.exp(-t_arr / tau)) + const
    # if s_0 > 0.0 and tau > 0.0 and eff > 0.0:
    #     s_t_arr = s_0 * (1.0 - 2.0 * eff * np.exp(-t_arr / tau)) + const
    # else:
    #     s_t_arr = np.tile(np.inf, len(t_arr))
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
    opts = fc.join_(D_OPTS, opts)
    if verbose >= VERB_LVL['medium']:
        print('Opts:\t{}'.format(json.dumps(opts)))
    if os.path.isdir(data_dirpath):
        pattern = slice(*opts['pattern'])
        sources, params = [], {}
        last_acq, new_acq = None, None
        data_filepath_list = fc.listdir(
            data_dirpath, opts['data_ext'])[pattern]
        for data_filepath in data_filepath_list:
            info = mrt.naming.parse_filename(
                fc.change_ext(fc.os.path.basename(data_filepath), '',
                                   mrt.util.EXT['niz']))
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
                                      fc.add_extsep(opts['meta_ext']))
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
                seps = itertools.accumulate(grouping) if grouping else []
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
                print('DEBUG')
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

                  compute_func(images, params, compute_args, compute_kws)
                  -> img_list, img_type_list
                - compute_args (list): additional positional parameters for
                  compute_func
                - compute_kws (Mappable|None): additional keyword parameters for
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
    # TODO: implement affine_func, affine_args, affine_kws?
    # get the num, name and seq from first source file
    opts = fc.join_(D_OPTS, opts)

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
    if fc.check_redo(sources, targets, force):
        if verbose > VERB_LVL['none']:
            print('{}:\t{}'.format('Object', os.path.basename(info['name'])))
        if verbose >= VERB_LVL['medium']:
            print('Opts:\t{}'.format(json.dumps(opts)))
        images, affines = [], []
        mask = tuple(
            (slice(*dim) if dim is not None else slice(None))
            for dim in opts['mask'])
        for source in sources:
            if verbose > VERB_LVL['none']:
                print('Source:\t{}'.format(os.path.basename(source)))
            if verbose > VERB_LVL['none']:
                print('Params:\t{}'.format(params))
            image, affine, header = mrt.input_output.load(source, meta=True)
            # fix mask if shapes are different
            if opts['adapt_mask']:
                mask = tuple(
                    (mask[i] if i < len(mask) else slice(None))
                    for i in range(len(image.shape)))
            images.append(image[mask])
            affines.append(affine)
        if 'compute_func' in opts:
            compute_func = eval(opts['compute_func'])
            if 'compute_args' not in opts:
                opts['compute_args'] = []
            if 'compute_kws' not in opts:
                opts['compute_kws'] = {}
            img_list, aff_list, img_type_list, params_list = compute_func(
                images, affines, params,
                *opts['compute_args'], **opts['compute_kws'])
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
        sources_kws,
        compute_func,
        compute_args,
        compute_kws,
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
        sources_kws (Mappable|None): Keyword parameters passed to get_sources_func.
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
        data_dirpath, meta_dirpath, *sources_args, **sources_kws)
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
                *compute_args, **compute_kws)
            fc.elapsed('Time: ')
            if verbose >= VERB_LVL['medium']:
                fc.report(only_last=True)
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
                sources_func, sources_args, sources_kws,
                compute_func, compute_args, compute_kws,
                new_in_dirpath, new_out_dirpath, recursive,
                meta_subpath, data_subpath, verbose)


# ======================================================================
if __name__ == '__main__':
    msg(__doc__.strip())

fc.elapsed('pymrt.computation')
