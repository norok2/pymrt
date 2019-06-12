#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PyMRT: voxel-by-voxel correlation analysis for MRI data.
"""

# ======================================================================
# :: Future Imports
from __future__ import (
    absolute_import, division, print_function, unicode_literals)

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
# import argparse  # Parser for command-line options, arguments and subcommands
# import re  # Regular expression operations
# import subprocess  # Subprocess management
import multiprocessing  # Process-based parallelism
# import inspect  # Inspect live objects
import csv  # CSV File Reading and Writing [CSV: Comma-Separated Values]
import json  # JSON encoder and decoder [JSON: JavaScript Object Notation]

# :: External Imports
import numpy as np  # NumPy (multidimensional numerical arrays library)
import scipy as sp  # SciPy (signal and image processing library)
# import matplotlib as mpl  # Matplotlib (2D/3D plotting library)
# import sympy as sym  # SymPy (symbolic CAS library)
# import PIL  # Python Image Library (image manipulation toolkit)
# import SimpleITK as sitk  # Image ToolKit Wrapper
import nibabel as nib  # NiBabel (NeuroImaging I/O Library)
# import nipy  # NiPy (NeuroImaging in Python)
# import nipype  # NiPype (NiPy Pipelines and Interfaces)
import flyingcircus as fc  # Everything you always wanted to have in Python.*

# :: External Imports Submodules
# import scipy.constants  # SciPy: Mathematal and Physical Constants
import scipy.ndimage  # SciPy: ND-image Manipulation
import scipy.stats  # SciPy: Statistical functions

# :: Local Imports
import pymrt as mrt
import pymrt.utils
import pymrt.naming
import pymrt.registration
import pymrt.segmentation
import pymrt.input_output

from pymrt import VERB_LVL, D_VERB_LVL
from pymrt import msg
from pymrt.config import EXT_CMD

# ======================================================================
# :: parsing constants
D_SEP = '_'
PARAM_SEP = '_'
PARAM_VAL_SEP = '='
INFO_SEP = '__'

# map image type identifiers
MAP_ID = {
    't1': 'T1',
    't2': 'T2',
    't2s': 'T2S',
    'chi': 'CHI',
    'b0': 'B0',
    'b1t': 'B1T',
    'pd': 'PD',
    'm0': 'MO',
    't1w': 'T1w',
    't2w': 'T2w',
    'pdw': 'PDw',
}

# magnitude/phase image type identifiers
TYPE_ID = {
    'mag': 'MAG',
    'phs': 'PHS',
    'real': 'RE',
    'imag': 'IM',
    'complex': 'CX',
    'temp': 'TMP',
    'none': None,
}

# service image type identifiers
SERVICE_ID = {
    'mask': 'MSK'}

# MP2RAGE-related image type identifiers
MP2RAGE_ID = {
    'inv1m': 'INV1',
    'inv1p': 'INV1_PHS',
    'inv2m': 'INV2',
    'inv2p': 'INV2_PHS',
    't1': 'T1_Images',
    'uni': 'UNI_Images'}

# useful groupings
SRC_IMG_TYPE = set(list(MAP_ID.values()) + [TYPE_ID['none']])
KNOWN_IMG_TYPES = set(
    list(MAP_ID.values()) + list(TYPE_ID.values()) +
    list(SERVICE_ID.values()) + list(MP2RAGE_ID.values()))

# suffix of new reconstructed image from Siemens
ID = {
    'series': 's',
    'reco': 'rr',
}
NEW_RECO_ID = 'rr'
SERIES_NUM_ID = 's'

# prefixes for target files
MASK_FILENAME = 'mask'

EXT = {
    'reg_ref': '0_REG_REF',
    'corr_ref': '0_CORR_REF'}


# ======================================================================
def _get_ref_list(
        dirpath,
        target_list=None,
        subdir=None,
        ref_ext=''):
    """
    Find reference NIfTI image files in a directory.

    If no reference file is found, use files from target list as reference.
    A file is flagged as reference if there exist a file with the same name,
    but filename extension as specified by ref_ext.

    Args:
        dirpath (str): Path where to look for reference files.
        target_list (list[str]: List of possible reference files.
            If search fails, use first of these.
        subdir (str): Subdir to append to the specified path.
            Useful for recursive calls.
        ref_ext (str): Filename extension of the reference file flag.

    Returns:
        ref_filepaths (list[str]): List of paths to reference files
        ref_src_filepaths (list[str]): List of paths to reference source files
    """
    ref_src_filepaths = fc.base.listdir(dirpath, ref_ext)
    if ref_src_filepaths:
        ref_filepaths = []
        for ref_src in ref_src_filepaths:
            # extract dirpath
            ref_dirpath = os.path.dirname(ref_src)
            if subdir:
                ref_dirpath = os.path.join(ref_dirpath, subdir)
            # extract filename
            ref_filename = fc.base.change_ext(
                os.path.basename(ref_src), mrt.utils.EXT['niz'], ref_ext)
            ref_filepath = os.path.join(ref_dirpath, ref_filename)
            ref_filepaths.append(ref_filepath)
    elif target_list:
        ref_filepaths = target_list
    else:
        ref_filepaths = fc.base.listdir(dirpath, mrt.utils.EXT['niz'])
    if not ref_filepaths:
        raise RuntimeError('No reference file(s) found')
    return ref_filepaths, ref_src_filepaths


# ======================================================================
def _compute_affine_fsl(
        in_filepath,
        ref_filepath,
        aff_filepath,
        msk_filepath=None,
        flirt_kws=None,
        flirt__kws=None,
        force=False,
        verbose=D_VERB_LVL):
    """
    Calculate registration matrix.

    Args:
        in_filepath (str): Path to input image.
        ref_filepath (str): Path to reference image.
        aff_filepath (str): Path to file where to store registration matrix.
        msk_filepath (str): Path to mask image.
        flirt_kws (dict|None): Keyword arguments for `flirt`.
            Keyword arguments from this dictionary are passed directly to
            FSL's `flirt`.
        flirt__kws (dict|None): Keyword arguments for `flirt` (filtered).
            Keywords arguments from this dictionary should contain only
            strings containing Python code as values, which will be passed to
            Python's `eval` function.
        force (bool): Force calculation of output.
        verbose (int): Set level of verbosity.

    Returns:
        None.
    """
    if fc.base.check_redo([in_filepath, ref_filepath], [aff_filepath],
                          force):
        msg('Affine: {}'.format(os.path.basename(aff_filepath)),
            verbose, D_VERB_LVL)
        ext_cmd = EXT_CMD['fsl/5.0/flirt']
        cmd_args = {
            'in': in_filepath,
            'ref': ref_filepath,
            'omat': aff_filepath}
        if msk_filepath and os.path.isfile(msk_filepath):
            cmd_args['refweight'] = msk_filepath
        if flirt_kws:
            cmd_args.update(flirt_kws)
        if flirt__kws:
            for key, val in flirt__kws.items():
                cmd_args[str(key)] = eval(val)
        cmd = ' '.join(
            [ext_cmd] + ['-{} {}'.format(k, v) for k, v in cmd_args.items()])
        fc.base.execute(cmd, verbose=verbose)


# ======================================================================
def _apply_affine_fsl(
        in_filepath,
        ref_filepath,
        out_filepath,
        aff_filepath,
        force=False,
        verbose=D_VERB_LVL):
    """
    Apply FSL registration matrix.

    Args:
        in_filepath (str): Path to input file
        ref_filepath (str): Path to reference file
        out_filepath (str): Path to output file
        aff_filepath (str): Path to registration matrix file
        force (bool): Force calculation of output.
        verbose (int): Set level of verbosity.

    Returns:
        None.
    """
    if fc.base.check_redo(
            [in_filepath, ref_filepath, aff_filepath], [out_filepath],
            force):
        msg('Regstr: {}'.format(os.path.basename(out_filepath)),
            verbose, D_VERB_LVL)
        ext_cmd = EXT_CMD['fsl/5.0/flirt']
        cmd_options = {
            'in': in_filepath,
            'ref': ref_filepath,
            'out': out_filepath,
            'applyxfm': '',
            'init': aff_filepath,
        }
        cmd = ' '.join(
            [ext_cmd] +
            ['-{} {}'.format(k, v) for k, v in cmd_options.items()])
        fc.base.execute(cmd, verbose=verbose)


def register_ants(
        in_filepath,
        ref_filepath,
        out_filepath,
        opts_kws,
        force=False,
        verbose=D_VERB_LVL):
    return


# ======================================================================
def register_fsl(
        in_filepath,
        ref_filepath,
        out_filepath,
        ref_mask_filepath=None,
        affine_prefix='affine',
        flirt_kws=None,
        flirt__kws=None,
        helper_img_type=None,
        force=False,
        verbose=D_VERB_LVL):
    """
    Register input file to reference.

    Args:
        in_filepath (str): Path to input file.
        ref_filepath (str): Path to reference file.
        out_filepath (str): Path to output file.
        ref_mask_filepath (str|None): Path to mask for the reference file.
        affine_prefix (str): Prefix to be used when generating affine filename.
        flirt_kws (dict|None): Keyword arguments for `flirt`.
            Keyword arguments from this dictionary are passed directly to
            FSL's `flirt`.
        flirt__kws (dict|None): Keyword arguments for `flirt` (filtered)
            Keywords arguments from this dictionary should contain only
            strings containing Python code as values, which will be passed to
            Python's `eval` function.
        helper_img_type (str|None): The image type for helping registration.
            If None, no helper image is used.
        force (bool): Force calculation of output.
        verbose (int): Set level of verbosity.

    Returns:
        None.
    """
    if helper_img_type:
        in_tmp_filepath = mrt.naming.change_img_type(in_filepath,
                                                     helper_img_type)
        ref_tmp_filepath = mrt.naming.change_img_type(ref_filepath,
                                                      helper_img_type)
        if not os.path.exists(in_tmp_filepath):
            in_tmp_filepath = in_filepath
        if not os.path.exists(ref_tmp_filepath):
            ref_tmp_filepath = ref_filepath
    else:
        in_tmp_filepath = in_filepath
        ref_tmp_filepath = ref_filepath
    xfm_filepath = os.path.join(
        os.path.dirname(out_filepath),
        mrt.naming.combine_filename(affine_prefix,
                                    (ref_filepath, in_filepath)) +
        fc.base.add_extsep(mrt.utils.EXT['text']))
    _compute_affine_fsl(
        in_tmp_filepath, ref_tmp_filepath, xfm_filepath,
        ref_mask_filepath, flirt_kws, flirt__kws,
        force=force, verbose=verbose)
    _apply_affine_fsl(
        in_filepath, ref_filepath, out_filepath, xfm_filepath,
        force, verbose)


# ======================================================================
def register(
        in_filepath,
        ref_filepath,
        out_filepath,
        in_weight_filepath=None,
        ref_weight_filepath=None,
        force=False,
        verbose=D_VERB_LVL):
    """
    Register input file to reference.

    Args:
        in_filepath (str): Path to input file.
        ref_filepath (str): Path to reference file.
        out_filepath (str): Path to output file.
        force (bool): Force calculation of output.
        verbose (int): Set level of verbosity.

    Returns:
        None.
    """

    def _quick_reg(array_list, *_args, **_kws):
        img = array_list[0]
        ref = array_list[1]
        # # at first translate...
        # linear, shift = mrt.registration.affine_registration(
        #     img, ref, transform='translation',
        #     init_guess=('weights', 'weights'))
        # # print(shift)
        # img = sp.ndimage.affine_transform(img, linear, shift)
        # ... then reorient
        linear, shift = mrt.registration.affine_registration(
            img, ref, transform='reflection_simple')
        img = sp.ndimage.affine_transform(img, linear, shift)
        # ... and finally perform finer registration
        linear, shift = mrt.registration.affine_registration(img, ref, *_args,
                                                             **_kws)
        img = sp.ndimage.affine_transform(img, linear, shift)
        return img

    if fc.base.check_redo([in_filepath, ref_filepath], [out_filepath],
                          force):
        mrt.input_output.simple_filter_n_1(
            [in_filepath, ref_filepath], out_filepath, _quick_reg,
            transform='rigid', interp_order=1, init_guess=('none', 'none'))

    msg('Regstr: {}'.format(os.path.basename(out_filepath)))


# ======================================================================
def apply_mask(
        in_filepath,
        mask_filepath,
        out_filepath,
        mask_val=0.0,
        force=False,
        verbose=D_VERB_LVL):
    """
    Apply a mask.

    Args:
        in_filepath (str): Path to input file.
        mask_filepath (str): Path to reference file.
        out_filepath (str): Path to output file.
        mask_val (int|float): Value of masked out voxels.
        force (bool): Force calculation of output.
        verbose (int): Set level of verbosity.

    Returns:
        None.
    """

    def _mask_reframe(arr_list, mask_val):
        img, mask = arr_list
        container = sp.ndimage.find_objects(mask.astype(bool).astype(int))[0]
        img[~mask.astype(bool)] = mask_val
        if container:
            img = img[container]
        img = fc.extra.frame(img, 0.1, 0.0)
        return img

    if fc.base.check_redo(
            [in_filepath, mask_filepath], [out_filepath], force):
        msg('RunMsk: {}'.format(os.path.basename(out_filepath)))
        mrt.input_output.simple_filter_n_1(
            [in_filepath, mask_filepath], out_filepath,
            _mask_reframe, mask_val)


# ======================================================================
def calc_mask(
        in_filepath,
        out_dirpath=None,
        threshold='twice_first_peak',
        threshold_kws=None,
        comparison='>',
        smoothing=1.0,
        erosion_iter=4,
        dilation_iter=2,
        bet_params='-v',
        helper_img_type=None,
        force=False,
        verbose=D_VERB_LVL):
    """
    Extract a mask from an image.

    FSL's BET brain extraction algorithm can be used.

    | Workflow is:
    * Brain extraction with FSL's BET (if any)
    * Extract mask using geometry algorithm

    TODO: fix docs!!

    Parameters
    ==========
    in_filepath : str
        Path to image from which mask is to be extracted.
    out_dirpath : str (optional)
        Path to directory where to store results.
    bet_params : str (optional)
        Parameters to be used with FSL's BET brain extraction algorithm.
    threshold : (0,1)-float (optional)
        Value threshold relative to the range of the values.
    percentile_range : 2 tuple (0,1)-float (optional)
        Percentile values to be used as minimum and maximum for calculating
        the absolute threshold value from the relative one, in order to
        improve robustness against histogram outliers. Values must be in the
        [0, 1] range.
    comparison : str (optional)
        Comparison mode: [=, !=, >, <, >=, <=]
    smoothing : float (optional)
        Sigma to be used for Gaussian filtering. If zero, no filtering done.
    size_threshold : (0,1)-float (optional)
        Size threshold relative to the largest size of the array shape.
    erosion_iter : int (optional)
        Number of binary erosion iteration in mask post-processing.
    dilation_iter : int (optional)
        Number of binary dilation iteration in mask post-processing.
    force : boolean (optional)
        Force calculation of output.
    verbose : int (optional)
        Set level of verbosity.

    Returns
    =======
    out_filepath : str
        Path to output file.

    """
    # todo: move to mrt.input_output.
    if not out_dirpath:
        out_dirpath = os.path.dirname(in_filepath)
    if os.path.isdir(out_dirpath):
        out_filename = os.path.basename(
            mrt.naming.change_img_type(in_filepath, SERVICE_ID['mask']))
        out_dirpath = os.path.join(out_dirpath, out_filename)

    in_tmp_filepath = (
        mrt.naming.change_img_type(in_filepath, helper_img_type)
        if helper_img_type else in_filepath)

    msg('I: FSL BET2 params: {}'.format(bet_params),
        verbose, VERB_LVL['medium'])

    if bet_params:
        # set optimized version of mask final calculation on BET output
        # threshold = 'twice_first_peak'
        # threshold_kws = None
        # comparison = '>'
        # smoothing = 1.0
        # erosion_iter = 3
        # dilation_iter = 2
        # perform BET extraction
        ext_cmd = EXT_CMD['fsl/5.0/bet']
        bet_tmp_filepath = mrt.naming.change_img_type(out_dirpath, 'BRAIN')
        if fc.base.check_redo([in_tmp_filepath], [bet_tmp_filepath], force):
            msg('TmpMsk: {}'.format(os.path.basename(bet_tmp_filepath)))
            cmd_tokens = [
                ext_cmd, in_tmp_filepath, bet_tmp_filepath, bet_params]
            cmd = ' '.join(cmd_tokens)
            ret_code, p_stdout, p_stderr = fc.base.execute(
                cmd, verbose=verbose)
        in_tmp_filepath = bet_tmp_filepath

    if threshold_kws is None:
        threshold_kws = {}
    # extract mask using a threshold
    if fc.base.check_redo([in_tmp_filepath], [out_dirpath], force):
        _calc_mask_kws = dict(
            threshold=threshold,
            threshold_kws=dict(threshold_kws),
            comparison=comparison,
            smoothing=smoothing,
            erosion_iter=erosion_iter,
            dilation_iter=dilation_iter
        )
        msg('I: compute_mask params: '.format(_calc_mask_kws.items()),
            verbose, VERB_LVL['medium'])
        msg('GetMsk: {}'.format(os.path.basename(out_dirpath)))
        mrt.input_output.simple_filter_1_1(
            in_tmp_filepath, out_dirpath,
            mrt.segmentation.auto_mask, **_calc_mask_kws)
    return out_dirpath


# ======================================================================
def calc_difference(
        in1_filepath,
        in2_filepath,
        out_filepath,
        force=False,
        verbose=D_VERB_LVL):
    """
    Calculate the difference (i2 - i1) of two NIfTI images.

    Parameters:
        in1_filepath (str): Path to first input file
        in2_filepath (str): Path to second input file
        out_filepath (str): Path to output file
        force (bool): Force calculation of output (optional)
        verbose (int): Set level of verbosity (optional)

    Returns:
        None

    """
    if fc.base.check_redo([in1_filepath, in2_filepath], [out_filepath],
                          force):
        msg('DifImg: {}'.format(os.path.basename(out_filepath)))
        mrt.input_output.simple_filter_n_1(
            [in1_filepath, in2_filepath], out_filepath,
            (lambda images: images[1] - images[0]))


# ======================================================================
def calc_correlation(
        in1_filepath,
        in2_filepath,
        out_filepath,
        mask_filepath=None,
        removes=None,
        val_interval=None,
        trunc=None,
        force=False,
        verbose=D_VERB_LVL):
    """
    | Calculate correlation coefficients:
    * Pearson's Correlation Coefficient
    * Correlation Distance Average and Standard Deviation:
    * Relative Error Average and Standard Deviation:

    Parameters
    ==========
    in1_filepath : str
        Path to first input file.
    in2_filepath : str
        Path to second input file.
    out_filepath : str
        Path to output file.
    mask_filepath : str (optional)
        Path to mask file.
    mask_nan : bool (optional)
        Mask NaN values.
    mask_inf : bool (optional)
        Mask Inf values.
    removes : list of int or float
        List of values to mask.
    val_interval : 2-tuple (optional)
        The (min, max) values range.
    trunc : int or None (optional)
        Defines the maximum length to be used by numeric values in output.
    force : boolean (optional)
        Force calculation of output.
    verbose : int (optional)
        Set level of verbosity.

    Returns
    =======
    None.

    """
    if mask_filepath is None:
        mask_filepath = ''
    if os.path.exists(mask_filepath):
        in_filepath_list = [in1_filepath, in2_filepath, mask_filepath]
    else:
        in_filepath_list = [in1_filepath, in2_filepath]
    if fc.base.check_redo(in_filepath_list, [out_filepath], force):
        msg('Correl: {}'.format(os.path.basename(out_filepath)))
        img1_nii = nib.load(in1_filepath)
        img2_nii = nib.load(in2_filepath)
        img1 = img1_nii.get_data()
        img2 = img2_nii.get_data()
        msg('Mask:   {}'.format(os.path.basename(mask_filepath)),
            verbose, VERB_LVL['high'])
        if mask_filepath:
            mask_nii = nib.load(mask_filepath)
            mask = mask_nii.get_data().astype(bool)
        else:
            mask = np.ones_like(img1 * img2).astype(bool)
        if val_interval is None:
            val_interval = fc.extra.minmax(np.stack((img1, img2)))
        mask *= (img1 > val_interval[0])
        mask *= (img1 < val_interval[1])
        mask *= (img2 > val_interval[0])
        mask *= (img2 < val_interval[1])
        if removes is None:
            removes = []
        # calculate stats of difference image
        d_arr = img1[mask] - img2[mask]
        d_dict = fc.extra.calc_stats(d_arr, removes)
        # calculate stats of the absolute difference image
        e_arr = np.abs(d_arr)
        e_dict = fc.extra.calc_stats(e_arr, removes)
        # calculate Pearson's Correlation Coefficient
        pcc_val, pcc_p_val = \
            sp.stats.pearsonr(img1[mask].ravel(), img2[mask].ravel())
        pcc2_val = pcc_val * pcc_val
        # calculate linear polynomial fit
        n = 1
        poly_params = \
            np.polyfit(img1[mask].ravel(), img2[mask].ravel(), n)[::-1]
        #        # calculate Theil robust slope estimator (WARNING: too much
        #  memory!)
        #        theil_coeff, theil_offset, = sp.stats.mstats.theilslopes(
        #            img2[mask].ravel(),
        # img1[mask].ravel())
        # mutual information
        mi = fc.extra.norm_mutual_information(img1[mask], img2[mask])
        # voxel counts
        num = np.sum(mask.astype(bool))
        num_tot = np.size(mask)
        num_ratio = num / num_tot
        # save results to csv
        filenames = [
            fc.base.change_ext(
                os.path.basename(path), '', mrt.utils.EXT['niz'])
            for path in [in2_filepath, in1_filepath]]
        lbl_len = max([len(name) for name in filenames])
        label_list = ['avg', 'std', 'min', 'max', 'sum']
        val_filter = (lambda x: fc.base.compact_num_str(x, trunc)) \
            if trunc else (lambda x: x)
        d_arr_val = [val_filter(d_dict[key]) for key in label_list]
        e_arr_val = [val_filter(e_dict[key]) for key in label_list]
        values = \
            ['{: <{size}s}'.format(name, size=lbl_len) for name in
             filenames] + \
            list(d_arr_val) + list(e_arr_val) + \
            [val_filter(val)
             for val in [pcc_val, pcc2_val, pcc_p_val, mi,
                         poly_params[1], poly_params[0],
                         #                theil_coeff, theil_offset,
                         num, num_tot, num_ratio]]
        labels = \
            ['{: <{size}s}'.format('x_corr_file', size=lbl_len),
             '{: <{size}s}'.format('y_corr_file', size=lbl_len)] + \
            ['D-' + lbl for lbl in label_list] + \
            ['E-' + lbl for lbl in label_list] + \
            ['r', 'r2', 'p-val', 'nmi',
             'lin-cof', 'lin-off',
             #             'thl-cof', 'thl-off',
             'N_eff', 'N_tot', 'N_ratio']
        with open(out_filepath, 'w') as csvfile:
            csvwriter = csv.writer(csvfile,
                                   delimiter=str(fc.base.CSV_DELIMITER))
            csvwriter.writerow([fc.base.CSV_COMMENT_TOKEN + in2_filepath])
            csvwriter.writerow([fc.base.CSV_COMMENT_TOKEN + in1_filepath])
            csvwriter.writerow(labels)
            csvwriter.writerow(values)


# ======================================================================
def combine_correlation(
        filepath_list,
        out_dirpath,
        out_filename='correlation',
        selected_cols=None,
        force=False,
        verbose=D_VERB_LVL):
    """
    Group correlation files together.

    Parameters
    ==========
    filepath_list : list of string
        List of paths to correlation files.
    out_dirpath : str
        Path to directory where to store results.
    out_filename : str (optional)
        Filename (without extension) where to store results.
    selected_cols : list of int or None (optional)
        List of colums to be used in the grouped correlation file.
    force : boolean (optional)
        Force calculation of output.
    verbose : int (optional)
        Set level of verbosity.

    Returns
    =======
    None.

    """

    def same(list1, list2):
        """Check if list items are the same, except for blank stripping."""
        return all(item1.strip() == item2.strip()
                   for item1, item2 in zip(list1, list2))

    filepath_list.sort()
    # :: get base dir
    base_dir = ''
    for filepath in filepath_list:
        with open(filepath, 'r') as csvfile:
            csvreader = csv.reader(csvfile,
                                   delimiter=str(fc.base.CSV_DELIMITER))
            for row in csvreader:
                if row[0].startswith(fc.base.CSV_COMMENT_TOKEN):
                    base_dir = os.path.dirname(os.path.commonprefix(
                        (base_dir, row[0]))) + os.path.sep \
                        if base_dir else row[0]
    # :: summarize correlation results
    labels, rows, max_cols = [], [], []
    for filepath in filepath_list:
        with open(filepath, 'r') as csvfile:
            csvreader = csv.reader(csvfile,
                                   delimiter=str(fc.base.CSV_DELIMITER))
            for row in csvreader:
                if row[0].startswith(fc.base.CSV_COMMENT_TOKEN):
                    sub_dir = os.path.dirname(row[0][len(base_dir):])
                elif not labels:
                    labels = ['subdir'] + row if sub_dir else row
                elif not (same(labels, row) or same(labels, ['subdir'] + row)):
                    if sub_dir:
                        data = [sub_dir] + row
                    else:
                        data = row
                    if not max_cols:
                        max_cols = [0] * len(data)
                    for i, item in enumerate(data):
                        if max_cols[i] < len(item):
                            max_cols[i] = len(item)
                    rows.append(data)
    # :: fix column width
    for i, label in enumerate(labels):
        labels[i] = '{: <{size}s}'.format(label, size=max_cols[i])
    for j, row in enumerate(rows):
        for i, col in enumerate(row):
            rows[j][i] = '{: <{size}s}'.format(col, size=max_cols[i])
    # :: write grouped correlation to new file
    out_filepath = os.path.join(
        out_dirpath, out_filename + fc.base.add_extsep(mrt.utils.EXT['tab']))
    if fc.base.check_redo(filepath_list, [out_filepath], force):
        with open(out_filepath, 'w') as csvfile:
            csvwriter = csv.writer(csvfile,
                                   delimiter=str(fc.base.CSV_DELIMITER))
            if not selected_cols:
                selected_cols = range(len(labels))
            csvwriter.writerow([base_dir])
            csvwriter.writerow(
                [item for i, item in enumerate(labels) if
                 i in selected_cols])
            for row in rows:
                csvwriter.writerow(
                    [col for i, col in enumerate(row) if
                     i in selected_cols])
    return out_filepath


# ======================================================================
def plot_correlation(
        img1_filepath,
        img2_filepath,
        mask_filepath,
        val_type,
        val_interval,
        val_units,
        out_dirpath,
        corr_prefix='corr',
        force=False,
        verbose=D_VERB_LVL):
    """
    Plot the voxel-by-voxel correlation 2D histogram.

    Parameters
    ==========
    corr_filepath : str
        Path to correlation file to use for plotting.
    mask_filepath : str
        Path to mask file.
    val_type : str
        Name of the data to be processed.
    val_interval : float 2-tuple
        Interval of the data to be processed.
    val_units : str
        Units of measurements of the data to be processed.
    out_dirpath : str
        Path to directory where to store results.
    force : boolean (optional)
        Force calculation of output.
    verbose : int (optional)
        Set level of verbosity.

    Returns
    =======
    None.

    """
    filename = mrt.naming.combine_filename(
        corr_prefix, (img1_filepath, img2_filepath))
    save_filepath = os.path.join(
        out_dirpath, filename + fc.base.add_extsep(mrt.utils.EXT['plot']))
    in_filepath_list = [img1_filepath, img2_filepath]
    if mask_filepath:
        in_filepath_list.append(mask_filepath)
    if fc.base.check_redo(in_filepath_list, [save_filepath], force):
        msg('PltCor: {}'.format(os.path.basename(save_filepath)))
        img1_label = mrt.naming.filename2label(
            img1_filepath, ext=mrt.utils.EXT['niz'], max_length=32)
        img2_label = mrt.naming.filename2label(
            img2_filepath, ext=mrt.utils.EXT['niz'], max_length=32)
        title = 'Voxel-by-Voxel Correlation'
        if not val_type:
            val_type = 'Image'
        x_lbl = '{} / {} ({})'.format(val_type, val_units, img1_label)
        y_lbl = '{} / {} ({})'.format(val_type, val_units, img2_label)
        # plot the 2D histogram
        mrt.input_output.plot_histogram2d(
            img1_filepath, img2_filepath, mask_filepath, mask_filepath,
            hist_interval=(0.0, 1.0), bins=512, array_interval=val_interval,
            scale='log10', title=title,
            labels=(x_lbl, y_lbl), bisector=':k',
            cbar_kws={},
            save_filepath=save_filepath, ax=None, force=force, verbose=verbose)


# ======================================================================
def plot_histogram(
        img_filepath,
        mask_filepath,
        val_type,
        val_interval,
        val_units,
        out_dirpath,
        out_filepath_prefix='hist',
        force=False,
        verbose=D_VERB_LVL):
    """
    Plot the histogram of values for the image.

    Parameters
    ==========
    img_filepath : str
        Path to image file to use for plotting.
    mask_filepath : str
        Path to mask file.
    val_type : str
        Name of the data to be processed.
    val_interval : float 2-tuple
        Interval of the data to be processed.
    val_units : str
        Units of measurements of the data to be processed.
    out_dirpath : str
        Path to directory where to store results.
    out_filepath_prefix : str (optional)
        Prefix for output figure.
    force : boolean (optional)
        Force calculation of output.
    verbose : int (optional)
        Set level of verbosity.

    Returns
    =======
    None.

    """
    save_filepath = os.path.join(
        out_dirpath,
        out_filepath_prefix + INFO_SEP +
        fc.base.change_ext(os.path.basename(img_filepath),
                           mrt.utils.EXT['plot'],
                           mrt.utils.EXT['niz']))
    in_filepath_list = [img_filepath]
    if mask_filepath:
        in_filepath_list.append(mask_filepath)
    if fc.base.check_redo(in_filepath_list, [save_filepath], force):
        msg('PltHst: {}'.format(os.path.basename(save_filepath)))
        if not val_type:
            val_type = ''
        plot_title = '{} ({})'.format(
            val_type, mrt.naming.filename2label(img_filepath, max_length=32))
        mrt.input_output.plot_histogram1d(
            img_filepath, mask_filepath, hist_interval=(0.0, 1.0), bins=1024,
            array_interval=val_interval, title=plot_title,
            labels=(val_units, None),
            save_filepath=save_filepath, ax=None, force=force, verbose=verbose)


# ======================================================================
def plot_sample(
        img_filepath,
        val_type,
        val_interval,
        val_units,
        out_dirpath,
        out_filepath_prefix='sample',
        axis=2,
        index=None,
        force=False,
        verbose=D_VERB_LVL):
    """
    Plot a sample of the image.

    Parameters
    ==========
    img_filepath : str
        Path to image file to use for plotting.
    val_type : str
        Name of the data to be processed.
    val_interval : float 2-tuple
        Interval of the data to be processed.
    val_units : str
        Units of measurements of the data to be processed.
    out_dirpath : str
        Path to directory where to store results.
    out_filepath_prefix : str (optional)
        Prefix for output figure.
    axis : int (optional)
        Direction of the sampling.
    index : int or None (optional)
        Depth of the sampling. If None, mid-value is used.
    force : boolean (optional)
        Force calculation of output.
    verbose : int (optional)
        Set level of verbosity.

    Returns
    =======
    None.

    """
    save_filepath = os.path.join(
        out_dirpath,
        out_filepath_prefix + INFO_SEP +
        fc.base.change_ext(os.path.basename(img_filepath),
                           mrt.utils.EXT['plot'],
                           mrt.utils.EXT['niz']))
    if fc.base.check_redo([img_filepath], [save_filepath], force):
        msg('PltFig: {}'.format(os.path.basename(save_filepath)))
        if not val_type:
            val_type = 'Image'
        plot_title = '{} / {} ({})'.format(
            val_type, val_units,
            mrt.naming.filename2label(img_filepath, max_length=32))
        mrt.input_output.plot_sample2d(
            img_filepath, axis, index, title=plot_title,
            array_interval=val_interval,
            cbar_kws={},
            save_filepath=save_filepath, ax=None, force=force, verbose=verbose)


# ======================================================================
def registering(
        in_filepath_list,
        ref_filepath=None,
        ref_mask_filepath=None,
        out_dirpath='registration',
        register_func=register,
        register_args=None,
        register_kws=None,
        log_filename='registration.log',
        use_mp=True,
        force=False,
        verbose=D_VERB_LVL):
    """
    Register images to reference.

    Args:
        in_filepath_list (list[str]): Input filepath list
        ref_filepath (str): Path to the registration reference image file.
            If None, use first entry of input filepath list
        ref_mask_filepath (str): Path to the mask for the reference image file.
            If None, no mask is used.
        out_dirpath (str): Path to directory where to store results
        register_func (func): Function to be used for registration
        register_args (list): Positional parameters passed to register_func
        register_kws (dict): Keyword parameters passed to register_func
        log_filename (str): Path to file where to log registration steps
        use_mp (bool): Use multiprocessing for faster computation
        force (bool): Force calculation of output
        verbose (int): Set level of verbosity

    Returns:
        out_filepath_list (list[str]): Output (registered) filepath list.

    """
    # TODO: improve registration procedure and flexibility: fsl -> ANTS?

    # ensure existing output path
    if not os.path.exists(out_dirpath):
        os.makedirs(out_dirpath)
    # select a reference filepath if it does not exists
    if not ref_filepath:
        ref_filepath = in_filepath_list[0]
    # log the name of the reference image
    log_filepath = os.path.join(out_dirpath, log_filename)
    if fc.base.check_redo(
            in_filepath_list + [ref_filepath], [log_filepath], force):
        with open(log_filepath, 'w') as log_file:
            log_file.write(ref_filepath)
            log_file.close()
    # set up multiprocess framework
    if use_mp:
        # parallel
        n_proc = multiprocessing.cpu_count() + 1
        pool = multiprocessing.Pool(processes=n_proc)
        proc_result_list = []
    register_args = tuple(register_args) if register_args else ()
    register_kws = dict(register_kws) if register_kws else {}
    # perform registration
    out_filepath_list = []
    for in_filepath in in_filepath_list:
        out_filepath = os.path.join(
            out_dirpath, os.path.basename(in_filepath))
        out_filepath_list.append(out_filepath)
        if ref_filepath != in_filepath:
            register_kws.update({
                'in_filepath': in_filepath,
                'ref_filepath': ref_filepath,
                'out_filepath': out_filepath,
                'force': force,
                'verbose': verbose})
            if use_mp:
                # parallel
                proc_result = pool.apply_async(
                    register_func, register_args, register_kws)
                proc_result_list.append(proc_result)
            else:
                # serial
                register_func(*register_args, **register_kws)
        else:
            if fc.base.check_redo([in_filepath], [out_filepath], force):
                msg('Regstr: {}'.format(os.path.basename(out_filepath)))
                msg('I: copying without registering.')
                shutil.copy(in_filepath, out_filepath)
    if use_mp:
        res_list = []
        for proc_result in proc_result_list:
            res_list.append(proc_result.get())
    return out_filepath_list


# ======================================================================
def masking(
        in_filepath_list,
        mask_filepath,
        out_dirpath='masking',
        mask_val=0.0,
        use_mp=True,
        force=False,
        verbose=D_VERB_LVL):
    """
    Mask input images using selected mask file.

    Parameters:
        in_filepath_list (list[string]): List of filepaths used as input.
        mask_filepath : str
            Path to mask image file.
        out_dirpath : str (optional)
            Path to directory where to store results.
        mask_val : int or float
            Value of masked out voxels.
        use_mp : boolean (optional)
            Use multiprocessing for faster computation.
        force : boolean (optional)
            Force calculation of output.
        verbose : int (optional)
            Set level of verbosity.

    Returns:
        out_filepath_list (list[str]): List of path to output files.
    """
    # ensure existing output path
    if not os.path.exists(out_dirpath):
        os.makedirs(out_dirpath)
    if use_mp:
        # parallel
        n_proc = multiprocessing.cpu_count() + 1
        pool = multiprocessing.Pool(processes=n_proc)
        proc_result_list = []
    out_filepath_list = []
    for in_filepath in in_filepath_list:
        out_filepath = os.path.join(out_dirpath, os.path.basename(in_filepath))
        out_filepath_list.append(out_filepath)
        apply_mask_args = []
        apply_mask_kws = {
            'in_filepath': in_filepath,
            'mask_filepath': mask_filepath,
            'out_filepath': out_filepath,
            'mask_val': mask_val,
            'force': force,
            'verbose': verbose,
        }
        if use_mp:
            # parallel
            proc_result = pool.apply_async(
                apply_mask, apply_mask_args, apply_mask_kws)
            proc_result_list.append(proc_result)
        else:
            # serial
            apply_mask(**apply_mask_kws)
    if use_mp:
        res_list = []
        for proc_result in proc_result_list:
            res_list.append(proc_result.get())
    return out_filepath_list


# ======================================================================
def prepare_comparison(
        in_filepaths,
        ref_filepaths,
        out_dirpath,
        skip_equal=True,
        skip_symmetric=False,
        diff_prefix='diff',
        corr_prefix='corr'):
    """
    Get list items to be compared.

    Args:
        in_filepaths (list[str]): List of filepaths used as input.
        ref_filepaths (list[str]): List of filepaths used as reference.
        out_dirpath (str):
        skip_equal (bool):
        skip_symmetric (bool):
        diff_prefix (str):
        corr_prefix (str):

    Returns:

    """

    def _symmetric(item1, item2, source):
        result = False
        for item in source:
            tmp1, tmp2 = item[:2]
            if tmp1 == item2 and tmp2 == item1:
                result = True
                break
        return result

    cmp_list = []
    combinator = itertools.product(ref_filepaths, in_filepaths)
    for ref_filepath, in_filepath in combinator:
        if skip_equal and in_filepath == ref_filepath:
            continue
        if skip_symmetric and _symmetric(in_filepath, ref_filepath, cmp_list):
            continue
        diff_filepath = os.path.join(
            out_dirpath,
            mrt.naming.combine_filename(diff_prefix,
                                        (ref_filepath, in_filepath)) +
            fc.base.add_extsep(mrt.utils.EXT['niz']))
        corr_filepath = os.path.join(
            out_dirpath,
            mrt.naming.combine_filename(corr_prefix,
                                        (ref_filepath, in_filepath)) +
            fc.base.add_extsep(mrt.utils.EXT['tab']))
        cmp_list.append(
            (in_filepath, ref_filepath, diff_filepath, corr_filepath))
    return cmp_list


# ======================================================================
def comparing(
        in_filepaths,
        ref_filepaths=None,
        out_dirpath='comparing',
        mask_filepath=None,
        skip_equal=True,
        removes=None,
        val_interval=None,
        trunc=None,
        diff_prefix='diff',
        corr_prefix='corr',
        use_mp=True,
        force=False,
        verbose=D_VERB_LVL):
    """
    Compare input files to reference files.
    Calculate difference and correlation coefficients.

    Args:
        in_filepaths (list[str]): List of filepaths used as input.
        ref_filepaths (list[str]): List of filepaths used as reference.
        out_dirpath (str): Subpath where to store results.
        mask_filepath (str): Path to mask image file.
        skip_equal (bool): Skip comparison if input and reference are equal.
        mask_nan (bool): Ignore NaN values during comparison.
        mask_inf (bool): Ignore Inf values during comparison.
        mask_vals : list of int or float
            List of values to mask.
        val_interval : 2-tuple (optional)
            The (min, max) values range.
        trunc : int or None (optional)
            Defines the maximum length to be used by numeric values in output.
        diff_prefix :
            Prefix to use for difference files.
        corr_prefix :
            Prefix to use for correlation files.
        use_mp : boolean (optional)
            Use multiprocessing for faster computation.
        force : boolean (optional)
            Force calculation of output.
        verbose : int (optional)
            Set level of verbosity.

    Returns:
        cmp_list (list[list[str]]):
    """
    # ensure existing output path
    if not os.path.exists(out_dirpath):
        os.makedirs(out_dirpath)
    if use_mp:
        # parallel
        n_proc = multiprocessing.cpu_count() + 1
        pool = multiprocessing.Pool(processes=n_proc)
        proc_result_list = []
    cmp_list = prepare_comparison(
        in_filepaths, ref_filepaths, out_dirpath,
        skip_equal=True, skip_symmetric=True,
        diff_prefix='diff', corr_prefix='corr')
    for in_filepath, ref_filepath, diff_filepath, corr_filepath in cmp_list:
        if not removes:
            removes = [np.nan, np.inf, -np.inf, 0.0]
        if use_mp:
            # parallel
            # calc difference
            proc_result = pool.apply_async(
                calc_difference,
                (in_filepath, ref_filepath, diff_filepath, force, verbose))
            proc_result_list.append(proc_result)
            # calc correlation
            proc_result = pool.apply_async(
                calc_correlation,
                (in_filepath, ref_filepath, corr_filepath, mask_filepath,
                 removes, val_interval, trunc,
                 force, verbose))
            proc_result_list.append(proc_result)
        else:
            # serial
            calc_difference(
                in_filepath, ref_filepath, diff_filepath, force, verbose)
            calc_correlation(
                in_filepath, ref_filepath, corr_filepath, mask_filepath,
                removes, val_interval, trunc,
                force, verbose)
    if use_mp:
        res_list = []
        for proc_result in proc_result_list:
            res_list.append(proc_result.get())
    return cmp_list


# ======================================================================
def check_correlation(
        dirpath,
        val_name=None,
        val_interval=None,
        val_units=None,
        mask_filepath=None,
        reg_ref_ext=EXT['reg_ref'],
        corr_ref_ext=EXT['corr_ref'],
        tmp_dir='tmp',
        reg_dir='reg',
        msk_dir='msk',
        cmp_dir='cmp',
        fig_dir='fig',
        force=False,
        verbose=D_VERB_LVL):
    """
    Check the voxel correlation for a list of homogeneous images.

    Args:
        dirpath (str): Path to directory to process.
        val_name (str): Name of the data to be processed.
        val_interval (tuple[float]): Interval (min, max) of the processed data.
        val_units (str): Units of measurements of the data to be processed.
        mask_filepath (str): Path to mask file. If None, extract it.
        reg_ref_ext (str): File extension of registration reference.
        corr_ref_ext (str): File extension of correlation reference(s).
        tmp_dir (str): Subpath where to store temporary files.
        reg_dir (str): Subpath where to store registration files.
        msk_dir (str): Subpath where to store registration files.
        cmp_dir (str): Subpath where to store masking files.
        fig_dir (str): Subpath where to store plotting files (i.e. figures).
        force (bool): Force calculation of output.
        verbose (int): Set level of verbosity.

    Returns:
        targets (list[str]): List of processed image files.
        corrs (list[str]): List of files containing correlation
        computations
    """
    msg('Target: {}'.format(dirpath))
    # :: manage image type, inteval and units
    if not val_name:
        val_name = os.path.split(dirpath)[-1]
        if val_name not in SRC_IMG_TYPE:
            val_name = None
        msg('W: image type not specified.', verbose, VERB_LVL['medium'])
        msg('I: guessed image type: {}'.format(val_name),
            verbose, VERB_LVL['high'])
    else:
        msg('I: {} {} {}'.format(val_name, val_interval, val_units),
            verbose, VERB_LVL['medium'])
    if not val_interval:
        msg('W: values interval not specified.', verbose, VERB_LVL['medium'])
        msg('I: values interval guessed from image-specific values.',
            verbose, VERB_LVL['high'])
    else:
        val_interval = sorted(val_interval)
        if np.ptp(val_interval) == 0.0:
            raise ValueError('Values interval has size 0. Aborting!')
    if not val_units:
        val_units = 'a.u.'
        msg('W: values units not specified.', verbose, VERB_LVL['medium'])
        msg('I: guessed image type: {}'.format(val_name),
            verbose, VERB_LVL['high'])
    # :: populate a list of images to analyze
    targets, corrs = [], []
    if os.path.exists(dirpath):
        filepath_list = fc.base.listdir(dirpath, mrt.utils.EXT['niz'])
        sources = [fc.base.realpath(filepath) for filepath in filepath_list
                   if not val_name or
                   mrt.naming.parse_filename(filepath)['type'] == val_name]
        if len(sources) > 0:
            # :: create output directories
            # NOTE: use tmp/reg/msk/cmp/fig_path in code
            path_list = []
            for subdir in [tmp_dir, reg_dir, msk_dir, cmp_dir, fig_dir]:
                if subdir:
                    subdir = os.path.join(dirpath, subdir)
                path_list.append(subdir)
            tmp_path, reg_path, msk_path, cmp_path, fig_path = path_list
            for path in path_list:
                if path:
                    if not os.path.exists(path):
                        os.makedirs(path)

            # :: ensure the presence of a reference file
            ref_list, ref_src_list = \
                _get_ref_list(dirpath, sources, None, reg_ref_ext)
            ref = ref_list[0]
            ref_src = ref_src_list[0] if len(ref_src_list) > 0 else ''
            # registration instructions
            msg('I: RegRef: {}'.format(ref_src if ref_src else None),
                verbose, VERB_LVL['medium'])
            try:
                with open(ref_src, 'r') as ref_file:
                    reg_info = json.load(ref_file)
            except (IOError, ValueError) as e_msg:
                msg('W: loading JSON: {}'.format(e_msg))
                reg_info = {}
            msg('I: RegRefInfo: {}'.format(reg_info),
                verbose, VERB_LVL['medium'])

            # ensure the presence of a mask
            if msk_dir:
                # if mask_filepath was not specified, set up a new name
                if not mask_filepath:
                    mask_filepath = fc.base.change_ext(
                        MASK_FILENAME, mrt.utils.EXT['niz'])
                # add current directory if it was not specified
                if not os.path.isfile(mask_filepath):
                    mask_filepath = os.path.join(dirpath, mask_filepath)
                # if mask not found, create one from registration reference
                if not os.path.isfile(mask_filepath):
                    if 'calc_mask' not in reg_info:
                        reg_info['calc_mask'] = {}
                    mask_filepath = calc_mask(
                        ref, tmp_path, verbose=verbose, force=force,
                        **reg_info['calc_mask'])
                mask_filepath = fc.base.realpath(mask_filepath)
            # else:
            #     mask_filepath = None
            msg('I: {}'.format(mask_filepath), verbose, VERB_LVL['high'])
            msg('I: {}'.format(sources), verbose, VERB_LVL['high'])
            if mask_filepath in sources:
                sources.remove(mask_filepath)
            msg('I: mask: {}'.format(mask_filepath),
                verbose, VERB_LVL['medium'])
            # :: co-register targets
            if reg_dir:
                if 'func_register' not in reg_info:
                    reg_info['func_register'] = 'register'
                if reg_info['func_register'] not in reg_info:
                    reg_info[reg_info['func_register']] = {}
                targets = registering(
                    sources, ref, mask_filepath, reg_path,
                    register_func=eval(reg_info['func_register']),
                    register_args=(),
                    register_kwargs=reg_info[reg_info['func_register']],
                    use_mp=False, force=force, verbose=verbose)
            else:
                targets = sources
            # :: mask targets
            if msk_path:
                targets = masking(
                    targets, mask_filepath, msk_path, use_mp=False,
                    force=force, verbose=verbose)
                # make sure the mask has correct shape
                new_mask = os.path.join(
                    msk_path, os.path.basename(mask_filepath))
                msg('I: newly shaped mask: {}'.format(new_mask),
                    verbose, VERB_LVL['medium'])
                apply_mask(mask_filepath, mask_filepath, new_mask)
                mask_filepath = new_mask
            # perform comparison
            if cmp_path:
                ref_list, ref_src_list = _get_ref_list(
                    dirpath, targets, msk_dir, corr_ref_ext)
                cmp_list = comparing(
                    targets, ref_list, cmp_path, mask_filepath,
                    use_mp=False, val_interval=val_interval,
                    force=force, verbose=verbose)
                # group resulting correlations
                corrs = [item[3] for item in cmp_list]
                combine_correlation(
                    corrs, cmp_path, force=force, verbose=verbose)
            # plotting
            if fig_path:
                for target in targets:
                    plot_sample(
                        target, val_name, val_interval, val_units, fig_path,
                        force=force, verbose=verbose)
                    plot_histogram(
                        target, mask_filepath, val_name, val_interval,
                        val_units, fig_path,
                        force=force, verbose=verbose)
                # use last plotted image to calculate approximate diff_interval
                if val_interval is None:
                    stats_dict = mrt.input_output.calc_stats(target)
                    val_interval = (stats_dict['min'], stats_dict['max'])
                diff_interval = fc.extra.combine_interval(
                    val_interval, val_interval, '-')
                for in_filepath, ref_filepath, diff_filepath, corr_filepath \
                        in cmp_list:
                    plot_sample(
                        diff_filepath, val_name, diff_interval, val_units,
                        fig_path,
                        force=force, verbose=verbose)
                    plot_histogram(
                        diff_filepath, mask_filepath,
                        val_name, diff_interval, val_units, fig_path,
                        force=force, verbose=verbose)
                    plot_correlation(
                        in_filepath, ref_filepath, mask_filepath,
                        val_name, val_interval, val_units,
                        fig_path,
                        force=force, verbose=verbose)
            msg('Done:   {}'.format(dirpath))
        else:
            msg('W: no input file found.', verbose, VERB_LVL['medium'])
            msg('I: descending into subdirectories.', verbose,
                VERB_LVL['high'])
            sub_dirpath_list = fc.base.listdir(dirpath, None)
            for sub_dirpath in sub_dirpath_list:
                tmp_target_list, tmp_corr_list = check_correlation(
                    sub_dirpath,
                    val_name, val_interval, val_units,
                    mask_filepath,
                    reg_ref_ext, corr_ref_ext,
                    tmp_dir, reg_dir, msk_dir, cmp_dir, fig_dir,
                    force, verbose)
                targets += tmp_target_list
                corrs += tmp_corr_list
            # group resulting correlations
            if corrs:
                combine_correlation(
                    corrs, dirpath, force=force, verbose=verbose)
    return targets, corrs


# ======================================================================
if __name__ == '__main__':
    msg(__doc__.strip())
