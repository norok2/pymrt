#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
mri_tools: voxel-by-voxel correlation analysis for MRI data.
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
import shutil  # High-level file operations
# import math  # Mathematical functions
# import time  # Time access and conversions
# import datetime  # Basic date and time types
# import operator  # Standard operators as functions
# import collections  # High-performance container datatypes
import itertools  # Functions creating iterators for efficient looping
# import functools  # Higher-order functions and operations on callable objects
# import argparse  # Parser for command-line options, arguments and subcommands
# import re  # Regular expression operations
# import subprocess  # Subprocess management
import multiprocessing  # Process-based parallelism
# import inspect  # Inspect live objects
import csv  # CSV File Reading and Writing [CSV: Comma-Separated Values]
# import json  # JSON encoder and decoder [JSON: JavaScript Object Notation]


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

# :: External Imports Submodules
import matplotlib.pyplot as plt  # Matplotlib's pyplot: MATLAB-like syntax
# import mayavi.mlab as mlab  # Mayavi's mlab: MATLAB-like syntax
# import scipy.constants  # SciPy: Mathematal and Physical Constants
import scipy.ndimage  # SciPy: ND-image Manipulation
import scipy.stats  # SciPy: Statistical functions
# :: Local Imports
import mri_tools.modules.base as mrb
import mri_tools.modules.utils as mru
import mri_tools.modules.geometry as mrg
# import mri_tools.modules.plot as mrp
import mri_tools.modules.registration as mrr
import mri_tools.modules.segmentation as mrs
# import mri_tools.modules.computation as mrc
# import mri_tools.modules.correlation as mrl
import mri_tools.modules.nifti as mrn
# import mri_tools.modules.sequences as mrq
# from mri_tools.modules.debug import dbg
# from mri_tools.modules.sequences import mp2rage

from mri_tools import VERB_LVL
from mri_tools import D_VERB_LVL
from mri_tools.config import EXT_CMD

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
    'm0': 'MO'}

# magnitude/phase image type identifiers
TYPE_ID = {
    'mag': 'MAG',
    'phs': 'PHS',
    'real': 'RE',
    'imag': 'IM',
    'complex': 'CX',
    'temp': 'TMP',
    'none': None}

# service image type identifiers
SERVICE_ID = {
    'helper': 'HLP',  # TODO: different support for registr. helpers and masks
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
SRC_IMG_TYPE = set(MAP_ID.values() + [TYPE_ID['none']])
KNOWN_IMG_TYPES = set(
    MAP_ID.values() + TYPE_ID.values() + SERVICE_ID.values() +
    MP2RAGE_ID.values())

# suffix of new reconstructed image from Siemens
NEW_RECO_ID = 'rr'
SERIES_NUM_ID = 's'

# prefixes for target files
MASK_FILENAME = 'mask'

D_EXT = {
    'registration reference': '0_REG_REF',
    'correlation reference': '0_CORR_REF'}


# ======================================================================
def get_ref_list(
        dirpath,
        target_list=None,
        subdir=None,
        ref_ext=''):
    """
    Find reference NIfTI image files in a directory.
    If no reference file is found, use files from target list as reference.
    A file is flagged as reference if there exist a file with the same name,
    but filename extension as specified by ref_ext.

    Parameters
    ==========
    dirpath : str
        Path where to look for reference files.
    target_list : list of string (optional)
        List of possible reference files. If search fails, use first of these.
    ref_ext : str
        Filename extension of the reference file flag.

    Returns
    =======
    ref_filepath_list : list of string or string
        List of path to reference files.

    """
    ref_flag_list = mrb.listdir(dirpath, ref_ext)
    if ref_flag_list:
        ref_filepath_list = []
        for ref_flag in ref_flag_list:
            # extract dirpath
            ref_dirpath = os.path.dirname(ref_flag)
            if subdir:
                ref_dirpath = os.path.join(ref_dirpath, subdir)
            # extract filename
            ref_filename = mrn.filename_addext(
                os.path.basename(ref_flag)[:-len(mrb.add_extsep(ref_ext))])
            ref_filepath = os.path.join(ref_dirpath, ref_filename)
            ref_filepath_list.append(ref_filepath)
    elif target_list:
        ref_filepath_list = target_list
    else:
        ref_filepath_list = mrb.listdir(dirpath, mrn.D_EXT)
    if not ref_filepath_list:
        raise RuntimeError('Could not find any suitable reference file.')
    return ref_filepath_list


# ======================================================================
def compute_affine_fsl(
        in_filepath,
        ref_filepath,
        aff_filepath,
        msk_filepath=None,
        method='corratio',
        dof=12,
        force=False,
        verbose=D_VERB_LVL):
    """
    Calculate registration matrix.

    Parameters
    ==========
    in_filepath : str
        Path to input image.
    ref_filepath : str
        Path to reference image.
    aff_filepath : str
        Path to file where to store registration matrix.
    dof : int (optional)
        | Number of degrees of freedom of the affine transformation
        | 6: only rotations and translations (rigid body)
        | 9: only rotations, translations and magnification
        | 12: all possible linear transformations
    force : boolean (optional)
        Force calculation of output.
    verbose : int (optional)
        Set level of verbosity.

    Returns
    =======
    None.

    """
    if mrb.check_redo([in_filepath, ref_filepath], [aff_filepath], force):
        if verbose > VERB_LVL['none']:
            print('Affine:\t{}'.format(os.path.basename(aff_filepath)))
        ext_cmd = EXT_CMD['fsl/4.1/flirt']
        cmd_args = {
            'in': in_filepath,
            'ref': ref_filepath,
            'omat': aff_filepath,
            'refweight': msk_filepath if msk_filepath else '',
            'dof': dof,
            'cost': method,
            'searchrx': '-180 180',
            'searchry': '-180 180',
            'searchrz': '-180 180',
        }
        cmd = ' '.join(
            [ext_cmd] + ['-{} {}'.format(k, v) for k, v in cmd_args.items()])
        if verbose >= VERB_LVL['high']:
            print('> ', cmd)
        p_stdout, p_stderr = mrb.execute(cmd)
        if verbose >= VERB_LVL['debug']:
            print(p_stdout)
            print(p_stderr)


# ======================================================================
def apply_affine_fsl(
        in_filepath,
        ref_filepath,
        out_filepath,
        aff_filepath,
        force=False,
        verbose=D_VERB_LVL):
    """
    Apply registration matrix.

    Parameters
    ==========
    in_filepath : str
        Path to input file.
    ref_filepath : str
        Path to reference file.
    out_filepath : str
        Path to output file.
    regmat_filepath : str
        Path to registration matrix file.
    force : boolean (optional)
        Force calculation of output.
    verbose : int (optional)
        Set level of verbosity.

    Returns
    =======
    None.

    """
    if mrb.check_redo(
            [in_filepath, ref_filepath, aff_filepath], [out_filepath],
            force):
        if verbose > VERB_LVL['none']:
            print('Regstr:\t{}'.format(os.path.basename(out_filepath)))
        ext_cmd = EXT_CMD['fsl/4.1/flirt']
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
        if verbose >= VERB_LVL['high']:
            print('> ', cmd)
        mrb.execute(cmd)


# ======================================================================
def register(
        in_filepath,
        ref_filepath,
        out_filepath,
        force=False,
        verbose=D_VERB_LVL):
    """
    Register input file to reference.

    Parameters
    ==========
    in_filepath : str
        Path to input file.
    ref_filepath : str
        Path to reference file.
    out_filepath : str
        Path to output file.
    aff_filepath : str
        Path to registration matrix file.
    ref_mask_filepath : str
    force : boolean (optional)
        Force calculation of output.
    verbose : int (optional)
        Set level of verbosity.

    Returns
    =======
    None.

    """

    def _quick_reg(array_list, *args, **kwargs):
        img = array_list[0]
        ref = array_list[1]
        # # at first translate...
        # linear, shift = mrr.affine_registration(
        #     img, ref, transform='translation',
        #     init_guess=('weights', 'weights'))
        # # print(shift)
        # img = mrg.affine_transform(img, linear, shift)
        # ... then reorient
        linear, shift = mrr.affine_registration(
            img, ref, transform='reflection_simple')
        img = mrg.affine_transform(img, linear, shift)
        # ... and finally perform finer registration
        linear, shift = mrr.affine_registration(img, ref, *args, **kwargs)
        img = mrg.affine_transform(img, linear, shift)
        return img

    mrn.simple_filter_n(
        [in_filepath, ref_filepath], out_filepath, _quick_reg,
        transform='rigid', interp_order=1, init_guess=('none', 'none'))




    if verbose > VERB_LVL['none']:
        print('Regstr:\t{}'.format(os.path.basename(out_filepath)))


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

    Parameters
    ==========
    in_filepath : str
        Path to input file.
    mask_filepath : str
        Path to reference file.
    out_filepath : str
        Path to output file.
    mask_val : int or float
        Value of masked out voxels.
    force : boolean (optional)
        Force calculation of output.
    verbose : int (optional)
        Set level of verbosity.

    Returns
    =======
    None.

    """
    def _mask_reframe(arr_list, mask_val):
            img, mask = arr_list
            container = sp.ndimage.find_objects(mask.astype(int))[0]
            img[~mask.astype(bool)] = mask_val
            img = img[container]
            img = mrg.frame(img, 0.1, 0.0)
            return img

    if mrb.check_redo([in_filepath, mask_filepath], [out_filepath], force):
        if verbose > VERB_LVL['none']:
            print('RunMsk:\t{}'.format(os.path.basename(out_filepath)))
        mrn.simple_filter_n(
            [in_filepath, mask_filepath], out_filepath,
            _mask_reframe, mask_val)


# ======================================================================
def calc_mask(
        in_filepath,
        out_filepath=None,
        threshold=0.01,
        comparison='>',
        mode='relative',
        smoothing=0.0,
        erosion_iter=0,
        dilation_iter=0,
        bet_params='',
        force=False,
        verbose=D_VERB_LVL):
    """
    Extract a mask from an image.

    FSL's BET brain extraction algorithm can be used.

    | Workflow is:
    * Brain extraction with FSL's BET (if any)
    * Extract mask using mri_tools.modules.geometry algorithm

    Parameters
    ==========
    in_filepath : str
        Path to image from which mask is to be extracted.
    out_filepath : str (optional)
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

    def _calc_mask(
            array,
            threshold,
            comparison,
            mode,
            smoothing,
            erosion_iter,
            dilation_iter):
        # :: preprocess
        if smoothing > 0.0:
            array = sp.ndimage.gaussian_filter(array, smoothing)
        # :: masking
        array = mrs.mask_threshold(array, threshold, comparison, mode)
        # :: postprocess
        if erosion_iter > 0:
            array = sp.ndimage.binary_erosion(array, iterations=erosion_iter)
        if dilation_iter > 0:
            array = sp.ndimage.binary_dilation(array, iterations=dilation_iter)
        return array.astype(float)

    # todo: move to mrn?
    if not out_filepath:
        out_filepath = os.path.dirname(in_filepath)
    if os.path.isdir(out_filepath):
        out_filename = os.path.basename(
            mru.change_img_type(in_filepath, SERVICE_ID['mask']))
        out_filepath = os.path.join(out_filepath, out_filename)
    if bet_params:
        ext_cmd = EXT_CMD['']
        tmp_filepath = mru.change_img_type(out_filepath, 'BRAIN')
        if mrb.check_redo([in_filepath], [tmp_filepath], force):
            if verbose > VERB_LVL['none']:
                print('TmpMsk:\t{}'.format(os.path.basename(tmp_filepath)))
            cmd_tokens = [ext_cmd, in_filepath, tmp_filepath, bet_params]
            cmd = ' '.join(cmd_tokens)
            if verbose >= VERB_LVL['high']:
                print('> ', cmd)
            p_stdout, p_stderr = mrb.execute(cmd)
            if verbose >= VERB_LVL['debug']:
                print(p_stdout)
                print(p_stderr)
    else:
        tmp_filepath = in_filepath
    # extract mask using a threshold
    if mrb.check_redo([tmp_filepath], [out_filepath], force):
        params = [
            threshold, comparison, mode,
            smoothing,
            erosion_iter, dilation_iter]
        if verbose > VERB_LVL['none']:
            print('GetMsk:\t{}'.format(os.path.basename(out_filepath)))
        mrn.simple_filter(
            tmp_filepath, out_filepath, _calc_mask, *params)
    return out_filepath


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
    if mrb.check_redo([in1_filepath, in2_filepath], [out_filepath], force):
        if verbose > VERB_LVL['none']:
            print('DifImg:\t{}'.format(os.path.basename(out_filepath)))
        mrn.simple_filter_n(
            [in1_filepath, in2_filepath], out_filepath,
            (lambda images: images[1] - images[0]))


# ======================================================================
def calc_correlation(
        in1_filepath,
        in2_filepath,
        out_filepath,
        mask_filepath=None,
        mask_nan=True,
        mask_inf=True,
        mask_val_list=None,
        val_range=None,
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
    mask_val_list : list of int or float
        List of values to mask.
    val_range : 2-tuple (optional)
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
    if mrb.check_redo(in_filepath_list, [out_filepath], force):
        if verbose > VERB_LVL['none']:
            print('Correl:\t{}'.format(os.path.basename(out_filepath)))
        img1_nii = nib.load(in1_filepath)
        img2_nii = nib.load(in2_filepath)
        img1 = img1_nii.get_data()
        img2 = img2_nii.get_data()
        if verbose >= VERB_LVL['high']:
            print('Mask:\t{}'.format(os.path.basename(mask_filepath)))
        if mask_filepath:
            mask_nii = nib.load(mask_filepath)
            mask = mask_nii.get_data().astype(bool)
        else:
            mask = np.ones_like(img1 * img2).astype(bool)
        mask *= (img1 > val_range[0]).astype(bool)
        mask *= (img1 < val_range[1]).astype(bool)
        mask *= (img2 > val_range[0]).astype(bool)
        mask *= (img2 < val_range[1]).astype(bool)
        if not mask_val_list:
            mask_val_list = []
        # calculate stats of difference image
        d_arr = img1[mask] - img2[mask]
        d_dict = mrb.calc_stats(
            d_arr, mask_nan, mask_inf, mask_val_list)
        # calculate stats of the absolute difference image
        e_arr = np.abs(d_arr)
        e_dict = mrb.calc_stats(
            e_arr, mask_nan, mask_inf, mask_val_list)
        # calculate Pearson's Correlation Coefficient
        pcc_val, pcc_p_val = \
            sp.stats.pearsonr(img1[mask].ravel(), img2[mask].ravel())
        pcc2_val = pcc_val * pcc_val
        # calculate linear polynomial fit
        linear_coeff, linear_offset = np.polyfit(
            img1[mask].ravel(), img2[mask].ravel(), 1)
        #        # calculate Theil robust slope estimator (WARNING: too much
        #  memory!)
        #        theil_coeff, theil_offset, = sp.stats.mstats.theilslopes(
        #            img2[mask].ravel(),
        # img1[mask].ravel())
        # voxel counts
        num = np.sum(mask.astype(bool))
        num_tot = np.size(mask)
        num_ratio = num / num_tot
        # save results to csv
        filename_max_len = max([len(mrn.filename_noext(os.path.basename(path)))
                                for path in [in2_filepath, in1_filepath]])
        label_list = ['avg', 'std', 'min', 'max', 'sum']
        val_filter = (lambda x: mrb.compact_num_str(x, trunc)) \
            if trunc else (lambda x: x)
        d_arr_val = [val_filter(d_dict[key]) for key in label_list]
        e_arr_val = [val_filter(e_dict[key]) for key in label_list]
        values = \
            ['{: <{size}s}'.format(
                mrn.filename_noext(os.path.basename(path)),
                size=filename_max_len)
             for path in [in2_filepath, in1_filepath]] + \
            list(d_arr_val) + list(e_arr_val) + \
            [val_filter(val)
             for val in [pcc_val, pcc2_val, pcc_p_val,
                         linear_coeff, linear_offset,
                         #                theil_coeff, theil_offset,
                         num, num_tot, num_ratio]]
        labels = \
            ['{: <{size}s}'.format('x_corr_file', size=filename_max_len),
             '{: <{size}s}'.format('y_corr_file', size=filename_max_len)] + \
            ['D-' + lbl for lbl in label_list] + \
            ['E-' + lbl for lbl in label_list] + \
            ['r', 'r2', 'p-val',
             'lin-cof', 'lin-off',
             #             'thl-cof', 'thl-off',
             'N_eff', 'N_tot', 'N_ratio']
        with open(out_filepath, 'wb') as csvfile:
            csvwriter = csv.writer(csvfile, delimiter=str(mrb.CSV_DELIMITER))
            csvwriter.writerow([mrb.COMMENT_TOKEN + in2_filepath])
            csvwriter.writerow([mrb.COMMENT_TOKEN + in1_filepath])
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
        with open(filepath, 'rb') as csvfile:
            csvreader = csv.reader(csvfile, delimiter=str(mrb.CSV_DELIMITER))
            for row in csvreader:
                if row[0].startswith(mrb.COMMENT_TOKEN):
                    base_dir = os.path.dirname(os.path.commonprefix(
                        (base_dir, row[0]))) + os.path.sep \
                        if base_dir else row[0]
    # :: summarize correlation results
    labels, rows, max_cols = [], [], []
    for filepath in filepath_list:
        with open(filepath, 'rb') as csvfile:
            csvreader = csv.reader(csvfile, delimiter=str(mrb.CSV_DELIMITER))
            for row in csvreader:
                if row[0].startswith(mrb.COMMENT_TOKEN):
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
                    for idx, item in enumerate(data):
                        if max_cols[idx] < len(item):
                            max_cols[idx] = len(item)
                    rows.append(data)
    # :: fix column width
    for idx, label in enumerate(labels):
        labels[idx] = '{: <{size}s}'.format(label, size=max_cols[idx])
    for jdx, row in enumerate(rows):
        for idx, col in enumerate(row):
            rows[jdx][idx] = '{: <{size}s}'.format(col, size=max_cols[idx])
    # :: write grouped correlation to new file
    out_filepath = os.path.join(
        out_dirpath, out_filename + mrb.add_extsep(mrb.CSV_EXT))
    if mrb.check_redo(filepath_list, [out_filepath], force):
        with open(out_filepath, 'wb') as csvfile:
            csvwriter = csv.writer(csvfile, delimiter=str(mrb.CSV_DELIMITER))
            if not selected_cols:
                selected_cols = range(len(labels))
            csvwriter.writerow([base_dir])
            csvwriter.writerow(
                [item for idx, item in enumerate(labels)
                 if idx in selected_cols])
            for row in rows:
                csvwriter.writerow(
                    [col for idx, col in enumerate(row)
                     if idx in selected_cols])
    return out_filepath


# ======================================================================
def plot_correlation(
        img1_filepath,
        img2_filepath,
        mask_filepath,
        data_type,
        data_range,
        data_units,
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
    data_type : str
        Name of the data to be processed.
    data_range : float 2-tuple
        Range of the data to be processed.
    data_units : str
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
    save_path = os.path.join(
        out_dirpath, mru.combine_filename(
            corr_prefix, (img1_filepath, img2_filepath)) + mrb.add_extsep(
            mrb.PNG_EXT))
    in_filepath_list = [img1_filepath, img2_filepath]
    if mask_filepath:
        in_filepath_list.append(mask_filepath)
    if mrb.check_redo(in_filepath_list, [save_path], force):
        if verbose > VERB_LVL['none']:
            print('PltCor:\t{}'.format(os.path.basename(save_path)))
        img1_label = mru.filename2label(img1_filepath, max_length=32)
        img2_label = mru.filename2label(img2_filepath, max_length=32)
        title = 'Voxel-by-Voxel Correlation'
        if not data_type:
            data_type = 'Image'
        x_lbl = '{} / {} ({})'.format(data_type, data_units, img1_label)
        y_lbl = '{} / {} ({})'.format(data_type, data_units, img2_label)
        # plot the 2D histogram
        mrn.plot_histogram2d(
            img1_filepath, img2_filepath, mask_filepath, mask_filepath,
            hist_range=(0.0, 1.0), bins=512, array_range=data_range,
            scale='log10', title=title, cmap=plt.cm.hot_r,
            labels=(x_lbl, y_lbl), bisector=':k',
            colorbar_opts={},
            save_path=save_path, close_figure=not plt.isinteractive())


# ======================================================================
def plot_histogram(
        img_filepath,
        mask_filepath,
        data_type,
        data_range,
        data_units,
        out_dirpath,
        out_filepath_prefix='hist',
        force=False,
        verbose=D_VERB_LVL):
    """
    Plot the histogram of values for the image

    Parameters
    ==========
    img_filepath : str
        Path to image file to use for plotting.
    mask_filepath : str
        Path to mask file.
    data_type : str
        Name of the data to be processed.
    data_range : float 2-tuple
        Range of the data to be processed.
    data_units : str
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
    save_path = os.path.join(out_dirpath, out_filepath_prefix + INFO_SEP +
                             mrn.filename_noext(
                                 os.path.basename(img_filepath)) +
                             mrb.add_extsep(mrb.PNG_EXT))
    in_filepath_list = [img_filepath]
    if mask_filepath:
        in_filepath_list.append(mask_filepath)
    if mrb.check_redo(in_filepath_list, [save_path], force):
        if verbose > VERB_LVL['none']:
            print('PltHst:\t{}'.format(os.path.basename(save_path)))
        if not data_type:
            data_type = ''
        plot_title = '{} ({})'.format(
            data_type, mru.filename2label(img_filepath, max_length=32))
        mrn.plot_histogram1d(
            img_filepath, mask_filepath, hist_range=(0.0, 1.0), bins=1024,
            array_range=data_range, title=plot_title,
            labels=(data_units, None), save_path=save_path,
            close_figure=not plt.isinteractive())


# ======================================================================
def plot_sample(
        img_filepath,
        data_type,
        data_range,
        data_units,
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
    data_type : str
        Name of the data to be processed.
    data_range : float 2-tuple
        Range of the data to be processed.
    data_units : str
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
    save_path = os.path.join(out_dirpath, out_filepath_prefix + INFO_SEP +
                             mrn.filename_noext(
                                 os.path.basename(img_filepath)) +
                             mrb.add_extsep(mrb.PNG_EXT))
    if mrb.check_redo([img_filepath], [save_path], force):
        if verbose > VERB_LVL['none']:
            print('PltFig:\t{}'.format(os.path.basename(save_path)))
        if not data_type:
            data_type = 'Image'
        plot_title = '{} / {} ({})'.format(
            data_type, data_units,
            mru.filename2label(img_filepath, max_length=32))
        mrn.plot_sample2d(
            img_filepath, axis, index, title=plot_title, array_range=data_range,
            colorbar_opts={},
            close_figure=not plt.isinteractive(), save_path=save_path)


# ======================================================================
def registering(
        in_filepath_list,
        ref_filepath=None,
        ref_mask_filepath=None,
        out_dirpath='coregistration',
        info_file='registration_info.txt',
        options=6,
        regmat_prefix='regmat',
        use_mp=True,
        force=False,
        verbose=D_VERB_LVL):
    """
    Register images to reference.
    TODO: improve registration procedure and flexibility: fsl -> ANTS?

    Parameters
    ==========
    in_filepath_list : list of string
        List of filepaths used as input.
    ref_filepath : str (optional)
        Path to reference image file. If not set, first input file is used.
    out_dirpath : str (optional)
        Path to directory where to store results.
    use_mp : boolean (optional)
        Use multiprocessing for faster computation.
    force : boolean (optional)
        Force calculation of output.
    verbose : int (optional)
        Set level of verbosity.

    Returns
    =======
    out_filepath_list : list of string
        List of path to output files.

    """
    # ensure existing output path
    if not os.path.exists(out_dirpath):
        os.makedirs(out_dirpath)
    # select a reference filepath if it does not exists
    if not ref_filepath:
        ref_filepath = in_filepath_list[0]
    reginfo_filepath = os.path.join(out_dirpath, info_file)
    if mrb.check_redo(
                    in_filepath_list + [ref_filepath], [reginfo_filepath],
            force):
        out_file = open(reginfo_filepath, 'w')
        out_file.write(ref_filepath)
        out_file.close()
    if use_mp:
        # parallel
        n_proc = multiprocessing.cpu_count()
        pool = multiprocessing.Pool(processes=n_proc)
        proc_result_list = []
    out_filepath_list = []
    for in_filepath in in_filepath_list:
        regmat = os.path.join(
            out_dirpath,
            mru.combine_filename(regmat_prefix, (ref_filepath, in_filepath)) +
            mrb.add_extsep(mrb.TXT_EXT))
        out_filepath = os.path.join(
            out_dirpath, os.path.basename(in_filepath))
        out_filepath_list.append(out_filepath)
        if ref_filepath != in_filepath:
            if use_mp:
                # parallel
                proc_result = pool.apply_async(
                    register,
                    (in_filepath, ref_filepath, out_filepath, force, verbose))
                proc_result_list.append(proc_result)
            else:
                # serial
                register(
                    in_filepath, ref_filepath, out_filepath, force, verbose)
        else:
            if mrb.check_redo([in_filepath], [out_filepath], force):
                if verbose > VERB_LVL['none']:
                    print('Regstr:\t{}'.format(os.path.basename(out_filepath)))
                if verbose >= VERB_LVL['high']:
                    print('II: copying without registering.')
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

    Parameters
    ==========
    in_filepath_list : list of string
        List of filepaths used as input.
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

    Returns
    =======
    out_filepath_list : list of string
        List of path to output files.

    """
    # ensure existing output path
    if not os.path.exists(out_dirpath):
        os.makedirs(out_dirpath)
    if use_mp:
        # parallel
        n_proc = multiprocessing.cpu_count()
        pool = multiprocessing.Pool(processes=n_proc)
        proc_result_list = []
    out_filepath_list = []
    for in_filepath in in_filepath_list:
        out_filepath = os.path.join(out_dirpath, os.path.basename(in_filepath))
        out_filepath_list.append(out_filepath)
        if use_mp:
            # parallel
            args = (
                in_filepath, mask_filepath, out_filepath, mask_val,
                force, verbose)
            kwargs = {}
            proc_result = pool.apply_async(apply_mask, args, kwargs)
            proc_result_list.append(proc_result)
        else:
            # serial
            apply_mask(
                in_filepath, mask_filepath, out_filepath, mask_val,
                force, verbose)
    if use_mp:
        res_list = []
        for proc_result in proc_result_list:
            res_list.append(proc_result.get())
    return out_filepath_list


# ======================================================================
def get_comparing_list(
        in_filepath_list,
        ref_filepath_list,
        out_dirpath,
        skip_equal=True,
        skip_symmetric=False,
        diff_prefix='diff',
        corr_prefix='corr'):
    """Get list items to be compared."""

    def _symmetric(item1, item2, source):
        result = False
        for item in source:
            tmp1, tmp2 = item[:2]
            if tmp1 == item2 and tmp2 == item1:
                result = True
                break
        return result

    cmp_list = []
    combinator = itertools.product(ref_filepath_list, in_filepath_list)
    for ref_filepath, in_filepath in combinator:
        if skip_equal and in_filepath == ref_filepath:
            continue
        if skip_symmetric and _symmetric(in_filepath, ref_filepath, cmp_list):
            continue
        diff_filepath = os.path.join(
            out_dirpath,
            mru.combine_filename(diff_prefix, (
                ref_filepath, in_filepath)) +
            mrb.add_extsep(mrn.D_EXT))
        corr_filepath = os.path.join(
            out_dirpath,
            mru.combine_filename(corr_prefix, (
                ref_filepath, in_filepath)) +
            mrb.add_extsep(mrb.CSV_EXT))
        cmp_list.append(
            (in_filepath, ref_filepath, diff_filepath, corr_filepath))
    return cmp_list


# ======================================================================
def comparing(
        in_filepath_list,
        ref_filepath_list=None,
        out_dirpath='comparing',
        mask_filepath=None,
        skip_equal=True,
        mask_nan=True,
        mask_inf=True,
        mask_vals=None,
        val_range=None,
        trunc=None,
        diff_prefix='diff',
        corr_prefix='corr',
        use_mp=True,
        force=False,
        verbose=D_VERB_LVL):
    """
    Compare input files to reference files.
    Calculate difference and correlation coefficients.

    Parameters
    ==========
    in_filepath_list : list of string
        List of filepaths used as input.
    ref_filepath_list : list of string (optional)
        List of filepaths used as reference.
    out_dirpath : str (optional)
        Path to directory where to store results.
    mask_filepath : str (optional)
        Path to mask image file.
    skip_equal : boolean (optional)
        Skip comparison if input and reference are equal.
    mask_nan : bool (optional)
        Mask NaN values.
    mask_inf : bool (optional)
        Mask Inf values.
    mask_vals : list of int or float
        List of values to mask.
    val_range : 2-tuple (optional)
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

    Returns
    =======
    diff_filepath_list : list of string
        List of path to difference image files.
    corr_filepath_list : list of string
        List of path to voxel correlation coefficient data files.

    """
    # ensure existing output path
    if not os.path.exists(out_dirpath):
        os.makedirs(out_dirpath)
    if use_mp:
        # parallel
        n_proc = multiprocessing.cpu_count()
        pool = multiprocessing.Pool(processes=n_proc)
        proc_result_list = []
    cmp_list = get_comparing_list(
        in_filepath_list, ref_filepath_list, out_dirpath,
        skip_equal=True, skip_symmetric=True,
        diff_prefix='diff', corr_prefix='corr')
    for in_filepath, ref_filepath, diff_filepath, corr_filepath in cmp_list:
        if not mask_vals:
            mask_vals = [0.0]
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
                 mask_nan, mask_inf, mask_vals, val_range, trunc,
                 force, verbose))
            proc_result_list.append(proc_result)
        else:
            # serial
            calc_difference(
                in_filepath, ref_filepath, diff_filepath, force, verbose)
            calc_correlation(
                in_filepath, ref_filepath, corr_filepath, mask_filepath,
                mask_nan, mask_inf, mask_vals, val_range, trunc,
                force, verbose)
    if use_mp:
        res_list = []
        for proc_result in proc_result_list:
            res_list.append(proc_result.get())
    return cmp_list


# ======================================================================
def check_correlation(
        dirpath,
        type=None,
        val_range=None,
        val_units=None,
        mask_filepath=None,
        reg_ref_ext=D_EXT['registration reference'],
        corr_ref_ext=D_EXT['correlation reference'],
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
        type (str): Name of the data to be processed.
        val_range (tuple[float]): Range (min, max) of the data to be processed.
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
        (list[str], list[str]):
            - *target_list*: List of processed image files.
            - *corr_list*: List of files containing correlation computations.

    """
    if verbose > VERB_LVL['none']:
        print('Target: {}'.format(dirpath))
    # :: manage image type, range and units
    if not type:
        type = os.path.split(dirpath)[-1]
        if type not in SRC_IMG_TYPE:
            type = None
        if verbose >= VERB_LVL['medium']:
            print('W: image type not specified.')
            print('I: guessed image type: {}'.format(type))
    else:
        if verbose >= VERB_LVL['medium']:
            print('I: ', type, val_range, val_units)
    if not val_range:
        if verbose >= VERB_LVL['medium']:
            print('W: values range not specified.')
            print('I: values range not guessed. Using image-specific values.')
    if not val_units:
        val_units = 'a.u.'
        if verbose >= VERB_LVL['medium']:
            print('W: values units not specified.')
            print('I: guessed image type: {}'.format(type))
    # :: populate a list of images to analyze
    target_list, corr_list = [], []
    if os.path.exists(dirpath):
        filepath_list = mrb.listdir(dirpath, mrn.D_EXT)
        source_list = [filepath for filepath in filepath_list
                       if not type or
                       mru.parse_filename(filepath)['type'] == type]
        if len(source_list) > 0:
            # :: create output directories
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
            # todo: allow code in the registration reference file to contain
            #  registration instructions
            if reg_dir or msk_dir:
                ref = get_ref_list(dirpath, source_list, None, reg_ref_ext)[0]

            # ensure the presence of a mask
            if msk_dir:
                # if mask_filepath was not specified, set up a new name
                if not mask_filepath:
                    mask_filepath = mrn.filename_addext(MASK_FILENAME)
                # add current directory if it was not specified
                if not os.path.exists(mask_filepath):
                    mask_filepath = os.path.join(dirpath, mask_filepath)
                # if mask not found, create one
                if not os.path.exists(mask_filepath):
                    # if available use helper
                    mask_src = mru.change_img_type(ref, SERVICE_ID['helper'])
                    # if not, revert to ref image
                    if not os.path.exists(mask_src):
                        mask_src = ref
                    if verbose >= VERB_LVL['medium']:
                        print('I: FSL BET2 params: {}'.format(bet_params))
                    if bet_params:
                        threshold = 0.5
                        comparison = '>'
                        mode = 'relative'
                        smoothing = 1.0
                        erosion_iter = 6
                        dilation_iter = 2
                    else:
                        threshold = 0.5
                        comparison = '>'
                        mode = 'relative'
                        smoothing = 1.0
                        erosion_iter = 2
                        dilation_iter = 2
                    params = [
                        threshold, comparison, mode,
                        smoothing,
                        erosion_iter, dilation_iter, bet_params]
                    if verbose >= VERB_LVL['medium']:
                        print('I: compute_mask params: ', *params)
                    mask_filepath = calc_mask(
                        mask_src, tmp_path, *params)
            else:
                mask_filepath = ''
            if verbose >= VERB_LVL['medium']:
                print('I: using mask: {}'.format(mask_filepath))
            # :: co-register targets
            if reg_dir:
                target_list = registering(
                    source_list, ref, mask_filepath, reg_path, use_mp=True,
                    force=force, verbose=verbose)
            else:
                target_list = source_list
            # :: mask targets
            if msk_path:
                target_list = masking(
                    target_list, mask_filepath, msk_path, use_mp=True,
                    force=force, verbose=verbose)
                # make sure the mask as correct shape
                new_mask = os.path.join(
                    dirpath, tmp_dir, os.path.basename(mask_filepath))
                apply_mask(mask_filepath, mask_filepath, new_mask)
                mask_filepath = new_mask
            # perform comparison
            if cmp_path:
                ref_list = get_ref_list(
                    dirpath, target_list, msk_dir, corr_ref_ext)
                cmp_list = comparing(
                    target_list, ref_list, cmp_path, mask_filepath,
                    use_mp=False, val_range=val_range,
                    force=force, verbose=verbose)
                # group resulting correlations
                corr_list = [item[3] for item in cmp_list]
                combine_correlation(
                    corr_list, cmp_path, force=force, verbose=verbose)
            # plotting
            if fig_path:
                for target in target_list:
                    plot_sample(
                        target, type, val_range, val_units, fig_path,
                        force=force, verbose=verbose)
                    plot_histogram(
                        target, mask_filepath, type, val_range,
                        val_units, fig_path,
                        force=force, verbose=verbose)
                # use last plotted image to calculate approximate diff_range
                if val_range is None:
                    stats_dict = mrn.calc_stats(target)
                    val_range = (stats_dict['min'], stats_dict['max'])
                diff_range = mrb.combined_range(val_range, val_range, '-')
                for in_filepath, ref_filepath, diff_filepath, corr_filepath \
                        in cmp_list:
                    plot_sample(
                        diff_filepath, type, diff_range, val_units,
                        fig_path,
                        force=force, verbose=verbose)
                    plot_histogram(
                        diff_filepath, mask_filepath,
                        type, diff_range, val_units, fig_path,
                        force=force, verbose=verbose)
                    plot_correlation(
                        in_filepath, ref_filepath, mask_filepath,
                        type, val_range, val_units,
                        fig_path,
                        force=force, verbose=verbose)
        else:
            if verbose >= VERB_LVL['medium']:
                print('W: no input file found.')
                print('I: descending into subdirectories.')
            sub_dirpath_list = mrb.listdir(dirpath, None)
            for sub_dirpath in sub_dirpath_list:
                tmp_target_list, tmp_corr_list = check_correlation(
                    sub_dirpath,
                    type, val_range, val_units,
                    mask_filepath,
                    reg_ref_ext, corr_ref_ext,
                    tmp_dir, reg_dir, msk_dir, cmp_dir, fig_dir,
                    force, verbose)
                target_list += tmp_target_list
                corr_list += tmp_corr_list
            # group resulting correlations
            if corr_list:
                combine_correlation(corr_list, dirpath,
                                    force=force, verbose=verbose)
    return target_list, corr_list


# ======================================================================
if __name__ == '__main__':
    print(__doc__)