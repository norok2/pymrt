#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
mr_lib: useful utilities for MRI data analysis.
"""


# ======================================================================
# :: Future Imports
from __future__ import division
from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals


__version__ = '0.2.0.350'
# $Source$


# ======================================================================
# :: Custom Module Details
AUTHOR = 'Riccardo Metere'
CONTACT = 'metere@cbs.mpg.de'
DATE_INFO = {'day': 18, 'month': 'Sep', 'year': 2014}
DATE = ' '.join([str(v) for k, v in sorted(DATE_INFO.items())])
LICENSE = 'License GPLv3: GNU General Public License version 3'
COPYRIGHT = 'Copyright (C) ' + str(DATE_INFO['year'])
# first non-empty line of __doc__
DOC_FIRSTLINE = [line for line in __doc__.splitlines() if line][0]


# ======================================================================
# :: Python Standard Library Imports
import os  # Miscellaneous operating system interfaces
# import shutil  # High-level file operations
# import math  # Mathematical functions
# import time  # Time access and conversions
# import datetime  # Basic date and time types
# import operator  # Standard operators as functions
# import collections  # High-performance container datatypes
# import itertools  # Functions creating iterators for efficient looping
# import functools  # Higher-order functions and operations on callable objects
# import argparse  # Parser for command-line options, arguments and subcommands
import re  # Regular expression operations
# import subprocess  # Subprocess management
# import multiprocessing  # Process-based parallelism
# import inspect  # Inspect live objects
# import csv  # CSV File Reading and Writing [CSV: Comma-Separated Values]
# import json  # JSON encoder and decoder [JSON: JavaScript Object Notation]


# :: External Imports
# import numpy as np  # NumPy (multidimensional numerical arrays library)
# import scipy as sp  # SciPy (signal and image processing library)
# import matplotlib as mpl  # Matplotlib (2D/3D plotting library)
# import sympy as sym  # SymPy (symbolic CAS library)
# import PIL  # Python Image Library (image manipulation toolkit)
# import SimpleITK as sitk  # Image ToolKit Wrapper
# import nibabel as nib  # NiBabel (NeuroImaging I/O Library)
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
import mri_tools.lib.nifti as mrn
# from mri_tools import INFO
# from mri_tools import VERB_LVL
# from mri_tools import D_VERB_LVL
# from mri_tools import _firstline

# from dcmpi.lib.common import D_NUM_DIGITS
D_NUM_DIGITS = 3


# ======================================================================
# :: parsing constants
D_SEP = '_'
PARAM_SEP = ','
PARAM_KEY_VAL_SEP = '='
INFO_SEP = '__'

# suffix of new reconstructed image from Siemens
NEW_RECO_ID = 'rr'
SERIES_NUM_ID = 's'


# ======================================================================
def get_param_val(
        param_str,
        param_lbl='',
        case_sensitive=False):
    """
    Extract numerical value from string information.
    This expects an appropriate string, as retrieved by parse_filename().

    Parameters
    ==========
    param_str : string
        The string containing the information.
    param_lbl : string (optional)
        The string containing the label of the parameter.
    case_sensitive : boolean (optional)
        Parsing of the string is case-sensitive.

    Returns
    =======
    param_val : int or float
        The value of the parameter.

    See Also
    ========
    set_param_val

    """
    if param_str:
        if not case_sensitive:
            param_str = param_str.lower()
            param_lbl = param_lbl.lower()
        if param_str.startswith(param_lbl):
            param_val = param_str[len(param_lbl):]
        elif param_str.endswith(param_lbl):
            param_val = param_str[:-len(param_lbl)]
        else:
            param_val = None
        try:
            param_val = int(param_val)
        except (ValueError, TypeError):
            try:
                param_val = float(param_val)
            except (ValueError, TypeError):
                param_val = None
    else:
        param_val = None
    return param_val


# ======================================================================
def set_param_val(
        param_val,
        param_lbl,
        param_sep='',
        case='lower'):
    """
    Extract numerical value from string information.
    This expects an appropriate string, as retrieved by parse_filename().

    Parameters
    ==========
    param_val : int or float
        The value of the parameter.
    param_lbl : string
        The string containing the label of the parameter.
    case : boolean (optional)
        The text case of parameter label is set to this.
        Possible values are: None, 'lower', 'upper'

    Returns
    =======
    param_str : string
        The string containing the information.

    See Also
    ========
    get_param_val

    """
    if case == 'lower':
        param_lbl = param_lbl.lower()
    elif case == 'upper':
        param_lbl = param_lbl.upper()
    param_str = '{}{}'.format(param_lbl, param_val)
    return param_str


# ======================================================================
def parse_filename(filepath):
    """
    Extract specific information from SIEMENS data file name/path.
    Expected format is: [s<###>__]<series_name>[__<#>][__<type>].nii.gz

    Parameters
    ==========
    filepath : string
        Full path of the image filename.

    Returns
    =======
    info : dictionary
        Dictionary containing:
            | 'num' : int : identification number of the series.
            | 'name' : string : series name.
            | 'seq' : int or None : sequential number of the series.
            | 'type' : string : image type

    See Also
    ========
    to_filename

    """
    filename = os.path.basename(filepath)
    filename_noext = mrn.filename_noext(filename)
    tokens = filename_noext.split(INFO_SEP)
    info = {}
    # initialize end of name indexes
    idx_begin_name = 0
    idx_end_name = len(tokens)
    # check if contains scan ID
    info['num'] = get_param_val(tokens[0], SERIES_NUM_ID)
    idx_begin_name += (1 if info['num'] is not None else 0)
    # check if contains Sequential Number
    info['seq'] = None
    if len(tokens) > 1:
        for token in tokens[-1:-3:-1]:
            if mrb.is_number(token):
                info['seq'] = get_param_val(token, '')
                break
    idx_end_name -= (1 if info['seq'] is not None else 0)
    # check if contains Image type
    info['type'] = tokens[-1] if idx_end_name - idx_begin_name > 1 else None
    idx_end_name -= (1 if info['type'] is not None else 0)
    # determine series name
    info['name'] = INFO_SEP.join(tokens[idx_begin_name:idx_end_name])
    return info


# ======================================================================
def to_filename(
        info,
        dirpath=None,
        ext=mrn.D_EXT):
    """
    Reconstruct file name/path with SIEMENS-like structure.
    Produced format is: [s<num>__]<series_name>[__<seq>][__<type>].nii.gz

    Parameters
    ==========
    info : dictionary
        Dictionary containing:
            | 'num' : int or None: Identification number of the scan.
            | 'name' : string : Series name.
            | 'seq' : int or None : Sequential number of the volume.
            | 'type' : string or None: Image type
    dirpath : string (optional)
        The base directory path for the filename.
    ext : string (optional)
        Extension to append to the newly generated filename or filepath.

    Returns
    =======
    filepath : string
        Full path of the image filename.

    See Also
    ========
    parse_filename

    """
    tokens = []
    if 'num' in info and info['num'] is not None:
        tokens.append('{}{:0{size}d}'.format(
            SERIES_NUM_ID, info['num'], size=D_NUM_DIGITS))
    if 'name' in info:
        tokens.append(info['name'])
    if 'seq' in info and info['seq'] is not None:
        tokens.append('{:d}'.format(info['seq']))
    if 'type' in info and info['type'] is not None:
        tokens.append(info['type'])
    filepath = INFO_SEP.join(tokens)
    filepath += (mrb.add_extsep(ext) if ext else '')
    filepath = os.path.join(dirpath, filepath) if dirpath else filepath
    return filepath


# ======================================================================
def parse_series_name(name, p_sep=PARAM_SEP, kv_sep=PARAM_KEY_VAL_SEP):
    """
    Extract specific information from series name.

    Parameters
    ==========
    name : string
        Full name of the image series.

    Returns
    =======
    base : string
        Base name of the series, i.e. without parsed parameters.
    params : (string, float or int) dictionary
        List of parameters in the (label, value) format.

    See Also
    ========
    to_series_name

    """
    params, base_list = {}, []
    tokens = name.split(p_sep)
    for token in tokens:
        is_param = False
        if kv_sep and kv_sep in token:
            param_val, param_id = token.split(kv_sep)
            params[param_id] = param_val
            is_param = True
        else:
            param_id = re.findall('^[a-zA-Z]*', token)[0]
            param_val = get_param_val(token, param_id)
            if param_val != None:
                params[param_id] = param_val
                is_param = True
        if not is_param:
            base_list.append(token)
    base = p_sep.join(base_list)
    return base, params


# ======================================================================
def to_series_name(
        base,
        params):
    """
    Reconstruct series name from specific information.

    Parameters
    ==========
    base : string
        Base name of the series, i.e. without parsed parameters.
    params : (string, float or int) dictionary
        List of parameters in the (label, value) format.
    strict : bool (optional)
        If strict is True, only known parameters are included.

    Returns
    =======
    name : string
        Full name of the image series.

    See Also
    ========
    parse_series_name

    """
    tokens = [base]
    params = sorted(params.items())
    tags = [key for key, val in params if not val]
    for key, val in params:
        if val:
            tokens.append(set_param_val(val, key))
    for tag in tags:
        tokens.append(key)
    name = PARAM_SEP.join(tokens)
    return name


# ======================================================================
def change_img_type(
        filepath,
        new_type):
    """
    Change the image type of a given image file.

    Parameters
    ==========
    filepath : string
        The filepath of the base image.
    new_type : string
        The new image type identifier.

    Returns
    =======
    new_filepath : string
        The filepath of the image with new type.

    """
    dirpath = os.path.dirname(filepath)
    info = parse_filename(filepath)
    info['img_type'] = new_type
    new_filepath = to_filename(info, dirpath)
    return new_filepath


# ======================================================================
def change_param_val(
        filepath,
        param_lbl,
        new_param_val):
    """
    Change the parameter value of a given filepath.

    Parameters
    ==========
    filepath : string
        The filepath of the base image.
    new_type : string
        The new image type identifier.

    Returns
    =======
    new_name : string
        The filepath of the image with new type.

    """
    dirpath = os.path.dirname(filepath)
    info = parse_filename(filepath)
    base, params = parse_series_name(info['name'])
    params[param_lbl] = new_param_val
    info['name'] = to_series_name(base, params)
    new_filepath = to_filename(info, dirpath)
    return new_filepath


# ======================================================================
def combine_filename(
        prefix,
        filename_list):
    """
    Create a new filename, based on a combination of filenames.

    Parameters
    ==========


    Returns
    =======
    filename : string

    """
    filename = prefix
    for filenames in filename_list:
        filename += 2 * INFO_SEP + \
            mrn.filename_noext(os.path.basename(filenames))
    return filename


# ======================================================================
def filename2label(
        filepath,
        exclude_list=None,
        max_length=None):
    """
    Create a sensible but shorter label from filename.

    Parameters
    ==========
    filepath : string
        Path fo the file from which a label is to be extracted.
    exclude_list : list of string (optional)
        List of string to exclude from filepath.
    max_length : int (optional)
        Maximum length of the label.

    Returns
    =======
    label : string
        The extracted label.

    """
    info = parse_filename(filepath)
    tokens = info['name'].split(INFO_SEP)
    # remove unwanted information
    exclude_list = []
    tokens = [token for token in tokens if token not in exclude_list]
    label = INFO_SEP.join(tokens)
    if max_length:
        label = label[:max_length]
    return label


## ======================================================================
#def calc_averages(
#        filepath_list,
#        out_dirpath,
#        threshold=0.05,
#        rephasing=True,
#        registration=False,
#        limit_num=None,
#        force=False,
#        verbose=D_VERB_LVL):
#    """
#    Calculate the average of MR complex images.
#
#    TODO: clean up code / fix documentation
#
#    Parameters
#    ==========
#    """
#    def _compute_regmat(par):
#        """Multiprocessing-friendly version of 'compute_regmat()'."""
#        return compute_regmat(*par)
#
#    tmp_dirpath = os.path.join(out_dirpath, 'tmp')
#    if not os.path.exists(tmp_dirpath):
#        os.makedirs(tmp_dirpath)
#    # sort by scan number
#    get_num = lambda filepath: parse_filename(filepath)['num']
#    filepath_list.sort(key=get_num)
#    # generate output name
#    sum_num, sum_avg = 0, 0
#    for filepath in filepath_list:
#        info = parse_filename(filepath)
#        base, params = parse_series_name(info['name'])
#        sum_num += info['num']
#        if PARAM_ID['avg'] in params:
#            sum_avg += params[PARAM_ID['avg']]
#        else:
#            sum_avg += 1
#    params[PARAM_ID['avg']] = sum_avg // 2
#    name = to_series_name(base, params)
#    new_info = {
#        'num': sum_num,
#        'name': name,
#        'img_type': TYPE_ID['temp'],
#        'te_val': info['te_val']}
#    out_filename = to_filename(new_info)
#    out_tmp_filepath = os.path.join(out_dirpath, out_filename)
#    out_mag_filepath = change_img_type(out_tmp_filepath, TYPE_ID['mag'])
#    out_phs_filepath = change_img_type(out_tmp_filepath, TYPE_ID['phs'])
#    out_filepath_list = [out_tmp_filepath, out_mag_filepath, out_phs_filepath]
#    # perform calculation
#   if mrb.check_redo(filepath_list, out_filepath_list, force) and sum_avg > 1:
#        # stack multiple images together
#        # assume every other file is a phase image, starting with magnitude
#        img_tuple_list = []
#        mag_filepath = phs_filepath = None
#        for filepath in filepath_list:
#            if verbose > VERB_LVL['none']:
#                print('Source:\t{}'.format(os.path.basename(filepath)))
#            img_type = parse_filename(filepath)['img_type']
#            if img_type == TYPE_ID['mag'] or not mag_filepath:
#                mag_filepath = filepath
#            elif img_type == TYPE_ID['phs'] or not phs_filepath:
#                phs_filepath = filepath
#            else:
#                raise RuntimeWarning('Filepath list not valid for averaging.')
#            if mag_filepath and phs_filepath:
#                img_tuple_list.append([mag_filepath, phs_filepath])
#                mag_filepath = phs_filepath = None
#
##        # register images
##        regmat_filepath_list = [
##            os.path.join(
##            tmp_dirpath,
##            mrn.filename_noext(os.path.basename(img_tuple[0])) +
##            mrb.add_extsep(mrb.TXT_EXT))
##            for img_tuple in img_tuple_list]
##        iter_param_list = [
##            (img_tuple[0], img_tuple_list[0][0], regmat)
##            for img_tuple, regmat in
##            zip(img_tuple_list, regmat_filepath_list)]
##        pool = multiprocessing.Pool(multiprocessing.cpu_count())
##        pool.map(_compute_regmat, iter_param_list)
##        reg_filepath_list = []
##        for idx, img_tuple in enumerate(img_tuple_list):
##            regmat = regmat_filepath_list[idx]
##            for filepath in img_tuple:
##                out_filepath = os.path.join(
##                    tmp_dirpath, os.path.basename(filepath))
##                apply_regmat(
##                    filepath, img_tuple_list[0][0], out_filepath, regmat)
##                reg_filepath_list.append(out_filepath)
##        # combine all registered images together
##        img_tuple_list = []
##        for filepath in reg_filepath_list:
##            if img_type == TYPE_ID['mag'] or not mag_filepath:
##                mag_filepath = filepath
##            elif img_type == TYPE_ID['phs'] or not phs_filepath:
##                phs_filepath = filepath
##            else:
##               raise RuntimeWarning('Filepath list not valid for averaging.')
##            if mag_filepath and phs_filepath:
##                img_tuple_list.append([mag_filepath, phs_filepath])
##                mag_filepath = phs_filepath = None
#
#        # create complex images and disregard inappropriate
#        img_list = []
#        avg_power = 0.0
#        num = 0
#        shape = None
#        for img_tuple in img_tuple_list:
#            mag_filepath, phs_filepath = img_tuple
#            img_mag_nii = nib.load(mag_filepath)
#            img_mag = img_mag_nii.get_data()
#            img_phs_nii = nib.load(mag_filepath)
#            img_phs = img_phs_nii.get_data()
#            affine_nii = img_mag_nii.get_affine()
#            if not shape:
#                shape = img_mag.shape
#            if avg_power:
#                rel_power = np.abs(avg_power - np.sum(img_mag)) / avg_power
#            if (not avg_power or rel_power < threshold) \
#                    and shape == img_mag.shape:
#                img_list.append(mrb.polar2complex(img_mag, img_phs))
#                num += 1
#                avg_power = (avg_power * (num - 1) + np.sum(img_mag)) / num
#        out_mag_filepath = change_param_val(
#            out_mag_filepath, PARAM_ID['avg'], num)
#
#        # k-space constant phase correction
#        img0 = img_list[0]
#        ft_img0 = np.fft.fftshift(np.fft.fftn(img0))
#        k0_max = np.unravel_index(np.argmax(ft_img0), ft_img0.shape)
#        for idx, img in enumerate(img_list):
#            ft_img = np.fft.fftshift(np.fft.fftn(img))
#            k_max = np.unravel_index(np.argmax(ft_img), ft_img.shape)
#            dephs = np.angle(ft_img0[k0_max] / ft_img[k_max])
#            img = np.fft.ifftn(np.fft.ifftshift(ft_img * np.exp(1j * dephs)))
#            img_list[idx] = img
#
#        img = mrb.ndstack(img_list, -1)
#        img = np.mean(img, -1)
#        mrn.img_maker(out_mag_filepath, np.abs(img), affine_nii)
##        mrn.img_maker(out_phs_filepath, np.angle(img), affine_nii)
#
##        fixed = np.abs(img_list[0])
##        for idx, img in enumerate(img_list):
##            affine = mrb.affine_registration(np.abs(img), fixed, 'rigid')
##            img_list[idx] = mrb.apply_affine(img_list[idx], affine)
##        mrn.img_maker(out_filepath, np.abs(img), affine_nii)
##        print(img.shape, img.nbytes / 1024 / 1024)  # DEBUG
##        # calculate the Fourier transform
##        for img in img_list:
##            fft_list.append(np.fft.fftshift(np.fft.fftn(img)))
##        fixed = np.abs(img[:, :, :, 0])
##        mrb.plot_sample2d(fixed, -1)
##        tmp = tmp * np.exp(1j*0.5)
##        moving = sp.ndimage.shift(fixed, [1.0, 5.0, 0.0])
##        mrb.plot_sample2d(moving, -1)
#
##        print(linear, shift)
##        moved = sp.ndimage.affine_transform(moving, linear, offset=-shift)
##        mrb.plot_sample2d(moved, -1)
##        mrn.img_maker(out_filepath, moving, affine)
##        mrn.img_maker(mag_filepath, fixed, affine)
##        mrn.img_maker(phs_filepath, moved-fixed, affine)
##        for idx in range(len(img_list)):
##            tmp_img = img[:, :, :, idx]
##            tmp_fft = fft[:, :, :, idx]
##            mrb.plot_sample2d(np.real(tmp_fft), -1)
##            mrb.plot_sample2d(np.imag(tmp_fft), -1)
##            mrb.plot_sample2d(np.abs(img[:, :, :, idx]), -1)
##            mrb.plot_sample2d(np.angle(img[:, :, :, idx]), -1)
#
#        # calculate output
#        if verbose > VERB_LVL['none']:
#            print('Target:\t{}'.format(os.path.basename(out_mag_filepath)))
#            print('Target:\t{}'.format(os.path.basename(out_phs_filepath)))
#    return mag_filepath, phs_filepath


# ======================================================================
if __name__ == '__main__':
    print(__doc__)
