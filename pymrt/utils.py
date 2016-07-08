#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
pymrt.utils: useful utilities for MRI data analysis.
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
# import unittest  # Unit testing framework
import doctest  # Test interactive Python examples


# :: External Imports
import numpy as np  # NumPy (multidimensional numerical arrays library)
# import scipy as sp  # SciPy (signal and image processing library)
# import matplotlib as mpl  # Matplotlib (2D/3D plotting library)
# import sympy as sym  # SymPy (symbolic CAS library)
# import PIL  # Python Image Library (image manipulation toolkit)
# import SimpleITK as sitk  # Image ToolKit Wrapper
# import nibabel as nib  # NiBabel (NeuroImaging I/O Library)
# import nipy  # NiPy (NeuroImaging in Python)
# import nipype  # NiPype (NiPy Pipelines and Interfaces)

# :: External Imports Submodules
# import scipy.optimize  # SciPy: Optimization Algorithms
# import scipy.integrate  # SciPy: Integrations facilities
# import scipy.constants  # SciPy: Mathematal and Physical Constants
# import scipy.ndimage  # SciPy: ND-image Manipulation
# import scipy.stats  # SciPy: Statistical functions

# :: Local Imports
import pymrt.base as mrb
import pymrt.input_output as mrio
from pymrt.debug import dbg

# from pymrt import INFO
# from pymrt import VERB_LVL
# from pymrt import D_VERB_LVL
# from pymrt import get_first_line

# from dcmpi.lib.common import D_NUM_DIGITS
D_NUM_DIGITS = 3  # synced with: dcmpi.common.D_NUM_DIGITS

# ======================================================================
# :: parsing constants
D_SEP = '_'
PARAM_BASE_SEP = '_'
PARAM_SEP = ','
PARAM_KEY_VAL_SEP = '='
INFO_SEP = '__'
TOKEN_SEP = '_'
KEY_VAL_SEP = '='

# suffix of new reconstructed image from Siemens
NEW_RECO_ID = 'rr'
SERIES_NUM_ID = 's'


# ======================================================================
def str_to_key_val(
        text,
        kv_sep=KEY_VAL_SEP,
        case_sensitive=False):
    """
    Extract numerical value from string information.
    This expects a string containing a single parameter in the form `key=val`.

    Parameters:
        text (str): The input string.
        kv_sep (str): The string separating the key and the value.
        case_sensitive : bool (optional)
            Parsing of the string is case-sensitive.

    Returns:
        param_val (int|float|str|None): The value of the parameter.

    Examples:
        >>> str_to_key_val('key=1000')
        (u'key', 1000)
        >>> str_to_key_val('key1000', '')
        (u'key', 1000)

    See Also:
        set_param_val, parse_series_name
    """
    if text:
        if not case_sensitive:
            text = text.lower()
        if kv_sep and kv_sep in text:
            key, val = text.split(kv_sep)
        elif kv_sep == '':
            key = re.findall(r'^[a-zA-Z\-]*', text)[0]
            val = text[len(key):]
        else:
            key = text
            val = None
    else:
        key, val = None, None
    if val:
        val = mrb.auto_convert(val)
    return key, val


# ======================================================================
def key_val_to_str(
        key,
        val=None,
        kv_sep=KEY_VAL_SEP,
        case='lower'):
    """
    Extract numerical value from string information.
    This expects an appropriate string, as retrieved by parse_filename().

    Args:
        val (int|float|None): The value of the parameter.
        key (str): The string containing the label of the parameter.
        kv_sep (str): String separating key from value in parameters.
        case (str): Set the case of the parameter label.
            - 'lower': force the key to be in lower case
            - 'upper': force the key to be in upper case
            - any other value has no effect and it is silently ignored

    Returns:
        text (str): The string containing the information.

    Examples:
        >>> key_val_to_str('key', 1000)
        'key=1000'
        >>> key_val_to_str('key', 1000, '')
        'key1000'

    See Also:
        str_to_key_val, to_series_name
    """
    if case == 'lower':
        key = key.lower()
    elif case == 'upper':
        key = key.upper()
    if val is not None:
        text = kv_sep.join((key, str(val)))
    else:
        text = key
    return text


# ======================================================================
def str_to_info(
        text,
        sep=TOKEN_SEP,
        kv_sep=KEY_VAL_SEP):
    """
    Extract information from Siemens DICOM names.
    Expected format is: [s<###>_]<series_name>[_<#>][_<type>]

    Args:
        text (str):
        sep (str):
        kv_sep (str):

    Returns:
        info (dict):

    Examples:
        >>> d = str_to_info('S001_TEST_NAME_TE=10.6_2_T1')
        >>> for k in sorted(d.keys()): print(k, ':', d[k])  # display dict
        _# : 1
        2 : None
        name : None
        t1 : None
        te : 10.6
        test : None
    """
    tokens = text.split(sep)
    info = {
        '#': mrb.auto_convert(tokens[0][len(SERIES_NUM_ID):])
        if SERIES_NUM_ID.lower() in tokens[0].lower() else None}
    for token in tokens[1:]:
        key, val = str_to_key_val(token, kv_sep)
        info[key] = val
    return info


# ======================================================================
def info_to_str(
        info,
        sep=TOKEN_SEP,
        kv_sep=KEY_VAL_SEP):
    """

    Args:
        info (dict):
        sep (str):
        kv_sep (str):

    Returns:
        text (str):

    Examples:
        >>> info = {'_#': 10, 'me-mp2rage'}
    """
    tokens = []
    for key, val in info.items():
        tokens.append(key_val_to_str(key, val, kv_sep))
    return sep.join(tokens)


# ======================================================================
def filepath_to_info(
        filepath,
        file_ext=mrb.EXT['niz'],
        sep=TOKEN_SEP,
        kv_sep=KEY_VAL_SEP):
    """

    Args:
        filepath ():
        file_ext ():
        sep ():
        kv_sep ():

    Returns:
        info (dict): Information extracted from the
    """
    filename = mrb.change_ext(os.path.basename(filepath), '', file_ext)
    return str_to_info(filename)


# ======================================================================
def info_to_filepath(
        info,
        dirpath='.',
        file_ext=mrb.EXT['niz'],
        sep=TOKEN_SEP,
        kv_sep=TOKEN_SEP):
    filename = mrb.change_ext(info_to_str(info, sep, kv_sep), file_ext, '')
    return os.path.join(
        dirpath, + mrb.add_extsep(file_ext))


# ======================================================================
def filepath_set_info(
        in_filepath,
        key,
        val=None,
        force=True):
    """

    Args:
        in_filepath (str):
        key (str):
        val (int|float|str|None):
        force (bool): Force overwrite of previously existing key:val pair.

    Returns:
        out_filepath(str):
    """
    info = filepath_to_info(in_filepath)
    if key not in info or force:
        info[key] = val
    return info_to_filepath(info)


# ======================================================================
def get_param_val(
        param_str,
        param_key='',
        case_sensitive=False):
    """
    Extract numerical value from string information.
    This expects a string containing a single parameter.

    Parameters
    ==========
    name : str
        The string containing the information.
    param_key : str (optional)
        The string containing the label of the parameter.
    case_sensitive : bool (optional)
        Parsing of the string is case-sensitive.

    Returns
    =======
    param_val : int or float
        The value of the parameter.

    See Also
    ========
    set_param_val, parse_series_name

    """
    if param_str:
        if not case_sensitive:
            param_str = param_str.lower()
            param_key = param_key.lower()
        if param_str.startswith(param_key):
            param_val = param_str[len(param_key):]
        elif param_str.endswith(param_key):
            param_val = param_str[:-len(param_key)]
        else:
            param_val = None
    else:
        param_val = None
    return param_val


# ======================================================================
def set_param_val(
        param_val,
        param_key,
        kv_sep=PARAM_KEY_VAL_SEP,
        case='lower'):
    """
    Extract numerical value from string information.
    This expects an appropriate string, as retrieved by parse_filename().

    Args:
        param_val (int|float|None): The value of the parameter.
        param_key (str): The string containing the label of the parameter.
        kv_sep (str): String separating key from value in parameters.
        case ('lower'|'upper'|None): Set the case of the parameter label.

    Returns:
        str: The string containing the information.

    .. _refs:
        get_param_val, to_series_name
    """
    if case == 'lower':
        param_key = param_key.lower()
    elif case == 'upper':
        param_key = param_key.upper()
    if param_val is not None:
        param_str = kv_sep.join((param_key, str(param_val)))
    else:
        param_str = param_key
    return param_str


# ======================================================================
def parse_filename(
        filepath,
        i_sep=INFO_SEP,
        p_sep=PARAM_SEP,
        kv_sep=PARAM_KEY_VAL_SEP,
        b_sep=PARAM_BASE_SEP):
    """
    Extract specific information from SIEMENS data file name/path.
    Expected format is: [s<###>__]<series_name>[__<#>][__<type>].nii.gz

    Parameters
    ==========
    filepath : str
        Full path of the image filename.

    Returns
    =======
    info : dictionary
        Dictionary containing:
            | 'num' : int : identification number of the series.
            | 'name' : str : series name.
            | 'seq' : int or None : sequential number of the series.
            | 'type' : str : image type

    See Also
    ========
    to_filename

    """
    filename = os.path.basename(filepath)
    filename_noext = mrb.change_ext(filename, '', mrb.EXT['niz'])
    if i_sep != p_sep and i_sep != kv_sep and i_sep != b_sep:
        tokens = filename_noext.split(i_sep)
        info = {}
        # initialize end of name indexes
        idx_begin_name = 0
        idx_end_name = len(tokens)
        # check if contains scan ID
        info['num'] = mrb.auto_convert(get_param_val(tokens[0], SERIES_NUM_ID))
        idx_begin_name += (1 if info['num'] is not None else 0)
        # check if contains Sequential Number
        info['seq'] = None
        if len(tokens) > 1:
            for token in tokens[-1:-3:-1]:
                if mrb.is_number(token):
                    info['seq'] = mrb.auto_convert(token)
                    break
        idx_end_name -= (1 if info['seq'] is not None else 0)
        # check if contains Image type
        info['type'] = tokens[-1] if idx_end_name - idx_begin_name > 1 else None
        idx_end_name -= (1 if info['type'] is not None else 0)
        # determine series name
        info['name'] = i_sep.join(tokens[idx_begin_name:idx_end_name])
    else:
        raise TypeError('Cannot parse this file name.')
    return info


# ======================================================================
def to_filename(
        info,
        dirpath=None,
        ext=mrb.EXT['niz']):
    """
    Reconstruct file name/path with SIEMENS-like structure.
    Produced format is: [s<num>__]<series_name>[__<seq>][__<type>].nii.gz

    Parameters
    ==========
    info : dictionary
        Dictionary containing:
            | 'num' : int or None: Identification number of the scan.
            | 'name' : str : Series name.
            | 'seq' : int or None : Sequential number of the volume.
            | 'type' : str or None: Image type
    dirpath : str (optional)
        The base directory path for the filename.
    ext : str (optional)
        Extension to append to the newly generated filename or filepath.

    Returns
    =======
    filepath : str
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
def parse_series_name(
        name,
        p_sep=PARAM_SEP,
        kv_sep=PARAM_KEY_VAL_SEP,
        b_sep=PARAM_BASE_SEP):
    """
    Extract specific information from series name.

    Parameters
    ==========
    name : str
        Full name of the image series.
    p_sep : str (optional)
        String separating parameters.
    kv_sep : str (optional)
        String separating key from value in parameters.
    b_sep : str (optional)
        String separating the parameters from the base name.

    Returns
    =======
    base : str
        Base name of the series, i.e. without parsed parameters.
    params : (string, float or int) dictionary
        List of parameters in the (label, value) format.

    See Also
    ========
    to_series_name

    """
    if p_sep != b_sep and b_sep in name:
        base, tokens = name.split(b_sep)
        tokens = tokens.split(p_sep)
    elif p_sep in name:
        tmp = name.split(p_sep)
        base = tmp[0]
        tokens = tmp[1:]
    else:
        base = name
        tokens = ()
    params = {}
    for token in tokens:
        if kv_sep and kv_sep in token:
            param_id, param_val = token.split(kv_sep)
        else:
            param_id = re.findall('^[a-zA-Z\-]*', token)[0]
            param_val = get_param_val(token, param_id)
        params[param_id] = mrb.auto_convert(param_val) if param_val else None
    return base, params


# ======================================================================
def to_series_name(
        base,
        params,
        p_sep=PARAM_SEP,
        kv_sep=PARAM_KEY_VAL_SEP,
        b_sep=PARAM_BASE_SEP,
        value_case='lower',
        tag_case='lower'):
    """
    Reconstruct series name from specific information.

    Parameters
    ==========
    base : str
        Base name of the series, i.e. without parsed parameters.
    params : (string, float or int) dictionary
        List of parameters in the (label, value) format.
    p_sep : str (optional)
        String separating parameters.
    kv_sep : str (optional)
        String separating key from value in parameters.
    b_sep : str (optional)
        String separating the parameters from the base name.
    value_case : 'lower', 'upper' or None (optional)
        TODO
    tag_case : 'lower', 'upper' or None (optional)
        TODO

    Returns
    =======
    name : str
        Full name of the image series.

    See Also
    ========
    parse_series_name

    """
    values = []
    tags = []
    for key, val in params.items():
        if val is not None:
            values.append(set_param_val(val, key, kv_sep, value_case))
        else:
            tags.append(set_param_val(val, key, kv_sep, tag_case))
    params = p_sep.join(sorted(values) + sorted(tags))
    name = b_sep.join((base, params))
    return name


# ======================================================================
def change_img_type(
        filepath,
        img_type):
    """
    Change the image type of an image filename in a filepath.

    Parameters
    ==========
    filepath : str
        The filepath of the base image.
    img_type : str
        The new image type identifier.

    Returns
    =======
    filepath : str
        The filepath of the image with the new type.

    """
    dirpath = os.path.dirname(filepath)
    info = parse_filename(os.path.basename(filepath))
    info['type'] = img_type
    filepath = to_filename(info, dirpath)
    return filepath


# ======================================================================
def change_param_val(
        filepath,
        param_key,
        param_val):
    """
    Change the parameter value of an image filename in a filepath.

    Parameters
    ==========
    filepath : str
        The image filepath.
    param_key : str
        The identifier of the parameter to change.
    new_param_val : str
        The new value of the parameter to change.

    Returns
    =======
    new_name : str
        The filepath of the image with new type.

    """
    dirpath = os.path.dirname(filepath)
    info = parse_filename(filepath)
    base, params = parse_series_name(info['name'])
    params[param_key] = param_val
    info['name'] = to_series_name(base, params)
    filepath = to_filename(info, dirpath)
    return filepath


# ======================================================================
def extract_param_val(
        filepath,
        param_key):
    """
    Extract the parameter value from an image file name/path.

    Args:
        filepath (str): The image filepath.
        param_key (str): The identifier of the parameter to extract.

    Returns:
        The value of the extracted parameter.

    """
    # todo: add support for lists
    info = parse_filename(filepath)
    base, params = parse_series_name(info['name'])
    param_val = params[param_key] if param_key in params else False
    return param_val


# ======================================================================
def combine_filename(
        prefix,
        filenames):
    """
    Create a new filename, based on a combination of filenames.

    Args:
        prefix:
        filenames:

    Returns:

    """
    # todo: fix doc
    filename = prefix
    for name in filenames:
        filename += 2 * INFO_SEP + \
                    mrb.change_ext(os.path.basename(name), '', mrb.EXT['niz'])
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
    filepath : str
        Path fo the file from which a label is to be extracted.
    exclude_list : list of string (optional)
        List of string to exclude from filepath.
    max_length : int (optional)
        Maximum length of the label.

    Returns
    =======
    label : str
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


'''
## ======================================================================
# def calc_averages(
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
#        """Multiprocessing-friendly version of 'compute_affine_fsl()'."""
#        return compute_affine_fsl(*par)
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
##            mrio.del_ext(os.path.basename(img_tuple[0])) +
##            mrb.add_extsep(mrb.EXT['txt']))
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
##                apply_affine_fsl(
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
#        mrio.save(out_mag_filepath, np.abs(img), affine_nii)
##        mrio.save(out_phs_filepath, np.angle(img), affine_nii)
#
##        fixed = np.abs(img_list[0])
##        for idx, img in enumerate(img_list):
##            affine = mrb.affine_registration(np.abs(img), fixed, 'rigid')
##            img_list[idx] = mrb.apply_affine(img_list[idx], affine)
##        mrio.save(out_filepath, np.abs(img), affine_nii)
##        print(img.shape, img.nbytes / 1024 / 1024)  # DEBUG
##        # calculate the Fourier transform
##        for img in img_list:
##            fft_list.append(np.fft.fftshift(np.fft.fftn(img)))
##        fixed = np.abs(img[:, :, :, 0])
##        mrb.sample2d(fixed, -1)
##        tmp = tmp * np.exp(1j*0.5)
##        moving = sp.ndimage.shift(fixed, [1.0, 5.0, 0.0])
##        mrb.sample2d(moving, -1)
#
##        print(linear, shift)
##        moved = sp.ndimage.affine_transform(moving, linear, offset=-shift)
##        mrb.sample2d(moved, -1)
##        mrio.save(out_filepath, moving, affine)
##        mrio.save(mag_filepath, fixed, affine)
##        mrio.save(phs_filepath, moved-fixed, affine)
##        for idx in range(len(img_list)):
##            tmp_img = img[:, :, :, idx]
##            tmp_fft = fft[:, :, :, idx]
##            mrb.sample2d(np.real(tmp_fft), -1)
##            mrb.sample2d(np.imag(tmp_fft), -1)
##            mrb.sample2d(np.abs(img[:, :, :, idx]), -1)
##            mrb.sample2d(np.angle(img[:, :, :, idx]), -1)
#
#        # calculate output
#        if verbose > VERB_LVL['none']:
#            print('Target:\t{}'.format(os.path.basename(out_mag_filepath)))
#            print('Target:\t{}'.format(os.path.basename(out_phs_filepath)))
#    return mag_filepath, phs_filepath
'''


# ======================================================================
if __name__ == '__main__':
    print(__doc__)
    doctest.testmod()
