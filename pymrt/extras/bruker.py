#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PyMRT: parse Bruker raw data.

The module is NumPy-aware.
"""

# ======================================================================
# :: Future Imports
from __future__ import (
    division, absolute_import, print_function, unicode_literals)

# ======================================================================
# :: Python Standard Library Imports
import os  # Miscellaneous operating system interfaces
# import shutil  # High-level file operations
import math  # Mathematical functions
# import time  # Time access and conversions
import datetime  # Basic date and time types
# import operator  # Standard operators as functions
# import collections  # Container datatypes
# import argparse  # Parser for command-line options, arguments and subcommands
# import itertools  # Functions creating iterators for efficient looping
# import subprocess  # Subprocess management
# import multiprocessing  # Process-based parallelism
# import csv  # CSV File Reading and Writing [CSV: Comma-Separated Values]
# import json  # JSON encoder and decoder [JSON: JavaScript Object Notation]
import struct  # Interpret strings as packed binary data
import doctest  # Test interactive Python examples
import glob  # Unix style pathname pattern expansion
import warnings  # Warning control

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
# import matplotlib.pyplot as plt  # Matplotlib's pyplot: MATLAB-like syntax
# import scipy.optimize  # SciPy: Optimization Algorithms
# import scipy.integrate  # SciPy: Integrations facilities
# import scipy.constants  # SciPy: Mathematal and Physical Constants
# import scipy.ndimage  # SciPy: ND-image Manipulation

# :: Local Imports
import pymrt as mrt
import pymrt.utils
import pymrt.naming
import pymrt.input_output
# import pymrt.geometry
from pymrt.extras import jcampdx

# from pymrt import INFO, DIRS
from pymrt import VERB_LVL, D_VERB_LVL, VERB_LVL_NAMES
from pymrt import elapsed, print_elapsed
from pymrt import msg, dbg


# ======================================================================
def _get_shape(base_shape, *extras):
    extras = tuple(mrt.utils.auto_repeat(extras, 1))
    return tuple(base_shape) + (extras if np.prod(extras) > 1 else ())


# ======================================================================
def _get_load_bin_info_fid(acqp, method):
    _mode = {'little': '<', 'big': '>'}
    _dtype = {
        'GO_32BIT_SGN_INT': 'int',
        'GO_32_BIT_SGN_INT': 'int',
        'GO_16BIT_SGN_INT': 'short',
        'GO_16_BIT_SGN_INT': 'short',
        'GO_32BIT_FLOAT': 'float',
        'GO_32_BIT_FLOAT': 'float'}

    # for ParaVision 6.0.1, who knows for the rest
    info = dict(
        dtype=_dtype[acqp['GO_raw_data_format']],
        mode=_mode[acqp['BYTORDA']],
        user_filter=acqp['ACQ_user_filter'] == 'Yes',
        cx_interleaved=True,
    )
    return info


# ======================================================================
def _get_load_bin_info_reco(reco, method):
    _dtype = {
        '_8BIT_UNSGN_INT': 'uchar',
        '_16BIT_SGN_INT': 'short',
        '_32BIT_SGN_INT': 'int',
        '_32BIT_FLOAT': 'float'}
    _mode = {'littleEndian': '<', 'bigEndian': '>'}

    # for ParaVision 6.0.1, who knows for the rest
    info = dict(
        dtype=_dtype[reco['RECO_wordtype']],
        mode=_mode[reco['RECO_byte_order']],
        user_filter=False,
        cx_interleaved=False,
    )
    return info


# ======================================================================
def _load_bin(
        filepath,
        dtype='int',
        mode='<',
        user_filter=False,
        cx_interleaved=True,
        dry=False):
    byte_size = struct.calcsize(mrt.utils.DTYPE_STR[dtype])
    with mrt.utils.zopen(filepath, 'rb') as file_obj:
        file_size = file_obj.seek(0, 2)
        file_obj.seek(0)
        if not dry:
            if user_filter:
                raise NotImplementedError
            arr = np.array(mrt.utils.read_stream(
                file_obj, dtype, mode, file_size // byte_size))
            if cx_interleaved:
                arr = arr[0::2] + 1j * arr[1::2]
        else:
            arr = np.zeros(file_size)
    return arr


# ======================================================================
def _to_patterns(name, exts):
    return [mrt.utils.change_ext('*/' + name, ext) for ext in exts]


# ======================================================================
def _get_single(dirpath, name, exts):
    filepaths = mrt.utils.flistdir(dirpath, _to_patterns(name, exts))
    if len(filepaths) == 1:
        filepath = filepaths[0]
    elif len(filepaths) > 1:
        raise FileExistsError(
            'Multiple `{name}` files found.'.format(name=name))
    else:
        raise FileNotFoundError(
            'No `{name}` file found.'.format(name=name))
    return filepath


# ======================================================================
def _get_scan_num_sample_id(comments):
    lines = comments.split('$$ ')
    lines = [line for line in lines if line.strip().endswith('/acqp')]
    source_filepath = mrt.utils.multi_split_path(lines[0])
    scan_num = source_filepath[-3]
    sample_id = source_filepath[-4]
    scan_num = '{s}{num:03d}'.format(
        s=mrt.naming.SERIES_NUM_ID, num=int(scan_num))
    return scan_num, sample_id


# ======================================================================
def _get_reco_num(comments):
    lines = comments.split('$$ ')
    lines = [line for line in lines if line.strip().endswith('/reco')]
    source_filepath = mrt.utils.multi_split_path(lines[0])
    reco_num = source_filepath[-3]
    reco_num = '{s}{num:02d}'.format(
        s=mrt.naming.NEW_RECO_ID[0], num=int(reco_num))
    return reco_num


# ======================================================================
def _reco_from_fid(
        arr,
        acqp,
        method,
        verbose=D_VERB_LVL):
    is_cartesian = True
    if is_cartesian:
        load_info = _get_load_bin_info_fid(acqp, method)
        dtype_size = struct.calcsize(
            load_info['mode'] + mrt.utils.DTYPE_STR[load_info['dtype']])
        block_size = acqp['GO_block_size']
        if block_size == 'continuous':
            block_size = None
        elif block_size == 'Standard_KBlock_Format':
            block_size = (1024 // dtype_size // 2)
        else:
            block_size = int(block_size)

        # todo: ACQ_phase_factor

        # number of images per experiment (e.g. multi-echo)
        num_images = acqp['NI']
        msg('num_images={}'.format(num_images), verbose, VERB_LVL['debug'])

        # inner cycle repetition (before phase-encoding) to be averaged
        num_accum = acqp['NAE']
        msg('num_accum={}'.format(num_accum), verbose, VERB_LVL['debug'])

        # outer cycle repetition (after phase-encoding) to be averaged
        num_avg = acqp['NA']
        msg('num_avg={}'.format(num_avg), verbose, VERB_LVL['debug'])

        # image repetitions that are NOT to be averaged
        num_rep = acqp['NR']
        msg('num_rep={}'.format(num_rep), verbose, VERB_LVL['debug'])

        # number of dummy scans
        # num_ds = acqp['DS']
        # msg('num_ds={}'.format(num_ds), verbose, VERB_LVL['debug'])

        base_shape = method['PVM_Matrix']
        msg('base_shape={}'.format(base_shape), verbose, VERB_LVL['debug'])
        
        # acq_size = acqp['ACQ_size']
        # msg('acq_size={}'.format(acq_size), verbose, VERB_LVL['debug'])

        try:
            # fix checkerboard phase artifact
            arr = arr * np.exp(
                np.pi * np.arange(arr.size).reshape(arr.shape) % 2)

            fp = '/home/raid1/metere/hd3/sandbox/hmri/_/test{n}{o}{s}.nii.gz'
            # fid_shape = -1, fid_shape[0], fid_shape[1], fid_shape[2]
            # i = 0
            # for fid_shape in mrt.utils.unique_permutations(fid_shape):
            #     for order in 'C',:
            #         arr = arr.reshape(fid_shape, order=order)
            #         print(i, fid_shape, order)
            #         minor = fid_shape.index(min(fid_shape))
            #         arr = np.swapaxes(arr, minor, -1)
            #         print(arr.shape)
            #         mrt.input_output.save(
            #             fp.format(n=i, o=order, s=fid_shape), np.abs(arr))
            #         i += 1

            fid_shape = (
                tuple(((mrt.utils.num_align(base_shape[0], block_size)
                        if block_size else base_shape[0]),
                       num_images, num_accum)) +
                tuple(base_shape[1:]) + (num_avg, num_rep) + (-1,))

            # ro_size = base_shape[0]
            # pe_size = base_shape[1]
            # sl_size = base_shape[2]
            # me_size = num_images
            # if block_size:
            #     adc_size = mrt.utils.num_align(ro_size, block_size)
            # else:
            #     adc_size = ro_size
            # fid_shape = (adc_size, me_size, pe_size, sl_size, -1)
            print(fid_shape)

            arr = arr.reshape(fid_shape, order='F')
            arr = np.moveaxis(arr, 1, -1)  # for num_images
            arr = np.moveaxis(arr, 1, -1)  # for num_accum
            arr = np.squeeze(arr)  # remove singlet dimensions
            arr = np.delete(arr, slice(base_shape[0], None), 0)

            print(arr.shape, arr.dtype)
            print(np.sum(arr.real), np.sum(arr.imag))

            # perform spatial FFT
            ft_axes = tuple(range(len(base_shape)))
            arr = np.fft.ifftshift(
                np.fft.ifftn(arr, axes=ft_axes), axes=ft_axes)

            # average  experiments
            if num_accum > 1:
                # arr = np.sum(arr, axis=(-2))
                arr = arr[..., 1]

            # extras = num_images, num_avg, num_accum, num_rep
            # fid_shape = tuple(fid_shape[::-1]) + (
            #     (other_size,) if other_size > 1 else ())
            # arr = arr[::-1].reshape(fid_shape)
            mrt.input_output.save(fp.format(n='', o='M', s=''), np.abs(arr))
            mrt.input_output.save(fp.format(n='', o='P', s=''), np.angle(arr))
            quit()
        except ValueError:
            fid_shape = mrt.utils.factorize_k(arr.size, 3)
            warning = ('Could not determine correct shape for FID. '
                       'Using `{}`'.format(fid_shape))
            warnings.warn(warning)
            arr = arr.reshape(fid_shape)
    else:
        raise NotImplementedError
    return arr


def _reco_from_bin(arr, reco, method):
    warning = 'This is EXPERIMENTAL!'
    warnings.warn(warning)

    is_complex = reco['RECO_image_type'] == 'COMPLEX_IMAGE'
    if is_complex:
        arr = arr[arr.size // 2:] + 1j * arr[:arr.size // 2]

    shape = reco['RECO_size']
    # reco['RecoObjectsPerRepetition'], reco['RecoNumRepetitions']),
    shape = (-1,) + tuple(np.roll(shape, 1))
    # print(shape)
    arr = np.swapaxes(arr.reshape(shape), 0, -1)
    return arr


# ======================================================================
def batch_extract(
        dirpath,
        out_filename='niz/{scan_num}__{acq_method}_{scan_name}_{reco_flag}',
        out_dirpath=None,
        custom_reco=None,
        custom_reco_kws=None,
        fid_name='fid',
        dseq_name='2dseq',
        acqp_name='acqp',
        method_name='method',
        reco_name='reco',
        allowed_ext=('', 'gz'),
        force=False,
        verbose=D_VERB_LVL):
    """

    Args:
        dirpath ():
        out_filename ():
        out_dirpath ():
        custom_reco (str|None):
            Determines how results will be saved.
            Available options are:
             - 'mag_phs': saves magnitude and phase.
             - 're_im': saves real and imaginary parts.
             - 'cx': saves the complex data.
        fid_name ():
        dseq_name ():
        acqp_name ():
        method_name ():
        reco_name ():
        allowed_ext ():
        force ():
        verbose ():

    Returns:

    """
    if allowed_ext is None:
        allowed_ext = ''
    elif isinstance(allowed_ext, str):
        allowed_ext = (allowed_ext,)
    fid_filepaths = sorted(
        mrt.utils.flistdir(dirpath, _to_patterns(fid_name, allowed_ext)))

    for fid_filepath in sorted(fid_filepaths):
        msg('FID: {}'.format(fid_filepath),
            verbose, D_VERB_LVL)
        fid_dirpath = os.path.dirname(fid_filepath)
        if out_dirpath is None:
            out_dirpath = dirpath
        out_filepath = os.path.join(out_dirpath, out_filename)

        acqp_filepath = _get_single(
            fid_dirpath, acqp_name, allowed_ext)
        method_filepath = _get_single(
            fid_dirpath, method_name, allowed_ext)

        dseq_filepaths = sorted(
            mrt.utils.flistdir(
                fid_dirpath, _to_patterns(dseq_name, allowed_ext)))
        reco_filepaths = sorted(
            mrt.utils.flistdir(
                fid_dirpath, _to_patterns(reco_name, allowed_ext)))

        acqp_s, acqp, acqp_c = jcampdx.read(acqp_filepath)
        method_s, method, method_c = jcampdx.read(method_filepath)
        scan_num, sample_id = _get_scan_num_sample_id(acqp_c)
        scan_name = mrt.utils.safe_filename(acqp['ACQ_scan_name'])
        acq_method = mrt.utils.safe_filename(acqp['ACQ_method'])
        reco_flag = mrt.naming.NEW_RECO_ID

        if custom_reco:
            load_info = _get_load_bin_info_fid(acqp, method)

            if custom_reco == 'cx':
                reco_flag = mrt.naming.ITYPES['cx']
                cx_filepath = mrt.utils.change_ext(
                    out_filepath.format_map(locals()), mrt.utils.EXT['niz'])
                if not os.path.isdir(os.path.dirname(cx_filepath)):
                    os.makedirs(os.path.dirname(cx_filepath))

                if mrt.utils.check_redo(
                        [fid_filepath, acqp_filepath, method_filepath],
                        [cx_filepath], force):
                    arr = _load_bin(fid_filepath, **load_info)
                    arr = _reco_from_fid(arr, acqp, method)
                    mrt.input_output.save(cx_filepath, arr)
                    msg('CX:  {}'.format(os.path.basename(cx_filepath)),
                        verbose, D_VERB_LVL)

            elif custom_reco == 'mag_phs':
                reco_flag = mrt.naming.ITYPES['mag']
                mag_filepath = mrt.utils.change_ext(
                    out_filepath.format_map(locals()), mrt.utils.EXT['niz'])
                if not os.path.isdir(os.path.dirname(mag_filepath)):
                    os.makedirs(os.path.dirname(mag_filepath))

                reco_flag = mrt.naming.ITYPES['phs']
                phs_filepath = mrt.utils.change_ext(
                    out_filepath.format_map(locals()), mrt.utils.EXT['niz'])
                if not os.path.isdir(os.path.dirname(phs_filepath)):
                    os.makedirs(os.path.dirname(phs_filepath))

                if mrt.utils.check_redo(
                        [fid_filepath, acqp_filepath, method_filepath],
                        [mag_filepath, phs_filepath], force):
                    reco_flag = mrt.naming.ITYPES['mag']

                    arr = _load_bin(fid_filepath, **load_info)
                    arr = _reco_from_fid(arr, acqp, method)
                    mrt.input_output.save(mag_filepath, np.abs(arr))
                    msg('MAG: {}'.format(os.path.basename(mag_filepath)),
                        verbose, D_VERB_LVL)
                    mrt.input_output.save(phs_filepath, np.angle(arr))
                    msg('PHS: {}'.format(os.path.basename(phs_filepath)),
                        verbose, D_VERB_LVL)

            elif custom_reco == 're_im':
                reco_flag = mrt.naming.ITYPES['re']
                re_filepath = mrt.utils.change_ext(
                    out_filepath.format_map(locals()), mrt.utils.EXT['niz'])
                if not os.path.isdir(os.path.dirname(re_filepath)):
                    os.makedirs(os.path.dirname(re_filepath))

                reco_flag = mrt.naming.ITYPES['im']
                im_filepath = mrt.utils.change_ext(
                    out_filepath.format_map(locals()), mrt.utils.EXT['niz'])
                if not os.path.isdir(os.path.dirname(im_filepath)):
                    os.makedirs(os.path.dirname(im_filepath))

                if mrt.utils.check_redo(
                        [fid_filepath, acqp_filepath, method_filepath],
                        [re_filepath, im_filepath], force):
                    arr = _load_bin(fid_filepath, **load_info)
                    arr = _reco_from_fid(arr, acqp, method)
                    mrt.input_output.save(re_filepath, np.abs(arr))
                    msg('RE: {}'.format(os.path.basename(re_filepath)),
                        verbose, D_VERB_LVL)
                    mrt.input_output.save(im_filepath, np.angle(arr))
                    msg('IM: {}'.format(os.path.basename(im_filepath)),
                        verbose, D_VERB_LVL)

        else:
            warning = 'Voxel data and shapes may be incorrect.'
            warnings.warn(warning)
            for dseq_filepath, reco_filepath \
                    in zip(dseq_filepaths, reco_filepaths):
                reco_s, reco, reco_c = jcampdx.read(reco_filepath)
                reco_flag = _get_reco_num(reco_c)

                cx_filepath = mrt.utils.change_ext(
                    out_filepath.format_map(locals()), mrt.utils.EXT['niz'])
                if not os.path.isdir(os.path.dirname(cx_filepath)):
                    os.makedirs(os.path.dirname(cx_filepath))

                load_info = _get_load_bin_info_reco(reco, method)

                if mrt.utils.check_redo(
                        [dseq_filepath, reco_filepath], [cx_filepath], force):
                    arr = _load_bin(dseq_filepath, **load_info)
                    arr = _reco_from_bin(arr, reco, method)
                    mrt.input_output.save(cx_filepath, arr)
                    msg('NIZ: {}'.format(os.path.basename(cx_filepath)),
                        verbose, D_VERB_LVL)


# ======================================================================
if __name__ == '__main__':
    msg(__doc__.strip())
    doctest.testmod()

    batch_extract(
        '/home/raid1/metere/hd3/sandbox/hmri'
        '/Specimen_170814_1_0_Study_20170814_080054/2',
        custom_reco='mag_phs',
        force=True)

# ======================================================================
elapsed(os.path.basename(__file__))
