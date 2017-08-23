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
# import math  # Mathematical functions
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
def _load_bin(filepath, dtype='int', mode='<', is_complex=True, dry=False):
    count = 1
    fmt = mode + str(count) + mrt.utils.DTYPE_STR[dtype]
    byte_count = struct.calcsize(fmt)
    with mrt.utils.zopen(filepath, 'rb') as file_obj:
        file_size = file_obj.seek(0, 2)
        file_obj.seek(0)
        if dry:
            arr = np.zeros(file_size // byte_count)
        else:
            arr, n = mrt.utils.read_stream(
                file_obj, dtype, file_size // byte_count, mode)
            assert (n == file_size)  # check that data read is consistent
            arr = np.array(arr)
        if is_complex:
            arr = arr[::2] + 1j * arr[1::2]
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
def _reco(fid_arr, acqp, method):
    is_cartesian = True
    if is_cartesian:
        # num_images = acqp['NI']
        # num_avgs = acqp['NA']
        # num_exps = acqp['NAE']
        # num_rep = acqp['NR']
        # base_shape = tuple(acqp['ACQ_size'])
        base_shape = tuple(method['PVM_Matrix'])
        try:
            fid_arr = fid_arr.reshape(base_shape + (-1,))
        except ValueError:
            pass
        print(base_shape, fid_arr.size, fid_arr.shape)
        print(mrt.utils.factorize(fid_arr.size))
    else:
        raise NotImplementedError
    return fid_arr


# ======================================================================
def batch_extract(
        dirpath,
        out_filename='niz/{scan_num}__{acq_method}_{scan_name}_{reco_num}',
        out_dirpath=None,
        custom_reco=False,
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
        out_filepath ():
        out_dirpath ():
        reco ():
        fid_name ():
        acqp_name ():
        method_name ():
        allowed_ext ():

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
        reco_num = mrt.naming.NEW_RECO_ID

        if custom_reco:
            niz_filepath = mrt.utils.change_ext(
                out_filepath.format_map(locals()), mrt.utils.EXT['niz'])
            if not os.path.isdir(os.path.dirname(niz_filepath)):
                os.makedirs(os.path.dirname(niz_filepath))

            if mrt.utils.check_redo(
                    [fid_filepath, acqp_filepath, method_filepath],
                    [niz_filepath], force):
                arr = _load_bin(fid_filepath, dtype='int', is_complex=True)
                arr = _reco(arr, acqp, method)
                mrt.input_output.save(niz_filepath, arr)
                msg('NIZ: {}'.format(os.path.basename(niz_filepath)),
                    verbose, D_VERB_LVL)

        else:
            warnings.warn('Voxel data and shapes may be incorrect.')
            for dseq_filepath, reco_filepath \
                    in zip(dseq_filepaths, reco_filepaths):
                reco_s, reco, reco_c = jcampdx.read(reco_filepath)
                reco_num = _get_reco_num(reco_c)

                niz_filepath = mrt.utils.change_ext(
                    out_filepath.format_map(locals()), mrt.utils.EXT['niz'])
                if not os.path.isdir(os.path.dirname(niz_filepath)):
                    os.makedirs(os.path.dirname(niz_filepath))

                if mrt.utils.check_redo(
                        [dseq_filepath, reco_filepath], [niz_filepath], force):
                    arr = _load_bin(
                        dseq_filepath, dtype='short', is_complex=False)

                    base_shape = reco['RECO_size']
                    base_shape = np.roll(base_shape, 1)

                    arr = np.swapaxes(
                        arr.reshape((-1,) + tuple(base_shape)), 0, -1)

                    mrt.input_output.save(niz_filepath, arr)
                    msg('NIZ: {}'.format(os.path.basename(niz_filepath)),
                        verbose, D_VERB_LVL)


# ======================================================================
if __name__ == '__main__':
    msg(__doc__.strip())
    doctest.testmod()

    batch_extract(
        '/home/raid1/metere/hd3/sandbox/hmri'
        '/Specimen_170814_1_0_Study_20170814_080054',
        custom_reco=True,
        force=False)

# ======================================================================
elapsed(os.path.basename(__file__))
