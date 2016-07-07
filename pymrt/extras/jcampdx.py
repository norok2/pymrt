#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
pymrt: read files with a JCAMP-DX-like structure.

The module is NumPy-aware.
"""

# ======================================================================
# :: Future Imports
from __future__ import division
from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals

# ======================================================================
# :: Python Standard Library Imports
# import os  # Miscellaneous operating system interfaces
# import shutil  # High-level file operations
# import math  # Mathematical functions
import time  # Time access and conversions
import datetime  # Basic date and time types
# import operator  # Standard operators as functions
# import collections  # High-performance container datatypes
# import argparse  # Parser for command-line options, arguments and subcommands
# import itertools  # Functions creating iterators for efficient looping
# import subprocess  # Subprocess management
# import multiprocessing  # Process-based parallelism
# import csv  # CSV File Reading and Writing [CSV: Comma-Separated Values]
# import json  # JSON encoder and decoder [JSON: JavaScript Object Notation]

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
# import pymrt.base as mrb
# import pymrt.utils as mru
# import pymrt.input_output as mrio
# import pymrt.geometry as mrg
# from pymrt.sequences import mp2rage


# ======================================================================
def _strip_comments(data, comment_start_str='$$'):
    """
    Strip comments, i.e. lines starting with: $$
    """
    lines = data.split('\n')
    code_lines, comment_lines = [], []
    for line in lines:
        if line.startswith(comment_start_str):
            comment_lines.append(line)
        else:
            code_lines.append(line)
    return '\n'.join(code_lines), '\n'.join(comment_lines)


# ======================================================================
def _auto_convert(val_str):
    """
    Convert value to numeric if possible, or strip '<' and '>' from strings.
    """
    if _has_str_decorator(val_str):
        val = val_str[1:-1]
    else:
        try:
            val = int(val_str)
        except ValueError:
            try:
                val = float(val_str)
            except ValueError:
                val = val_str
    return val


# ======================================================================
def _has_str_decorator(val):
    """
    Determine if string is delimited by '<' and '>'.
    """
    str_start_id, str_end_id = '<', '>'
    return val.startswith(str_start_id) and val.endswith(str_end_id)


# ======================================================================
def _parse_record(record):
    """
    Parse record to adapt for JCAMP-DX-like specification.
    """
    val = record.strip()
    val = val.replace('\n', '').strip()
    group_start_id, group_end_id = '(', ')'
    list_sep = ', '
    if val.startswith(group_start_id) and val.endswith(group_end_id):
        val_list = val[1:-1].split(list_sep)
        val = [_auto_convert(value) for value in val_list]
    elif val.startswith(group_start_id):
        val_hdr_str = \
            val[val.index(group_start_id) + 1:val.index(group_end_id)]
        val_data_str = val[val.index(group_end_id) + 1:]
        if _has_str_decorator(val_data_str):
            val = _auto_convert(val_data_str)
        else:
            val_hdr = [int(dim)
                       for dim in val_hdr_str.split(',')]
            val_data = [_auto_convert(value)
                        for value in val_data_str.split(' ')]
            val = np.array(val_data).reshape(val_hdr)
    else:
        val = _auto_convert(val)
    return val


# ======================================================================
def read(filepath):
    """
    Read files with JCAMP-DX-like structure.

    Args:
        filepath (str|unicode): Path to file to parse.

    Returns:
        ldr_std (dict): Dictionary of the standard Labelled-Data-Records found.
        ldr_user (dict): Dictionary of User-defined Labelled-Data-Records found.
        comments (str): Comment lines.
    """
    ldr_sep, ldr_usr_sep, ldr_dict_sep = '##', '$', '='
    with open(filepath, 'rb') as in_file:
        ldr_std, ldr_user = {}, {}
        data = in_file.read()
        data, comments = _strip_comments(data)
        ldrs = [ldr for ldr in data.split(ldr_sep) if ldr]
        ldr_list = []
        for ldr in ldrs:
            ldr_name, ldr_val = ldr.split(ldr_dict_sep)
            ldr_val = _parse_record(ldr_val)
            if ldr.startswith(ldr_usr_sep):
                ldr_user[ldr_name.strip(ldr_usr_sep)] = ldr_val
            else:
                ldr_std[ldr_name] = ldr_val
            ldr_list.append(ldr)
    return ldr_std, ldr_user, comments


# ======================================================================
def test():
    """
    Test module functionalities with files provided in the package.

    Args:
        None.

    Returns:
        None.
    """
    test_filepath_list = []  # 'test/file1.jcampdx']
    try:
        for test_filepath in test_filepath_list:
            read(test_filepath)
    except Exception as exception:  # This has to catch all exceptions.
        print(exception)
        print('Test not passed.')
    else:
        print('All test were passed successfully.')


# ======================================================================
if __name__ == '__main__':
    print(__doc__)
    # time the tests
    begin_time = time.time()
    test()
    end_time = time.time()
    print('ExecTime: ', datetime.timedelta(0, end_time - begin_time))
