#!/usr/
# -*- coding: utf-8 -*-
"""
pymrt: read files from LCModel.
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
# :: Header information
HDR = {
    'metabolites': 'Conc.  %SD   /Cre   Metabolite',
    'data_mag': ' points ',
    'data_phs': 'phased data points follow',
    'data_fit': 'points of the fit to the data follow',
    'data_bg': 'background values follow',

    'extra': {
        'snr': 'S/N',
        'fwhm': 'FWHM',
        'echo_time': 'echot',
    },
}


# ======================================================================
def _percent_to_float(text):
    """
    Convert percent text to float.

    Args:
        text (text): The number to convert.

    Returns:
        val (float): The converted Value.

    Examples:
        >>>
    """
    try:
        val = float(text.strip('%')) / 100
    except ValueError:
        val = np.nan
    return val


# ======================================================================
def read(
        filepath,
        extras=('snr', 'fwhm')):
    """
    Read LCModel results text file.

    Args:
        filepath (str): The filepath to read
        extras (tuple[str]): The extra information to extract

    Returns:
        lcmodel (dict): The extracted information.
             The concentration information are stored in `metabolites`.
             The data for the spectra is stored respectively in:
             `data_mag`, `data_mag`, `data_mag`, and `data_mag`.
             The extra information is stored in the corresponding entries.
    """
    lcmodel = {}
    with open(filepath, 'r') as file:
        # read file until it finds the concentrations labels
        for line in file:
            if HDR['metabolites'] in line:
                lcmodel['metabolites'] = {}
                break
        # read concentrations
        for line in file:
            try:
                if line.strip():
                    val, err, over_cre, name = line.split()
                    lcmodel['metabolites'][name] = {
                        'val': float(val),
                        'percent_err': _percent_to_float(err),
                        'over_cre': float(over_cre)}
            except ValueError:
                break

        # read file until it finds the first spectra labels
        labels = ['data_mag', 'data_phs', 'data_fit', 'data_bg']
        for label in labels:
            vals = []
            if label == labels[0]:
                for line in file:
                    if HDR[label] in line:
                        break
            for line in file:
                try:
                    vals += [float(val) for val in line.split()]
                except ValueError:
                    lcmodel[label] = np.array(vals)
                    break

        # read extra information
        file.seek(0)
        for line in file:
            for extra in extras:
                if extra in HDR['extra'] and HDR['extra'][extra] in line:
                    tokens = line.split()
                    i = 0
                    for i, token in enumerate(tokens):
                        if token == HDR['extra'][extra]:
                            break
                    for token in tokens[i:]:
                        try:
                            value = float(token)
                        except ValueError:
                            continue
                    lcmodel[extra] = value if value else None
    return lcmodel


# ======================================================================
def test():
    """
    Test module functionalities with files provided in the package.

    Args:
        None.

    Returns:
        None.
    """
    test_filepath_list = []  # 'test/file1.coord']
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

    s = '/scr/beryllium1/mr16/RM/lcmLONGc_128avg' \
        '/LONGc_128avg160419_RL6T_MET20_Step01_WAT.coord'

    print(read(s))
