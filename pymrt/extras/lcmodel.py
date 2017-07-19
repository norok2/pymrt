#!/usr/
# -*- coding: utf-8 -*-
"""
PyMRT: read_output files from LCModel.
"""

# ======================================================================
# :: Future Imports
from __future__ import(
    division, absolute_import, print_function, unicode_literals)

# ======================================================================
# :: Python Standard Library Imports
# import os  # Miscellaneous operating system interfaces
# import shutil  # High-level file operations
# import math  # Mathematical functions
import time  # Time access and conversions
import datetime  # Basic date and time types
# import operator  # Standard operators as functions
# import collections  # Container datatypes
# import argparse  # Parser for command-line options, arguments and subcommands
import re  # Regular expression operations
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
import pymrt as mrt
import pymrt.utils

# import pymrt.naming
# import pymrt.input_output
# import pymrt.geometry
# from pymrt.sequences import mp2rage
from pymrt import elapsed, print_elapsed

# ======================================================================
# :: Header information
HDR_INPUT = {
    'section_id': '$',
    'section_end': 'END',
    'num': {'section': 'SEQPAR', 'label': 'NumberOfPoints'}
}

HDR_OUTPUT = {
    'metabolites': 'Conc.  %SD   /Cre   Metabolite',
    'data_cs': 'points on ppm-axis',  # chemical shift in ppm
    'data_s': 'phased data points follow',
    'data_fit': 'points of the fit to the data follow',
    'data_bg': 'background values follow',
}


# ======================================================================
def percent_to_float(text):
    """
    Convert percent text to float.

    Args:
        text (text): The number to convert.

    Returns:
        val (float): The converted Value.

    Examples:
        >>> percent_to_float('50%')
        0.5
    """
    try:
        val = float(text.strip('%')) / 100
    except ValueError:
        val = np.nan
    return val


# ======================================================================
def read_input(
        filepath):
    """
    Read LCModel input files (without extension).

    Args:
        filepath (str): The input filepath.

    Returns:
        content (dict): The extracted information.
            The input data is stored in `data`.
            Any extra information is stored into the corresponding sections.
    """
    content = {}
    with open(filepath, 'r') as file:
        i = 0
        section = 'base'
        for line in file:
            if HDR_INPUT['section_id'] in line:
                section = line.strip().strip(HDR_INPUT['section_id'])
                if section == HDR_INPUT['section_end']:
                    section = 'base'
                else:
                    content[section] = {}
            try:
                tokens = [item.strip() for item in line.split('=')]
                label = tokens[0]
                value = ' '.join(tokens[1:])
                if section == HDR_INPUT['num']['section'] and \
                                label == HDR_INPUT['num']['label']:
                    content['data'] = np.zeros((int(value), 2))
            except ValueError:
                pass
            else:
                if section and label and value:
                    content[section][label] = value
            try:
                content['data'][i, :] = [float(val) for val in line.split()]
            except (ValueError, TypeError):
                pass
            else:
                i += 1
    return content


# ======================================================================
def read_output(
        filepath):
    """
    Read LCModel output files (with extension `.grid_coord` and `.txt`).

    Args:
        filepath (str): The filepath to read_output

    Returns:
        content (dict): The extracted information.
             The concentrations and any extra infos are stored respectively in:
              - `metabolites`
              - `extra`
             The spectral data are stored respectively in:
              - `data_cs`: the chemical shift in ppm (the x-axis);
              - `data_s`: the spectrum in arb.units (the y-axis);
              - `data_fit`: the spectrum obtained from the fitting in arb.units;
              - `data_bg`: the background of the spectrum  in arb.units.
    """
    content = {}
    with open(filepath, 'r') as file:
        # read_output file until it finds the concentrations labels
        for line in file:
            if HDR_OUTPUT['metabolites'] in line:
                content['metabolites'] = {}
                break
        # read_output concentrations
        for line in file:
            try:
                if line.strip():
                    val, err, over_cre, name = line.split()
                    content['metabolites'][name] = {
                        'val': float(val),
                        'percent_err': percent_to_float(err),
                        'over_cre': float(over_cre)}
            except ValueError:
                break

        # read_output file until it finds the first spectra labels
        labels = ['data_cs', 'data_s', 'data_fit', 'data_bg']
        for label in labels:
            vals = []
            if label == labels[0]:
                for line in file:
                    if HDR_OUTPUT[label] in line:
                        break
            for line in file:
                try:
                    vals.extend([float(val) for val in line.split()])
                except ValueError:
                    content[label] = np.array(vals)
                    break

        # todo: improve support for all output
        # read_output extra information
        file.seek(0)
        for line in file:
            if '=' in line or ':' in line or ',' in line:
                if 'extra' not in content:
                    content['extra'] = {}
                label = None
                items = re.split(r'=| {2}|, |: ', line)
                for item in items:
                    item = item.strip()
                    if item:
                        while item.count('\'') % 2 != 0:
                            item += file.readline().strip()
                        if not label:
                            label = item.strip()
                        if label not in content['extra']:
                            content['extra'][label] = []
                        add_item = \
                            label != item and \
                            label in content['extra'] and \
                            not content['extra'][label] or \
                            mrt.utils.is_number(item)
                        if add_item:
                            content['extra'][label].append(item)
                        else:
                            label = item.strip()
    return content


# ======================================================================
def test():
    """
    Test module functionalities with files provided in the package.

    Args:
        None.

    Returns:
        None.
    """
    test_filepath_list = []  # 'test/file1.grid_coord']
    try:
        for test_filepath in test_filepath_list:
            read_output(test_filepath)
    except Exception as exception:  # This has to catch all exceptions.
        print(exception)
        print('Test not passed.')
    else:
        print('All test were passed successfully.')


# ======================================================================
if __name__ == '__main__':
    msg(__doc__.strip())

    test()

    # todo: move this test to unittest
    s = '/scr/beryllium1/mr16/RM/lcmLONGc_128avg' \
        '/LONGc_128avg160419_RL6T_MET20_Step01_WAT.txt'

    d = read_output(s)
    for k, v in sorted(d['metabolites'].items()):
        print('{} : {}'.format(k, v))
    print()

    for k, v in sorted(d['extra'].items()):
        print('{} : {}'.format(k, v))
    print()
    if 'data_cs' in d:
        import matplotlib.pyplot as plt

        plt.figure()
        plt.plot(d['data_cs'], d['data_s'], '-b')
        plt.plot(d['data_cs'], d['data_fit'], '-r')
        plt.plot(d['data_cs'], d['data_bg'], '-y')
        plt.show()

    s = '/scr/beryllium1/mr16/RM/lcmLONGc_112avg' \
        '/LONGc_112avg160419_RL6T_MET20_Step01'
    c = read_input(s)
    print(c['data'].shape, c)

    elapsed('test lcmodel i/o')
    print_elapsed()
    print()
