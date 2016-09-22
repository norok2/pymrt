#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DESCRIPTION OF THE FILE CONTENT IS MISSING!
"""

# ======================================================================
# :: Future Imports
from __future__ import (
    division, absolute_import, print_function, unicode_literals)

# ======================================================================
# :: Python Standard Library Imports
# import os  # Miscellaneous operating system interfaces
# import sys  # System-specific parameters and functions
# import shutil  # High-level file operations
# import platform  # Access to underlying platform’s identifying data
# import math  # Mathematical functions
# import time  # Time access and conversions
# import datetime  # Basic date and time types
# import re  # Regular expression operations
# import operator  # Standard operators as functions
# import collections  # High-performance container datatypes
import argparse  # Parser for command-line options, arguments and subcommands

# import itertools  # Functions creating iterators for efficient looping
# import functools  # Higher-order functions and operations on callable objects
# import subprocess  # Subprocess management
# import multiprocessing  # Process-based parallelism
# import csv  # CSV File Reading and Writing [CSV: Comma-Separated Values]
# import json  # JSON encoder and decoder [JSON: JavaScript Object Notation]
# import shlex  # Simple lexical analysis

# Python interface to Tcl/Tk
# try:
#     import tkinter as tk
#     import tkinter.ttk as ttk
#     import tkinter.messagebox as messagebox
#     import tkinter.filedialog as filedialog
#     import tkinter.simpledialog as simpledialog
# except ImportError:
#     import Tkinter as tk
#     import ttk
#     import tkMessageBox as messagebox
#     import tkFileDialog as filedialog
#     import tkSimpleDialog as simpledialog

# Configuration file parser
# try:
#     import configparser
# except ImportError:
#     import ConfigParser as configparser

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
# import dicom as pydcm  # PyDicom (Read, modify and write DICOM files.)

# :: External Imports Submodules
# import matplotlib.pyplot as plt  # Matplotlib's pyplot: MATLAB-like syntax
# import mayavi.mlab as mlab  # Mayavi's mlab: MATLAB-like syntax
# import scipy.optimize  # SciPy: Optimization Algorithms
# import scipy.integrate  # SciPy: Integrations facilities
# import scipy.constants  # SciPy: Mathematal and Physical Constants
# import scipy.ndimage  # SciPy: ND-image Manipulation

# ======================================================================
# :: Version
__version__ = '0.0.0.0'

# ======================================================================
# :: Project Details
INFO = {
    'author': 'Riccardo Metere <metere@cbs.mpg.de>',
    'copyright': 'Copyright (C) 2015',
    'license': 'GNU General Public License version 3 or later (GPLv3+)',
    'notice':
        """
This program is free software and it comes with ABSOLUTELY NO WARRANTY.
It is covered by the GNU General Public License version 3 (GPLv3).
You are welcome to redistribute it under its terms and conditions.
        """,
    'version': __version__
}

# ======================================================================
# :: supported verbosity levels (level 4 skipped on purpose)
VERB_LVL = {'none': 0, 'low': 1, 'medium': 2, 'high': 3, 'debug': 5}
D_VERB_LVL = VERB_LVL['low']


# ======================================================================
# :: additional globals

import os  # Miscellaneous operating system interfaces
import numpy as np  # NumPy (multidimensional numerical arrays library)


# ======================================================================
def read(
        basename,
        dirpath='.'):
    """
    Read a CFL header+data pair.

    Args:
        basename (str): The base filename.
            Corresponding '.hdr' and '.cfl' files must exist.
        dirpath (str): The working directory.

    Returns:
        array (ndarray): The data read.

    Examples:

    """
    filepath = os.path.join(dirpath, basename)

    # load header
    with open(filepath + '.hdr', 'r') as header_file:
        header_file.readline()  # skip comment line
        dim_line = header_file.readline()

    # obtain the shape of the image
    shape = [int(i) for i in dim_line.strip().split(' ')]
    # remove trailing singleton dimensions from shape
    while shape[-1] == 1:
        shape.pop(-1)
    # calculate the data size
    data_size = int(np.prod(shape))

    # load data
    with open(filepath + ".cfl", "r") as data_file:
        array = np.fromfile(
            data_file, dtype=np.complex64, count=data_size)

    # note: BART uses FORTRAN-style memory allocation
    return array.reshape(shape, order='F')


# ======================================================================
def write(
        array,
        basename,
        dirpath='.'):
    """
    Write a CFL header+data pair.

    Args:
        array (ndarray): The array to save.
        basename (str): The base filename.
            Corresponding '.hdr' and '.cfl' files will be created/overwritten.
        dirpath (str): The working directory.

    Returns:
        None.
    """
    filepath = os.path.join(dirpath, basename)

    # save header
    with open(filepath + '.hdr', 'w') as header_file:
        header_file.write(str('# Dimensions\n'))
        header_file.write(str(' '.join([str(n) for n in array.shape])) + '\n')

    # save data
    with open(filepath + '.cfl', 'w') as data_file:
        # note: BART uses FORTRAN-style memory allocation
        array.astype(np.complex64, 'F').tofile(data_file)


# ======================================================================
if __name__ == '__main__':
    pass
