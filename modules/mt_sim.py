#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
mr_lib: Matrix Algebra Formalism to analize Magnetization Transfer Experiments.
"""


# ======================================================================
# :: Future Imports
from __future__ import division
from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals


__version__ = '0.0.0.1'
# $Source$


# ======================================================================
# :: Custom Module Details
AUTHOR = 'Riccardo Metere'
CONTACT = 'metere@cbs.mpg.de'
DATE_INFO = {'day': 19, 'month': 'Sep', 'year': 2014}
DATE = ' '.join([str(v) for k, v in sorted(DATE_INFO.items())])
LICENSE = 'License GPLv3: GNU General Public License version 3'
COPYRIGHT = 'Copyright (C) ' + str(DATE_INFO['year'])
# first non-empty line of __doc__
DOC_FIRSTLINE = [line for line in __doc__.splitlines() if line][0]


# ======================================================================
# :: Python Standard Library Imports
# import os  # Miscellaneous operating system interfaces
# import shutil  # High-level file operations
# import math  # Mathematical functions
import time  # Time access and conversions
import datetime  # Basic date and time types
# import operator  # Standard operators as functions
import collections  # High-performance container datatypes
import argparse  # Parser for command-line options, arguments and subcommands
# import itertools  # Functions creating iterators for efficient looping
# import functools  # Higher-order functions and operations on callable objects
# import subprocess  # Subprocess management
# import multiprocessing  # Process-based parallelism
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

# :: Local Imports



# ======================================================================
# :: supported verbosity levels (level 4 skipped on purpose)
VERB_LVL = {'none': 0, 'low': 1, 'medium': 2, 'high': 3, 'debug': 5}
D_VERB_LVL = VERB_LVL['low']


# ======================================================================
def my_func(my_param):
    """
    Sample function.

    Parameters
    ==========
    my_param : type
        Sample parameter.

    Returns
    =======
    my_rr : rtype
        Sample return.

    """
    my_rr = my_param
    return my_rr


# ======================================================================
def handle_arg():
    """
    Handle command-line application arguments.
    """
    # :: Define DEFAULT values
    # verbosity
    d_verbose = D_VERB_LVL
    # :: Create Argument Parser
    arg_parser = argparse.ArgumentParser(
        description=__doc__,
        epilog='v.{} - {} {} <{}>\n{}'.format(
            __version__, COPYRIGHT, AUTHOR, CONTACT, LICENSE),
        formatter_class=argparse.RawDescriptionHelpFormatter)
    # :: Add POSIX standard arguments
    arg_parser.add_argument(
        '--ver', '--version',
        version='%(prog)s {}\n{}\n{} {} <{}>\n{}'.format(
            __version__, DOC_FIRSTLINE, COPYRIGHT, AUTHOR, CONTACT, LICENSE),
        action='version')
    arg_parser.add_argument(
        '-v', '--verbose',
        action='count', default=d_verbose,
        help='increase the level of verbosity [%(default)s]')
    # :: Add additional arguments
    arg_parser.add_argument(
        '-f', '--force',
        action='store_true',
        help='force new processing [%(default)s]')
    # nothing here yet!
    # avoid mandatory arguments whenever possible
    return arg_parser


# ======================================================================
if __name__ == '__main__':
    # :: handle program parameters
    ARG_PARSER = handle_arg()
    ARGS = ARG_PARSER.parse_args()
    # :: print debug info
    if ARGS.verbose == VERB_LVL['debug']:
        ARG_PARSER.print_help()
        print()
        print('II:', 'Parsed Arguments:', ARGS)
    print(__doc__)
    begin_time = time.time()
    # :: TODO: add your timed code here
    end_time = time.time()
    if ARGS.verbose > VERB_LVL['low']:
        print('ExecTime: ', datetime.timedelta(0, end_time - begin_time))

