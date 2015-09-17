#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Extract a mask from an image (with specialized options for brain extraction).

| Workflow is:
* Brain extraction with FSL's BET (if any)
* Gaussian filter (smoothing) with specified sigma
* histogram deviation reduction by a specified factor
* masking values using a relative threshold (and thresholding method)
* binary erosion(s) witha specified number of iterations.
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
DATE_INFO = {'day': 18, 'month': 'Sep', 'year': 2014}
DATE = ' '.join([str(v) for k, v in sorted(DATE_INFO.items())])
LICENSE = 'License GPLv3: GNU General Public License version 3'
COPYRIGHT = 'Copyright (C) ' + str(DATE_INFO['year'])
# first non-empty line of __doc__
DOC_FIRSTLINE = [line for line in __doc__.splitlines() if line][0]


# ======================================================================
# :: Python Standard Library Imports
import os  # Miscellaneous operating system interfaces
import shutil  # High-level file operations
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
import mri_tools.lib.base as mrb
import mri_tools.lib.utils as mru
import mri_tools.lib.nifti as mrn
# import mri_tools.lib.geom_mask as mrgm
# import mri_tools.lib.mp2rage as mp2rage


# ======================================================================
# :: supported verbosity levels (level 4 skipped on purpose)
VERB_LVL = {'none': 0, 'low': 1, 'medium': 2, 'high': 3, 'debug': 5}
D_VERB_LVL = VERB_LVL['low']


# ======================================================================
def compute_mask(
        in_filepath, out_filepath, bet_params, smoothing,
        hist_dev_factor, rel_threshold, erosion_iter, force, verbose):
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
    if out_filepath and verbose >= VERB_LVL['none']:
        out_dirpath = os.path.dirname(out_filepath)
        print('OutDir:\t{}'.format(out_dirpath))
    mru.compute_mask(
        in_filepath, out_filepath, bet_params, smoothing, hist_dev_factor,
        rel_threshold, erosion_iter, force, verbose)


# ======================================================================
def handle_arg():
    """
    Handle command-line application arguments.
    """
    # :: Define DEFAULT values
    # verbosity
    d_verbose = D_VERB_LVL
    # input filepath
    d_input = None
    # output filepath
    d_output = None
    # FSL's BET brain extraction algorithm parameters
    d_bet_params = ''
    # smoothing
    d_smoothing = 0.0
    # histogram deviation factor
    d_hist_dev_factor = 5.0
    # relative threshold in the (0, 1) range
    d_rel_threshold = 0.1
    # number of erosion iterations
    d_erosion_iter = 0
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
    arg_parser.add_argument(
        '-i', '--input', metavar='FILE',
        default=d_input, required=True,
        help='set input filepath [%(default)s]')
    arg_parser.add_argument(
        '-o', '--output', metavar='FILE',
        default=d_output,
        help='set output filepath [%(default)s]')
    arg_parser.add_argument(
        '-b', '--bet_params', metavar='STR',
        default=d_bet_params,
        help='parameters to FSL\'s BET tool. Empty STR to skip [%(default)s]')
    arg_parser.add_argument(
        '-s', '--smoothing', metavar='SIGMA',
        type=float, default=d_smoothing,
        help='value of Gaussian smoothing\'s sigma [%(default)s]')
    arg_parser.add_argument(
        '-r', '--hist_dev_factor', metavar='X',
        type=float, default=d_hist_dev_factor,
        help='histogram deviation factor [%(default)s]')
    arg_parser.add_argument(
        '-t', '--rel_threshold', metavar='R',
        type=float, default=d_rel_threshold,
        help='relative thresholding value [%(default)s]')
    arg_parser.add_argument(
        '-e', '--erosion_iter', metavar='N',
        type=int, default=d_erosion_iter,
        help='number of postprocess binary erosion iterations [%(default)s]')
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
    if ARGS.verbose > VERB_LVL['low']:
        print(__doc__)
    begin_time = time.time()
    compute_mask(
        ARGS.input,
        ARGS.output,
        ARGS.bet_params,
        ARGS.smoothing,
        ARGS.hist_dev_factor,
        ARGS.rel_threshold,
        ARGS.erosion_iter,
        ARGS.force,
        ARGS.verbose)
    end_time = time.time()
    if ARGS.verbose > VERB_LVL['low']:
        print('ExecTime: ', datetime.timedelta(0, end_time - begin_time))
