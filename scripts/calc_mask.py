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

# ======================================================================
# :: Python Standard Library Imports
import os  # Miscellaneous operating system interfaces
# import math  # Mathematical functions
import time  # Time access and conversions
import datetime  # Basic date and time types
# import operator  # Standard operators as functions
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
# import mri_tools.base as mrb
# import mri_tools.utils as mru
# import mri_tools.nifti as mrn
# import mri_tools.computation as mrc
import mri_tools.correlation as mrl
# import mri_tools.geometry as mrg
# from mri_tools.sequences import mp2rage
# import dcmpi.common as dcmlib

from mri_tools import INFO
from mri_tools import VERB_LVL
from mri_tools import D_VERB_LVL


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
    d_percentile_range = (0.0, 1.0)
    # relative value threshold in the (0, 1) range
    d_val_threshold = 0.1
    # comparison method
    d_comparison = '>'
    # relative size threshold in the (0, 1) range
    d_size_threshold = 0.02
    # number of erosion iterations
    d_erosion_iter = 0
    # number of dilation iterations
    d_dilation_iter = 0
    # :: Create Argument Parser
    arg_parser = argparse.ArgumentParser(
        description=__doc__,
        epilog='v.{} - {}\n{}'.format(
            INFO['version'], ', '.join(INFO['authors']), INFO['license']),
        formatter_class=argparse.RawDescriptionHelpFormatter)
    # :: Add POSIX standard arguments
    arg_parser.add_argument(
        '--ver', '--version',
        version='%(prog)s - ver. {}\n{}\n{} {}\n{}'.format(
            INFO['version'],
            next(line for line in __doc__.splitlines() if line),
            INFO['copyright'], ', '.join(INFO['authors']),
            INFO['notice']),
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
        '-r', '--percentile_range', metavar=('MIN', 'MAX'),
        type=float, nargs=2, default=d_percentile_range,
        help='percentiles to obtain values range for threshold [%(default)s]')
    arg_parser.add_argument(
        '-a', '--val_threshold', metavar='R',
        type=float, default=d_val_threshold,
        help='value threshold (relative to values range) [%(default)s]')
    arg_parser.add_argument(
        '-c', '--comparison', metavar='STR',
        default=d_comparison,
        help='comparison directive [%(default)s]')
    arg_parser.add_argument(
        '-z', '--size_threshold', metavar='Z',
        type=float, default=d_size_threshold,
        help='size threshold (relative to matrix size) [%(default)s]')
    arg_parser.add_argument(
        '-e', '--erosion_iter', metavar='N',
        type=int, default=d_erosion_iter,
        help='number of postprocess binary erosion iterations [%(default)s]')
    arg_parser.add_argument(
        '-d', '--dilation_iter', metavar='N',
        type=int, default=d_erosion_iter,
        help='number of postprocess binary dilation iterations [%(default)s]')
    return arg_parser


# ======================================================================
def main():
    # :: handle program parameters
    arg_parser = handle_arg()
    args = arg_parser.parse_args()
    # :: print debug info
    if args.verbose == VERB_LVL['debug']:
        arg_parser.print_help()
        print()
        print('II:', 'Parsed Arguments:', args)
    if args.verbose > VERB_LVL['low']:
        print(__doc__)
    begin_time = time.time()

    if args.output and args.verbose >= VERB_LVL['none']:
        args.output = os.path.dirname(os.path.realpath(args.output))
        print('OutDir:\t{}'.format(args.output))
    mrl.calc_mask(
        args.input,
        args.output,
        args.val_threshold,
        args.comparison,
        'absolute',
        args.smoothing,
        args.erosion_iter,
        args.dilation_iter,
        args.bet_params,
        args.force,
        args.verbose)

    end_time = time.time()
    if args.verbose > VERB_LVL['low']:
        print('ExecTime: ', datetime.timedelta(0, end_time - begin_time))


# ======================================================================
if __name__ == '__main__':
    main()
