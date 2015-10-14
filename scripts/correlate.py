#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Check voxel-by-voxel correlation after registration and masking.
"""


# ======================================================================
# :: Future Imports
from __future__ import division
from __future__ import absolute_import
from __future__ import print_function
#from __future__ import unicode_literals


# ======================================================================
# :: Python Standard Library Imports
# import os  # Miscellaneous operating system interfaces
# import shutil  # High-level file operations
# import math  # Mathematical functions
import time  # Time access and conversions
import datetime  # Basic date and time types
# import operator  # Standard operators as functions
import argparse  # Parser for command-line options, arguments and subcommands
# import itertools  # Functions creating iterators for efficient looping
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
# import mri_tools.modules.base as mrb
# import mri_tools.modules.utils as mru
# import mri_tools.modules.nifti as mrn
# import mri_tools.modules.compute as mrc
import mri_tools.modules.correlation as mrl
# import mri_tools.modules.geometry as mrg
# from mri_tools.modules.sequences import mp2rage
# import dcmpi.lib.common as dcmlib

from mri_tools import INFO
from mri_tools import VERB_LVL
from mri_tools import D_VERB_LVL
from mri_tools import get_first_line


# ======================================================================
def handle_arg():
    """
    Handle command-line application arguments.
    """
    # :: Define DEFAULT values
    # verbosity
    d_verbose = D_VERB_LVL
    # working directory
    d_working_dir = '.'
    # data units
    d_type = None
    # data range
    d_range = None
    # data units
    d_units = None
    # mask file
    d_mask_file = None
    # registration reference extension
    d_reg_ref_ext = mrl.D_EXT['registration reference']
    # correlation reference extension
    d_corr_ref_ext = mrl.D_EXT['correlation reference']
    # temporary dir
    d_tmp_dir = 'tmp'
    # registration dir
    d_reg_dir = 'reg'
    # masking dir
    d_msk_dir = 'msk'
    # comparing dir
    d_cmp_dir = 'cmp'
    # figures dir
    d_fig_dir = 'fig'
    # FSL's BET brain extraction algorithm parameters
    d_bet_params = ''
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
            INFO['version'], get_first_line(__doc__),
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
        '-d', '--dir', metavar='DIR',
        default=d_working_dir,
        help='set working directory [%(default)s]')
    arg_parser.add_argument(
        '-t', '--type', metavar='TYPE',
        default=d_type,
        help='set data type [%(default)s]')
    arg_parser.add_argument(
        '-a', '--range', metavar=('MIN', 'MAX'),
        type=float, nargs=2, default=d_range,
        help='set parameters for the range of values [%(default)s]')
    arg_parser.add_argument(
        '-u', '--units', metavar='UNITS',
        default=d_units,
        help='set units for values [%(default)s]')
    arg_parser.add_argument(
        '-m', '--mask', metavar='MASK_FILE',
        default=d_mask_file,
        help='set exact mask file [%(default)s]')
    arg_parser.add_argument(
        '--reg_ref_ext', metavar='FILE_EXT',
        default=d_reg_ref_ext,
        help='file extension of registration reference flag [%(default)s]')
    arg_parser.add_argument(
        '--corr_ref_ext', metavar='FILE_EXT',
        default=d_corr_ref_ext,
        help='file extension of correlation reference flag [%(default)s]')
    arg_parser.add_argument(
        '--tmp_dir', metavar='SUBDIR',
        default=d_tmp_dir,
        help='subdirectory where to store temporary files [%(default)s]')
    arg_parser.add_argument(
        '--reg_dir', metavar='SUBDIR',
        default=d_reg_dir,
        help='subdirectory where to store registration files [%(default)s]')
    arg_parser.add_argument(
        '--msk_dir', metavar='SUBDIR',
        default=d_msk_dir,
        help='subdirectory where to store masking files [%(default)s]')
    arg_parser.add_argument(
        '--cmp_dir', metavar='SUBDIR',
        default=d_cmp_dir,
        help='subdirectory where to store comparing files [%(default)s]')
    arg_parser.add_argument(
        '--fig_dir', metavar='SUBDIR',
        default=d_fig_dir,
        help='subdirectory where to store figures [%(default)s]')
    arg_parser.add_argument(
        '--toggle_helpers',
        action='store_true',
        help='toggle helper files usage for registration [%(default)s]')
    arg_parser.add_argument(
        '--bet_params', metavar='STR',
        default=d_bet_params,
        help='parameters to FSL\'s BET tool. Empty str to skip [%(default)s]')
    return arg_parser


# ======================================================================
if __name__ == '__main__':
    # :: handle program parameters
    ARG_PARSER = handle_arg()
    ARGS = ARG_PARSER.parse_args()
    # :: print debug info
    if ARGS.verbose == VERB_LVL['debug']:
        ARG_PARSER.print_help()
        print('II:', 'Parsed Arguments:', ARGS)
    if ARGS.verbose > VERB_LVL['low']:
        print(__doc__)
    begin_time = time.time()
    mrr.check_correlation(
        ARGS.dir, ARGS.type, ARGS.range, ARGS.units,
        ARGS.mask, ARGS.reg_ref_ext, ARGS.corr_ref_ext,
        ARGS.tmp_dir, ARGS.reg_dir, ARGS.msk_dir, ARGS.cmp_dir, ARGS.fig_dir,
        ARGS.toggle_helpers, ARGS.bet_params, ARGS.force, ARGS.verbose)
    end_time = time.time()
    if ARGS.verbose > VERB_LVL['low']:
        print('ExecTime: ', datetime.timedelta(0, end_time - begin_time))
