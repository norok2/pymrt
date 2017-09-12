#!python
# -*- coding: utf-8 -*-
"""
Check voxel-by-voxel correlation after registration and masking.
"""

# ======================================================================
# :: Future Imports
from __future__ import(
    division, absolute_import, print_function, unicode_literals)

# ======================================================================
# :: Python Standard Library Imports
import os  # Miscellaneous operating system interfaces
# import shutil  # High-level file operations
# import math  # Mathematical functions
# import time  # Time access and conversions
# import datetime  # Basic date and time types
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
# import scipy.optimize  # SciPy: Optimization Algorithms
# import scipy.integrate  # SciPy: Integrations facilities
# import scipy.constants  # SciPy: Mathematal and Physical Constants
# import scipy.ndimage  # SciPy: ND-image Manipulation

# :: Local Imports
import pymrt as mrt
import pymrt.utils
# import pymrt.naming
# import pymrt.input_output
# import pymrt.compute as pmc
import pymrt.correlation as pml
# import pymrt.geometry
# from pymrt.sequences import mp2rage
# import dcmpi.lib.common as dcmlib

from pymrt import INFO
from pymrt import VERB_LVL, D_VERB_LVL
from pymrt import msg, dbg
from pymrt import elapsed, report


# ======================================================================
def handle_arg():
    """
    Handle command-line application arguments.
    """
    # :: Create Argument Parser
    arg_parser = argparse.ArgumentParser(
        description=__doc__,
        epilog='v.{} - {}\n{}'.format(
            INFO['version'], INFO['author'], INFO['license']),
        formatter_class=argparse.RawDescriptionHelpFormatter)
    # :: Add POSIX standard arguments
    arg_parser.add_argument(
        '--ver', '--version',
        version='%(prog)s - ver. {}\n{}\n{} {}\n{}'.format(
            INFO['version'],
            next(line for line in __doc__.splitlines() if line),
            INFO['copyright'], INFO['author'], INFO['notice']),
        action='version')
    arg_parser.add_argument(
        '-v', '--verbose',
        action='count', default=D_VERB_LVL,
        help='increase the level of verbosity [%(default)s]')
    arg_parser.add_argument(
        '-q', '--quiet',
        action='store_true',
        help='override verbosity settings to suppress output [%(default)s]')
    # :: Add additional arguments
    arg_parser.add_argument(
        '-f', '--force',
        action='store_true',
        help='force new processing [%(default)s]')
    arg_parser.add_argument(
        '-d', '--dirpath', metavar='DIR',
        default='.',
        help='set working directory [%(default)s]')
    arg_parser.add_argument(
        '-n', '--val_name', metavar='NAME',
        default=None,
        help='set values name ( [%(default)s]')
    arg_parser.add_argument(
        '-a', '--val_interval', metavar=('MIN', 'MAX'),
        type=float, nargs=2, default=None,
        help='set interval of values [%(default)s]')
    arg_parser.add_argument(
        '-u', '--val_units', metavar='UNITS',
        default=None,
        help='set units for values [%(default)s]')
    arg_parser.add_argument(
        '-m', '--mask_filepath', metavar='FILE',
        default=None,
        help='set exact mask file [%(default)s]')
    arg_parser.add_argument(
        '--reg_ref_ext', metavar='FILE_EXT',
        default=pml.EXT['reg_ref'],
        help='file extension of registration reference flag [%(default)s]')
    arg_parser.add_argument(
        '--corr_ref_ext', metavar='FILE_EXT',
        default=pml.EXT['corr_ref'],
        help='file extension of correlation reference flag [%(default)s]')
    arg_parser.add_argument(
        '--tmp_dir', metavar='SUBDIR',
        default='tmp',
        help='subdirectory where to store temporary files [%(default)s]')
    arg_parser.add_argument(
        '--reg_dir', metavar='SUBDIR',
        default='reg',
        help='subdirectory where to store registration files [%(default)s]')
    arg_parser.add_argument(
        '--msk_dir', metavar='SUBDIR',
        default='msk',
        help='subdirectory where to store masking files [%(default)s]')
    arg_parser.add_argument(
        '--cmp_dir', metavar='SUBDIR',
        default='cmp',
        help='subdirectory where to store comparing files [%(default)s]')
    arg_parser.add_argument(
        '--fig_dir', metavar='SUBDIR',
        default='fig',
        help='subdirectory where to store figures [%(default)s]')
    return arg_parser


# ======================================================================
def main():
    # :: handle program parameters
    arg_parser = handle_arg()
    args = arg_parser.parse_args()
    # fix verbosity in case of 'quiet'
    if args.quiet:
        args.verbose = VERB_LVL['none']
    # :: print debug info
    if args.verbose >= VERB_LVL['debug']:
        arg_parser.print_help()
        msg('\nARGS: ' + str(vars(args)), args.verbose, VERB_LVL['debug'])
    msg(__doc__.strip())

    kws = vars(args)
    kws.pop('quiet')
    pml.check_correlation(**kws)

    elapsed(os.path.basename(__file__))
    msg(report(), args.verbose, VERB_LVL['medium'])


# ======================================================================
if __name__ == '__main__':
    main()
