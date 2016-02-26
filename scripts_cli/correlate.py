#!python
# -*- coding: utf-8 -*-
"""
Check voxel-by-voxel correlation after registration and masking.
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
import mri_tools.base as mrb
# import mri_tools.utils as mru
# import mri_tools.input_output as mrio
# import mri_tools.compute as mrc
import mri_tools.correlation as mrl
# import mri_tools.geometry as mrg
# from mri_tools.sequences import mp2rage
# import dcmpi.lib.common as dcmlib

from mri_tools import INFO
from mri_tools import VERB_LVL
from mri_tools import D_VERB_LVL


# ======================================================================
def handle_arg():
    """
    Handle command-line application arguments.
    """
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
        action='count', default=D_VERB_LVL,
        help='increase the level of verbosity [%(default)s]')
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
        '-t', '--type', metavar='TYPE',
        default=None,
        help='set data type [%(default)s]')
    arg_parser.add_argument(
        '-a', '--interval', metavar=('MIN', 'MAX'),
        type=float, nargs=2, default=None,
        help='set parameters for the range of values [%(default)s]')
    arg_parser.add_argument(
        '-u', '--units', metavar='UNITS',
        default=None,
        help='set units for values [%(default)s]')
    arg_parser.add_argument(
        '-m', '--mask', metavar='MASK_FILE',
        default=None,
        help='set exact mask file [%(default)s]')
    arg_parser.add_argument(
        '--reg_ref_ext', metavar='FILE_EXT',
        default=mrl.D_EXT['reg_ref'],
        help='file extension of registration reference flag [%(default)s]')
    arg_parser.add_argument(
        '--corr_ref_ext', metavar='FILE_EXT',
        default=mrl.D_EXT['corr_ref'],
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
    # :: print debug info
    if args.verbose == VERB_LVL['debug']:
        arg_parser.print_help()
        print('II:', 'Parsed Arguments:', args)
    if args.verbose > VERB_LVL['low']:
        print(__doc__)

    kwargs = {
        'dirpath': args.dirpath,
        'val_type': args.type,
        'val_interval': args.interval,
        'val_units': args.units,
        'mask_filepath': args.mask,
        'reg_ref_ext': args.reg_ref_ext,
        'corr_ref_ext': args.corr_ref_ext,
        'tmp_dir': args.tmp_dir,
        'reg_dir': args.reg_dir,
        'msk_dir': args.msk_dir,
        'cmp_dir': args.cmp_dir,
        'fig_dir': args.fig_dir,
        'force': args.force,
        'verbose': args.verbose,
    }
    mrl.check_correlation(**kwargs)

    if args.verbose > VERB_LVL['low']:
        mrb.elapsed(os.path.basename(__file__))
        mrb.print_elapsed()


# ======================================================================
if __name__ == '__main__':
    main()
