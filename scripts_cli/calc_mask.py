#!python
# -*- coding: utf-8 -*-
"""
Extract a mask from an image (with specialized options for brain extraction).

Workflow is:
- Gaussian filter (smoothing) with specified sigma
- masking of values according to specific method or threshold value
- binary erosion with a specified number of iterations.
- binary dilation with a specified number of iterations.
"""

# ======================================================================
# :: Future Imports
from __future__ import (
    division, absolute_import, print_function, unicode_literals, )

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
import json  # JSON encoder and decoder [JSON: JavaScript Object Notation]

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
import pymrt.input_output
import pymrt.segmentation
import pymrt.naming

from pymrt import INFO
from pymrt import VERB_LVL, D_VERB_LVL
from pymrt import msg, dbg


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
        '-i', '--input', metavar='FILE',
        default=None, required=True,
        help='set input filepath [%(default)s]')
    arg_parser.add_argument(
        '-o', '--output', metavar='FILE',
        default=None,
        help='set output filepath [%(default)s]')
    arg_parser.add_argument(
        '-s', '--smoothing', metavar='SIGMA',
        type=float, default=0.0,
        help='value of Gaussian smoothing\'s sigma [%(default)s]')
    arg_parser.add_argument(
        '-a', '--threshold', metavar='METHOD|X',
        default='otsu',
        help='thresholding method or value [%(default)s]')
    arg_parser.add_argument(
        '-A', '--threshold_opts', metavar='JSON',
        default='{"bins":"sqrt"}',
        help='comparison directive [%(default)s]')
    arg_parser.add_argument(
        '-c', '--comparison', metavar='">"|">="|"<"|"<="|"="|"!="',
        default='>',
        help='comparison directive [%(default)s]')
    arg_parser.add_argument(
        '-e', '--erosion_iter', metavar='N',
        type=int, default=0,
        help='number of postprocess binary erosion iterations [%(default)s]')
    arg_parser.add_argument(
        '-d', '--dilation_iter', metavar='N',
        type=int, default=0,
        help='number of postprocess binary dilation iterations [%(default)s]')
    arg_parser.add_argument(
        '-t', '--dtype', metavar='TYPE',
        default='int',
        help='data type of the output [%(default)s]')
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
    begin_time = datetime.datetime.now()

    if not args.output:
        root, base, ext = fc.util.split_path(args.input)
        args.output = fc.util.join_path(root, 'mask__' + base, ext)

    kws = fc.util.set_func_kws(
        mrt.segmentation.auto_mask, vars(args))

    kws['threshold_kws'] = json.loads(args.threshold_opts)
    kws['threshold'] = fc.util.auto_convert(kws['threshold'])

    data, meta = mrt.input_output.load(args.input, meta=True)
    data = mrt.segmentation.auto_mask(data, **kws).astype(args.dtype)
    mrt.input_output.save(
        args.output, data, **{k: v for k, v in meta.items()})

    end_time = datetime.datetime.now()
    if args.verbose > VERB_LVL['low']:
        msg('ExecTime: {}'.format(end_time - begin_time))


# ======================================================================
if __name__ == '__main__':
    main()
