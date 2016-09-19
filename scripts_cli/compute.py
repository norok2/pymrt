#!python
# -*- coding: utf-8 -*-
"""
Perform a custom calculation on a set of data matching specified criteria.

A number of calculation algorithms and criteria can be tuned using the
appropriate combination of values in the 'options' command-line argument.
Some computations, that are of particular interest in MRI, are readily
available through the 'method' command-line argument.

See also: pymrt.compute
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
import argparse  # Parser for command-line options, arguments and subcommands
# import itertools  # Functions creating iterators for efficient looping
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
import pymrt.base as pmb
# import pymrt.naming as pmn
# import pymrt.input_output as pmio
import pymrt.computation as pmc
# import pymrt.correlate as pmr
# import pymrt.geometry as pmg
# from pymrt.sequences import mp2rage
# import dcmpi.common as dcmlib

from pymrt import INFO
from pymrt import VERB_LVL, D_VERB_LVL
from pymrt import msg, dbg


# ======================================================================
def _func_name(label, name):
    return '_'.join((label, name))


# ======================================================================
def handle_arg():
    """
    Handle command-line application arguments.
    """
    # :: Create Argument Parser
    arg_parser = argparse.ArgumentParser(
        description=__doc__,
        epilog='v.{} - {}\n{}'.format(
            INFO['version'], ', '.join(INFO['authors']),
            INFO['license']),
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
        '-i', '--input', metavar='DIR',
        default='.',
        help='set input directory [%(default)s]')
    arg_parser.add_argument(
        '-o', '--output', metavar='DIR',
        default='.',
        help='set output directory [%(default)s]')
    arg_parser.add_argument(
        '-r', '--recursive',
        action='store_true',
        help='Force descending into subdirectories [%(default)s]')
    arg_parser.add_argument(
        '-e', '--meta_subpath', metavar='DIR',
        default=pmc.META_EXT,
        help='set extra input subdirectory [%(default)s]')
    arg_parser.add_argument(
        '-a', '--data_subpath', metavar='DIR',
        default='nii',
        help='set extra input subdirectory [%(default)s]')
    arg_parser.add_argument(
        '-m', '--method', metavar='METHOD',
        default='generic',
        help='set computation target and method [%(default)s]')
    arg_parser.add_argument(
        '-n', '--options', metavar='OPTS',
        default='{}',
        help='set JSON-formatted options dictionary [%(default)s]')
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

    if args.verbose >= VERB_LVL['medium']:
        print("II: Using method/options: '{}' / '{}'".format(
            args.method, args.options))

    if args.method:
        preset_func_name = _func_name('preset', args.method)
        sources_func_name = _func_name('sources', args.method)
        compute_func_name = _func_name('compute', args.method)

        # use preset if available
        opts = json.loads(args.options) if args.options else {}
        if preset_func_name in vars(pmc):
            new_opts = vars(pmc)[preset_func_name]()
            new_opts.update(opts)
            opts = new_opts
        # source extraction
        sources_func = vars(pmc)[sources_func_name] \
            if sources_func_name in vars(pmc) else pmc.sources_generic
        sources_args = [opts, args.force, args.verbose]
        sources_kwargs = {}
        # computation
        compute_func = vars(pmc)[compute_func_name] \
            if compute_func_name in vars(pmc) else pmc.compute_generic
        compute_args = [opts, args.force, args.verbose]
        compute_kwargs = {}
        # inform on the actual functions
        if args.verbose > VERB_LVL['none']:
            print('II: Mode: {} / {} / {}'.format(
                args.method, sources_func.__name__, compute_func.__name__))
            print('II: Opts: {}'.format(json.dumps(opts)))
        # proceed with computation on selected sources
        if opts != {}:
            pmc.compute(
                sources_func, sources_args, sources_kwargs,
                compute_func, compute_args, compute_kwargs,
                args.input, args.output, args.recursive,
                args.meta_subpath, args.data_subpath, args.verbose)
        else:
            print('EE: Mode / options combination not supported.')
    else:
        print('WW: Method not specified.')

    pmb.elapsed('compute')
    if args.verbose > VERB_LVL['none']:
        pmb.print_elapsed()


# ======================================================================
if __name__ == '__main__':
    main()
