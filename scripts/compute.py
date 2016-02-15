#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Perform a custom calculation on a set of data matching specified criteria.

A number of calculation algorithms and criteria can be tuned using the
appropriate combination of values in the 'options' command-line argument.
Some computations, that are of particular interest in MRI, are readily
available through the 'method' command-line argument.

See also: mri_tools.compute
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
# import mayavi.mlab as mlab  # Mayavi's mlab: MATLAB-like syntax
# import scipy.optimize  # SciPy: Optimization Algorithms
# import scipy.integrate  # SciPy: Integrations facilities
# import scipy.constants  # SciPy: Mathematal and Physical Constants
# import scipy.ndimage  # SciPy: ND-image Manipulation

# :: Local Imports
# import mri_tools.base as mrb
# import mri_tools.utils as mru
# import mri_tools.nifti as mrn
import mri_tools.computation as mrc
# import mri_tools.correlate as mrr
# import mri_tools.geometry as mrg
# from mri_tools.sequences import mp2rage
import dcmpi.common as dcmlib

from mri_tools import INFO
from mri_tools import VERB_LVL
from mri_tools import D_VERB_LVL
from mri_tools.base import elapsed, print_elapsed


# ======================================================================
def _func_name(label, name):
    return '_'.join((label, name))


# ======================================================================
def handle_arg():
    """
    Handle command-line application arguments.
    """
    # :: Define DEFAULT values
    # verbosity
    d_verbose = D_VERB_LVL
    # input directory
    d_input_dir = '.'
    # output directory
    d_output_dir = '.'
    # calculation method
    d_method = 'generic'
    # method options
    d_options = '{}'
    # extra metadata subpath
    d_meta_subpath = dcmlib.ID['info']
    # extra data subpath
    d_data_subpath = dcmlib.ID['nifti']
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
            action='count', default=d_verbose,
            help='increase the level of verbosity [%(default)s]')
    # :: Add additional arguments
    arg_parser.add_argument(
            '-f', '--force',
            action='store_true',
            help='force new processing [%(default)s]')
    arg_parser.add_argument(
            '-i', '--input', metavar='DIR',
            default=d_input_dir,
            help='set input directory [%(default)s]')
    arg_parser.add_argument(
            '-o', '--output', metavar='DIR',
            default=d_output_dir,
            help='set output directory [%(default)s]')
    arg_parser.add_argument(
            '-r', '--recursive',
            action='store_true',
            help='Force descending into subdirectories [%(default)s]')
    arg_parser.add_argument(
            '-e', '--meta_subpath', metavar='DIR',
            default=d_meta_subpath,
            help='set extra input subdirectory [%(default)s]')
    arg_parser.add_argument(
            '-a', '--data_subpath', metavar='DIR',
            default=d_data_subpath,
            help='set extra input subdirectory [%(default)s]')
    arg_parser.add_argument(
            '-m', '--method', metavar='METHOD',
            default=d_method,
            help='set computation target and method [%(default)s]')
    arg_parser.add_argument(
            '-n', '--options', metavar='OPTS',
            default=d_options,
            help='set JSON-formatted options dictionary [%(default)s]')
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

    if ARGS.verbose >= VERB_LVL['medium']:
        print("II: Using method/options: '{}' / '{}'".format(
                ARGS.method, ARGS.options))

    if ARGS.method:
        preset_func_name = _func_name('preset', ARGS.method)
        sources_func_name = _func_name('sources', ARGS.method)
        compute_func_name = _func_name('compute', ARGS.method)

        # use preset if available
        opts = json.loads(ARGS.options) if ARGS.options else {}
        if preset_func_name in vars(mrc):
            new_opts = vars(mrc)[preset_func_name]()
            new_opts.update(opts)
            opts = new_opts
        # source extraction
        sources_func = vars(mrc)[sources_func_name] \
            if sources_func_name in vars(mrc) else mrc.sources_generic
        sources_args = [opts, ARGS.force, ARGS.verbose]
        sources_kwargs = {}
        # computation
        compute_func = vars(mrc)[compute_func_name] \
            if compute_func_name in vars(mrc) else mrc.compute_generic
        compute_args = [opts, ARGS.force, ARGS.verbose]
        compute_kwargs = {}
        # inform on the actual functions
        if ARGS.verbose > VERB_LVL['none']:
            print('II: Mode: {} / {} / {}'.format(
                    ARGS.method, sources_func.__name__, compute_func.__name__))
            print('II: Opts: {}'.format(json.dumps(opts)))
        # proceed with computation on selected sources
        if opts != {}:
            mrc.compute(
                    sources_func, sources_args, sources_kwargs,
                    compute_func, compute_args, compute_kwargs,
                    ARGS.input, ARGS.output, ARGS.recursive,
                    ARGS.meta_subpath, ARGS.data_subpath, ARGS.verbose)
        else:
            print("EE: Mode / options combination not supported.")
    else:
        print('WW: Method not specified.')

    elapsed('compute')
    if ARGS.verbose > VERB_LVL['none']:
        print_elapsed()
