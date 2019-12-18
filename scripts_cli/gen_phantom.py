#!python
# -*- coding: utf-8 -*-
"""
Module Name and Description

Long module description.
"""

# ======================================================================
# :: Future Imports
from __future__ import(
    division, absolute_import, print_function, unicode_literals, )

# ======================================================================
# :: Python Standard Library Imports
# import os  # Miscellaneous operating system interfaces
# import shutil  # High-level file operations
# import math  # Mathematical functions
# import time  # Time access and conversions
# import datetime  # Basic date and time types
# import operator  # Standard operators as functions
# import collections  # Container datatypes
import argparse  # Parser for command-line options, arguments and subcommands
# import itertools  # Functions creating iterators for efficient looping
# import subprocess  # Subprocess management
# import multiprocessing  # Process-based parallelism
# import csv  # CSV File Reading and Writing [CSV: Comma-Separated Values]
# import json  # JSON encoder and decoder [JSON: JavaScript Object Notation]

# :: External Imports
import numpy as np  # NumPy (multidimensional numerical arrays library)
# import scipy as sp  # SciPy (signal and image processing library)
# import matplotlib as mpl  # Matplotlib (2D/3D plotting library)
# import sympy as sym  # SymPy (symbolic CAS library)
# import PIL  # Python Image Library (image manipulation toolkit)
# import SimpleITK as sitk  # Image ToolKit Wrapper
import nibabel as nib  # NiBabel (NeuroImaging I/O Library)
# import nipy  # NiPy (NeuroImaging in Python)
# import nipype  # NiPype (NiPy Pipelines and Interfaces)
import raster_geometry  # Create/manipulate N-dim raster geometric shapes.

# :: External Imports Submodules
# import matplotlib.pyplot as plt  # Matplotlib's pyplot: MATLAB-like syntax
# import scipy.optimize  # SciPy: Optimization Algorithms
# import scipy.integrate  # SciPy: Integrations facilities
# import scipy.constants  # SciPy: Mathematal and Physical Constants
# import scipy.ndimage  # SciPy: ND-image Manipulation

# :: Local Imports
import pymrt as mrt  # Python Magnetic Resonance Tools: the multi-tool of MRI.
# import pymrt.utils
# import pymrt.naming
# import pymrt.plot as pmp
# import pymrt.registration
# import pymrt.segmentation
# import pymrt.computation as pmc
# import pymrt.correlation as pml
import pymrt.input_output
# import pymrt.sequences as pmq
# from pymrt.debug import dbg
# from pymrt.sequences import mp2rage

from pymrt import INFO
from pymrt import VERB_LVL, D_VERB_LVL
from pymrt import msg, dbg, fmt, fmtm

# ======================================================================
# :: Custom defined constants
APPEND_MODE = {'sum': 'sum', 'prod': 'prod'}
PHANTOMS = ('cuboid', 'ellipsoid', 'rhomboid', 'cylinder')


# TODO: fix documentation

# ======================================================================
def gen_phantom(
        i_filepath,
        shape,
        append,
        phantom,
        position,
        lengths,
        fill,
        o_filepath,
        verbose):
    """
    Create an image containing a simulated phantom with a geometric shape.

    Parameters
    ==========
    i_filepath : str
        Input filepath.
    o_filepath : str
        Output filepath.

    Returns
    =======
    None

    """
    # :: determine starting image
    if i_filepath:
        nii = nib.load(i_filepath)
        img = nii.get_data()
        aff = nii.get_affine()
        hdr = nii.get_header()
        shape = img.shape
    else:
        if append == APPEND_MODE['sum']:
            img = np.zeros(shape)
        elif append == APPEND_MODE['prod']:
            img = np.ones(shape)
        else:
            raise ValueError
        # affine matrix should be of shape: (N_DIM + 1, N_DIM + 1)
        aff = np.eye(len(shape) + 1)
        hdr = None
    # :: position of the phantom center relative to the mask center
    position = position
    # :: create the mask
    if phantom == 'cuboid':
        mask = raster_geometry.cuboid(shape, position, lengths)
    elif phantom == 'ellipsoid':
        mask = raster_geometry.ellipsoid(shape, position, lengths)
    elif phantom == 'rhomboid':
        mask = raster_geometry.rhomboid(shape, position, lengths)
    elif phantom == 'cylinder':
        mask = raster_geometry.cylinder(shape, position, lengths[0], lengths[1])
    # create an image from the mask
    img_append = raster_geometry.set_values(mask, fill)
    if append == APPEND_MODE['sum']:
        img += img_append
    elif append == APPEND_MODE['prod']:
        img *= img_append
    else:
        raise ValueError
    print(
        ('Created a {}-sized {}' +
         '\n- centered at position {}' +
         '\n- inside a {} voxmap' +
         '\n- with {} internal/external filling' +
         '\n- appended (mode: \'{}\') to: \'{}\''
         '\n- saving to: \'{}\'').format(
            lengths, phantom,
            position,
            shape,
            fill,
            append, i_filepath,
            o_filepath))
    mrt.input_output.save(o_filepath, img, aff, hdr)


# ======================================================================
def handle_arg():
    """
    Handle command-line application arguments.
    """
    # :: Define DEFAULT values
    # verbosity
    d_verbose = D_VERB_LVL
    # number of dimensions of the image
    d_dim = 3
    # size of the resulting image
    d_sizes = tuple([raster_geometry.D_SHAPE] * d_dim)
    # phantom to create
    d_phantom = PHANTOMS[0]
    # proportional position of the center relative to the middle
    d_position = (0.0, 0.0, 0.0)
    # lengths of the resulting object
    d_lengths = tuple([raster_geometry.D_LENGTH_1] * d_dim)
    # lengths of the resulting object
    d_angles = tuple([0.0] * d_dim)
    # intensity values (internal, external)
    d_intensities = (1.0, 0.0)
    # appending mode
    d_append = list(APPEND_MODE)[0]
    # input file
    d_infile = None
    # output file
    d_outfile = 'output.nii.gz'
    # :: Create Argument Parser
    arg_parser = argparse.ArgumentParser(
        description=__doc__,
        epilog=fmtm('v.{version} - {author}\n{license}', INFO),
        formatter_class=argparse.RawDescriptionHelpFormatter)
    # :: Add POSIX standard arguments
    arg_parser.add_argument(
        '--ver', '--version',
        version=fmt(
            '%(prog)s - ver. {version}\n{}\n{copyright} {author}\n{notice}',
            next(line for line in __doc__.splitlines() if line), **INFO),
        action='version')
    arg_parser.add_argument(
        '-v', '--verbose',
        action='count', default=d_verbose,
        help='increase the level of verbosity [%(default)s]')
    arg_parser.add_argument(
        '-q', '--quiet',
        action='store_true',
        help='override verbosity settings to suppress output [%(default)s]')
    # :: Add additional arguments
    group_input = arg_parser.add_mutually_exclusive_group()
    group_input.add_argument(
        '-s', '--sizes', metavar='SIZE',
        type=int, nargs='+', default=d_sizes,
        help='set the size of the image to create [%(default)s]')
    group_input.add_argument(
        '-i', '--infile', metavar='FILEPATH',
        default=d_infile,
        help='set input file (overrides image creation) [%(default)s]', )
    arg_parser.add_argument(
        '-a', '--append',
        choices=APPEND_MODE, default=d_append,
        help='set appending mode [%(default)s]')
    arg_parser.add_argument(
        '-f', '--phantom',
        choices=PHANTOMS, default=d_phantom,
        help='choose the phantom to create [%(default)s]')
    arg_parser.add_argument(
        '-p', '--position',
        type=float, nargs='+', default=d_position,
        help='proportional position of the center of the phantom (in the \
            range [-1,1]) relative to the middle of the image [%(default)s]')
    arg_parser.add_argument(
        '-l', '--lengths', metavar='LENGTH',
        type=float, nargs='+', default=d_lengths,
        help='set the lengths required to define the object (semisides, \
            semiaxes, semidiagonals or height/radius) [%(default)s]')
    arg_parser.add_argument(
        '-g', '--angles', metavar='ANGLE',
        type=float, nargs='+', default=d_angles,
        help='set the angles required to define the object [%(default)s]')
    arg_parser.add_argument(
        '-n', '--intensities', metavar=('INTERNAL', 'EXTERNAL'),
        type=float, nargs=2, default=d_intensities,
        help='set the internal and external intensity values [%(default)s]')
    arg_parser.add_argument(
        '-o', '--outfile', metavar='FILEPATH',
        default=d_outfile,
        help='set output file [%(default)s]', )
    return arg_parser


# ======================================================================
def main():
    # handle program parameters
    arg_parser = handle_arg()
    args = arg_parser.parse_args()
    # fix verbosity in case of 'quiet'
    if args.quiet:
        args.verbose = VERB_LVL['none']
    # :: print debug info
    if args.verbose >= VERB_LVL['debug']:
        arg_parser.print_help()
        msg('\nARGS: ' + str(vars(args)), args.verbose, VERB_LVL['debug'])

    gen_phantom(
        args.infile, args.sizes, args.append, args.phantom, args.position,
        args.lengths, args.intensities, args.outfile, args.verbose)


# ======================================================================
if __name__ == '__main__':
    main()
