#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Module Name and Description

Long module description.
"""


# ======================================================================
# :: Future Imports
from __future__ import division
from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals


__version__ = '1.0.0.8'
# $Source$


# ======================================================================
# :: Custom Module Details
AUTHOR = 'Riccardo Metere'
CONTACT = 'metere@cbs.mpg.de'
DATE_INFO = {'day': 4, 'month': 'Feb', 'year': 2014}
DATE = ' '.join([str(v) for k, v in sorted(DATE_INFO.items())])
COPYRIGHT = 'Copyright (C) ' + str(DATE_INFO['year'])
LICENSE = 'License GPLv3: GNU General Public License version 3'
# first non-empty line of __doc__
DOC_FIRSTLINE = [line for line in __doc__.splitlines() if line][0]


# ======================================================================
# :: Python Standard Library Imports
# import os  # Operating System facilities
# import math  # Mathematical Functions
import collections  # Collections of Items
import argparse  # Python Standard Library: Argument Parsing

# :: External Imports
import numpy as np  # NumPy (multidimensional numerical arrays library)
# import scipy as sp  # SciPy (signal and image processing library)
# import matplotlib as mpl  # Matplotlib (2D/3D plotting library)
# import matplotlib.pyplot as plt  # Matplotlib's pyplot: MATLAB-like syntax
# import mayavi.mlab as mlab  # Mayavi's mlab: MATLAB-like syntax
# import sympy as sym  # SymPy (symbolic CAS library)
# import PIL  # Python Image Library (image manipulation toolkit)
# import SimpleITK as sitk  # Image ToolKit Wrapper
import nibabel as nib  # NiBabel (NeuroImaging I/O Library)
# import nipy  # NiPy (NeuroImaging in Python)
# import nipype  # NiPype (NiPy Pipelines and Interfaces)

# :: Local Imports
import mri_tools.lib.geom_mask as mrgm  # Generate masks of geometrical shapes
import mri_tools.lib.nifti as mrn  # Custom-made Nifti1 utils

# ======================================================================
# :: supported verbosity levels (level 4 skipped on purpose)
VERB_LVL = {'none': 0, 'low': 1, 'medium': 2, 'high': 3, 'debug': 5}

# ======================================================================
# :: Custom defined constants
APPEND_SUM = '+'
APPEND_PROD = '*'
PHANTOM_CUBOID = 'cuboid'
PHANTOM_ELLIPSOID = 'ellipsoid'
PHANTOM_RHOMBOID = 'rhomboid'
PHANTOM_CYLINDER = 'cylinder'


# ======================================================================
def make_phantom(
        i_filepath,
        shape,
        append,
        phantom,
        position,
        lengths,
        fill,
        o_filepath,
        verbose=VERB_LVL['none']):
    """
    Create a phantom using custom_lib/geom_mask package.

    Parameters
    ==========
    None

    Returns
    =======
    None

    """
    # :: determine starting image
    if i_filepath:
        img = nib.load(i_filepath)
        img_data = img.get_data()
        img_affine = img.get_affine()
        img_header = img.get_header()
        shape = img_data.shape
    else:
        if append == APPEND_SUM:
            img_data = np.zeros(shape)
        elif append == APPEND_PROD:
            img_data = np.ones(shape)
        else:
            raise ValueError
        # affine matrix should be of shape: (N_DIM + 1, N_DIM + 1)
        img_affine = np.eye(len(shape) + 1)
        img_header = None
    # :: position of the phantom center relative to the mask center
    position = position
    # :: create the mask
    if phantom == PHANTOM_CUBOID:
        mask = geom_mask.cuboid(shape, position, lengths)
    elif phantom == PHANTOM_ELLIPSOID:
        mask = geom_mask.ellipsoid(shape, position, lengths)
    elif phantom == PHANTOM_RHOMBOID:
        mask = geom_mask.rhomboid(shape, position, lengths)
    elif phantom == PHANTOM_CYLINDER:
        mask = geom_mask.cylinder(shape, position, lengths[0], lengths[1])
    # create an image from the mask
    img_append = geom_mask.render(mask, fill)
    if append == APPEND_SUM:
        img_data += img_append
    elif append == APPEND_PROD:
        img_data *= img_append
    else:
        raise ValueError
    print(('Created a {}-sized {}' +
        '\n- centered at position {}' +
        '\n- inside a {} voxmap' +
        '\n- with {} internal/external filling' +
        '\n- appended (mode: \'{}\') to: \'{}\''
        '\n- saving to: \'{}\'')
        .format(lengths, phantom,
                position,
                shape,
                fill,
                append, i_filepath,
                o_filepath))
    nifti.img_maker(o_filepath, img_data, img_affine, img_header)


# ======================================================================
def handle_arg():
    """
    Handle command-line application arguments.
    """
    # :: Define CHOICE values
    c_phantom = (PHANTOM_CUBOID, PHANTOM_ELLIPSOID, PHANTOM_RHOMBOID,
        PHANTOM_CYLINDER)
    c_append = (APPEND_SUM, APPEND_PROD)
    # :: Define DEFAULT values
    # verbosity
    d_verbose = VERB_LVL['none']
    # number of dimensions of the image
    d_dim = 3
    # size of the resulting image
    d_sizes = tuple([geom_mask.D_SHAPE] * d_dim)
    # phantom to create
    d_phantom = c_phantom[0]
    # proportional position of the center relative to the middle
    d_position = (0.0, 0.0, 0.0)
    # lengths of the resulting object
    d_lengths = tuple([geom_mask.D_LENGTH_1] * d_dim)
    # lengths of the resulting object
    d_angles = tuple([0.0] * d_dim)
    # intensity values (internal, external)
    d_intensities = (1.0, 0.0)
    # appenging mode
    d_append = c_append[0]
    # input file
    d_infile = None
    # output file
    d_outfile = 'output.nii.gz'
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
    group_input = arg_parser.add_mutually_exclusive_group()
    group_input.add_argument(
        '-s', '--sizes', metavar='SIZE',
        type=int, nargs='+', default=d_sizes,
        help='set the size of the image to create [%(default)s]')
    group_input.add_argument(
        '-i', '--infile', metavar='FILEPATH',
        default=d_infile,
        help='set input file (overrides image creation) [%(default)s]',)
    arg_parser.add_argument(
        '-a', '--append',
        choices=c_append, default=d_append,
        help='set appending mode [%(default)s]')
    arg_parser.add_argument(
        '-f', '--phantom',
        choices=c_phantom, default=d_phantom,
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
        help='set output file [%(default)s]',)
    return arg_parser


# ======================================================================
if __name__ == '__main__':
    # handle program parameters
    ARG_PARSER = handle_arg()
    ARGS = ARG_PARSER.parse_args()
    # :: print debug info
    if ARGS.verbose == VERB_LVL['debug']:
        ARG_PARSER.print_help()
        print()
        print('II:', 'Parsed Arguments:', ARGS)
    make_phantom(
        ARGS.infile, ARGS.sizes, ARGS.append, ARGS.phantom, ARGS.position,
        ARGS.lengths, ARGS.intensities, ARGS.outfile, ARGS.verbose)
