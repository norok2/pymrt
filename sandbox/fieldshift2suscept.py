#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Calculate the susceptibility from the z-component field perturbation.

Consider a sample with a certain susceptibility spatial distribution chi(r)
(r is the position vector) submerged in a large z-oriented constant magnetic
field B0_z of known intensity. Assume chi(r) << 1.
The susceptibiliy distribution chi(r) can be calculated from the field shift
DB_z(r) in the Fourier domain:
chi(k) = (DB_z(k) / B0_z) (1 / C(k))
where:
- chi(k) is the Fourier transform of chi(r)
- C(k) is the Fourier transform of the dipole convolution kernel
The C-factor is dependent upon orientation relative to B0z in the Fourier
space. Such relation can be expressed more conveniently as a function of two
subsequent rotations in the direct space:
- a rotation of angle th (theta) around x-axis
- a rotation of angle ph (phi) around y-axis
(It assumed that x-, y- and z- axes are on the 1st, 2nd and 3rd dimensions of
the volume.)
The full expression is given by:
              / 1   (kz cos(th) cos(ph) - ky sin(th) cos(ph) + kx sin(ph))^2 \
 C(k,th,ph) = | - - -------------------------------------------------------- |
              \ 3                      kx^2 + ky^2 + kz^2                    /

[ref: S. Wharton, R. Bowtell, NeuroImage 53 (2010) 515-525]
[ref: J.P. Marques, R. Bowtell, Concepts in MR B 25B (2005) 65-78]
"""

# ======================================================================
# :: Future Imports
from __future__ import(
    division, absolute_import, print_function, unicode_literals)

__version__ = '0.0.0.13'
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
# import os  # Miscellaneous operating system interfaces
# import shutil  # High-level file operations
# import math  # Mathematical functions
import time  # Time access and conversions
import datetime  # Basic date and time types
# import operator  # Standard operators as functions
import collections  # High-performance container datatypes
import argparse  # Parser for command-line options, arguments and sub-commands
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
import pymrt.input_output as pmio

# ======================================================================
# :: supported verbosity levels (level 4 skipped on purpose)
VERB_LVL = {'none': 0, 'low': 1, 'medium': 2, 'high': 3, 'debug': 5}
D_VERB_LVL = VERB_LVL['low']


# ======================================================================
def suscept2fieldshift(
        delta_bdz_r,
        b0z,
        theta,
        phi,
        threshold=None):
    """
    Calculate the z-comp. field perturbation due to susceptibility variation.

    Parameters
    ==========
    delta_bdz_r : ndarray
        The z-component of field shift  DBdz(r) over B0_z
    b0z : float
        Main magnetic field B0 in Tesla
    theta : float
        Angle of 1st rotation (along x-axis) in deg
    phi : float
        Angle of 2nd rotation (along y-axis) in deg

    Returns
    =======
    chi_r : ndarray
        The susceptibility distribution chi(r) in SI units

    """

    # convert input angles to radians
    theta = np.deg2rad(theta)
    phi = np.deg2rad(phi)
    # field shift along z in Tesla in Fourier space (zero-centered)
    delta_bdz_k = np.fft.fftshift(np.fft.fftn(delta_bdz_r))
    # :: dipole convolution kernel C = C(k, theta, phi)
    #     / 1   (kz cos(th) cos(ph) - ky sin(th) cos(ph) + kx sin(ph))^2 \
    # C = | - - -------------------------------------------------------- |
    #     \ 3                      kx^2 + ky^2 + kz^2                    /
    k_range = [slice(-k_size / 2.0, +k_size / 2.0) \
               for k_size in delta_bdz_k.shape]
    kk_x, kk_y, kk_z = np.ogrid[k_range]
    dipole_kernel_k = (
        1.0 / 3.0 -
        (kk_z * np.cos(theta) * np.cos(phi)
         - kk_y * np.sin(theta) * np.cos(phi)
         + kk_x * np.sin(phi)) ** 2 /
        (kk_x ** 2 + kk_y ** 2 + kk_z ** 2))
    # remove singularity for |k|^2 == 0 with DBdz(k) = B0z * chi(k) / 3
    dipole_singularity = np.isnan(dipole_kernel_k)
    dipole_kernel_k[dipole_singularity] = 1.0 / 3.0
    # susceptibility in SI units in Fourier space
    chi_k = delta_bdz_k / dipole_kernel_k
    # remove singularity of susceptibility
    if threshold:
        mask = np.abs(dipole_kernel_k) < threshold
        chi_k[mask] = 0.0
    else:
        chi_singularity = np.isnan(chi_k)
        chi_k[chi_singularity] = 0.0
    # susceptibility in SI units in direct space (zero-centered)
    chi_r = np.fft.ifftn(np.fft.ifftshift(chi_k))
    # convert to real values
    chi_r = np.real(chi_r)
    return chi_r


# ======================================================================
def handle_arg():
    """
    Handle command-line application arguments.
    """
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
        action='count', default=D_VERB_LVL,
        help='increase the level of verbosity [%(default)s]')
    # :: Add additional arguments
    arg_parser.add_argument(
        '-b', '--b0z', metavar='B0_z',
        type=float, default=7.0,
        help='set the magnetic field B0 along z-axis in Tesla [%(default)s]', )
    arg_parser.add_argument(
        '-t', '--theta', metavar='TH',
        type=float, default=0.0,
        help='set angle of 1st rotation (along x-axis) in deg [%(default)s]', )
    arg_parser.add_argument(
        '-p', '--phi', metavar='PH',
        type=float, default=0.0,
        help='set angle of 2nd rotation (along y-axis) in deg [%(default)s]', )
    arg_parser.add_argument(
        '-i', '--input', metavar='FILE',
        default='tmp_fieldshift.nii.gz',
        help='set input file [%(default)s]', )
    arg_parser.add_argument(
        '-o', '--output', metavar='FILE',
        default='tmp_suscept.nii.gz',
        help='set output file [%(default)s]', )
    return arg_parser


# ======================================================================
if __name__ == '__main__':
    # handle program parameters
    ARG_PARSER = handle_arg()
    ARGS = ARG_PARSER.parse_args()
    # print calling arguments
    if ARGS.verbose >= VERB_LVL['debug']:
        print(ARGS)
    # perform calculation
    begin_time = datetime.datetime.now()
    # pmio.simple_filter_1_1(
    #     ARGS.input, ARGS.output,
    #     suscept2fieldshift, ARGS.b0z, ARGS.theta, ARGS.phi)
    end_time = datetime.datetime.now()
    if ARGS.verbose > VERB_LVL['none']:
        print('ExecTime: {}'.format(end_time - begin_time))
