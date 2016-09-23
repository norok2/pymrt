#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Calculate the z-component field perturbation due to susceptibility variation.

Consider a sample with a certain susceptibility spatial distribution chi(r)
(r is the position vector) submerged in a large z-oriented constant magnetic
field B0_z of known intensity. Assume chi(r) << 1.
The field shift DB_z(r) due to the susceptibility distribution can be easily
calculated in the Fourier domain:
DB_z(k) = B0_z chi(k) C(k)
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
              ( 1   (kz cos(th) cos(ph) - ky sin(th) cos(ph) + kx sin(ph))^2 )
 C(k,th,ph) = ( - - ==================================================------ )
              ( 3                      kx^2 + ky^2 + kz^2                    )
The resulting field shift is then given by:
DB_z(r) = DB_z(r) / B0_z

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
import datetime  # Basic date and time types
import argparse  # Parser for command-line options, arguments and sub-commands


# :: External Imports
import numpy as np  # NumPy (multidimensional numerical arrays library)


# :: External Imports Submodules
import matplotlib.pyplot as plt  # Matplotlib's pyplot: MATLAB-like syntax

# :: Local Imports
import pymrt.input_output as pmio

# ======================================================================
# :: supported verbosity levels (level 4 skipped on purpose)
VERB_LVL = {'none': 0, 'low': 1, 'medium': 2, 'high': 3, 'debug': 5}
D_VERB_LVL = VERB_LVL['low']


# ======================================================================
def suscept2fieldshift(
        chi_r,
        b0z,
        theta,
        phi):
    """
    Calculate the z-comp. field perturbation due to susceptibility variation.

    Parameters
    ==========
    chi_r : ndarray
        Input voxmap containing susceptibility distribution chi(r) in SI units
    b0z : float
        Main magnetic field B0 in Tesla
    theta : float
        Angle of 1st rotation (along x-axis) in deg
    phi : float
        Angle of 2nd rotation (along y-axis) in deg

    Returns
    =======
    delta_bdz_r : numpy 3D-array
        Output voxmap containing z-component of field shift  DBdz(r) over B0_z

    """

    # convert input angles to radians
    theta = np.deg2rad(theta)
    phi = np.deg2rad(phi)
    # susceptibility in Fourier space (zero-centered)
    chi_k = np.fft.fftshift(np.fft.fftn(chi_r))
    # :: dipole convolution kernel C = C(k, theta, phi)
    #     / 1   (kz cos(th) cos(ph) - ky sin(th) cos(ph) + kx sin(ph))^2 \
    # C = | - - ==================================================------ |
    #     \ 3                      kx^2 + ky^2 + kz^2                    /
    k_range = [slice(-k_size / 2.0, +k_size / 2.0) for k_size in chi_k.shape]
    kk_x, kk_y, kk_z = np.ogrid[k_range]
    dipole_kernel_k = (1.0 / 3.0 -
                       (kk_z * np.cos(theta) * np.cos(phi)
                        - kk_y * np.sin(theta) * np.cos(phi)
                        + kk_x * np.sin(phi)) ** 2 /
                       (kk_x ** 2 + kk_y ** 2 + kk_z ** 2))
    # fix singularity for |k|^2 == 0 with DBdz(k) = B0z * chi(k) / 3
    singularity = np.isnan(dipole_kernel_k)
    dipole_kernel_k[singularity] = 1.0 / 3.0
    # field shift along z in Tesla in Fourier space
    delta_bdz_k = b0z * chi_k * dipole_kernel_k
    # field shift along z in Tesla in direct space (zero-centered)
    delta_bdz_r = np.fft.ifftn(np.fft.ifftshift(delta_bdz_k))
    delta_bdz_r = np.real(delta_bdz_r) / b0z
    return delta_bdz_r


# ======================================================================
def handle_arg():
    """
    Handle command-line application arguments.
    """
    # :: Define DEFAULT values
    # verbosity
    d_verbose = D_VERB_LVL
    # main magnetic field B0
    d_b0 = 7.0  # T
    # polar angle
    d_theta = 0.0  # deg
    # azimuthal angle
    d_phi = 0.0  # deg
    # input file
    d_input_file = 'tmp_suscept.nii.gz'
    # output file
    d_output_file = 'tmp_fieldshift.nii.gz'
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
    arg_parser.add_argument(
        '-b', '--b0z', metavar='B0_z',
        type=float, default=d_b0,
        help='set the magnetic field B0 along z-axis in Tesla [%(default)s]', )
    arg_parser.add_argument(
        '-t', '--theta', metavar='TH',
        type=float, default=d_theta,
        help='set angle of 1st rotation (along x-axis) in deg [%(default)s]', )
    arg_parser.add_argument(
        '-p', '--phi', metavar='PH',
        type=float, default=d_phi,
        help='set angle of 2nd rotation (along y-axis) in deg [%(default)s]', )
    arg_parser.add_argument(
        '-i', '--input', metavar='FILE',
        default=d_input_file,
        help='set input file [%(default)s]', )
    arg_parser.add_argument(
        '-o', '--output', metavar='FILE',
        default=d_output_file,
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
    #     suscept2fieldshift, [ARGS.b0z, ARGS.theta, ARGS.phi])
    end_time = datetime.datetime.now()
    if ARGS.verbose > VERB_LVL['none']:
        print('ExecTime: {}'.format(end_time - begin_time))
