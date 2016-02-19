#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Template: Module Name and Description

Long module description.
"""


# ======================================================================
# :: Future Imports
from __future__ import division
from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals


# ======================================================================
# :: Custom Module Details
AUTHOR = 'Riccardo Metere'
CONTACT = 'metere@cbs.mpg.de'
DATE_INFO = {'day': 1, 'month': 'Jul', 'year': 2014}
DATE = ' '.join([str(v) for k, v in sorted(DATE_INFO.items())])
COPYRIGHT = 'Copyright (C) ' + str(DATE_INFO['year'])
LICENSE = 'License GPLv3: GNU General Public License version 3'
# first non-empty line of __doc__
DOCget_first_line = [line for line in __doc__.splitlines() if line][0]


# ======================================================================
# :: Python Standard Library Imports
# import os  # Operating System facilities
# import math  # Mathematical Functions
import collections  # Collections of Items
import argparse  # Argument Parsing

# :: External Imports
import numpy as np  # NumPy (multidimensional numerical arrays library)
# import scipy as sp  # SciPy (signal and image processing library)
# import matplotlib as mpl  # Matplotlib (2D/3D plotting library)
import matplotlib.pyplot as plt  # Matplotlib's pyplot: MATLAB-like syntax
# import mayavi.mlab as mlab  # Mayavi's mlab: MATLAB-like syntax
# import sympy as sym  # SymPy (symbolic CAS library)
# import PIL  # Python Image Library (image manipulation toolkit)
# import SimpleITK as sitk  # Image ToolKit Wrapper
# import nibabel as nib  # NiBabel (NeuroImaging I/O Library)
# import nipy  # NiPy (NeuroImaging in Python)
# import nipype  # NiPype (NiPy Pipelines and Interfaces)

# :: Local Imports
# import mri_tools.modules.base as mrb
# import mri_tools.modules.utils as mru
# import mri_tools.modules.nifti as mrn
# import mri_tools.modules.geometry as mrg
# from mri_tools.modules.sequences import mp2rage
from mri_tools import __version__

# ======================================================================
# :: supported verbosity levels (level 4 skipped on purpose)
VERB_LVL = {'none': 0, 'low': 1, 'medium': 2, 'high': 3, 'debug': 5}


# ======================================================================
def my_func(my_param):
    """
    Sample function.

    Parameters
    ==========
    my_param : type
        Sample parameter.

    Returns
    =======
    my_rr : rtype
        Sample return.

    """
    my_rr = my_param
    return my_rr


# ======================================================================
def handle_arg():
    """
    Handle command-line application arguments.
    """
    # :: Define DEFAULT values
    # verbosity
    d_verbose = VERB_LVL['none']
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
            __version__, DOCget_first_line, COPYRIGHT, AUTHOR, CONTACT, LICENSE),
        action='version')
    arg_parser.add_argument(
        '-v', '--verbose',
        action='count', default=d_verbose,
        help='increase the level of verbosity [%(default)s]')
    # :: Add additional arguments
    # nothing here yet!
    # avoid mandatory arguments whenever possible
    return arg_parser


# ======================================================================
if __name__ == '__main__':
    # :: handle program parameters
    arg_parser = handle_arg()
    args = arg_parser.parse_args()
    # :: print debug info
    if args.verbose == VERB_LVL['debug']:
        arg_parser.print_help()
        print()
        print('II:', 'Parsed Arguments:', args)
    print(__doc__)

T1_val, TR_val = np.ogrid[700:2500, 1:20:0.1]
th_E_val = np.rad2deg(np.arccos(np.exp(-TR_val / T1_val)))
plt.ion()
plt.xlabel('TR / ms')
plt.ylabel('flip angle / deg')
plt.contourf(np.arange(1, 20, 0.1), np.arange(700, 2500), th_E_val, 50)
plt.colorbar()
plt.show(block=True)
print(T1_val.shape)
