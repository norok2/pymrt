#!python
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
# :: Python Standard Library Imports
# import os  # Operating System facilities
# import math  # Mathematical Functions
# import collections  # Collections of Items
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
# import mri_tools.base as mrb
# import mri_tools.utils as mru
# import mri_tools.input_output as mrio
# import mri_tools.geometry as mrg
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
    # nothing here yet!
    # avoid mandatory arguments whenever possible
    return arg_parser


# ======================================================================
def main():
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

# ======================================================================
if __name__ == '__main__':
    main()
