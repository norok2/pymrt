#!python
# -*- coding: utf-8 -*-
"""
Calculate the optimal signal conditions (e.g. Ernst angle) for FLASH sequences.

Long module description.
"""

# ======================================================================
# :: Future Imports
from __future__ import(
    division, absolute_import, print_function, unicode_literals)

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
# import sympy as sym  # SymPy (symbolic CAS library)
# import PIL  # Python Image Library (image manipulation toolkit)
# import SimpleITK as sitk  # Image ToolKit Wrapper
# import nibabel as nib  # NiBabel (NeuroImaging I/O Library)
# import nipy  # NiPy (NeuroImaging in Python)
# import nipype  # NiPype (NiPy Pipelines and Interfaces)

# :: Local Imports
# import pymrt.base as pmb
# import pymrt.naming as pmn
# import pymrt.input_output as pmio
# import pymrt.geometry as pmg
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
    arg_parser.add_argument(
        '-q', '--quiet',
        action='store_true',
        help='override verbosity settings to suppress output [%(default)s]')
    # :: Add additional arguments
    # nothing here yet!
    # avoid mandatory arguments whenever possible
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

    T1_val, TR_val = np.ogrid[700:2500, 1:20:0.1]
    th_E_val = np.rad2deg(np.arccos(np.exp(-TR_val / T1_val)))
    plt.ion()
    plt.xlabel('TR / ms')
    plt.ylabel('flip angle / deg')
    plt.contourf(np.arange(1, 20, 0.1), np.arange(700, 2500), th_E_val, 50)
    plt.colorbar()
    plt.show(block=True)
    print(T1_val.shape)

    import tkinter

    top = tkinter.Tk()
    # Code to add widgets will go here...
    top.mainloop()


# ======================================================================
if __name__ == '__main__':
    main()
