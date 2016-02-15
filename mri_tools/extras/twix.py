#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
mri_tools/twix: manage Siemens's TWIX (raw) data from MRI scanners.
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
# import itertools  # Functions creating iterators for efficient looping
# import functools  # Higher-order functions and operations on callable objects
# import argparse  # Parser for command-line options, arguments and sub-command
# import subprocess  # Subprocess management
# import multiprocessing  # Process-based parallelism
# import fractions  # Rational numbers
# mport csv  # CSV File Reading and Writing [CSV: Comma-Separated Values]
import json  # JSON encoder and decoder [JSON: JavaScript Object Notation]

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
import mri_tools.base as mrb


# from mri_tools import INFO
# from mri_tools import VERB_LVL
# from mri_tools import D_VERB_LVL
# from mri_tools import get_first_line


# ======================================================================
class Twix:
    """
    Manages the extraction of data from the disk.
    """

    def __init__(
            self,
            filepath,
            version=None):
        """
        

        Args:
            filepath:
            version:

        Returns:

        """
        self.file = open(filepath, 'rb')
        stream = bin(self.file.readlines(4))
        print(stream)

        # perform some magic
        if version is None:

            self.file.seek(0)

        if version.startswith('VB'):
            pass

        if version.startswith('VD'):
            print('Not supported yet')
        self.version = version



# ======================================================================
def read(
        filepath,
        version=None):
    """

    Args:
        filepath:

    Returns:
        twix (Twix): object for
    """
    return Twix(filepath, version)


# ======================================================================
def test():
    filepath = '/media/rick/VERBATIM HD/FMRIB/' + \
               'meas_MID389_gre_qmri_0_6mm_FA30_MTOff_LowRes_FID36111.dat'
    twix = read(filepath, 'VB17')
    print(twix.version)


# ======================================================================
if __name__ == "__main__":
    print(__doc__)
    begin_time = time.time()
    test()
    end_time = time.time()
    print('ExecTime: ', datetime.timedelta(0, end_time - begin_time))
