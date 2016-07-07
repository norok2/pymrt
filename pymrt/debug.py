#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
pymrt.base: debug utilities.
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
# import sys  # System-specific parameters and functions
# import shutil  # High-level file operations
# import math  # Mathematical functions
# import time  # Time access and conversions
# import datetime  # Basic date and time types
# import operator  # Standard operators as functions
# import collections  # High-performance container datatypes
# import itertools  # Functions creating iterators for efficient looping
# import functools  # Higher-order functions and operations on callable objects
# import argparse  # Parser for command-line options, arguments and sub-command
# import subprocess  # Subprocess management
# import multiprocessing  # Process-based parallelism
# import fractions  # Rational numbers
# import csv  # CSV File Reading and Writing [CSV: Comma-Separated Values]
import json  # JSON encoder and decoder [JSON: JavaScript Object Notation]
import inspect  # Inspect live objects
import stat  # Interpreting stat() results
# import unittest  # Unit testing framework
import doctest  # Test interactive Python examples

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
# from pymrt import INFO
# from pymrt import VERB_LVL
# from pymrt import D_VERB_LVL
# from pymrt import _EVENTS

# ======================================================================
def dbg(name):
    """
    Print content of a variable for debug purposes.

    Args:
        name: The name to be inspected.

    Returns:
        None.

    Examples:
        >>> my_dict = {'a': 1, 'b': 1}
        >>> dbg(my_dict)
        dbg(my_dict): {
            "a": 1,
            "b": 1
        }
        <BLANKLINE>
        >>> dbg(my_dict['a'])
        dbg(my_dict['a']): 1
        <BLANKLINE>
    """
    outer_frame = inspect.getouterframes(inspect.currentframe())[1]
    name_str = outer_frame[4][0][:-1]
    print(name_str, end=': ')
    print(json.dumps(name, sort_keys=True, indent=4))
    print()


# ======================================================================
if __name__ == '__main__':
    print(__doc__)
    doctest.testmod()
