#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Collection of frequently-used imports."""

# ======================================================================
# :: Future Imports
from __future__ import division
from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals

# ======================================================================
# :: Python Standard Library Imports
import os  # Miscellaneous operating system interfaces
import shutil  # High-level file operations
import math  # Mathematical functions
import cmath  # Mathematical functions for complex numbers
import time  # Time access and conversions
import datetime  # Basic date and time types
import operator  # Standard operators as functions
import collections  # Container datatypes
import random  # Generate pseudo-random numbers
import argparse  # Parser for command-line options, arguments and subcommands
import itertools  # Functions creating iterators for efficient looping
import functools  # Higher-order functions and operations on callable objects
import subprocess  # Subprocess management
import multiprocessing  # Process-based parallelism
import csv  # CSV File Reading and Writing [CSV: Comma-Separated Values]
import json  # JSON encoder and decoder [JSON: JavaScript Object Notation]
import inspect  # Inspect live objects
import profile  # Deterministic profiling of Python programs
import warnings  # Warning control
import unittest  # Unit testing framework
import doctest  # Test interactive Python examples

# Configuration file parser
try:
    import configparser
except ImportError:
    import ConfigParser as configparser

# :: External Imports
import numpy as np  # NumPy (multidimensional numerical arrays library)
import scipy as sp  # SciPy (signal and image processing library)
import sympy as sym  # SymPy (symbolic CAS library)
import matplotlib as mpl  # Matplotlib (2D/3D plotting library)
import seaborn as sns  # Seaborn: statistical data visualization
import PIL  # Python Image Library (image manipulation toolkit)
import nibabel as nib  # NiBabel (NeuroImaging I/O Library)
import nipy  # NiPy (NeuroImaging in Python)
import nipype  # NiPype (NiPy Pipelines and Interfaces)

# :: External Imports Submodules
import matplotlib.pyplot as plt  # Matplotlib's pyplot: MATLAB-like syntax
import mpl_toolkits.mplot3d as mpl3  # Matplotlib's 3D support

import scipy.optimize  # SciPy: Optimization
import scipy.integrate  # SciPy: Integration
import scipy.interpolate  # SciPy: Interpolation
import scipy.constants  # SciPy: Constants
import scipy.ndimage  # SciPy: Multidimensional image processing
import scipy.linalg  # SciPy: Linear Algebra
import scipy.stats  # SciPy: Statistical functions
import scipy.misc  # SciPy: Miscellaneous routines

# import sympy.mpmath  # SymPy: Function approximation

# :: Local Imports
import pymrt.base as pmb
import pymrt.naming as pmn
import pymrt.geometry as pmg
import pymrt.plot as pmp
import pymrt.registration as pmr
import pymrt.segmentation as pms
import pymrt.computation as pmc
import pymrt.correlation as pml
import pymrt.input_output as pmio
import pymrt.sequences as pmq
import pymrt.extras as pme

from pymrt.sequences import mp2rage
from pymrt.sequences import matrix_algebra
from pymrt.extras import twix
from pymrt.extras import jcampdx

from pymrt import INFO
from pymrt import VERB_LVL, D_VERB_LVL
from pymrt import msg, dbg
from pymrt import elapsed, print_elapsed
