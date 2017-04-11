#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Collection of frequently-used imports."""

# ======================================================================
# :: Future Imports
from __future__ import(
    division, absolute_import, print_function, unicode_literals)

# ======================================================================
# :: Python Standard Library Imports
import os  # Miscellaneous operating system interfaces
import sys  # System-specific parameters and functions
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
import urllib  # URL handling modules
import difflib  # Helpers for computing deltas
import email  # An email and MIME handling package
import shlex  # Simple lexical analysis
import hashlib  # Secure hashes and message digests
import warnings  # Warning control
import getpass  # Portable password input
import socket  # Low-level networking interface
import smtplib  # SMTP protocol client
import tempfile  # Generate temporary files and directories
import re  # Regular expression operations
import glob  # Unix style pathname pattern expansion

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
import pandas as pd  # pandas (Python Data Analysis Library)
import seaborn as sns  # Seaborn: statistical data visualization
import PIL  # Python Image Library (image manipulation toolkit)

import nibabel as nib  # NiBabel (NeuroImaging I/O Library)
import nipy  # NiPy (NeuroImaging in Python)
import nipype  # NiPype (NiPy Pipelines and Interfaces)

# :: External Imports Submodules
import matplotlib.pyplot as plt  # Matplotlib's pyplot: MATLAB-like syntax
import mpl_toolkits.mplot3d as mpl3  # Matplotlib's 3D support

import scipy.io  # SciPy: Input and output
import scipy.optimize  # SciPy: Optimization
import scipy.integrate  # SciPy: Integration
import scipy.interpolate  # SciPy: Interpolation
import scipy.constants  # SciPy: Constants
import scipy.ndimage  # SciPy: Multidimensional image processing
import scipy.linalg  # SciPy: Linear Algebra
import scipy.stats  # SciPy: Statistical functions
import scipy.misc  # SciPy: Miscellaneous routines
import scipy.signal  # SciPy: Signal Processing

# import sympy.mpmath  # SymPy: Function approximation

# :: Local Imports
import pymrt as mrt
import pymrt as mrt
import pymrt.utils
import pymrt.naming
import pymrt.geometry
import pymrt.plot
import pymrt.registration
import pymrt.segmentation
import pymrt.computation
import pymrt.correlation
import pymrt.input_output
import pymrt.sequences
import pymrt.extras

from pymrt.sequences import mp2rage
from pymrt.sequences import matrix_algebra
from pymrt.extras import twix
from pymrt.extras import jcampdx
from pymrt.recipes import *

from pymrt import INFO, DIRS
from pymrt import VERB_LVL, D_VERB_LVL
from pymrt import msg, dbg
from pymrt import elapsed, print_elapsed
