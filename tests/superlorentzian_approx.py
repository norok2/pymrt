#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test superlorentzian approximation
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
import cmath  # Mathematical functions for complex numbers
import time  # Time access and conversions
import datetime  # Basic date and time types
# import operator  # Standard operators as functions
# import collections  # High-performance container datatypes
# import argparse  # Parser for command-line options, arguments and subcommands

# import itertools  # Functions creating iterators for efficient looping
# import functools  # Higher-order functions and operations on callable objects
# import subprocess  # Subprocess management
# import multiprocessing  # Process-based parallelism
# import csv  # CSV File Reading and Writing [CSV: Comma-Separated Values]
# import json  # JSON encoder and decoder [JSON: JavaScript Object Notation]

# :: External Imports
import numpy as np  # NumPy (multidimensional numerical arrays library)
import scipy as sp  # SciPy (signal and image processing library)
# import matplotlib as mpl  # Matplotlib (2D/3D plotting library)
import sympy as sym  # SymPy (symbolic CAS library)
# import PIL  # Python Image Library (image manipulation toolkit)
# import SimpleITK as sitk  # Image ToolKit Wrapper
# import nibabel as nib  # NiBabel (NeuroImaging I/O Library)
# import nipy  # NiPy (NeuroImaging in Python)
# import nipype  # NiPype (NiPy Pipelines and Interfaces)
import cProfile as profile  # Deterministic Profiler

# :: External Imports Submodules
import matplotlib.pyplot as plt  # Matplotlib's pyplot: MATLAB-like syntax
import mpl_toolkits.mplot3d as mpl3  # Matplotlib's 3D support
# import scipy.optimize  # SciPy: Optimization
import scipy.integrate  # SciPy: Integration
import scipy.interpolate  # SciPy: Interpolation
import scipy.constants  # SciPy: Constants
# import scipy.ndimage  # SciPy: Multidimensional image processing
import scipy.linalg  # SciPy: Linear Algebra
import scipy.stats  # SciPy: Statistical functions
import scipy.misc  # SciPy: Miscellaneous routines
import sympy.mpmath  # SymPy: Function approximation

from numpy import pi, sin, cos, exp, sqrt
# from sympy import pi, sin, cos, exp, sqrt, re, im

# :: Local Imports
import mri_tools.base as mrb


# import mri_tools.geometry as mrg
# import mri_tools.plot as mrp
# import mri_tools.segmentation as mrs

# from mri_tools import INFO
# from mri_tools import VERB_LVL
# from mri_tools import D_VERB_LVL
# from mri_tools import get_first_line


# ======================================================================
def superlorentz_integrand(x, t):
    return sqrt(2.0 / pi) * \
           exp(-2.0 * (x / (3 * cos(t) ** 2 - 1)) ** 2.0) * sin(t) / \
           abs(3 * cos(t) ** 2 - 1)


# ======================================================================
def superlorentz(x):
    # scipy.integrate.quad returns both the value and the error, here ignored
    return sp.integrate.quad(
            lambda t: superlorentz_integrand(x, t), 0.0, pi / 2.0)[0]


def test_approx(
        func,
        num_points=np.round(np.linspace(50, 500, 13)),
        support=(-10, 10)):
    errors = np.zeros_like(num_points)
    for i, num_point in enumerate(num_points):
        x = np.linspace(support[0], support[1], num_point)
        y = np.vectorize(superlorentz)(x)
        y2 = func(x)
        errors[i] = np.mean(((y - y2) / (y + 1.0)) ** 2.0)
    return errors, num_points


# ======================================================================
def optimize_sampling():
    n = 60
    max_sampl = 65536

    x_i = np.logspace(-20, 2.0, max_sampl)
    y_i = np.vectorize(superlorentz)(x_i)

    min_sel_list = np.round(np.linspace(0, max_sampl * 0.8, n))
    max_sel_list = np.round(np.linspace(max_sampl * 0.97, max_sampl - 1, n))
    skip_sel_list = range(1, 2 * n + 1)

    for min_sel in min_sel_list:
        for max_sel in max_sel_list:
            for skip_sel in skip_sel_list:
                selection = slice(min_sel, max_sel, skip_sel)
                begin_time = time.time()
                errors, num_points = test_approx(
                        lambda x: np.interp(np.abs(x), x_i[selection], y_i[selection]))
                end_time = time.time()
                print('{:.3f}\t{:.3f}\t{:4d}\t{}\t{:.8e}'.format(
                        np.log10(x_i[min_sel]),
                        np.log10(x_i[max_sel]),
                        len(x_i[selection]),
                        datetime.timedelta(0, end_time - begin_time),
                        np.mean(errors)))
                plt.plot(num_points, errors)
    print(num_points)
    plt.show()


def optimize_interpolation():
    x_i = np.logspace(-10.0, 1.6, 64)
    y_i = np.vectorize(superlorentz)(x_i)

    x_j = np.linspace(-10.0, 10.0, 1024)
    y_j = np.vectorize(superlorentz)(x_j)

    x_k = np.logspace(-10.0, 1.6, 256)
    y_k = np.vectorize(superlorentz)(x_k)

    kwargs = {
        'bounds_error': False, 'fill_value': 0.0}
    f_i = scipy.interpolate.interp1d(x_i, y_i, **kwargs)
    f_j = scipy.interpolate.interp1d(x_j, y_j, **kwargs)

    approx_func_list = [
        ('np.interp   LOG  ', lambda x: np.interp(np.abs(x), x_k, y_k)),
        ('np.interp   LOG  ', lambda x: np.interp(np.abs(x), x_i, y_i)),
        ('sp.interp1d LOG *', lambda x: f_i(np.abs(x))),
        ('np.interp   LIN  ', lambda x: np.interp(np.abs(x), x_j, y_j)),
        ('sp.interp1d LIN *', f_j),
    ]

    approx_func_list += [
        ('sp.interp1d LOG {}'.format(order),
         lambda x: scipy.interpolate.interp1d(x_i, y_i, kind=order, **kwargs)(
                 np.abs(x)))
        for order in (1, 2, 5)]

    approx_func_list += [
        ('sp.interp1d LIN {}'.format(order),
         scipy.interpolate.interp1d(x_j, y_j, kind=order, **kwargs))
        for order in (1, 5)]

    fig = plt.figure()
    for name, func in approx_func_list:
        begin_time = time.time()
        errors, num_points = test_approx(func)
        end_time = time.time()
        print('{:20s}\t{}\t{:.8e}'.format(
                name,
                datetime.timedelta(0, end_time - begin_time),
                np.mean(errors)))
        plt.plot(num_points, errors)
    print(num_points)
    plt.show()


# ======================================================================
def main():
    # def superlorentzian(x):
    #     t = sym.symbols('t')
    #     y = sym.integrate(superlorentz_integrand(x, t), (t, 0.0, pi / 2.0))
    #     return y

    # x = sym.symbols('x')
    # y = superlorentzian(x)

    optimize_sampling()
    optimize_interpolation()


    # plotting two different approximations
    # fig = plt.figure()
    # plt.plot(x, y)
    # plt.plot(x, y2)
    # plt.show()


# ======================================================================
if __name__ == '__main__':
    print(__doc__)
    main()
    # profile.run('main()', sort=2)
