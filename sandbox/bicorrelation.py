#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
PyMRT: useful I/O utilities.

TODO:
- improve affine support
- improve header support
"""

# ======================================================================
# :: Future Imports
from __future__ import division
from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals

# ======================================================================
# :: Python Standard Library Imports
import os  # Miscellaneous operating system interfaces
# import shutil  # High-level file operations
# import math  # Mathematical functions
# import time  # Time access and conversions
# import datetime  # Basic date and time types
# import operator  # Standard operators as functions
# import collections  # High-performance container datatypes
# import itertools  # Functions creating iterators for efficient looping
# import functools  # Higher-order functions and operations on callable objects
# import argparse  # Parser for command-line options, arguments and subcommands
# import subprocess  # Subprocess management
# import multiprocessing  # Process-based parallelism
# import fractions  # Rational numbers
# import csv  # CSV File Reading and Writing [CSV: Comma-Separated Values]
# import json  # JSON encoder and decoder [JSON: JavaScript Object Notation]
# import inspect  # Inspect live objects

# :: External Imports
import numpy as np  # NumPy (multidimensional numerical arrays library)
import scipy as sp  # SciPy (signal and image processing library)
# import matplotlib as mpl  # Matplotlib (2D/3D plotting library)
# import sympy as sym  # SymPy (symbolic CAS library)
# import PIL  # Python Image Library (image manipulation toolkit)
# import SimpleITK as sitk  # Image ToolKit Wrapper
import nibabel as nib  # NiBabel (NeuroImaging I/O Library)
# import nipy  # NiPy (NeuroImaging in Python)
# import nipype  # NiPype (NiPy Pipelines and Interfaces)

# :: External Imports Submodules
import matplotlib.pyplot as plt  # Matplotlib's pyplot: MATLAB-like syntax
# import scipy.optimize  # SciPy: Optimization Algorithms
# import scipy.integrate  # SciPy: Integrations facilities
# import scipy.constants  # SciPy: Mathematal and Physical Constants
import scipy.ndimage  # SciPy: ND-image Manipulation
# :: Local Imports
import pymrt.base as mrb
# import pymrt.geometry as mrg
import pymrt.plot as mrp
# import pymrt.segmentation as mrs
import pymrt.input_output as mrio


# from pymrt import INFO
# from pymrt import VERB_LVL, D_VERB_LVL
# from pymrt import msg, dbg
# from pymrt import get_first_line

# ======================================================================
# :: Custom defined constants

# ======================================================================
# :: Default values usable in functions
def sobel(img, absolute=True):
    sobel = np.zeros_like(img)
    for i in range(img.ndim):
        tmp = scipy.ndimage.sobel(img, i)
        sobel += (tmp ** 2 if absolute else tmp)
    return np.sqrt(sobel) if absolute else sobel


# ======================================================================
def bicorr(
        dirpath,
        subpath,
        in1_filename='t1.nii.gz',
        in2_filename='t2s.nii.gz',
        mask_filename='mask.nii.gz',
        save_filename='hist2d__{}.png',
        in1_interval=(700, 3000),
        in2_interval=(1, 80)):
    in1_filepath = os.path.join(dirpath, subpath, in1_filename)
    in2_filepath = os.path.join(dirpath, subpath, in2_filename)
    mask_filepath = os.path.join(dirpath, subpath, mask_filename)
    save_filepath = os.path.join(dirpath, save_filename.format(subpath))
    if os.path.isfile(mask_filepath):
        print(save_filepath)
        mrio.plot_histogram2d(
            in1_filepath, in2_filepath, mask_filepath, mask_filepath,
            save_filepath=save_filepath, scale='log',
            # array_interval=((700, 3000), (700, 3000)))
            array_interval=(in1_interval, in2_interval))
    else:
        print('E: File not found `{}`'.format(mask_filepath ))


# ======================================================================
def sharpness(dirpath, mask_filename='mask.nii.gz', ):
    in_filepaths = mrb.listdir(dirpath, mrb.EXT['niz'])
    in_filepaths = [item for item in in_filepaths if mask_filename not in item]
    mask_filepath = os.path.join(dirpath, mask_filename)
    msk = mrio.load(mask_filepath).astype(bool)
    for in_filepath in in_filepaths:
        img = mrio.load(in_filepath)
        mrb.calc_stats(img[msk], title=os.path.basename(in_filepath))


# ======================================================================
def sobel_gen(
        dirpath,
        src_names=('t1', 't2s')):
    subpaths = mrb.listdir(dirpath, None, full_path=False)
    subpaths = [item for item in subpaths if 'sobel' not in item]
    # generate for sharpness calculation
    for subpath in subpaths:
        in_path = os.path.join(dirpath, subpath)
        out_path = os.path.join(dirpath, 'sobel')
        if not os.path.isdir(out_path):
            os.makedirs(out_path)
        filenames = mrb.listdir(in_path, mrb.EXT['niz'], full_path=False)
        for filename in filenames:
            if mrb.change_ext(filename, '', mrb.EXT['niz']) in src_names:
                in_filepath = os.path.join(in_path, filename)
                out_filepath = os.path.join(out_path, subpath + '__' + filename)
                mrio.simple_filter_1_1(in_filepath, out_filepath, sobel)
    # generate for bicorrelation
    for subpath in subpaths:
        in_path = os.path.join(dirpath, subpath)
        out_path = os.path.join(dirpath, 'sobel__' + subpath)
        if not os.path.isdir(out_path):
            os.makedirs(out_path)
        filenames = mrb.listdir(in_path, mrb.EXT['niz'], full_path=False)
        for filename in filenames:
            if mrb.change_ext(filename, '', mrb.EXT['niz']) in src_names:
                in_filepath = os.path.join(in_path, filename)
                out_filepath = os.path.join(out_path, filename)
                mrio.simple_filter_1_1(in_filepath, out_filepath, sobel)


# ======================================================================
def main():
    dirpath = '~/hd2/cache/me-mp2rage/misreg/bicorr/'
    # sharpness
    # sobel_gen(dirpath)
    # sharpness(os.path.join(dirpath, 'sobel'))
    # joint correlation
    subpaths = mrb.listdir(dirpath, None, full_path=False)
    for subpath in subpaths:
        if 'sobel__' in subpath:
            bicorr(
                dirpath, subpath,
                in1_interval=(1, 10000), in2_interval=(1, 300))
        elif 'sobel' not in subpath:
            # bicorr(
            #     dirpath, subpath,
            #     in1_interval=(700, 3000), in2_interval=(1, 80))
            pass


# ======================================================================
if __name__ == '__main__':
    main()
