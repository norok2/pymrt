#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PyMRT: generate output for LaTeX.
"""

# ======================================================================
# :: Future Imports
from __future__ import (
    division, absolute_import, print_function, unicode_literals, )

# ======================================================================
# :: Python Standard Library Imports
# import os  # Miscellaneous operating system interfaces
# import shutil  # High-level file operations
# import math  # Mathematical functions
# import time  # Time access and conversions
# import datetime  # Basic date and time types
# import operator  # Standard operators as functions
# import collections  # Container datatypes
import itertools  # Functions creating iterators for efficient looping
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
# import scipy.optimize  # SciPy: Optimization Algorithms
# import scipy.integrate  # SciPy: Integrations facilities
# import scipy.constants  # SciPy: Mathematal and Physical Constants
# import scipy.ndimage  # SciPy: ND-image Manipulation

# :: Local Imports
import pymrt as mrt
import pymrt.utils

from pymrt import INFO, PATH
# from pymrt import VERB_LVL, D_VERB_LVL, VERB_LVL_NAMES
from pymrt import elapsed, report
from pymrt import msg, dbg


# ======================================================================
def to_safe(text):
    """
    Escape LaTeX sensitive characters.

    Parameters
    ==========
    text : str
        The string to be scaped.

    Returns
    =======
    text : str
        The escaped string.

    """
    escape_list = ('%', '\\%'), ('#', '\\#')
    for escape in escape_list:
        text = text.replace(*escape)
    return text


# ======================================================================
def tabular(labels, rows, grouping, label_filters, row_filters, format_dict):
    output_str = ''
    # open wrapping...
    output_str += '{\n'
    # row coloring
    if 'rowcolors' in format_dict:
        output_str += '\\rowcolors{{{}}}{{{}}}{{{}}}\n'.format(
            len(labels) + 1, *format_dict['rowcolors'])
    # vertical stretching
    if 'vstretch' in format_dict:
        output_str += '\\renewcommand{{\\arraystretch}}{{{}}}%\n'.format(
            format_dict['vstretch'])
    # begin tabular environment
    output_str += '\\begin{{tabular}}{{{}}}\n'.format('r' * len(labels[0]))
    # separator
    output_str += '\\hline \\hline\n'
    # write labels
    for i, label in enumerate(labels):
        for j, col in enumerate(label):
            if label_filters:
                col = label_filters[i][j](col)
            output_str += to_safe('{}\t'.format(col))
            output_str += '& ' if j + 1 < len(label) else '\\\\\n'
    # separator
    output_str += '\hline\n'
    # write data
    seps = itertools.accumulate(grouping) if grouping else []
    for i, row in enumerate(rows):
        for j, col in enumerate(row):
            if row_filters:
                col = row_filters[j](col)
            output_str += to_safe('{}\t'.format(col))
            output_str += '& ' if j + 1 < len(row) else '\\\\\n'
        if i + 1 in seps and i + 1 < len(rows):
            output_str += '\\hline\n'
    # separator
    output_str += '\\hline \\hline\n'
    # end tabular environment
    output_str += '\\end{tabular}\n'
    # ...close wrapping
    output_str += '}\n\n'
    return output_str


# ======================================================================
def make(
        labels,
        rows,
        grouping=None,
        comments='',
        split=None,
        save_filepath=None,
        label_filters=None,
        row_filters=None,
        format_str='{"rowcolors": ["even", "odd"], "vstretch": 1.2}'):
    """
    Generate LaTeX code to write a standard table.

    Parameters
    ==========
    labels : (str list) list
        A list of rows, each containing a list of cols to be used as labels.
    rows : (str list) list
        A list of rows, each containing a list of cols to be used as data.
    grouping : int list (optional)
        The number of elements for each group. In between a separator is added.
    comments : str (optional)
        A string to be added as a comment.
    split : int or None (optional)
        The maximum number of rows in a table. Multiple tables are produced.
    save_filepath : str or None (optional)
        Filepath where to save the output LaTeX.
    label_filters : (func list) list or None
        List of functions f(x) -> x to be applied to each label element.
    row_filters : func list or None
        List of functions f(x) -> x to be applied to each column of the rows.
    format_str : str
        | JSON-compatible dictionary of additional formatting options.
        | Accepted values:
        * rowcolors: 2-str tuple containing the colors to be used.
        * vstretch: int of the multiplier to be used as stretching factor.

    Returns
    =======
    A string containing the generated table(s).

    """
    output_str = ''

    if any([len(row) != len(label) for row in rows for label in labels]):
        raise IndexError('Number of columns in labels and data must match.')

    # preamble comment
    output_str += '% WARNING! This table was generated automatically.\n'
    output_str += '%          Manual modifications might be lost.\n\n'
    # save additional comments
    for line in comments.split('\n'):
        output_str += '% ' + line + '\n'
    if comments:
        output_str += '\n'

    format_dict = json.loads(format_str)
    if split:
        num_chunks = int(np.ceil(len(rows) / split))
        chunks = []
        for i in range(num_chunks):
            begin_idx = i * split
            end_idx = (i + 1) * split if (i + 1) < num_chunks else None
            chunks.append(rows[begin_idx:end_idx])
        for chunk in chunks:
            output_str += tabular(labels, chunk, format_dict)
            output_str += '\n\n\n% -= tabular separator =- %\n\n\n'
    else:
        output_str += tabular(
            labels, rows, format_dict, label_filters, row_filters)

    if save_filepath:
        with open(save_filepath, 'w') as save_file:
            save_file.write(output_str)

    return output_str


# ======================================================================
def gen_matrix(
        x_arr,
        dx_arr=None,
        c_sep=r'&',
        r_sep=r'\\',
        container=r'array',
        prefix=r'\left[' + '\n',
        suffix=r'\right]' + '\n',
        prec=2):
    """
    Generate a LaTeX matrix from a numerical array.

    Optionally support specifying the error.

    Args:
        x_arr (np.ndarray): The input array of values.
        dx_arr (np.ndarray|None): The input array of errors.
        c_sep (str): The column separator.
        r_sep (str): The row separator.
        container (str): The container environment.
        prec (int): The approximation precision.
            If `dx_arr` is not None, determines the number of significant
            figures for the error (typically 1 or 2).
            Otherwise, indicates the number of digits after the decimal
            separator.
            Must be a non-negative number.

    Returns:
        text (str): The LaTeX-formatted matrix.
    """
    if dx_arr is None:
        dx_arr = np.zeros_like(x_arr)
    assert (x_arr.shape == dx_arr.shape)
    assert (x_arr.ndim == 2)
    text = ''
    for i, (x, dx) in enumerate(zip(x_arr.ravel(), dx_arr.ravel())):
        cell = ' {} \pm {} '.format(
            *mrt.utils.format_value_error(x, dx, prec))
        ending = c_sep if i % x_arr.shape[1] < x_arr.shape[1] - 1 else r_sep
        text += cell + ending
    columns = 'c' * x_arr.shape[1]
    text = (
        prefix +
        r'\begin{{{}}}{{{}}}'.format(container, columns) +
        text +
        r'\end{{{}}}'.format(container) +
        suffix)
    return text


# ======================================================================
elapsed(__file__[len(PATH['base']) + 1:])

# ======================================================================
if __name__ == '__main__':
    import doctest  # Test interactive Python examples

    msg(__doc__.strip())
    doctest.testmod()
    msg(report())
