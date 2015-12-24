#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
mri_tools: generate output for LaTeX.
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
import mri_tools.modules.base as mrb


# from mri_tools import INFO
# from mri_tools import VERB_LVL
# from mri_tools import D_VERB_LVL
# from mri_tools import get_first_line


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
def gen_table(
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

    def tabular(labels, rows, format_dict):
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
        for idx, label in enumerate(labels):
            for jdx, col in enumerate(label):
                if label_filters:
                    col = label_filters[idx][jdx](col)
                output_str += to_safe('{}\t'.format(col))
                output_str += '& ' if jdx + 1 < len(label) else '\\\\\n'
        # separator
        output_str += '\hline\n'
        # write data
        seps = mrb.accumulate(grouping) if grouping else []
        for idx, row in enumerate(rows):
            for jdx, col in enumerate(row):
                if row_filters:
                    col = row_filters[jdx](col)
                output_str += to_safe('{}\t'.format(col))
                output_str += '& ' if jdx + 1 < len(row) else '\\\\\n'
            if idx + 1 in seps and idx + 1 < len(rows):
                output_str += '\\hline\n'
        # separator
        output_str += '\\hline \\hline\n'
        # end tabular environment
        output_str += '\\end{tabular}\n'
        # ...close wrapping
        output_str += '}\n\n'
        return output_str

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
        for idx in range(num_chunks):
            begin_idx = idx * split
            end_idx = (idx + 1) * split if (idx + 1) < num_chunks else None
            chunks.append(rows[begin_idx:end_idx])
        for chunk in chunks:
            output_str += tabular(labels, chunk, format_dict)
            output_str += '\n\n\n% -= tabular separator =- %\n\n\n'
    else:
        output_str += tabular(labels, rows, format_dict)

    if save_filepath:
        with open(save_filepath, 'w') as save_file:
            save_file.write(output_str)

    return output_str


# ======================================================================
if __name__ == '__main__':
    print(__doc__)
