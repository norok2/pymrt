#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
mri_tools: useful basic utilities.
"""


# ======================================================================
# :: Future Imports
from __future__ import division
from __future__ import absolute_import
from __future__ import print_function
# from __future__ import unicode_literals


# ======================================================================
# :: Python Standard Library Imports
import os  # Miscellaneous operating system interfaces
import sys  # System-specific parameters and functions
# import shutil  # High-level file operations
# import math  # Mathematical functions
# import time  # Time access and conversions
# import datetime  # Basic date and time types
# import operator  # Standard operators as functions
#import collections  # High-performance container datatypes
import itertools  # Functions creating iterators for efficient looping
# import functools  # Higher-order functions and operations on callable objects
# import argparse  # Parser for command-line options, arguments and sub-command
import subprocess  # Subprocess management
# import multiprocessing  # Process-based parallelism
import fractions  # Rational numbers
import csv  # CSV File Reading and Writing [CSV: Comma-Separated Values]
# import json  # JSON encoder and decoder [JSON: JavaScript Object Notation]

# :: External Imports
import numpy as np  # NumPy (multidimensional numerical arrays library)
import scipy as sp  # SciPy (signal and image processing library)
# import matplotlib as mpl  # Matplotlib (2D/3D plotting library)
# import sympy as sym  # SymPy (symbolic CAS library)
# import PIL  # Python Image Library (image manipulation toolkit)
# import SimpleITK as sitk  # Image ToolKit Wrapper
# import nibabel as nib  # NiBabel (NeuroImaging I/O Library)
# import nipy  # NiPy (NeuroImaging in Python)
# import nipype  # NiPype (NiPy Pipelines and Interfaces)

# :: External Imports Submodules
import matplotlib.pyplot as plt  # Matplotlib's pyplot: MATLAB-like syntax
# import mayavi.mlab as mlab  # Mayavi's mlab: MATLAB-like syntax
import scipy.optimize  # SciPy: Optimization Algorithms
# import scipy.integrate  # SciPy: Integrations facilities
# import scipy.constants  # SciPy: Mathematal and Physical Constants
import scipy.ndimage  # SciPy: ND-image Manipulation

# :: Local Imports
# from mri_tools import INFO
from mri_tools import VERB_LVL
from mri_tools import D_VERB_LVL
# from mri_tools import _firstline


# ======================================================================
# :: Custom defined constants


# ======================================================================
# :: Default values usable in functions.
COMMENT_TOKEN = '#'
CSV_DELIMITER = '\t'
PNG_RANGE = (0.0, 255.0)
PNG_EXT = 'png'
EPS_EXT = 'eps'
SVG_EXT = 'svg'
TXT_EXT = 'txt'
CSV_EXT = 'csv'
DCM_EXT = 'ima'
D_TAB_SIZE = 8


# :: MatPlotLib-related constants
# colors and linestyles
PLOT_COLORS = ('r', 'g', 'b', 'c', 'm', 'y')
PLOT_LINESTYLES = ('-', '--', '-.', ':')
# standard plot resolutions
D_PLOT_DPI = 72


# :: TTY amenities
TTY_COLORS = {
    'r': 31, 'g': 32, 'b': 34, 'c': 36, 'm': 35, 'y': 33, 'w': 37, 'k': 30,
    'R': 41, 'G': 42, 'B': 44, 'C': 46, 'M': 45, 'Y': 43, 'W': 47, 'K': 40,
    }


# ======================================================================
def gcd(*num_list):
    """
    Find the greatest common divisor (GCD) of a list of numbers.

    Parameters
    ==========
    *num_list : list of int
        The input numbers.

    Returns
    =======
    gcd_val : int
        The value of the greatest common divisor (GCD).

    """
    gcd_val = num_list[0]
    for num in num_list[1:]:
        gcd_val = fractions.gcd(gcd_val, num)
    return gcd_val


# ======================================================================
def lcm(*num_list):
    """
    Find the least common multiple (LCM) of a list of numbers.

    Parameters
    ==========
    *num_list : list of int
        The input numbers.

    Returns
    =======
    lcm_val : int
        The value of the least common multiple (LCM)).

    """
    lcm_val = num_list[0]
    for num in num_list[1:]:
        lcm_val = lcm_val * num // fractions.gcd(lcm_val, num)
    return lcm_val


# ======================================================================
def merge_dicts(*dicts):
    """
    Merge dictionaries into a new dict (new keys overwrite the old ones).

    Parameters
    ==========
    dicts : dict tuple
       Dictionaries to be merged together.

    Returns
    =======
    merged : dict
         The merged dict (new keys overwrite the old ones).

    """
    merged = {}
    for item in dicts:
        merged.update(item)
    return merged


# ======================================================================
def accumulate(lst, func=lambda x, y: x + y):
    """
    Cumulatively apply the specified function to the elements of the list.

    Parameters
    ==========
    lst : list
        The list to process.
    func : func(x,y) -> z
        The function applied cumulatively to the first n items of the list.

    Returns
    =======
    lst : list
        The cumulative list.

    See Also
    ========
    itertools.accumulate

    """
    return [reduce(func, lst[:idx + 1]) for idx in range(len(lst))]


# ======================================================================
def multi_replace(text, replace_list):
    """
    Perform multiple replacements in a string.

    Parameters
    ==========
    text : str
        The input string.
    replace_list : (2-str tuple) tuple
        The listing of the replacements. Format: ((<old>, <new>), ...)

    Returns
    =======
    text : str
        The string after the performed replacements
    """
    return reduce(lambda s, r: s.replace(*r), replace_list, text),


# ======================================================================
def execute(cmd, use_pipes=True, dry=False, verbose=D_VERB_LVL):
    """
    Execute command and retrieve output at the end of execution.

    Parameters
    ==========
    cmd : str
        Command to execute.
    use_pipes : bool (optional)
        If True, get both stdout and stderr streams from the process.
    dry : bool (optional)
        If True, the command is printed instead of being executed (dry run).
    verbose : int (optional)
        Set level of verbosity.

    Returns
    =======
    p_stdout : string
        The stdout of the process after execution.
    p_stderr : string
        The stderr of the process after execution.

    """
    p_stdout = p_stderr = None
    if dry:
        print('Dry:\t{}'.format(cmd))
    else:
        if verbose > VERB_LVL['low']:
            print('Cmd:\t{}'.format(cmd))
        if use_pipes:
#            # :: deprecated
#            proc = os.popen3(cmd)
#            p_stdout, p_stderr = [item.read() for item in proc[1:]]
            # :: new style
            proc = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                shell=True, close_fds=True)
            p_stdout = proc.stdout.read()
            p_stderr = proc.stderr.read()
            if verbose > VERB_LVL['medium']:
                print('stdout:\t{}'.format(p_stdout))
            if verbose > VERB_LVL['medium']:
                print('stderr:\t{}'.format(p_stderr))
        else:
#            # :: deprecated
#            os.system(cmd)
            # :: new style
            subprocess.call(cmd, shell=True)
    return p_stdout, p_stderr


# ======================================================================
def groups_from(lst, grouping):
    """
    Generate a list of lists from a source list and grouping specifications.

    Parameters
    ==========
    lst : list
        The source list.
    grouping : (int)-list
        List of the number of elements that each group should contain.

    Returns
    =======
    groups : list
        List of lists obtained by grouping the elements of the source list.

    """
    group, groups = [], []
    jdx = 0
    count = grouping[jdx] if jdx < len(grouping) else len(lst) + 1
    for idx, item in enumerate(lst):
        if idx >= count:
            loop = True
            while loop:
                groups.append(group)
                group = []
                jdx += 1
                add = grouping[jdx] if jdx < len(grouping) else len(lst) + 1
                if add < 0:
                    add = len(lst) + 1
                count += add
                if add == 0:
                    loop = True
                else:
                    loop = False
        group.append(item)
    groups.append(group)
    return groups


# ======================================================================
def listdir(
        path,
        file_ext,
        pattern=slice(None, None, None),
        verbose=D_VERB_LVL):
    """
    Retrieve a sorted list of files matching specified extension and pattern.

    Parameters
    ==========
    path : str
        Path to search.
    file_ext : str
        File extension (eventually an empty string). None for directories
    pattern: slice (optional)
        Pattern for selecting filepaths (alphabetical ordering).
    verbose : bool (optional)
        Print useful information regarding the behavior of this function.

    Returns
    =======
    filepath_list : list
        Complete file paths matching specified extension and regular pattern.

    """
    # filter according to file_ext
    if file_ext is None:
        if verbose >= VERB_LVL['debug']:
            print('Scanning for DIRS on:\n{}'.format(path))
        # extracts only dirs
        filepath_list = [os.path.join(path, fi) for fi in os.listdir(path)
                         if os.path.isdir(os.path.join(path, fi))]
    else:
        if verbose >= VERB_LVL['debug']:
            print("Scanning for '{}' on:\n{}".format(file_ext, path))
        # extracts only those ending with specific file_ext
        filepath_list = [os.path.join(path, fi) for fi in os.listdir(path)
                         if fi.lower().endswith(file_ext.lower())]
    # sort filepath list
    filepath_list.sort()
    # return filepath list matching specified pattern
    return filepath_list[pattern]

# ======================================================================
def tty_colorify(
        text,
        color=None):
    """
    Add color TTY-compatible color code to a string, for pretty-printing.

    Parameters
    ==========
    text: str
        The text to be colored.
    color : str or int or None
        | A string or number for the color coding.
        | Lowercase letters modify the forground color.
        | Uppercase letters modify the background color.
        | Available colors:
        * r/R: red
        * g/G: green
        * b/B: blue
        * c/C: cyan
        * m/M: magenta
        * y/Y: yellow (brown)
        * k/K: black (gray)
        * w/W: white (gray)

    Returns
    =======
        The colored string.

    see also: TTY_COLORS
    """
    if color in TTY_COLORS:
        tty_color = TTY_COLORS[color]
    elif color in TTY_COLORS.values():
        tty_color = color
    else:
        tty_color = None
    if tty_color and sys.stdout.isatty():
        text = '\x1b[1;{color}m{}\x1b[1;m'.format(text, color=tty_color)
    return text


# ======================================================================
def add_extsep(ext):
    """
    Add a extsep char to a filename extension, if it does not have one.

    Parameters
    ==========
    ext : str
        Filename extension to which the dot has to be added.

    Returns
    =======
    dot_ext : str
        Filename extension with a prepending dot.

    """
    if not ext:
        dot_ext = ''
    elif ext[0] == os.path.extsep:
        dot_ext = ext
    else:
        dot_ext = os.path.extsep + ext
    return dot_ext


# ======================================================================
def compact_num_str(
        val,
        max_limit=D_TAB_SIZE - 1):
    """
    Convert a number into the most informative string within specified limit.

    Parameters
    ==========
    val : int or float
        The number to be converted to string.
    limit : int (optional
        The maximum number of characters allowed for the string.

    Returns
    =======
    val_str : str
        The string with the formatted number.

    """
    try:
        # this is to simplify formatting (and accepting even strings)
        val = float(val)
        # helpers
        extra_char_in_exp = 5
        extra_char_in_dec = 2
        extra_char_in_sign = 1
        # 'order' of zero is 1 for our purposes, because needs 1 char
        order = np.log10(abs(val)) if abs(val) > 0.0 else 1
        # adjust limit for sign
        limit = max_limit - extra_char_in_sign if val < 0.0 else max_limit
        # perform the conversion
        if order > float(limit) or order < -float(extra_char_in_exp - 1):
            limit -= extra_char_in_exp + 1
            val_str = '{:.{size}e}'.format(val, size=limit)
        elif -float(extra_char_in_exp - 1) <= order < 0.0:
            limit -= extra_char_in_dec
            val_str = '{:.{size}f}'.format(val, size=limit)
        elif val % 1.0 == 0.0:
            # currently, no distinction between int and float is made
            limit = 0
            val_str = '{:.{size}f}'.format(val, size=limit)
        else:
            limit -= (extra_char_in_dec + int(order))
            if limit < 0:
                limit = 0
            val_str = '{:.{size}f}'.format(val, size=limit)
    except (ValueError):
        print('EE:', 'Could not convert to float: {}'.format(val))
        val_str = 'NAN'
    return val_str


# ======================================================================
def has_decorator(text, pre_decor='"', post_decor='"'):
    """
    Determine if a string is delimited by some characters (decorators).
    """
    return text.startswith(pre_decor) and text.endswith(post_decor)


# ======================================================================
def strip_decorator(text, pre_decor='"', post_decor='"'):
    """
    Strip specific character sequences (decorators) from a string.
    """
    return text[len(pre_decor):-len(post_decor)]


# ======================================================================
def auto_convert(val_str, pre_decor=None, post_decor=None):
    """
    Convert value to numeric if possible, or strip delimiters from strings.
    """
    if pre_decor and post_decor and \
            has_decorator(val_str, pre_decor, post_decor):
        val = val_str[len(pre_decor):-len(post_decor)]
    else:
        try:
            val = int(val_str)
        except (ValueError):
            try:
                val = float(val_str)
            except (ValueError):
                try:
                    val = complex(val_str)
                except (ValueError):
                    val = val_str
    return val


# ======================================================================
def is_number(var):
    """
    Determine if a variable contains a number.

    Parameters
    ==========
    var : str
        The var

    Returns
    =======
    result : bool
        True if the values can be converted, False otherwise.

    """
    try:
        complex(var)
    except:
        result = False
    else:
        result = True
    return result


# ======================================================================
def significant_figures(val, num):
    """
    Format a number with the correct number of significant figures.

    Parameters
    ==========
    val : str or float or int
        The numeric value to be correctly formatted.
    num : str or int
        The number of significant figures to be displayed.

    Returns
    =======
    val : str
        String containing the properly formatted number.

    See Also
    ========
    The 'decimal' Python standard module.

    """
    val = float(val)
    num = int(num)
    order = int(np.floor(np.log10(abs(val)))) if abs(val) != 0.0 else 0
    dec = num - order - 1  # if abs(order) < abs(num) else 0
    typ = 'f' if order < num else 'g'
    prec = dec if order < num else num
#    print('val={}, num={}, ord={}, dec={}, typ={}, prec={}'.format(
#        val, num, order, dec, typ, prec))  # DEBUG
    val = '{:.{prec}{typ}}'.format(round(val, dec), prec=prec, typ=typ)
    return val


# ======================================================================
def format_value_error(
        val,
        err,
        num=2):
    """
    Write correct value/error pairs.

    Parameters
    ==========
    val : str or float or int
        The numeric value to be correctly formatted.
    err : str or float or int
        The numeric error to be correctly formatted.
    num : str or int (optional)
        The precision to be used for the error (usually 1 or 2).

    Returns
    =======
    val : str or float or int
        The numeric value correctly formatted.
    err : str or float or int
        The numeric error correctly formatted.

    """
    val = float(val)
    err = float(err)
    num = int(num)
    val_order = np.ceil(np.log10(np.abs(val))) if val != 0 else 0
    err_order = np.ceil(np.log10(np.abs(err))) if val != 0 else 0
    val_str = significant_figures(val, val_order - err_order + num)
    err_str = significant_figures(err, num)
    return val_str, err_str


# ======================================================================
def str2dict(
        in_str,
        entry_sep=',',
        key_val_sep='=',
        pre_decor='{',
        post_decor='}',
        strip_key_str=None,
        strip_val_str=None,
        convert=True):
    """
    Convert a string to a dictionary.

    Parameters
    ==========
    in_str : str
        The input string.
    entry_sep : str (optional)
        The entry separator.
    key_val_sep : str (optional)
        The key-value separator.
    pre_decor : str (optional)
        Beginning decorator string (starting the input string, not parsed).
    post_decor
        Ending decorator string (ending the input string, not parsed).
    strip_key_str : str (optional)
        | List of char to be stripped from both ends of the dictionary's key.
        | If None, whitespaces are stripped. Empty string for no stripping.
    strip_val_str : str (optional)
        | List of char to be stripped from both ends of the dictionary's value.
        | If None, whitespaces are stripped. Empty string for no stripping.
    convert : bool (optional)
        Enable automatic conversion of string to numeric.

    Returns
    =======
    out_dict : dict
        The output dictionary.

    See Also
    ========
        dict2str

    """
    if has_decorator(in_str, pre_decor, post_decor):
        in_str = strip_decorator(in_str, pre_decor, post_decor)
    entry_list = in_str.split(entry_sep)
    out_dict = {}
    for entry in entry_list:
        # fetch entry
        key_val_list = entry.split(key_val_sep)
        # parse entry
        if len(key_val_list) == 1:
            key, val = key_val_list[0], None
        elif len(key_val_list) == 2:
            key, val = key_val_list
            val = val.strip(strip_val_str)
        elif len(key_val_list) > 2:
            key, val = key_val_list[0], key_val_list[1:]
            val = [tmp_val.strip(strip_val_str) for tmp_val in val]
        else:
            key = None
        # strip dict key
        key = key.strip(strip_key_str)
        # add to dictionary
        if key:
            if convert:
                val = auto_convert(val)
            out_dict[key] = val
    return out_dict


# ======================================================================
def dict2str(
        in_dict,
        entry_sep=',',
        key_val_sep='=',
        pre_decor='{',
        post_decor='}',
        strip_key_str=None,
        strip_val_str=None,
        sort=None):
    """
    Convert a dictionary to a string.

    Parameters
    ==========
    in_dict : dict
        The input string.
    entry_sep : str (optional)
        The entry separator.
    key_val_sep : str (optional)
        The key-value separator.
    pre_decor : str (optional)
        Beginning decorator string (to be appended to the output).
    post_decor
        Ending decorator string ((to be appended to the output).
    convert : bool (optional)
        Enable automatic conversion of string to numeric.

    Returns
    =======
    out_str : str
        The output dictionary.

    See Also
    ========
        str2dict

    """
    if sort is True:
        key_list = sorted()
    else:
        key_list = in_dict.keys()
    out_list = []
    for key in key_list:
        key = key.strip(strip_key_str)
        val = str(in_dict[key]).strip(strip_val_str)
        out_list.append(key_val_sep.join([key, val]))
    out_str = pre_decor + entry_sep.join(out_list) + post_decor
    return out_str


# ======================================================================
def string_between(
        in_str,
        begin_str,
        end_str,
        incl_begin=False,
        incl_end=False,
        greedy=True):
    """
    Isolate the string contained between two tokens.

    Parameters
    ==========
    in_str : str
        String to parse.
    begin_string : str
        Token at the beginning.
    end_str : str
        Token at the ending.
    incl_begin : bool (optional)
        If True, include begin_string in the result.
    incl_end : bool (optional)
        If True, include end_str in the result.
    greedy : bool (optional)
        If True, output largest possible string.

    Returns
    =======
    out_str : str
        The string contained between the specified tokens (if any).
    """
    incl_begin = len(begin_str) if not incl_begin else 0
    incl_end = len(end_str) if incl_end else 0
    if begin_str in in_str and end_str in in_str:
        if greedy:
            out_str = in_str[
                in_str.find(begin_str) + incl_begin:
                in_str.rfind(end_str) + incl_end]
        else:
            out_str = in_str[
                in_str.rfind(begin_str) + incl_begin:
                in_str.find(end_str) + incl_end]
    else:
        out_str = ''
    return out_str


# ======================================================================
def check_redo(
        in_filepath_list,
        out_filepath_list,
        force=False):
    """
    Check if input files are newer than output files, to force calculation.

    Parameters
    ==========
    in_filepath_list : string list
        List of filepaths used as input of computation.
    out_filepath_list : string list
        List of filepaths used as output of computation.
    force : boolean
        Force computation to be re-done.

    Returns
    =======
    force : boolean
        Computation to be re-done.

    """
    if not in_filepath_list:
        raise IndexError('List of input files is empty.')
    for in_filepath in in_filepath_list:
        if not os.path.exists(in_filepath):
            raise IOError('Input file does not exists.')
    if not force:
        for out_filepath in out_filepath_list:
            if out_filepath:
                if not os.path.exists(out_filepath):
                    force = True
                    break
    if not force:
        for in_filepath, out_filepath in \
                itertools.product(in_filepath_list, out_filepath_list):
            if in_filepath and out_filepath:
                if os.path.getmtime(in_filepath) \
                        > os.path.getmtime(out_filepath):
                    force = True
                    break
    return force


# ======================================================================
def to_range(
        val,
        in_range=(0.0, 1.0),
        out_range=(0.0, 1.0)):
    """
    Linear convert value from input range to output range.

    Parameters
    ==========
    val : float
        Value to convert.
    in_range : float 2-tuple (optional)
        Range of the input value.
    out_range : float 2-tuple (optional)
        Range of the output value.

    Returns
    =======
    val : float
        The converted value.

    """
    in_min, in_max = in_range
    out_min, out_max = out_range
    return (val - in_min) / (in_max - in_min) * (out_max - out_min) + out_min


# ======================================================================
def range_size(val_range):
    """
    Calculate the size of a range given as a 2-tuple (A,B).

    Parameters
    ==========
    val_range : float 2-tuple
        Range for computation

    Returns
    =======
    val : float
        The converted value.

    """
    return val_range[1] - val_range[0]


# ======================================================================
def range_array(array):
    """
    Calculate the range of an array: (min, max).

    Parameters
    ==========
    array : ndarray
        Array for the computation of the range.

    Returns
    =======
    (min, max) : 2-tuple
        max(array), min(array)

    """
    range_arr = (np.min(array), np.max(array))
    return range_arr


# ======================================================================
def combined_range(
        range1,
        range2,
        operation):
    """
    Combine two ranges with some operation to obtain a new range.

    Parameters
    ==========
    range1 : float 2-tuple
        Range of first operand.
    range2 : float 2-tuple
        Range of second operand.
    operation : str
        String with operation to perform. Supported operations are:
            | '+' : addition
            | '-' : subtraction

    Returns
    =======
    new_range : float 2-tuple
        Range resulting from operation.

    """
    if operation == '+':
        new_range = (range1[0] + range2[0], range1[1] + range2[1])
    elif operation == '-':
        new_range = (range1[0] - range2[1], range1[1] - range2[0])
    else:
        new_range = (-np.inf, np.inf)
    return new_range


# ======================================================================
def mid_val_array(array):
    """
    Calculate the middle value vector.
    For example: [0, 1, 2, 3, 4] -> [0.5, 1.5, 2.5, 3.5]

    Parameters
    ==========
    array : N-array
        The input array.

    Returns
    =======
    array : (N-1)-array
        The output array.

    """
    return (array[1:] - array[:-1]) / 2.0 + array[:-1]


# ======================================================================
def polar2complex(modulus, argument):
    """
    Calculate complex number from the polar form.

    Parameters
    ==========
    modulus : float
        The modulus R of the complex number.
    argument : float
        The argument phi or phase of the complex number.

    Returns
    =======
    z : complex
        The complex number: z = R * cos(phi) + i * R * sin(phi)

    """
    return modulus * (np.cos(argument) + 1j * np.sin(argument))


# ======================================================================
def cartesian2complex(real, imag):
    """
    Calculate the complex number from the cartesian form.

    Parameters
    ==========
    real : float
        The real part z' of the complex number.
    imag : float
        The imaginary part z" of the complex number.

    Returns
    =======
    z : complex
        The complex number: z = z' + i * z"

    """
    return real + 1j * imag


# ======================================================================
def complex2cartesian(z):
    """
    Calculate the real and the imaginary part of a complex number.

    Parameters
    ==========
    z : complex
        The complex number: z = z' + i * z"

    Returns
    =======
    (real, imag) : (float, float)
        The real and imaginary part z' and z¨ of the complex number.

    """
    return np.real(z), np.imag(z)


# ======================================================================
def complex2polar(z):
    """
    Calculate the real and the imaginary part of a complex number.

    Parameters
    ==========
    z : complex
        The complex number: z = z' + i * z"

    Returns
    =======
    (modulus, argument) : (float, float)
        The modulus R and argument phi of the complex number.

    """
    return np.abs(z), np.angle(z)


# ======================================================================
def polar2cartesian(modulus, argument):
    """
    Calculate the real and the imaginary part of a complex number.

    Parameters
    ==========
    modulus : float
        The modulus R of the complex number.
    argument : float
        The argument phi or phase of the complex number.

    Returns
    =======
    (real, imag) : (float, float)
        The real and imaginary part z' and z¨ of the complex number.

    """
    return modulus * np.cos(argument), modulus * np.sin(argument)


# ======================================================================
def cartesian2polar(real, imag):
    """
    Calculate the real and the imaginary part of a complex number.

    Parameters
    ==========
    real : float
        The real part z' of the complex number.
    imag : float
        The imaginary part z" of the complex number.

    Returns
    =======
    (real, imag) : (float, float)
        The modulus R and argument phi of the complex number.

    """
    return np.sqrt(real ** 2 + imag ** 2), np.arctan2(real, imag)


# ======================================================================
def calc_stats(
        array,
        mask_nan=True,
        mask_inf=True,
        mask_vals=None,
        val_range=None,
        save_path=None,
        title=None,
        compact=False):
    """
    Calculate array statistical information (min, max, avg, std, sum).
    TODO: use a dictionary instead

    Parameters
    ==========
    array : ndarray
        The array to be investigated.
    mask_nan : bool (optional)
        Mask NaN values.
    mask_inf : bool (optional)
        Mask Inf values.
    mask_vals : list of int or float or None
        List of values to mask.
    val_range : 2-tuple (optional)
        The (min, max) values range.
    save_path : str or None (optional)
        The path to which the plot is to be saved. If unset, no output.
    title : str or None (optional)
        If title is not None, stats are printed to screen.
    compact : bool (optional)
        Use a compact format string for displaying results.

    Returns
    =======
    stats_dict : dict
        | 'min': minimum value
        | 'max': maximum value
        | 'avg': average or mean
        | 'std': standard deviation
        | 'sum': summation
        | 'num': number of elements

    """
    if mask_nan:
        array = array[~np.isnan(array)]
    if mask_inf:
        array = array[~np.isinf(array)]
    if not mask_vals:
        mask_vals = []
    for val in mask_vals:
        array = array[array != val]
    if val_range is None:
        val_range = range_array(array)
    array = array[array > val_range[0]]
    array = array[array < val_range[1]]
    stats_dict = {
        'avg': np.mean(array),
        'std': np.std(array),
        'min': np.min(array),
        'max': np.max(array),
        'sum': np.sum(array),
        'num': np.size(array), }
    label_list = ['avg', 'std', 'min', 'max', 'sum', 'num']
    val_list = []
    for label in label_list:
        val_list.append(compact_num_str(stats_dict[label]))
    if save_path:
        with open(save_path, 'wb') as csv_file:
            csv_writer = csv.writer(csv_file, delimiter=CSV_DELIMITER)
            csv_writer.writerow(label_list)
            csv_writer.writerow(val_list)
    if title:
        print_str = title + ': '
        for label in label_list:
            if compact:
                print_str += '{}={}, '.format(
                    label, compact_num_str(stats_dict[label]))
            else:
                print_str += '{}={}, '.format(label, stats_dict[label])
        print(print_str)
    return stats_dict


# ======================================================================
def slice_array(
        array,
        axis=0,
        index=None):
    """
    Slice a (N-1)D-array from an ND-array

    Parameters
    ==========
    array : ndarray
        The original array.
    axis : int (optional)
        The slicing axis.
    index : int (optional)
        The slicing index. If None, mid-value is taken.

    Returns
    =======
    sliced : ndarray
        The sliced (N-1)D-array.

    """
    # initialize slice index
    slab = [slice(None)] * len(array.shape)
    # ensure index is meaningful
    if not index:
        index = np.int(array.shape[axis] / 2.0)
    # check index
    if (index >= array.shape[axis]) or (index < 0):
        raise ValueError('Invalid array index in the specified direction')
    # determine slice index
    slab[axis] = index
    # slice the array
    return array[slab]


# ======================================================================
# def plot_with_adjusting_parameters()
# TODO: make a plot with possibility to adjust params


# ======================================================================
def plot_sample2d(
        array,
        axis=0,
        index=None,
        title=None,
        val_range=None,
        cmap=None,
        use_new_figure=True,
        close_figure=False,
        save_path=None):
    """
    Plot a 2D sample image of a 3D array.

    Parameters
    ==========
    array : ndarray
        The original 3D array.
    axis : int (optional)
        The slicing axis.
    index : int (optional)
        The slicing index. If None, mid-value is taken.
    title : str (optional)
        The title of the plot.
    val_range : 2-tuple (optional)
        The (min, max) values range.
    cmap : MatPlotLib ColorMap (optional)
        The colormap to be used for displaying the histogram.
    use_new_figure : bool (optional)
        Plot the histogram in a new figure.
    close_figure : bool (optional)
        Close the figure after saving (useful for batch processing).
    save_path : str (optional)
        The path to which the plot is to be saved. If unset, no output.

    Returns
    =======
    sample : ndarray
        The sliced (N-1)D-array.

    """
    if len(array.shape) != 3:
        raise IndexError('3D array required')
    sample = slice_array(array, axis, index)
    if use_new_figure:
        plt.figure()
    if title:
        plt.title(title)
    if val_range is None:
        val_range = range_array(array)
    plt.imshow(sample, cmap=cmap, vmin=val_range[0], vmax=val_range[1])
    plt.colorbar(use_gridspec=True)
    if save_path is not None:
        plt.tight_layout()
        plt.savefig(save_path, dpi=D_PLOT_DPI)
    if close_figure:
        plt.close()
    return sample


# ======================================================================
def plot_histogram1d(
        array,
        bin_size=1,
        hist_range=(0.0, 1.0),
        bins=None,
        array_range=None,
        scale='linear',
        title='Histogram',
        labels=('Value', 'Value Frequency'),
        style='-k',
        use_new_figure=True,
        close_figure=False,
        save_path=None):
    """
    Plot 1D histogram of array with MatPlotLib.

    Parameters
    ==========
    array : nd-array
        The array for which histogram is to be plotted.
    bin_size : int or float or float (optional)
        The size of the bins.
    hist_range : float 2-tuple (optional)
        The range of the histogram to display in percentage.
    bins : int (optional)
        The number of bins to use. If set, it overrides bin_size parameter.
    array_range : float 2-tuple (optional)
        Theoretical range of values for the array. If unset, uses min and max.
    scale : ['linear'|'log'|'log10'|'normed'] string (optional)
        The frequency value scaling.
    title : str (optional)
        The title of the plot.
    labels : str 2-tuple (optional)
        A 2-tuple of strings containing x-labels and y-labels.
    style : str (optional)
        Plotting style string (as accepted by MatPlotLib).
    use_new_figure : bool (optional)
        Plot the histogram in a new figure.
    close_figure : bool (optional)
        Close the figure after saving (useful for batch processing).
    save_path : str (optional)
        The path to which the plot is to be saved. If unset, no output.

    Returns
    =======
    hist : array
        The calculated histogram.

    """
    # setup array range
    if not array_range:
        array_range = (np.nanmin(array), np.nanmax(array))
    # setup bins
    if not bins:
        bins = int(range_size(array_range) / bin_size + 1)
    # setup histogram reange
    hist_range = tuple([to_range(val, out_range=array_range)
                        for val in hist_range])
    # calculate histogram
    if scale == 'normed':
        is_normed = True
    else:
        is_normed = False
    # prepare figure
    if use_new_figure:
        plt.figure()
    # create histogram
    hist, bin_edges = np.histogram(
        array, bins=bins, range=hist_range, normed=is_normed)
    # adjust scale
    if scale == 'log':
        hist[hist != 0.0] = np.log(hist[hist != 0.0])
    elif scale == 'log10':
        hist[hist != 0.0] = np.log10(hist[hist != 0.0])
    # plot figure
    plt.plot(mid_val_array(bin_edges), hist, style)
    # setup title and labels
    if title:
        plt.title(title)
    if labels[0]:
        plt.xlabel(labels[0])
    if labels[1]:
        plt.ylabel(labels[1] + ' ({})'.format(scale))
    else:
        plt.ylabel('{}'.format(scale))
    # save figure to file
    if save_path is not None:
        plt.tight_layout()
        plt.savefig(save_path, dpi=D_PLOT_DPI)
    # closing figure
    if close_figure:
        plt.close()
    return hist, bin_edges


# ======================================================================
def plot_histogram1d_list(
        array_list,
        bin_size=1,
        hist_range=(0.0, 1.0),
        bins=None,
        array_range=None,
        scale='linear',
        title='Histogram',
        labels=('Value', 'Value Frequency'),
        legends=None,
        styles=None,
        use_new_figure=True,
        close_figure=False,
        save_path=None):
    """
    Plot 1D histograms of multiple arrays with MatPlotLib.

    Parameters
    ==========
    array : nd-array
        The array for which histogram is to be plotted.
    bin_size : int or float (optional)
        The size of the bins.
    hist_range : float 2-tuple (optional)
        The range of the histogram to display in percentage.
    bins : int (optional)
        The number of bins to use. If set, it overrides bin_size parameter.
    array_range : float 2-tuple (optional)
        Theoretical range of values for the array. If unset, uses min and max.
    scale : ['linear'|'log'|'log10'|'normed'] string (optional)
        The frequency value scaling.
    title : str (optional)
        The title of the plot.
    labels : str 2-tuple (optional)
        The strings for x- and y-labels.
    styles : str list (optional)
        MatPlotLib's plotting style strings. If None, uses color cyclying.
    legends : str list (optional)
        Legend for each histogram. If None, no legend will be displayed.
    use_new_figure : bool (optional)
        Plot the histogram in a new figure.
    close_figure : bool (optional)
        Close the figure after saving (useful for batch processing).
    save_path : str (optional)
        The path to which the plot is to be saved. If unset, no output.

    Returns
    =======
    hist : array
        The calculated histogram.
    bin_edges : array
        The bin edges of the calculated histogram.
    plot : array
        The plot for further manipulation of the figure.

    """
    # setup array range
    if not array_range:
        array_range = (np.nanmin(array_list[0]), np.nanmax(array_list[0]))
        for array in array_list[1:]:
            array_range = (
                min(np.nanmin(array), array_range[0]),
                max(np.nanmax(array), array_range[1]))
    # setup bins
    if not bins:
        bins = int(range_size(array_range) / bin_size + 1)
    # setup histogram reange
    hist_range = tuple([to_range(val, out_range=array_range)
                        for val in hist_range])
    # calculate histogram
    if scale == 'normed':
        is_normed = True
    else:
        is_normed = False
    # prepare figure
    if use_new_figure:
        plot = plt.figure()
    # prepare style list
    if styles is None:
        styles = [linestyle + color
            for linestyle in PLOT_LINESTYLES for color in PLOT_COLORS]
    style_cycler = itertools.cycle(styles)
    # prepare histograms
    for idx, array in enumerate(array_list):
        hist, bin_edges = np.histogram(
            array, bins=bins, range=hist_range, normed=is_normed)
        # adjust scale
        if scale == 'log':
            hist[hist != 0.0] = np.log(hist[hist != 0.0])
        elif scale == 'log10':
            hist[hist != 0.0] = np.log10(hist[hist != 0.0])
        # prepare legend
        if legends is not None and idx < len(legends):
            legend = legends[idx]
        else:
            legend = '_nolegend_'
        # plot figure
        plt.plot(
            mid_val_array(bin_edges), hist, next(style_cycler), label=legend)
        plt.legend()
    # setup title and labels
    if title:
        plt.title(title)
    if labels[0]:
        plt.xlabel(labels[0])
    if labels[1]:
        plt.ylabel(labels[1] + ' ({})'.format(scale))
    else:
        plt.ylabel('{}'.format(scale))
    # save figure to file
    if save_path is not None:
        plt.tight_layout()
        plt.savefig(save_path, dpi=D_PLOT_DPI)
    # closing figure
    if close_figure:
        plt.close()
    return hist, bin_edges, plot


# ======================================================================
def plot_histogram2d(
        array1,
        array2,
        bin_size=1,
        hist_range=(0.0, 1.0),
        bins=None,
        array_range=None,
        use_separate_range=False,
        scale='linear',
        interpolation='bicubic',
        title='2D Histogram',
        labels=('Array 1 Values', 'Array 2 Values'),
        cmap=plt.cm.jet,
        bisector=None,
        use_new_figure=True,
        close_figure=False,
        save_path=None):
    """
    Plot 2D histogram of two arrays with MatPlotLib.

    Parameters
    ==========
    array1 : ndarray
        The array 1 for which the 2D histogram is to be plotted.
    array2 : ndarray
        The array 1 for which the 2D histogram is to be plotted.
    bin_size : int or float | int 2-tuple (optional)
        The size of the bins.
    hist_range : float 2-tuple | 2-tuple of float 2-tuple (optional)
        The range of the histogram to display in percentage.
    bins : int | int 2-tuple (optional)
        The number of bins to use. If set, it overrides bin_size parameter.
    array_range : float 2-tuple | 2-tuple of float 2-tuple (optional)
        Theoretical range of values for the array. If unset, uses min and max.
    use_separate_range : bool (optional)
        Select if display ranges in each dimension are determined separately.
    scale : ['linear'|'log'|'log10'|'normed'] string (optional)
        The frequency value scaling.
    interpolation : str (optional)
        Interpolation method (see imshow()).
    title : str (optional)
        The title of the plot.
    labels : str 2-tuple (optional)
        A 2-tuple of strings containing x-labels and y-labels.
    cmap : MatPlotLib ColorMap (optional)
        The colormap to be used for displaying the histogram.
    bisector : str or None (optional)
        If not None, show the first bisector using specified line style.
    use_new_figure : bool (optional)
        Plot the histogram in a new figure.
    close_figure : bool (optional)
        Close the figure after saving (useful for batch processing).
    save_path : str (optional)
        The path to which the plot is to be saved. If unset, no output.

    Returns
    =======
    hist2d : array
        The calculated 2D histogram.

    """
    # setup array range
    if not array_range:
        if use_separate_range:
            array_range = (
                (np.nanmin(array1), np.nanmax(array1)),
                (np.nanmin(array2), np.nanmax(array2)))
        else:
            array_range = (
                min(np.nanmin(array1), np.nanmin(array2)),
                max(np.nanmax(array1), np.nanmax(array2)))
    try:
        array_range[0].__iter__
    except AttributeError:
        array_range = (array_range, array_range)
    # setup bins
    if not bins:
        bins = tuple([int(range_size(a_range) / bin_size + 1)
                      for a_range in array_range])
    else:
        try:
            bins.__iter__
        except AttributeError:
            bins = (bins, bins)
    # setup histogram range
    try:
        hist_range[0].__iter__
    except AttributeError:
        hist_range = (hist_range, hist_range)
    hist_range = list(hist_range)
    for idx in range(2):
        hist_range[idx] = tuple([
            to_range(val, out_range=array_range[idx])
            for val in hist_range[idx]])
    hist_range = tuple(hist_range)
    # calculate histogram
    if scale == 'normed':
        is_normed = True
    else:
        is_normed = False
    # prepare histogram
    hist, x_edges, y_edges = np.histogram2d(
        array1.ravel(), array2.ravel(),
        bins=bins, range=hist_range, normed=is_normed)
    hist = hist.transpose()
    # adjust scale
    if scale == 'log':
        hist[hist != 0.0] = np.log(hist[hist != 0.0])
    elif scale == 'log10':
        hist[hist != 0.0] = np.log10(hist[hist != 0.0])
    # prepare figure
    if use_new_figure:
        plt.figure()
    # plot figure
    plt.imshow(
        hist, cmap=cmap, origin='lower', interpolation=interpolation,
        vmin=np.floor(np.min(hist)), vmax=np.ceil(np.max(hist)),
        extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]])
    # plot the color bar
    plt.colorbar(use_gridspec=True)
    # plot first bisector
    if bisector:
        plt.autoscale(False)
        x_val, y_val = [np.linspace(*val_range) for val_range in array_range]
        plt.plot(array_range[0], array_range[1], bisector, label='bisector')
    # setup title and labels
    plt.title(title + ' ({} scale)'.format(scale))
    plt.xlabel(labels[0])
    plt.ylabel(labels[1])
    # save figure to file
    if save_path is not None:
        plt.tight_layout()
        plt.savefig(save_path, dpi=D_PLOT_DPI)
    # closing figure
    if close_figure:
        plt.close()
    return hist


# ======================================================================
def euclid_dist(
        arr1,
        arr2,
        unsigned=True):
    """
    Calculate the element-wise correlation euclidean distance D,
    i.e. the distance between the identity line and the point of coordinates
    given by intensity.
        | D = sqrt(((A1 - A2) / 2)^2 + ((A2 - A1) / 2)^2)
        | D = abs(A2 - A1) / sqrt(2)

    Parameters
    ==========
    arr1 : ndarray
        The first array.
    arr2 : ndarray
        The second array.
    signed : bool (optional)
        Use signed distance.

    Returns
    =======
    arr : ndarray
        The resulting array.

    """
    arr = (arr2 - arr1) / np.sqrt(2.0)
    if unsigned:
        arr = np.abs(arr)
    return arr


# ======================================================================
def rel_err(
        arr1,
        arr2,
        use_average=True):
    """
    Calculate the element-wise relative error RE , i.e. the difference between
    the two arrays divided by their average or by the value of the 1st array.
        | E = (A2 - A1) / A1
        | E = 2 * (A2 - A1) / (A2 + A1)

    Parameters
    ==========
    arr1 : ndarray
        The first array.
    arr2 : ndarray
        The second array.

    Returns
    =======
    arr : ndarray
        The resulting array.

    """
    if arr2.dtype != np.complex:
        arr = (arr2 - arr1).astype(np.float)
    else:
        arr = (arr2 - arr1)
    if use_average:
        div = (arr1 + arr2) / 2.0
    else:
        div = arr1
    mask = (div != 0.0)
    arr[mask] = arr[mask] / div[mask]
    arr[~mask] = 0.0
    return arr


# ======================================================================
def ndstack(arr_list, axis=-1):
    """
    Stack a list of arrays of the same size along a specific axis.

    Parameters
    ==========
    arr_list : (N-1)-dim nd-array list
        A list of NumPy arrays of the same size.
    axis : int [0,N] (optional)
        Orientation along which array is concatenated in the N-dim space.

    Returns
    =======
    arr : N-dim nd-array
        The concatenated array.

    """
    arr = arr_list[0]
    n_dim = len(arr.shape) + 1
    if axis < 0:
        axis += n_dim
    if axis < 0:
        axis = 0
    if axis > n_dim:
        axis = n_dim
    # calculate new shape
    shape = arr.shape[:axis] + tuple([len(arr_list)]) + arr.shape[axis:]
    # stack arrays together
    arr = np.zeros(shape, dtype=arr.dtype)
    for idx, src in enumerate(arr_list):
        index = [slice(None)] * n_dim
        index[axis] = idx
        arr[tuple(index)] = src
    return arr


# ======================================================================
def ndsplit(arr, axis=-1):
    """
    Split an array along a specific axis into a list of arrays.

    Parameters
    ==========
    arr : N-dim nd-array
        The array to operate with.
    axis : int [0,N] (optional)
        Orientation along which array is split in the N-dim space.

    Returns
    =======
    arr_list : (N-1)-dim nd-array list
        A list of NumPy arrays of the same size.

    """
    n_dim = len(arr.shape)
    # split array apart
    arr_list = []
    for idx in range(arr.shape[axis]):
        # determine index for slicing
        index = [slice(None)] * n_dim
        index[axis] = idx
        arr_list.append(arr[index])
    return arr_list


# ======================================================================
def curve_fit(param_list):
    """
    Interface to use scipy.optimize.curve_fit with multiprocessing.
    If an error is encountered, optimized parameters and their covariance are
    set to 0.

    Parameters
    ==========
    param_list : list
        List of parameters to be passed to the function.

    Returns
    =======
    par_fit : array
        Optimized parameters minimizing least-square fitting.
    par_cov : 2d-array
        The estimated covariance of the optimized parameters.
        The diagonals provide the variance of the parameter estimate.

    """
    try:
        result = sp.optimize.curve_fit(*param_list)
    except (RuntimeError, RuntimeWarning, ValueError):
#        print('EE: Fitting error. Params were: {}', param_list)  # DEBUG
        err_val = 0.0
        n_fit_par = len(param_list[3])  # number of fitting parameters
        result = \
            np.tile(err_val, n_fit_par), \
            np.tile(err_val, (n_fit_par, n_fit_par))
    return result


# ======================================================================
def threshold2mask(
        arr,
        val,
        comparison):
    """
    Create a mask from image according to specific threshold.

    Parameters
    ==========
    arr : nd-array
        Array from which mask is created.
    val : numeric
        Value to be used for comparison.
    comparison : str
        Comparison mode: [=|>|<|>=|<=|~]

    Returns
    =======
    mask : nd-array
        Mask for which comparison is True.

    """
    if comparison == '=':
        mask = (arr == val)
    elif comparison == '>':
        mask = (arr > val)
    elif comparison == '<':
        mask = (arr < val)
    elif comparison == '>=':
        mask = (arr >= val)
    elif comparison == '<=':
        mask = (arr <= val)
    elif comparison == '~':
        mask = (arr != val)
    else:
        raise ValueError('Valid comparison modes are: [=|>|<|>=|<=|~]')
    return mask


# ======================================================================
def calc_mask(
        array,
        smoothing=1.0,
        hist_dev_factor=4.0,
        rel_threshold=0.01,
        comparison='>',
        erosion_iter=0):
    """
    Extract a mask from an array.
    | Workflow is:
    * Gaussian filter (smoothing) with specified sigma
    * histogram deviation reduction by a specified factor
    * masking values using a relative threshold (and thresholding method)
    * binary erosion(s) witha specified number of iterations.

    Parameters
    ==========
    array : nd-array
        Array from which mask is created.
    smoothing : float
        Sigma to be used for Gaussian filtering. If zero, no filtering done.
    hist_dev_factor : float
        Histogram deviation reduction factor (in std-dev units):
        values that are the specified number of standard deviations away from
        the average are not used for the absolute thresholding calculation.
    rel_threshold : (0,1)-float
        Relative threshold for masking out values.
    comparison : str
        Comparison mode: [=|>|<|>=|<=|~]
    erosion_iter : int
        Number of binary erosion iteration in mask post-processing.

    Returns
    =======
    array : nd-array
        The extracted mask.

    """
    # Gaussian smoothing
    if smoothing:
        array = sp.ndimage.gaussian_filter(array, sigma=smoothing)
    # histogram deviation reduction
    min_val, max_val = range_array(array)
    if hist_dev_factor:
        avg_val = np.mean(array)
        std_val = np.std(array)
        min_val = max(min_val, avg_val - hist_dev_factor * std_val)
        max_val = min(max_val, avg_val + hist_dev_factor * std_val)
    # absolute threshold value calculation
    threshold = min_val + (max_val - min_val) * rel_threshold
    # mask creation
    array = threshold2mask(array, threshold, comparison)
    # binary erosion
    if erosion_iter > 0:
        array = sp.ndimage.binary_erosion(array, iterations=erosion_iter)
    return array


# :: TODO: fix registration-related functions
# ======================================================================
def affine_registration(
        moving,
        fixed,
        transform='affine',
        interp_order=1,
        metric=None):
    """
    Register the 'moving' image to the 'fixed' image, using only the specified
    transformation and the given metric.

    Parameters
    ==========
    moving : ndarray
        The image to be registered.
    fixed : ndarray
        The reference (or template) image.
    transform : str (optional)
        The allowed transformations:
        | affine : general linear transformation and translation
        | similarity : scaling, rotation and translation
        | rigid : rotation and translation
        | scaling : only scaling (TODO: include translation)
        | translation : only translation

    Returns
    =======
    affine :
        The n+1 square matrix describing the affine transformation that
        minimizes the specified metric and can be used for registration.

    """

    def min_func_translation(shift, num_dim, moving, fixed, interp_order):
        """
        Function to minimize for translation transformation.
        """
        if all(np.abs(shift)) < np.max(moving.shape):
            moved = scipy.ndimage.shift(moving, shift, order=interp_order)
            diff = moved - fixed
        else:
            diff = np.tile(np.inf, len(moving))
        return np.abs(diff.ravel())

    def min_func_scaling(scaling, num_dim, moving, fixed, interp_order):
        """
        Function to minimize for scaling transformation.
        """
        if all(scaling) > 0.0:
            moved = scipy.ndimage.zoom(moving, scaling, order=interp_order)
            diff = moved - fixed
        else:
            diff = np.tile(np.inf, len(moving))
        return np.abs(diff.ravel())

    def min_func_rigid(par, num_dim, moving, fixed, axes_list, interp_order):
        """
        Function to minimize for rigid transformation.
        """
        shift = par[:num_dim]
        angle_list = par[num_dim:]
        if all(np.abs(shift)) < np.max(moving.shape) and \
                -2.0 * np.pi <= all(angle_list) <= 2.0 * np.pi:
            axes_list = list(itertools.combinations(range(num_dim), 2))
            moved = scipy.ndimage.shift(moving, shift, order=interp_order)
            for angle, axes in zip(angle_list, axes_list):
                moved = scipy.ndimage.rotate(moved, angle, axes, reshape=False)
            diff = moved - fixed
        else:
            diff = np.tile(np.inf, len(moving))
        return np.abs(diff.ravel())

    def min_func_affine(par, num_dim, moving, fixed, interp_order):
        """
        Function to minimize for affine transformation.
        """
        shift = par[:num_dim]
        linear = par[num_dim:].reshape((num_dim, num_dim))
        moved = scipy.ndimage.affine_transform(moving, linear, shift,
                                               order=interp_order)
        diff = moved - fixed
        return np.abs(diff.ravel())

    # determine number of dimensions
    num_dim = len(moving.shape)

    # calculate starting points
    shift = np.zeros(num_dim)  # TODO: use center-of-mass
    linear = np.eye(num_dim)  # TODO: use rotational tensor (of inerita)
    if transform == 'translation':
        par0 = shift
        res = scipy.optimize.leastsq(
            min_func_translation, par0,
            args=(num_dim, moving, fixed, interp_order))
        opt_par = res[0]
        shift = -opt_par
        linear = np.eye(num_dim)
    elif transform == 'scaling':  # TODO: improve scaling
        scaling = np.ones(num_dim)
        par0 = scaling
        res = scipy.optimize.leastsq(
            min_func_scaling, par0,
            args=(num_dim, moving, fixed, interp_order))
        opt_par = res[0]
        shift = np.zeros(num_dim)
        linear = np.diag(opt_par)
    elif transform == 'rigid':
        shift = np.zeros(num_dim)
        axes_list = list(itertools.combinations(range(num_dim), 2))
        angles = np.zeros(len(axes_list))
        par0 = np.concatenate((shift, angles))
        res = scipy.optimize.leastsq(
            min_func_rigid, par0,
            args=(num_dim, moving, fixed, axes_list, interp_order))
        opt_par = res[0]
        shift = opt_par[:num_dim]
        angles = opt_par[num_dim:]
        linear = angles2linear(angles, axes_list)
    elif transform == 'affine':
        par0 = np.concatenate((shift, linear.ravel()))
        res = scipy.optimize.leastsq(
            min_func_affine, par0,
            args=(num_dim, moving, fixed, interp_order))
        opt_par = res[0]
        shift = opt_par[:num_dim]
    affine = compose_affine(linear, shift)
    return affine


# ======================================================================
def apply_affine(
        array,
        affine,
        *opts):
    """
    Apply the specified affine transformation to the array.

    Parameters
    ==========
    img : ndarray
        The n-dimensional image to be transformed.
    affine : ndarray
        The n+1 square matrix describing the affine transformation.
    opts : ...
        Additional options to be passed to: scipy.ndimage.affine_transform

    Returns
    =======
    img : ndarray
        The transformed image.

    """
    linear, shift = decompose_affine(affine)
    if np.iscomplex(array).any():
        real = scipy.ndimage.affine_transform(
            np.real(array), linear, offset=shift)
        imag = scipy.ndimage.affine_transform(
            np.imag(array), linear, offset=shift)
        array = cartesian2complex(real, imag)
    else:
        array = scipy.ndimage.affine_transform(array, linear, offset=shift)
    return array


# ======================================================================
def decompose_affine(
        affine):
    """
    Decompose the affine matrix into a linear transformation and a translation.

    Parameters
    ==========
    affine : ndarray
        The n+1 square matrix describing the affine transformation.

    Returns
    =======
    linear : ndarray
        The n square matrix describing the linear transformation.
    shift : array
        The array containing the shift along each axis.

    """
    dims = affine.shape
    linear = affine[:dims[0] - 1, :dims[1] - 1]
    shift = affine[:-1, -1]
    return linear, shift


# ======================================================================
def compose_affine(
        linear,
        shift):
    """
    Combine a linear transformation and a translation into the affine matrix.

    Parameters
    ==========
    linear : ndarray
        The n square matrix describing the linear transformation.
    shift : array
        The array containing the shift along each axis.

    Returns
    =======
    affine : ndarray
        The n+1 square matrix describing the affine transformation.

    """
    dims = linear.shape
    affine = np.eye(dims[0] + 1)
    affine[:dims[0], :dims[1]] = linear
    affine[:-1, -1] = shift
    return affine


# ======================================================================
def angles2linear(
        angles,
        axes_list=None):
    """
    """
    # solution to: n! / 2! / (n - 2)! = N  (N: num of angles, n: num of dim)
    num_dim = ((1 + np.sqrt(1 + 8 * len(angles))) / 2)
    if np.modf(num_dim)[0] != 0.0:
        raise ValueError('Cannot get the dimension from the number of angles.')
    else:
        num_dim = int(num_dim)
    if not axes_list:
        axes_list = list(itertools.combinations(range(num_dim), 2))
    linear = np.eye(num_dim)
    for angle, axes in zip(angles, axes_list):
        rotation = np.eye(num_dim)
        rotation[axes[0], axes[0]] = np.cos(angle)
        rotation[axes[1], axes[1]] = np.cos(angle)
        rotation[axes[0], axes[1]] = -np.sin(angle)
        rotation[axes[1], axes[0]] = np.sin(angle)
        linear = np.dot(linear, rotation)
    return linear


# ======================================================================
if __name__ == '__main__':
    print(__doc__)

    ## Test significant_figures:
    #print(significant_figures(0.077355507524, 1))
    #print(significant_figures(0.077355507524, 2))
    #print(significant_figures(0.077355507524, 3))
    #
    #print(significant_figures(77355507524, 1))
    #print(significant_figures(77355507524, 2))
    #print(significant_figures(77355507524, 3))
    #
    #print(significant_figures(77, 1))
    #print(significant_figures(77, 2))
    #print(significant_figures(77, 3))
