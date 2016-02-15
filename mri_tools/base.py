#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
mri_tools: generic basic utilities.
"""

# ======================================================================
# :: Future Imports
from __future__ import division
from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals

# ======================================================================
# :: Python Standard Library Imports
import functools
import os  # Miscellaneous operating system interfaces
import sys  # System-specific parameters and functions
# import shutil  # High-level file operations
# import math  # Mathematical functions
import time  # Time access and conversions
import datetime  # Basic date and time types
# import operator  # Standard operators as functions
# import collections  # High-performance container datatypes
import itertools  # Functions creating iterators for efficient looping
# import functools  # Higher-order functions and operations on callable objects
# import argparse  # Parser for command-line options, arguments and sub-command
import subprocess  # Subprocess management
# import multiprocessing  # Process-based parallelism
import fractions  # Rational numbers
import csv  # CSV File Reading and Writing [CSV: Comma-Separated Values]
# import json  # JSON encoder and decoder [JSON: JavaScript Object Notation]
import inspect  # Inspect live objects

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
# import matplotlib.pyplot as plt  # Matplotlib's pyplot: MATLAB-like syntax
# import mayavi.mlab as mlab  # Mayavi's mlab: MATLAB-like syntax
import scipy.optimize  # SciPy: Optimization Algorithms
# import scipy.integrate  # SciPy: Integrations facilities
# import scipy.constants  # SciPy: Mathematal and Physical Constants
# import scipy.ndimage  # SciPy: ND-image Manipulation

# :: Local Imports
# from mri_tools import INFO
from mri_tools import VERB_LVL
from mri_tools import D_VERB_LVL
from mri_tools import _EVENTS

# from mri_tools import get_first_line

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
    func : func(x,y) -> z (optional)
        The function applied cumulatively to the first n items of the list.
        Defaults to cumulative sum.

    Returns
    =======
    lst : list
        The cumulative list.

    See Also
    ========
    itertools.accumulate

    """
    return [functools.reduce(func, lst[:idx + 1]) for idx in range(len(lst))]


# ======================================================================
def multi_replace(text, replace_list):
    """
    Perform multiple replacements in a string.

    Parameters
    ==========
    text : str
        The input string.
    replace_list : (2-str tuple) tuple
        The listing of the replacements.
        Format: ((<old>, <new>), ...)

    Returns
    =======
    text : str
        The string after the performed replacements.

    """
    return functools.reduce(lambda s, r: s.replace(*r), replace_list, text)


# ======================================================================
def cartesian(*arrays):
    """
    Generate a cartesian product of input arrays.

    Parameters
    ----------
    arrays (list of arrays): 1-D arrays to form the cartesian product of

    Returns
    -------
    out (ndarray): 2-D array of shape (M, len(arrays)) containing cartesian
        products formed of input arrays.

    Examples
    --------
    >>> cartesian(([1, 2, 3], [4, 5], [6, 7]))
    array([[1, 4, 6],
           [1, 4, 7],
           [1, 5, 6],
           [1, 5, 7],
           [2, 4, 6],
           [2, 4, 7],
           [2, 5, 6],
           [2, 5, 7],
           [3, 4, 6],
           [3, 4, 7],
           [3, 5, 6],
           [3, 5, 7]])

    """

    arrays = [np.asarray(x) for x in arrays]
    dtype = arrays[0].dtype

    n = np.prod([x.size for x in arrays])
    out = np.zeros([n, len(arrays)], dtype=dtype)

    m = n / arrays[0].size
    out[:, 0] = np.repeat(arrays[0], m)
    if arrays[1:]:
        out[0:m, 1:] = cartesian(arrays[1:])
        for j in range(1, arrays[0].size):
            out[j * m:(j + 1) * m, 1:] = out[0:m, 1:]
    return out


# ======================================================================
def set_keyword_parameters(
        func,
        values):
    """
    Set keyword parameters of a function to specific or default values.

    Parameters
    ==========
    func : function
        The function to be inspected.
    values : dict
        A dictionary containing the values to set.
        If a value is set to None, it will be replaced by the default value.
        To use the names defined locally, use: `locals()`

    Results
    =======
    kw_params : dict
        A dictionary of the keyword parameters to set.

    See Also
    ========
    inspect.getargspec,
    locals,
    globals

    """
    inspected = inspect.getargspec(func)
    defaults = dict(
        zip(reversed(inspected.args), reversed(inspected.defaults)))
    kw_params = {}
    for key in inspected.args:
        if key in values:
            kw_params[key] = values[key]
        elif key in defaults:
            kw_params[key] = defaults[key]
    return kw_params


# ======================================================================
def mdot(*array_list):
    """
    Cumulative application of `numpy.dot` operation.
    """
    # todo: fix doc
    array = array_list[0]
    for item in array_list[1:]:
        array = np.dot(array, item)
    return array


# ======================================================================
def ndot(array, dim=-1, step=1):
    """
    Cumulative application of `numpy.dot` operation.
    """

    if dim < 0:
        dim += array.ndim
    start = 0 if step > 0 else array.shape[dim] - 1
    stop = array.shape[dim] if step > 0 else -1
    prod = array[
        [slice(None) if j != dim else start for j in range(array.ndim)]]
    for i in range(start, stop, step)[1:]:
        indexes = [slice(None) if j != dim else i for j in range(array.ndim)]
        prod = np.dot(prod, array[indexes])
    return prod


def commutator(a, b):
    """
    Calculate the commutator of two arrays: [A,B] = AB - BA
    """
    # todo: fix doc
    return a.dot(b) - b.dot(a)


def anticommutator(a, b):
    """
    Calculate the anticommutator of two arrays: [A,B] = AB + BA
    """
    # todo: fix doc
    return a.dot(b) + b.dot(a)


# ======================================================================
def execute(cmd, use_pipes=True, dry=False, verbose=D_VERB_LVL):
    """
    Execute command and retrieve output at the end of execution.

    Args:
        command (str): Command to execute.
        use_pipes (bool): Get stdout and stderr streams from the process.
        dry (bool): Print rather than execute the command (dry run).
        verbose (int): Set level of verbosity

    Returns:
        p_stdout (str|None): if use_pipes the stdout of the process
        p_stderr (str|None): if use_pipes the stderr of the process

    """
    p_stdout = p_stderr = None
    if dry:
        print('Dry:\t{}'.format(cmd))
    else:
        if verbose >= VERB_LVL['high']:
            print('Cmd:\t{}'.format(cmd))
        if use_pipes:
            # # :: deprecated
            # proc = os.popen3(cmd)
            # p_stdout, p_stderr = [item.read() for item in proc[1:]]
            # :: new style
            proc = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                shell=True, close_fds=True)
            p_stdout = proc.stdout.read()
            p_stderr = proc.stderr.read()
            if verbose >= VERB_LVL['debug']:
                print('stdout:\t{}'.format(p_stdout))
                print('stderr:\t{}'.format(p_stderr))
        else:
            # # :: deprecated
            # os.system(cmd)
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
    j = 0
    count = grouping[j] if j < len(grouping) else len(lst) + 1
    for i, item in enumerate(lst):
        if i >= count:
            loop = True
            while loop:
                groups.append(group)
                group = []
                j += 1
                add = grouping[j] if j < len(grouping) else len(lst) + 1
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
        file_ext='',
        pattern=slice(None, None, None),
        full_path=True,
        verbose=D_VERB_LVL):
    """
    Retrieve a sorted list of files matching specified extension and pattern.

    Args:
        path (str): Path to search.
        file_ext (str|None): File extension. Empty string for all files.
            None for directories.
        pattern (slice): Selection pattern (assuming alphabetical ordering).
        full_path (bool): Include the full path.
        verbose (int): Set level of verbosity.

    Returns:
        list[str]: List of file names/paths.
    """
    if file_ext is None:
        if verbose >= VERB_LVL['debug']:
            print('Scanning for DIRS on:\n{}'.format(path))
        filepath_list = [
            os.path.join(path, filename) if full_path else filename
            for filename in os.listdir(path)
            if os.path.isdir(os.path.join(path, filename))]
    else:
        if verbose >= VERB_LVL['debug']:
            print("Scanning for '{}' on:\n{}".format(file_ext, path))
        # extracts only those ending with specific file_ext
        filepath_list = [
            os.path.join(path, filename) if full_path else filename
            for filename in os.listdir(path)
            if filename.lower().endswith(file_ext.lower())]
    return sorted(filepath_list)[pattern]


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
def change_ext(
        filepath,
        new_ext,
        old_ext=None,
        case_sensitive=False):
    """
    Substitute the old extension with a new one in a filepath.

    Parameters
    ==========
    filepath : str
        Input filepath.
    new_ext : str
        The new extension (with or without the dot).
    old_ext : str (optional)
        The old extension (with or without the dot). If None, will be guessed.

    Returns
    =======
    filepath : str
        Output filepath.

    """
    if old_ext is None:
        filepath, old_ext = os.path.splitext(filepath)
    else:
        old_ext = add_extsep(old_ext)
        if not case_sensitive:
            true_old_ext = filepath.lower().endswith(old_ext.lower())
        else:
            true_old_ext = filepath.endswith(old_ext)
        if true_old_ext:
            filepath = filepath[:-len(old_ext)]
    filepath += add_extsep(new_ext)
    return filepath


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
    except (TypeError, ValueError):
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
        except (TypeError, ValueError):
            try:
                val = float(val_str)
            except (TypeError, ValueError):
                try:
                    val = complex(val_str)
                except (TypeError, ValueError):
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
    try:
        val_str = significant_figures(val, val_order - err_order + num)
        err_str = significant_figures(err, num)
    except ValueError:
        val_str = str(val)
        err_str = str(err)
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
        sorting=None):
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
    key_list = sorted(in_dict.keys(), key=sorting)
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
    in_filepath_list : str list
        List of filepaths used as input of computation.
    out_filepath_list : str list
        List of filepaths used as output of computation.
    force : boolean
        Force computation to be re-done.

    Returns
    =======
    force : boolean
        Computation to be re-done.

    """
    # todo: include output_dir autocreation
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
def sgnlog(x, base=np.e):
    """
    Signed logarithm of x: log(abs(x) * sign(x)

    Args:
        x (float|ndarray): The input value(s)

    Returns:
        The signed logarithm
    """
    return np.log(np.abs(x)) / np.log(base) * np.sign(x)


# ======================================================================
def sgnlogspace(
        start,
        stop,
        num=50,
        endpoint=True,
        base=10.0):
    """
    Logarithmically spaced samples between signed start and stop endpoints.

    Args:
        start (float): The starting value of the sequence.
        stop (float): The end value of the sequence.
        num (int): Number of samples to generate. Must be non-negative.
        endpoint (bool): If True, stop is the last sample.
        base (float): The base of the log space. Must be non-negative.

    Returns:
        samples (ndarray): equally spaced samples on a log scale.
    """
    if start * stop < 0.0:
        bounds = (
            (start, -(np.exp(-np.log(np.abs(start))))),
            ((np.exp(-np.log(np.abs(stop)))), stop))
        args_bounds = tuple(
            tuple(np.log(np.abs(val)) / np.log(base) for val in arg_bounds)
            for arg_bounds in bounds)
        args_num = (num // 2, num - num // 2)
        args_sign = (np.sign(start), np.sign(stop))
        args_endpoint = True, endpoint
        logspaces = tuple(
            np.logspace(*(arg_bounds + (arg_num, arg_endpoint, base))) \
            * arg_sign
            for arg_bounds, arg_sign, arg_num, arg_endpoint
            in zip(args_bounds, args_sign, args_num, args_endpoint))
        samples = np.concatenate(logspaces)
    else:
        sign = np.sign(start)
        logspace_bound = \
            tuple(np.log(np.abs(val)) / np.log(base) for val in (start, stop))
        samples = np.logspace(*(logspace_bound + (num, endpoint, base))) * sign
    return samples


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
    if len(array) > 0:
        stats_dict = {
            'avg': np.mean(array),
            'std': np.std(array),
            'min': np.min(array),
            'max': np.max(array),
            'sum': np.sum(array),
            'num': np.size(array),}
    else:
        stats_dict = {
            'avg': None, 'std': None,
            'min': None, 'max': None,
            'sum': None, 'num': None}
    if save_path or title:
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
    slab = [slice(None)] * array.ndim
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
    n_dim = arr.ndim + 1
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
    for i, src in enumerate(arr_list):
        index = [slice(None)] * n_dim
        index[axis] = i
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
    # split array apart
    arr_list = []
    for i in range(arr.shape[axis]):
        # determine index for slicing
        index = [slice(None)] * arr.ndim
        index[axis] = i
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
        #        print('EE: Fitting error. Params were: {}', param_list)  #
        # DEBUG
        err_val = 0.0
        n_fit_par = len(param_list[3])  # number of fitting parameters
        result = \
            np.tile(err_val, n_fit_par), \
            np.tile(err_val, (n_fit_par, n_fit_par))
    return result


# ======================================================================
def elapsed(
        name,
        time_point=time.time(),
        events=_EVENTS):
    """
    Append a named event point to the events list.

    Args:
        name (str): The name of the event point
        time_point (float): The time in seconds since the epoch
        events (list[(str,time)]): A list of named event time points.
            Each event is a 2-tuple: (label, time)

    Returns:
        None
    """
    events.append((name, time_point))


# ======================================================================
def print_elapsed(
        events=_EVENTS,
        label='\nElapsed Time(s): '):
    """
    Print quick-and-dirty elapsed times between named event points.

    Args:
        events (list[str,time]): A list of named event time points.
            Each event is a 2-tuple: (label, time).
        label (str): heading of the elapsed time table.

    Returns:
        None
    """
    print(label, end='\n' if len(events) > 2 else '')
    first_elapsed = events[0][1]
    for i in range(len(events) - 1):
        name = events[i + 1][0]
        curr_elapsed = events[i + 1][1]
        last_elapsed = events[i][1]
        diff_first = datetime.timedelta(0, curr_elapsed - first_elapsed)
        diff_last = datetime.timedelta(0, curr_elapsed - last_elapsed)
        if diff_first == diff_last:
            diff_first = '-'
        print('{!s:24s} {!s:>24s}, {!s:>24s}'.format(
            name, diff_last, diff_first))


# ======================================================================
if __name__ == '__main__':
    print(__doc__)

# ======================================================================
elapsed('mri_tools.base')

# shape = np.array((100000, 20, 20))
# dim = 0
# arr = np.arange(np.prod(shape)).reshape(shape)
# print(arr.shape)
# _EVENTS += [('created', time.time())]
# prod1 = mdot(*[arr[j, :, :] for j in range(shape[dim])][::-1])
# _EVENTS += [('mdot', time.time())]
# prod2 = ndot(arr, dim, -1)
# _EVENTS += [('ndot', time.time())]
# print(np.sum(np.abs(prod1 - prod2)))
# print_elapsed()
# quit()
