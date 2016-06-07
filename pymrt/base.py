#!python
# -*- coding: utf-8 -*-
"""
pymrt: generic basic utilities.
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
import sys  # System-specific parameters and functions
# import shutil  # High-level file operations
import math  # Mathematical functions
import time  # Time access and conversions
import datetime  # Basic date and time types
# import operator  # Standard operators as functions
# import collections  # High-performance container datatypes
import itertools  # Functions creating iterators for efficient looping
import functools  # Higher-order functions and operations on callable objects
# import argparse  # Parser for command-line options, arguments and sub-command
import subprocess  # Subprocess management
# import multiprocessing  # Process-based parallelism
import fractions  # Rational numbers
import csv  # CSV File Reading and Writing [CSV: Comma-Separated Values]
# import json  # JSON encoder and decoder [JSON: JavaScript Object Notation]
import inspect  # Inspect live objects
import stat  # Interpreting stat() results

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
import scipy.optimize  # SciPy: Optimization Algorithms
# import scipy.integrate  # SciPy: Integrations facilities
# import scipy.constants  # SciPy: Mathematal and Physical Constants
# import scipy.ndimage  # SciPy: ND-image Manipulation

# :: Local Imports
# from pymrt import INFO
from pymrt import VERB_LVL
from pymrt import D_VERB_LVL
from pymrt import _EVENTS

# from pymrt import get_first_line

# ======================================================================
# :: Custom defined constants


# ======================================================================
# :: Default values usable in functions.
COMMENT_TOKEN = '#'
CSV_DELIMITER = '\t'
PNG_INTERVAL = (0.0, 255.0)
EXT = {
    'plot': 'png',
    'img': 'nii.gz',
    'text': 'txt',
    'tab': 'csv',
    'data': 'json',
}
D_TAB_SIZE = 8

# :: TTY amenities
TTY_COLORS = {
    'r': 31, 'g': 32, 'b': 34, 'c': 36, 'm': 35, 'y': 33, 'w': 37, 'k': 30,
    'R': 41, 'G': 42, 'B': 44, 'C': 46, 'M': 45, 'Y': 43, 'W': 47, 'K': 40,
}


# ======================================================================
def _is_hidden(filepath):
    """
    Heuristic to determine hidden files.

    Args:
        filepath (str): the input file path.

    Returns:
        is_hidden (bool): True if is hidden, False otherwise.

    Notes:
        Only works with UNIX-like files, relying on prepended '.'.
    """
    # if sys.version_info[0] > 2:
    #     filepath = filepath.encode('utf-8')
    # filepath = filepath.decode('utf-8')
    return os.path.basename(filepath).startswith('.')


# ======================================================================
def _is_special(stats_mode):
    """
    Heuristic to determine non-standard files.

    Args:
        filepath (str): the input file path.

    Returns:
        is_special (bool): True if is hidden, False otherwise.

    Notes:
        Its working relies on Python stat module implementation.
    """
    is_special = not stat.S_ISREG(stats_mode) and \
                 not stat.S_ISDIR(stats_mode) and \
                 not stat.S_ISLNK(stats_mode)
    return is_special


# ======================================================================
def gcd(*num_list):
    """
    Find the greatest common divisor (GCD) of a list of numbers.

    Args:
        *num_list (tuple[int]): The input numbers.

    Returns:
        gcd_val (int): The value of the greatest common divisor (GCD).
    """
    gcd_val = num_list[0]
    for num in num_list[1:]:
        gcd_val = math.gcd(gcd_val, num)
    return gcd_val


# ======================================================================
def lcm(*num_list):
    """
    Find the least common multiple (LCM) of a list of numbers.

    Args:
        *num_list (tuple[int]): The input numbers.

    Returns:
        gcd_val (int): The value of the least common multiple (LCM).
    """
    lcm_val = num_list[0]
    for num in num_list[1:]:
        lcm_val = lcm_val * num // fractions.gcd(lcm_val, num)
    return lcm_val


# ======================================================================
def merge_dicts(*dicts):
    """
    Merge dictionaries into a new dict (new keys overwrite the old ones).

    Args:
        dicts (tuple[dict]): Dictionaries to be merged together.

    Returns:
        merged (dict): The merged dict (new keys overwrite the old ones).
    """
    merged = {}
    for item in dicts:
        merged.update(item)
    return merged


# ======================================================================
def accumulate(lst, func=lambda x, y: x + y):
    """
    Cumulatively apply the specified function to the elements of the list.

    Args:
        lst (list): The list to process.
        func (callable): func(x,y) -> z
            The function applied cumulatively to the first n items of the list.
            Defaults to cumulative sum.

    Returns:
        lst (list): The cumulative list.

    See Also:
        itertools.accumulate

    """
    return [functools.reduce(func, lst[:idx + 1]) for idx in range(len(lst))]


# ======================================================================
def multi_replace(text, replace_list):
    """
    Perform multiple replacements in a string.

    Args:
        text (str): The input string
        replace_list (tuple[str,str]): The listing of the replacements.
            Format: ((<old>, <new>), ...).

    Returns:
        text (str): The string after the performed replacements.
    """
    return functools.reduce(lambda s, r: s.replace(*r), replace_list, text)


# ======================================================================
def cartesian(*arrays):
    """
    Generate a cartesian product of input arrays.

    Args:
        *arrays (tuple[ndarray]): 1-D arrays to form the cartesian product of

    Returns:
        out (ndarray): 2-D array of shape (M, len(arrays)) containing
            cartesian products formed of input arrays.

    Examples:
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

    Args:
        func (callable): The function to be inspected.
        values (dict): The (key, value) pairs to set.
            If a value is None, it will be replaced by the default value.
            To use the names defined locally, use: `locals()`.

    Results:
        kw_params (dict): A dictionary of the keyword parameters to set.

    See Also:
        inspect.getargspec, locals, globals.
    """
    # todo: refactor to get rid of deprecated getargspec
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
def mdot(*arrays):
    """
    Cumulative application of `numpy.dot` operation.

    Args:
        arrays (tuple[ndarray]): List of input arrays.

    Returns:
        array (ndarray): The result of the tensor product.
    """
    array = arrays[0]
    for item in arrays[1:]:
        array = np.dot(array, item)
    return array


# ======================================================================
def ndot(array, dim=-1, step=1):
    """
    Cumulative application of `numpy.dot` operation over a given axis.

    Args:
        array (ndarray): The input array.

    Returns:
        array (ndarray): The result of the tensor product.
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

    Args:
        a (ndarray): The first operand
        b (ndarray): The second operand

    Returns:
        c (ndarray): The operation result
    """
    return a.dot(b) - b.dot(a)


def anticommutator(a, b):
    """
    Calculate the anticommutator of two arrays: [A,B] = AB + BA

    Args:
        a (ndarray): The first operand
        b (ndarray): The second operand

    Returns:
        c (ndarray): The operation result
    """
    return a.dot(b) + b.dot(a)


# ======================================================================
def walk2(
        base,
        follow_links=False,
        follow_mounts=False,
        allow_special=False,
        allow_hidden=True,
        max_depth=-1,
        on_error=None):
    """
    Recursively walk through sub paths of a base directory

    Args:
        base (str): directory where to operate
        follow_links (bool): follow links during recursion
        follow_mounts (bool): follow mount points during recursion
        allow_special (bool): include special files
        allow_hidden (bool): include hidden files
        max_depth (int): maximum depth to reach. Negative for unlimited
        on_error (callable): function to call on error

    Returns:
        path (str): path to the next object
        stats (stat_result): structure containing file stats information
    """

    # def _or_not_and(flag, check):
    #     return flag or not flag and check

    def _or_not_and_not(flag, check):
        return flag or not flag and not check

    try:
        for name in os.listdir(base):
            path = os.path.join(base, name)
            stats = os.stat(path)
            mode = stats.st_mode
            # for some reasons, stat.S_ISLINK and os.path.islink results differ
            allow = \
                _or_not_and_not(follow_links, os.path.islink(path)) and \
                _or_not_and_not(follow_mounts, os.path.ismount(path)) and \
                _or_not_and_not(allow_special, _is_special(mode)) and \
                _or_not_and_not(allow_hidden, _is_hidden(path))
            if allow:
                yield path, stats
                if os.path.isdir(path):
                    if max_depth != 0:
                        next_level = walk2(
                            path, follow_links, follow_mounts,
                            allow_special, allow_hidden, max_depth - 1,
                            on_error)
                        for next_path, next_stats in next_level:
                            yield next_path, next_stats

    except OSError as error:
        if on_error is not None:
            on_error(error)
        return


# ======================================================================
def execute(cmd, use_pipes=True, dry=False, verbose=D_VERB_LVL):
    """
    Execute command and retrieve output at the end of execution.

    Args:
        command (str): Command to execute.
        use_pipes (bool): Get stdout and stderr streams from the process.
        dry (bool): Print rather than execute the command (dry run).
        verbose (int): Set level of verbosity.

    Returns:
        p_stdout (str|None): if use_pipes the stdout of the process.
        p_stderr (str|None): if use_pipes the stderr of the process.
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
    Generate a list of lists from a source list and grouping specifications

    Args:
        lst (list): The source list.
        grouping (list[int]): number of elements that each group contains.

    Returns:
        groups (list[list]): Grouped elements from the source list.
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
        list[str]: List of file names/paths
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


# # ======================================================================
# def tty_colorify(
#         text,
#         color=None):
#     """
#     Add color TTY-compatible color code to a string, for pretty-printing.
#
#     Parameters
#     ==========
#     text: str
#         The text to be colored.
#     color : str or int or None
#         | A string or number for the color coding.
#         | Lowercase letters modify the forground color.
#         | Uppercase letters modify the background color.
#         | Available colors:
#         * r/R: red
#         * g/G: green
#         * b/B: blue
#         * c/C: cyan
#         * m/M: magenta
#         * y/Y: yellow (brown)
#         * k/K: black (gray)
#         * w/W: white (gray)
#
#     Returns
#     =======
#         The colored string.
#
#     see also: TTY_COLORS
#     """
#     if color in TTY_COLORS:
#         tty_color = TTY_COLORS[color]
#     elif color in TTY_COLORS.values():
#         tty_color = color
#     else:
#         tty_color = None
#     if tty_color and sys.stdout.isatty():
#         text = '\x1b[1;{color}m{}\x1b[1;m'.format(text, color=tty_color)
#     return text


# ======================================================================
def add_extsep(ext):
    """
    Add a extsep char to a filename extension, if it does not have one.

    Args:
        ext (str): Filename extension to which the dot has to be added.

    Returns:
        ext (str): Filename extension with a prepending dot.
    """
    if not ext:
        ext = ''
    if not ext.startswith(os.path.extsep):
        ext = os.path.extsep + ext
    return ext


# ======================================================================
def change_ext(
        filepath,
        new_ext,
        old_ext=None,
        case_sensitive=False):
    """
    Substitute the old extension with a new one in a filepath.

    Args:
        filepath (str): Input filepath.
        new_ext (str): The new extension (with or without the dot).
        old_ext (str): The old extension (with or without the dot).
            If None, it will be guessed.
        case_sensitive (str): Case-sensitive match of old extension.

    Returns:
        filepath (str): Output filepath
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
    if new_ext:
        filepath += add_extsep(new_ext)
    return filepath


# ======================================================================
def compact_num_str(
        val,
        max_lim=D_TAB_SIZE - 1):
    """
    Convert a number into the most informative string within specified limit.

    Args:
        val (int|float): The number to be converted to string.
        max_lim (int): The maximum number of characters allowed for the string.

    Returns:
        val_str (str): The string with the formatted number.
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
        limit = max_lim - extra_char_in_sign if val < 0.0 else max_lim
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

    Args:
        text (str): The text input string.
        pre_decor (str): initial string decorator.
        post_decor (str): final string decorator.

    Returns:
        has_decorator (bool): True if text is delimited by the specified chars.
    """
    return text.startswith(pre_decor) and text.endswith(post_decor)


# ======================================================================
def strip_decorator(text, pre_decor='"', post_decor='"'):
    """
    Strip initial and final character sequences (decorators) from a string.

    Args:
        text (str): The text input string.
        pre_decor (str): initial string decorator.
        post_decor (str): final string decorator.

    Returns:
        text (str): the text without the specified decorators.
    """
    return text[len(pre_decor):-len(post_decor)]


# ======================================================================
def auto_convert(text, pre_decor=None, post_decor=None):
    """
    Convert value to numeric if possible, or strip delimiters from string.

    Args:
        text (str): The text input string.
        pre_decor (str): initial string decorator.
        post_decor (str): final string decorator.

    Returns:
        val (int|float|complex): The numeric value of the string.
    """
    if pre_decor and post_decor and \
            has_decorator(text, pre_decor, post_decor):
        val = strip_decorator(text, pre_decor, post_decor)
    else:
        try:
            val = int(text)
        except (TypeError, ValueError):
            try:
                val = float(text)
            except (TypeError, ValueError):
                try:
                    val = complex(text)
                except (TypeError, ValueError):
                    val = text
    return val


# ======================================================================
def is_number(var):
    """
    Determine if a variable contains a number.

    Args:
        var (str): The variable to test.

    Returns:
        result (bool): True if the values can be converted, False otherwise.
    """
    try:
        complex(var)
    except (TypeError, ValueError):
        result = False
    else:
        result = True
    return result


# ======================================================================
def significant_figures(val, num):
    """
    Format a number with the correct number of significant figures.

    Args:
        val (str|float|int): The numeric value to be correctly formatted.
        num (str|int): The number of significant figures to be displayed.

    Returns:
        val (str): String containing the properly formatted number.

    See Also:
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

    Args:
        val (str|float|int): The numeric value to be correctly formatted.
        err (str|float|int): The numeric error to be correctly formatted.
        num (str|int): The precision to be used for the error (usually 1 or 2).

    Returns:
        val_str (str): The string with the correctly formatted numeric value.
        err_str (str): The string with the correctly formatted numeric error.
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

    Args:
        in_str (str): The input string.
        entry_sep (str): The entry separator.
        key_val_sep (str): The key-value separator.
        pre_decor (str): initial decorator (to be removed before parsing).
        post_decor (str): final decorator (to be removed before parsing).
        strip_key_str (str): Chars to be stripped from both ends of the key.
            If None, whitespaces are stripped. Empty string for no stripping.
        strip_val_str (str): Chars to be stripped from both ends of the value.
            If None, whitespaces are stripped. Empty string for no stripping.
        convert (bool): Enable automatic conversion of string to numeric.

    Returns:
        out_dict (dict): The output dictionary generated from the string.

    See Also:
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

    Args:
        in_dict (dict): The input dictionary.
        entry_sep (str): The entry separator.
        key_val_sep (str): The key-value separator.
        pre_decor (str): initial decorator (to be appended to the output).
        post_decor (str): final decorator (to be appended to the output).
        strip_key_str (str): Chars to be stripped from both ends of the key.
            If None, whitespaces are stripped. Empty string for no stripping.
        strip_val_str (str): Chars to be stripped from both ends of the value.
            If None, whitespaces are stripped. Empty string for no stripping.
        sorting (callable): Function used as 'key' argument of 'sorted'
            for sorting the dictionary keys.

    Returns:
        out_str (str): The output string generated from the dictionary.

    See Also:
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
        text,
        begin_str,
        end_str,
        incl_begin=False,
        incl_end=False,
        greedy=True):
    """
    Isolate the string contained between two tokens

    Args:
        text (str): String to parse
        begin_str (str): Token at the beginning
        end_str (str): Token at the ending
        incl_begin (bool): Include 'begin_string' in the result
        incl_end (bool): Include 'end_str' in the result.
        greedy (bool): Output the largest possible string.

    Returns:
        text (str): The string contained between the specified tokens (if any)
    """
    incl_begin = len(begin_str) if not incl_begin else 0
    incl_end = len(end_str) if incl_end else 0
    if begin_str in text and end_str in text:
        if greedy:
            text = text[
                   text.find(begin_str) + incl_begin:
                   text.rfind(end_str) + incl_end]
        else:
            text = text[
                   text.rfind(begin_str) + incl_begin:
                   text.find(end_str) + incl_end]
    else:
        text = ''
    return text


# ======================================================================
def check_redo(
        in_filepaths,
        out_filepaths,
        force=False):
    """
    Check if input files are newer than output files, to force calculation.

    Args:
        in_filepaths (list[str]): Filepaths used as input of computation.
        out_filepaths (list[str]): Filepaths used as output of computation.
        force (bool): Force computation to be re-done.

    Returns:
        force (bool): True if the computation is to be re-done.

    Raises:
        IndexError: if the input filepath list is empty
        IOError: if any of the input files do not exist
    """
    # todo: include output_dir autocreation
    if not in_filepaths:
        raise IndexError('List of input files is empty.')
    for in_filepath in in_filepaths:
        if not os.path.exists(in_filepath):
            raise IOError('Input file does not exists.')
    if not force:
        for out_filepath in out_filepaths:
            if out_filepath:
                if not os.path.exists(out_filepath):
                    force = True
                    break
    if not force:
        for in_filepath, out_filepath in \
                itertools.product(in_filepaths, out_filepaths):
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
        endpoint (bool): The value of 'stop' is the last sample.
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
def minmax(array):
    """
    Calculate the minimum and maximum of an array: (min, max).

    Args:
        array (ndarray): The input array

    Returns:
        min, max (tuple[float]): the minimum and the maximum values of the
            array
    """
    return np.min(array), np.max(array)


# ======================================================================
def scale(
        val,
        in_interval=(0.0, 1.0),
        out_interval=(0.0, 1.0)):
    """
    Linear convert the value from input interval to output interval

    Args:
        val (float): Value to convert
        in_interval (float,float): Interval of the input value
        out_interval (float,float): Interval of the output value.

    Returns:
        val (float): The converted value
    """
    in_min, in_max = in_interval
    out_min, out_max = out_interval
    return (val - in_min) / (in_max - in_min) * (out_max - out_min) + out_min


# ======================================================================
def interval_size(interval):
    """
    Calculate the (signed) size of an interval given as a 2-tuple (A,B)

    Args:
        interval (float,float): Interval for computation

    Returns:
        val (float): The converted value
    """
    return interval[1] - interval[0]


# ======================================================================
def combine_interval(
        interval1,
        interval2,
        operation):
    """
    Combine two intervals with some operation to obtain a new interval.

    Args:
        interval1 (tuple[float]): Interval of first operand
        interval2 (tuple[float]): Interval of second operand
        operation (str): String with operation to perform. Supports:

            - '+' : addition
            - '-' : subtraction

    Returns:
        new_interval (tuple[float]): Interval resulting from operation
    """
    if operation == '+':
        new_interval = (
            interval1[0] + interval2[0], interval1[1] + interval2[1])
    elif operation == '-':
        new_interval = (
            interval1[0] - interval2[1], interval1[1] - interval2[0])
    else:
        new_interval = (-np.inf, np.inf)
    return new_interval


# ======================================================================
def midval(array):
    """
    Calculate the middle value vector.

    Args:
        array (ndarray): The input N-dim array

    Returns:
        array (ndarray): The output (N-1)-dim array

    Examples:
        >>>> midval(np.array([0, 1, 2, 3, 4]))
        array([ 0.5,  1.5,  2.5,  3.5])
    """
    return (array[1:] - array[:-1]) / 2.0 + array[:-1]


# ======================================================================
def unwrap(arr, voxel_sizes=None):
    """
    Superfast multi-dimensional Laplacian-based Fourier unwrapping.

    Args:
        arr (np.ndarray): The multi-dimensional array to unwrap.

    Returns:
        arr (np.ndarray): The multi-dimensional unwrapped array.

    See Also:
        Schofield, M. A. and Y. Zhu (2003). Optics Letters 28(14): 1194-1196.
    """
    if not voxel_sizes:
        voxel_sizes = np.ones_like(arr.shape)
    # calculate the Laplacian kernel
    k_range = [slice(-k_size / 2.0, +k_size / 2.0) for k_size in arr.shape]
    kk = np.ogrid[k_range]
    kk_2 = np.zeros_like(arr)
    for i, (dim, voxel_size) in enumerate(zip(arr.shape, voxel_sizes)):
        kk_2 += np.fft.fftshift(kk[i] / dim / voxel_size) ** 2
    # perform the Laplacian-based Fourier unwrapping
    arr = np.fft.fftn(
        np.cos(arr) * np.fft.ifftn(kk_2 * np.fft.fftn(np.sin(arr))) -
        np.sin(arr) * np.fft.ifftn(kk_2 * np.fft.fftn(np.cos(arr)))) / kk_2
    # removes the singularity generated by the division by kk_2
    arr[np.isinf(arr)] = 0.0
    arr = np.real(np.fft.ifftn(arr))
    return arr


# ======================================================================
def polar2complex(modulus, argument):
    """
    Calculate complex number from the polar form:
    z = R * exp(i * phi) = R * cos(phi) + i * R * sin(phi)

    Args:
        modulus (float): The modulus R of the complex number
        argument (float): The argument phi or phase of the complex number

    Returns:
        z (complex): The complex number z = R * exp(i * phi)
    """
    return modulus * np.exp(1j * argument)


# ======================================================================
def cartesian2complex(real, imag):
    """
    Calculate the complex number from the cartesian form: z = z' + i * z"

    Args:
        real (float): The real part z' of the complex number
        imag (float): The imaginary part z" of the complex number

    Returns:
        z (complex): The complex number: z = z' + i * z"
    """
    return real + 1j * imag


# ======================================================================
def complex2cartesian(z):
    """
    Calculate the real and the imaginary part of a complex number.

    Args:
        z (complex): The complex number: z = z' + i * z"

    Returns:
        tuple[float]:
            - real (float): The real part z' of the complex number
            - imag (float): The imaginary part z" of the complex number
    """
    return np.real(z), np.imag(z)


# ======================================================================
def complex2polar(z):
    """
    Calculate the real and the imaginary part of a complex number

    Args:
        z (complex): The complex number: z = z' + i * z"

    Returns:
        tuple[float]:
            - modulus (float): The modulus R of the complex number
            - argument (float): The argument phi or phase of the complex number
    """
    return np.abs(z), np.angle(z)


# ======================================================================
def polar2cartesian(modulus, argument):
    """
    Calculate the real and the imaginary part of a complex number

    Args:
        modulus (float): The modulus R of the complex number
        argument (float): The argument phi or phase of the complex number

    Returns:
        tuple[float]:
            - real (float): The real part z' of the complex number
            - imag (float): The imaginary part z" of the complex number
    """
    return modulus * np.cos(argument), modulus * np.sin(argument)


# ======================================================================
def cartesian2polar(real, imag):
    """
    Calculate the real and the imaginary part of a complex number.

    Args:
        real (float): The real part z' of the complex number
        imag (float): The imaginary part z" of the complex number

    Returns:
        tuple[float]:
            - modulus (float): The modulus R of the complex number
            - argument (float): The argument phi or phase of the complex number
    """
    return np.sqrt(real ** 2 + imag ** 2), np.arctan2(real, imag)


# ======================================================================
def calc_stats(
        array,
        mask_nan=True,
        mask_inf=True,
        mask_vals=None,
        val_interval=None,
        save_path=None,
        title=None,
        compact=False):
    """
    Calculate array statistical information (min, max, avg, std, sum, num)

    Args:
        array (ndarray): The array to be investigated
        mask_nan (bool): Mask NaN values
        mask_inf (bool): Mask Inf values
        mask_vals (list[int|float]|None): List of values to mask
        val_interval (tuple): The (min, max) values interval
        save_path (str|None): The path to which the plot is to be saved
            If None, no output
        title (str|None): If title is not None, stats are printed to screen
        compact (bool): Use a compact format string for displaying results

    Returns:
        stats_dict (dict):
            - 'min': minimum value
            - 'max': maximum value
            - 'avg': average or mean
            - 'std': standard deviation
            - 'sum': summation
            - 'num': number of elements
    """
    if mask_nan:
        array = array[~np.isnan(array)]
    if mask_inf:
        array = array[~np.isinf(array)]
    if not mask_vals:
        mask_vals = []
    for val in mask_vals:
        array = array[array != val]
    if val_interval is None:
        val_interval = minmax(array)
    array = array[array > val_interval[0]]
    array = array[array < val_interval[1]]
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
            csv_writer = csv.writer(csv_file, delimiter=str(CSV_DELIMITER))
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

    Args:
        array (ndarray): The input N-dim array
        axis (int): The slicing axis
        index (int): The slicing index. If None, mid-value is taken

    Returns:
        sliced (ndarray): The sliced (N-1)-dim array

    Raises:
        ValueError: if index is out of bounds
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
    Calculate the element-wise relative error

    Args:
        arr1 (ndarray): The input array with the exact values
        arr2 (ndarray): The input array with the approximated values
        use_average (bool): Use the input arrays average as the exact values

    Returns:
        array (ndarray): The relative error array
    """
    if arr2.dtype != np.complex:
        array = (arr2 - arr1).astype(np.float)
    else:
        array = (arr2 - arr1)
    if use_average:
        div = (arr1 + arr2) / 2.0
    else:
        div = arr1
    mask = (div != 0.0)
    array[mask] = array[mask] / div[mask]
    array[~mask] = 0.0
    return array


# ======================================================================
def euclid_dist(
        arr1,
        arr2,
        unsigned=True):
    """
    Calculate the element-wise correlation euclidean distance D,
    i.e. the distance between the identity line and the point of coordinates
    given by intensity.
        - D = abs(A2 - A1) / sqrt(2)

    Args:
        arr1 (ndarray): The first array
        arr2 (ndarray): The second array
        signed (bool): Use signed distance

    Returns:
        array (ndarray): The resulting array
    """
    array = (arr2 - arr1) / np.sqrt(2.0)
    if unsigned:
        array = np.abs(array)
    return array


# ======================================================================
def ndstack(arrays, axis=-1):
    """
    Stack a list of arrays of the same size along a specific axis

    Args:
        arrays (list[ndarray]): A list of (N-1)-dim arrays of the same size
        axis (int): Direction for the concatenation of the arrays

    Returns:
        array (ndarray): The concatenated N-dim array
    """
    array = arrays[0]
    n_dim = array.ndim + 1
    if axis < 0:
        axis += n_dim
    if axis < 0:
        axis = 0
    if axis > n_dim:
        axis = n_dim
    # calculate new shape
    shape = array.shape[:axis] + tuple([len(arrays)]) + array.shape[axis:]
    # stack arrays together
    array = np.zeros(shape, dtype=array.dtype)
    for i, src in enumerate(arrays):
        index = [slice(None)] * n_dim
        index[axis] = i
        array[tuple(index)] = src
    return array


# ======================================================================
def ndsplit(array, axis=-1):
    """
    Split an array along a specific axis into a list of arrays

    Args:
        array (ndarray): The N-dim array to split
        axis (int): Direction for the splitting of the array

    Returns:
        arrays (list[ndarray]): A list of (N-1)-dim arrays of the same size
    """
    # split array apart
    arrays = []
    for i in range(array.shape[axis]):
        # determine index for slicing
        index = [slice(None)] * array.ndim
        index[axis] = i
        arrays.append(array[index])
    return arrays


# ======================================================================
def curve_fit(args):
    """
    Interface to use scipy.optimize.curve_fit with multiprocessing.
    If an error is encountered, optimized parameters and their covariance are
    set to 0.

    Args:
        args (list): List of parameters to pass to the function

    Returns:
        par_fit (ndarray): Optimized parameters
        par_cov (ndarray): The covariance of the optimized parameters.
            The diagonals provide the variance of the parameter estimate
    """
    try:
        result = sp.optimize.curve_fit(*args)
    except (RuntimeError, RuntimeWarning, ValueError):
        #        print('EE: Fitting error. Params were: {}', param_list)  #
        # DEBUG
        err_val = 0.0
        n_fit_par = len(args[3])  # number of fitting parameters
        result = \
            np.tile(err_val, n_fit_par), \
            np.tile(err_val, (n_fit_par, n_fit_par))
    return result


# ======================================================================
def elapsed(
        name,
        time_point=None,
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
    if not time_point:
        time_point = time.time()
    events.append((name, time_point))


# ======================================================================
def print_elapsed(
        events=_EVENTS,
        label='\nElapsed Time(s): ',
        only_last=False):
    """
    Print quick-and-dirty elapsed times between named event points.

    Args:
        events (list[str,time]): A list of named event time points.
            Each event is a 2-tuple: (label, time)
        label (str): heading of the elapsed time table
        only_last (bool): print only the last event (useful inside a loop).

    Returns:
        None
    """
    if not only_last:
        print(label, end='\n' if len(events) > 2 else '')
        first_elapsed = events[0][1]
        for i in range(len(events) - 1):
            _id = i + 1
            name = events[_id][0]
            curr_elapsed = events[_id][1]
            prev_elapsed = events[_id - 1][1]
            diff_first = datetime.timedelta(0, curr_elapsed - first_elapsed)
            diff_last = datetime.timedelta(0, curr_elapsed - prev_elapsed)
            if diff_first == diff_last:
                diff_first = '-'
            print('{!s:24s} {!s:>24s}, {!s:>24s}'.format(
                name, diff_last, diff_first))
    else:
        _id = -1
        name = events[_id][0]
        curr_elapsed = events[_id][1]
        prev_elapsed = events[_id - 1][1]
        diff_last = datetime.timedelta(0, curr_elapsed - prev_elapsed)
        print('{!s}: {!s:>24s}'.format(name, diff_last))


# ======================================================================
if __name__ == '__main__':
    print(__doc__)

# ======================================================================
elapsed('pymrt.base')
