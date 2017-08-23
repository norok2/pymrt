#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pymrt.utils: generic basic utilities.
"""

# ======================================================================
# :: Future Imports
from __future__ import (
    division, absolute_import, print_function, unicode_literals)

# ======================================================================
# :: Python Standard Library Imports
import os  # Miscellaneous operating system interfaces
import io  # Core tools for working with streams
import sys  # System-specific parameters and functions
import math  # Mathematical functions
import itertools  # Functions creating iterators for efficient looping
import functools  # Higher-order functions and operations on callable objects
import subprocess  # Subprocess management
import fractions  # Rational numbers
import inspect  # Inspect live objects
import stat  # Interpreting stat() results
import doctest  # Test interactive Python examples
import shlex  # Simple lexical analysis
import warnings  # Warning control
import importlib  # The implementation of import
import gzip  # Support for gzip files
import json  # JSON encoder and decoder [JSON: JavaScript Object Notation]
import csv  # CSV File Reading and Writing [CSV: Comma-Separated Values]
import struct  # Interpret strings as packed binary data
import re  # Regular expression operations
import fnmatch  # Unix filename pattern matching

# :: External Imports
import numpy as np  # NumPy (multidimensional numerical arrays library)
import scipy as sp  # SciPy (signal and image processing library)

# :: External Imports Submodules
import scipy.optimize  # SciPy: Optimization Algorithms
import scipy.stats  # SciPy: Statistical functions
import scipy.signal  # SciPy: Signal Processing

from numpy.fft import fftshift, ifftshift
from scipy.fftpack import fftn, ifftn

# :: Internal Imports
import pymrt as mrt

# :: Local Imports
from pymrt import VERB_LVL, D_VERB_LVL, VERB_LVL_NAMES
from pymrt import elapsed, print_elapsed
from pymrt import msg, dbg

# ======================================================================
# :: Custom defined constants


# ======================================================================
# :: Default values usable in functions.
COMMENT_TOKEN = '#'
CSV_DELIMITER = '\t'
PNG_INTERVAL = (0.0, 255.0)
EXT = {
    'plot': 'png',
    'nii': 'nii',
    'niz': 'nii.gz',
    'text': 'txt',
    'tab': 'csv',
    'data': 'json',
}
D_TAB_SIZE = 8

# ======================================================================
# :: define C types

# : short form (base types used by `struct`)
_STRUCT_TYPES = (
    'x',  # pad bytes
    'c',  # char 1B
    'b',  # signed char 1B
    'B',  # unsigned char 1B
    '?',  # bool 1B
    'h',  # short int 2B
    'H',  # unsigned short int 2B
    'i',  # int 4B
    'I',  # unsigned int 4B
    'l',  # long 4B
    'L',  # unsigned long 4B
    'q',  # long long 8B
    'Q',  # unsigned long long 8B
    'f',  # float 4B
    'd',  # double 8B
    's', 'p',  # char[]
    'P',  # void * (only support mode: '@')
)

# : data type format conversion for `struct`
DTYPE_STR = {s: s for s in _STRUCT_TYPES}
# : define how to interpreted Python types
DTYPE_STR.update({
    bool: '?',
    int: 'i',
    float: 'f',
    str: 's',
})
# : define how to interpret human-friendly types
DTYPE_STR.update({
    'bool': '?',
    'char': 'b',
    'uchar': 'B',
    'short': 'h',
    'ushort': 'H',
    'int': 'i',
    'uint': 'I',
    'long': 'l',
    'ulong': 'L',
    'llong': 'q',
    'ullong': 'Q',
    'float': 'f',
    'double': 'd',
    'str': 's',
})


# ======================================================================
def _is_hidden(filepath):
    """
    Heuristic to determine hidden files.

    Args:
        filepath (str): the input filepath.

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
        filepath (str): The input filepath.

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
def read_stream(
        file_obj,
        dtype,
        count=1,
        mode='@',
        offset=None,
        whence=io.SEEK_SET):
    """

    Args:
        file_obj:
        dtype:
        count:
        mode:
        offset:
        whence:

    Returns:

    """
    if offset is not None:
        file_obj.seek(offset, whence)
    fmt = mode + str(count) + DTYPE_STR[dtype]
    byte_count = struct.calcsize(fmt)
    return struct.unpack_from(fmt, file_obj.read(byte_count)), byte_count


# ======================================================================
def read_cstr(
        file_obj,
        offset=None,
        whence=io.SEEK_SET):
    """

    Args:
        file_obj:
        offset:
        whence:

    Returns:

    """
    if offset is not None:
        file_obj.seek(offset, whence)
    buffer = []
    while True:
        c = file_obj.read(1).decode('ascii')
        if c is None or c == '\0':
            break
        else:
            buffer.append(c)
    return ''.join(buffer)


# ======================================================================
def auto_repeat(
        obj,
        n,
        force=False,
        check=False):
    """
    Automatically repeat the specified object n times.

    If the object is not iterable, a tuple with the specified size is returned.
    If the object is iterable, the object is left untouched.

    Args:
        obj: The object to operate with.
        n (int): The length of the output object.
        force (bool): Force the repetition, even if the object is iterable.
        check (bool): Ensure that the object has length n.

    Returns:
        val (tuple): Returns obj repeated n times.

    Raises:
        AssertionError: If force is True and the object does not have length n.

    Examples:
        >>> auto_repeat(1, 3)
        (1, 1, 1)
        >>> auto_repeat([1], 3)
        [1]
        >>> auto_repeat([1, 3], 2)
        [1, 3]
        >>> auto_repeat([1, 3], 2, True)
        ([1, 3], [1, 3])
        >>> auto_repeat([1, 2, 3], 2, True, True)
        ([1, 2, 3], [1, 2, 3])
        >>> auto_repeat([1, 2, 3], 2, False, True)
        Traceback (most recent call last):
            ...
        AssertionError
    """
    try:
        iter(obj)
    except TypeError:
        force = True
    finally:
        if force:
            obj = (obj,) * n
    if check:
        assert (len(obj) == n)
    return obj


# ======================================================================
def max_iter_len(items):
    """
    Determine the maximum length of an item within a collection of items.

    Args:
        items (iterable): The collection of items to inspect.

    Returns:
        num (int): The maximum length of the collection.
    """
    num = 1
    for val in items:
        try:
            iter(val)
        except TypeError:
            pass
        else:
            num = max(len(val), num)
    return num


# ======================================================================
def is_prime(num):
    """
    Determine if num is a prime number.

    A prime number is only divisible by 1 and itself.
    0 and 1 are considered special cases; in this implementations they are
    considered primes.

    It is implemented by directly testing for possible factors.

    Args:
        num (int): The number to check for primality.
            Only works for numbers larger than 1.

    Returns:
        is_divisible (bool): The result of the primality.

    Examples:
        >>> is_prime(100)
        False
        >>> is_prime(101)
        True
        >>> is_prime(-100)
        False
        >>> is_prime(-101)
        True
        >>> is_prime(2 ** 17)
        False
        >>> is_prime(17 * 19)
        False
        >>> is_prime(2 ** 17 - 1)
        True
        >>> is_prime(0)
        True
        >>> is_prime(1)
        True
    """
    # # : alternate implementation
    # is_divisible = num == 1 or num != 2 and num % 2 == 0
    # i = 3
    # while not is_divisible and i * i < num:
    #     is_divisible = num % i == 0
    #     # only odd factors needs to be tested
    #     i += 2
    # return not is_divisible

    # : fastest implementation
    num = abs(num)
    if num % 2 == 0 and num > 2:
        return False
    for i in range(3, int(num ** 0.5) + 1, 2):
        if num % i == 0:
            return False
    return True


# ======================================================================
def primes_in_range(
        stop,
        start=2):
    """
    Calculate the prime numbers in the range.

    Args:
        stop (int): The final value of the range.
            This value is excluded.
            If stop < start the values are switched.
        start (int): The initial value of the range.
            This value is included.
            If start > stop the values are switched.

    Yields:
        num (int): The next prime number.

    Examples:
        >>> list(primes_in_range(50))
        [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
        >>> list(primes_in_range(101, 150))
        [101, 103, 107, 109, 113, 127, 131, 137, 139, 149]
        >>> list(primes_in_range(1000, 1050))
        [1009, 1013, 1019, 1021, 1031, 1033, 1039, 1049]
        >>> list(primes_in_range(1050, 1000))
        [1009, 1013, 1019, 1021, 1031, 1033, 1039, 1049]
    """
    if start > stop:
        start, stop = stop, start
    if start % 2 == 0:
        if start == 2:
            yield start
        start += 1
    for num in range(start, stop, 2):
        if is_prime(num):
            yield num


# ======================================================================
def get_primes(num=2):
    """
    Calculate prime numbers.

    Args:
        num (int): The initial value

    Yields:
        num (int): The next prime number.

    Examples:
        >>> n = 15
        >>> primes = get_primes()
        >>> [next(primes) for i in range(n)]  # first n prime numbers
        [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
        >>> n = 10
        >>> primes = get_primes(101)
        >>> [next(primes) for i in range(n)]  # first n primes larger than 1000
        [101, 103, 107, 109, 113, 127, 131, 137, 139, 149]
        >>> n = 10
        >>> primes = get_primes(1000)
        >>> [next(primes) for i in range(n)]  # first n primes larger than 1000
        [1009, 1013, 1019, 1021, 1031, 1033, 1039, 1049, 1051, 1061]
    """
    while num <= 2:
        if is_prime(num):
            yield num
        num += 1
    if num % 2 == 0:
        num += 1
    while True:
        if is_prime(num):
            yield num
        num += 2


# ======================================================================
def factorize(num):
    """
    Find all factors of a number.

    Args:
        num (int): The number to factorize.

    Returns:
        numbers (list[int]): The factors of number.

    Example:
        >>> n = 100
        >>> f = factorize(n)
        >>> print(f)
        [2, 2, 5, 5]
        >>> n == np.prod(f)
        True
        >>> n= 1234567890
        >>> f = factorize(n)
        >>> print(f)
        [2, 3, 3, 5, 3607, 3803]
    """
    factors = []
    primes = get_primes()
    prime = next(primes)
    while prime * prime <= num:
        while num % prime == 0:
            num //= prime
            factors.append(prime)
        prime = next(primes)
    if num > 1:
        factors.append(num)
    return factors


# =====================================================================
def optimal_ratio(
        num,
        condition=None):
    """
    Find the optimal ratio for arranging elements into a matrix.

    Args:
        num (int): The number of elements to arrange.
        condition (callable): The optimality condition to use.
            This is passed as the `key` argument of `sorted`.

    Returns:
        num1 (int): The first number (num1 > num2).
        num2 (int): The second number (num2 < num1).

    Examples:
        >>> n1, n2 = 40, 48
        >>> [optimal_ratio(i) for i in range(n1, n2)]
        [(8, 5), (41, 1), (7, 6), (43, 1), (11, 4), (9, 5), (23, 2), (47, 1)]
        >>> [optimal_ratio(i, max) for i in range(n1, n2)]
        [(8, 5), (41, 1), (7, 6), (43, 1), (11, 4), (9, 5), (23, 2), (47, 1)]
        >>> [optimal_ratio(i, min) for i in range(n1, n2)]
        [(20, 2), (41, 1), (21, 2), (43, 1), (22, 2), (15, 3), (23, 2),\
 (47, 1)]
    """
    ratios = []
    if is_prime(num):
        return num, 1
    else:
        for i in range(2, int(num ** 0.5) + 1):
            if num % i == 0:
                ratios.append((num // i, i))
    return sorted(ratios, key=condition)[0]


# =====================================================================
def gcd(*nums):
    """
    Find the greatest common divisor (GCD) of a list of numbers.

    Args:
        *nums (iterable[int]): The input numbers.

    Returns:
        gcd_val (int): The value of the greatest common divisor (GCD).

    Examples:
        >>> gcd(12, 24, 18)
        6
        >>> gcd(12, 24, 18, 42, 600, 66, 666, 768)
        6
        >>> gcd(12, 24, 18, 42, 600, 66, 666, 768, 101)
        1
        >>> gcd(12, 24, 18, 3)
        3
    """
    gcd_val = nums[0]
    for num in nums[1:]:
        gcd_val = math.gcd(gcd_val, num)
    return gcd_val


# ======================================================================
def lcm(*nums):
    """
    Find the least common multiple (LCM) of a list of numbers.

    Args:
        *numbers (iterable[int]): The input numbers.

    Returns:
        gcd_val (int): The value of the least common multiple (LCM).

    Examples:
        >>> lcm(2, 3, 4)
        12
        >>> lcm(9, 8)
        72
        >>> lcm(12, 23, 34, 45, 56)
        985320
    """
    lcm_val = nums[0]
    for num in nums[1:]:
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

    Examples:
        >>> d1 = {1: 2, 3: 4, 5: 6}
        >>> d2 = {2: 1, 4: 3, 6: 5}
        >>> d3 = {1: 1, 3: 3, 6: 5}
        >>> dd = merge_dicts(d1, d2)
        >>> print(tuple(sorted(dd.items())))
        ((1, 2), (2, 1), (3, 4), (4, 3), (5, 6), (6, 5))
        >>> dd = merge_dicts(d1, d3)
        >>> print(tuple(sorted(dd.items())))
        ((1, 1), (3, 3), (5, 6), (6, 5))
    """
    merged = {}
    for item in dicts:
        merged.update(item)
    return merged


# ======================================================================
def accumulate(
        items,
        func=lambda x, y: x + y):
    """
    Cumulatively apply the specified function to the elements of the list.

    Args:
        items (iterable): The items to process.
        func (callable): func(x,y) -> z
            The function applied cumulatively to the first n items of the list.
            Defaults to cumulative sum.

    Returns:
        lst (list): The cumulative list.

    See Also:
        itertools.accumulate.
    Examples:
        >>> accumulate(list(range(5)))
        [0, 1, 3, 6, 10]
        >>> accumulate(list(range(5)), lambda x, y: (x + 1) * y)
        [0, 1, 4, 15, 64]
        >>> accumulate([1, 2, 3, 4, 5, 6, 7, 8], lambda x, y: x * y)
        [1, 2, 6, 24, 120, 720, 5040, 40320]
    """
    return [
        functools.reduce(func, list(items)[:i + 1])
        for i in range(len(items))]


# =====================================================================
def pseudo_ratio(x, y):
    """
    Calculate the pseudo-ratio of x, y: 1 / ((x / y) + (y / x))

    .. math::
        \\frac{1}{\\frac{x}{y}+\\frac{y}{x}} = \\frac{xy}{x^2+y^2}

    Args:
        x (int|float|np.ndarray): First input value.
        y (int|float|np.ndarray): Second input value.

    Returns:
        result: 1 / ((x / y) + (y / x))

    Examples:
        >>> pseudo_ratio(2, 2)
        0.5
        >>> pseudo_ratio(200, 200)
        0.5
        >>> pseudo_ratio(1, 2)
        0.4
        >>> pseudo_ratio(100, 200)
        0.4
        >>> items = 100, 200
        >>> (pseudo_ratio(*items) == pseudo_ratio(*items[::-1]))
        True
    """
    return (x * y) / (x ** 2 + y ** 2)


# =====================================================================
def gen_pseudo_ratio(*items):
    """
    Calculate the generalized pseudo-ratio of x_i: 1 / sum_ij [ x_i / x_j ]

    .. math::
        \\frac{1}{\\sum_{ij} \\frac{x_i}{x_j}}


    Args:
        *items (iterable[int|float|np.ndarray]): Input values.

    Returns:
        result: 1 / sum_ij [ x_i / x_j ]

    Examples:
        >>> gen_pseudo_ratio(2, 2, 2, 2, 2)
        0.05
        >>> gen_pseudo_ratio(200, 200, 200, 200, 200)
        0.05
        >>> gen_pseudo_ratio(1, 2)
        0.4
        >>> gen_pseudo_ratio(100, 200)
        0.4
        >>> items1 = [x * 10 for x in range(2, 10)]
        >>> items2 = [x * 1000 for x in range(2, 10)]
        >>> np.isclose(gen_pseudo_ratio(*items1), gen_pseudo_ratio(*items2))
        True
        >>> items = list(range(2, 10))
        >>> np.isclose(
        ...     gen_pseudo_ratio(*items), gen_pseudo_ratio(*items[::-1]))
        True
    """
    return 1 / np.sum(x / y for x, y in itertools.permutations(items, 2))


# ======================================================================
def multi_replace(
        text,
        replaces):
    """
    Perform multiple replacements in a string.

    Args:
        text (str): The input string.
        replaces (tuple[tuple[str]]): The listing of the replacements.
            Format: ((<old>, <new>), ...).

    Returns:
        text (str): The string after the performed replacements.

    Examples:
        >>> multi_replace('python.best', (('thon', 'mrt'), ('est', 'ase')))
        'pymrt.base'
        >>> multi_replace('x-x-x-x', (('x', 'est'), ('est', 'test')))
        'test-test-test-test'
        >>> multi_replace('x-x-', (('-x-', '.test'),))
        'x.test'
    """
    return functools.reduce(lambda s, r: s.replace(*r), replaces, text)


# ======================================================================
def common_subseq_2(
        seq1,
        seq2,
        sorting=None):
    """
    Find the longest common consecutive subsequence(s).
    This version works for two iterables.

    This is known as the `longest common substring` problem, or LCS for short.

    Args:
        seq1 (iterable): The first input sequence.
            Must be of the same type as seq2.
        seq2 (iterable): The second input sequence.
            Must be of the same type as seq1.
        sorting (callable): Sorting function passed to 'sorted' via `key` arg.

    Returns:
        commons (list[iterable]): The longest common subsequence(s).

    Examples:
        >>> common_subseq_2('academy','abracadabra')
        ['acad']
        >>> common_subseq_2('los angeles','lossless')
        ['los', 'les']
        >>> common_subseq_2('los angeles','lossless',lambda x: x)
        ['les', 'los']
        >>> common_subseq_2((1, 2, 3, 4, 5),(0, 1, 2))
        [(1, 2)]
    """
    # note: [[0] * (len(seq2) + 1)] * (len(seq1) + 1) will not work!
    counter = [[0 for j in range(len(seq2) + 1)] for i in range(len(seq1) + 1)]
    longest = 0
    commons = []
    for i, item in enumerate(seq1):
        for j, jtem in enumerate(seq2):
            if item == jtem:
                tmp = counter[i][j] + 1
                counter[i + 1][j + 1] = tmp
                if tmp > longest:
                    commons = []
                    longest = tmp
                    commons.append(seq1[i - tmp + 1:i + 1])
                elif tmp == longest:
                    commons.append(seq1[i - tmp + 1:i + 1])
    if sorting is None:
        return commons
    else:
        return sorted(commons, key=sorting)


# ======================================================================
def common_subseq(
        seqs,
        sorting=None):
    """
    Find the longest common consecutive subsequence(s).
    This version works for an iterable of iterables.

    This is known as the `longest common substring` problem, or LCS for short.

    Args:
        seqs (iterable[iterable]): The input sequences.
            All the items must be of the same type.
        sorting (callable): Sorting function passed to 'sorted' via `key` arg.

    Returns:
        commons (list[iterable]): The longest common subsequence(s).

    Examples:
        >>> common_subseq(['academy', 'abracadabra', 'cadet'])
        ['cad']
        >>> common_subseq(['los angeles', 'lossless', 'les alos'])
        ['los', 'les']
        >>> common_subseq(['los angeles', 'lossless', 'les alos', 'losles'])
        ['los', 'les']
        >>> common_subseq(['los angeles', 'lossless', 'dolos'])
        ['los']
        >>> common_subseq([(1, 2, 3, 4, 5), (1, 2, 3), (0, 1, 2)])
        [(1, 2)]
    """
    commons = [seqs[0]]
    for text in seqs[1:]:
        tmps = []
        for common in commons:
            tmp = common_subseq_2(common, text, sorting)
            if len(tmps) == 0 or len(tmp[0]) == len(tmps[0]):
                tmps.extend(common_subseq_2(common, text, sorting))
        commons = tmps
    return commons


# ======================================================================
def safe_filename(
        text,
        allowed='a-zA-Z0-9._-',
        replacing='_',
        group_consecutive=True):
    """
    Return a string containing a safe filename.

    Args:
        text (str): The input string.
        allowed (str):  The valid characters.
            Must comply to Python's regular expression syntax.
        replacing (str): The replacing text.
        group_consecutive (str): Group consecutive non-allowed.
            If True, consecutive non-allowed characters are replaced by a
            single instance of `replacing`.
            Otherwise, each character is replaced individually.

    Returns:
        text (str): The filtered text.

    Examples:
        >>> safe_filename('pymrt.txt')
        'pymrt.txt'
        >>> safe_filename('pymrt+12.txt')
        'pymrt_12.txt'
        >>> safe_filename('pymrt+12.txt')
        'pymrt_12.txt'
        >>> safe_filename('pymrt+++12.txt')
        'pymrt_12.txt'
        >>> safe_filename('pymrt+++12.txt', group_consecutive=False)
        'pymrt___12.txt'
        >>> safe_filename('pymrt+12.txt', allowed='a-zA-Z0-9._+-')
        'pymrt+12.txt'
        >>> safe_filename('pymrt+12.txt', replacing='-')
        'pymrt-12.txt'
    """
    return re.sub(
        r'[^{allowed}]{greedy}'.format(
            allowed=allowed, greedy='+' if group_consecutive else ''),
        replacing, text)


# ======================================================================
def auto_open(filepath, *args, **kwargs):
    """
    Auto-magically open a compressed file.

    Supports `gzip` and `bzip2`.

    Note: all compressed files should be opened as binary.
    Opening in text mode is not supported.

    Args:
        filepath (str): The file path.
        *args (iterable): Positional arguments passed to `open()`.
        **kwargs (dict): Keyword arguments passed to `open()`.

    Returns:
        file_obj: A file object.

    Raises:
        IOError: on failure.

    See Also:
        open(), gzip.open(), bz2.open()

    Examples:
        >>> file_obj = auto_open(__file__, 'rb')
    """
    zip_module_names = 'gzip', 'bz2'
    file_obj = None
    for zip_module_name in zip_module_names:
        try:
            zip_module = importlib.import_module(zip_module_name)
            file_obj = zip_module.open(filepath, *args, **kwargs)
            file_obj.read(1)
        except (OSError, IOError, AttributeError, ImportError):
            file_obj = None
        else:
            file_obj.seek(0)
            break
    if not file_obj:
        file_obj = open(filepath, *args, **kwargs)
    return file_obj


# ======================================================================
def zopen(filepath, *args, **kwargs):
    """
    Auto-magically open a gzip-compressed file.

    Note: all compressed files should be opened as binary.
    Opening in text mode is not supported.

    Args:
        filepath (str): The file path.
        *args (iterable): Positional arguments passed to `open()`.
        **kwargs (dict): Keyword arguments passed to `open()`.

    Returns:
        file_obj: A file object.

    Raises:
        IOError: on failure.

    See Also:
        open(), gzip.open()

    Examples:
        >>> file_obj = zopen(__file__, 'rb')
    """
    file_obj = open(filepath, *args, **kwargs)

    # test if file is gzip using magic type (first 2 bytes)
    magic = file_obj.read(2)
    file_obj.seek(0)
    if magic == b'\x1f\x8b':
        file_obj = gzip.GzipFile(fileobj=file_obj)

    return file_obj


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
def mdot(*arrs):
    """
    Cumulative application of multiple `numpy.dot` operation.

    Args:
        *arrs (tuple[ndarray]): Tuple of input arrays.

    Returns:
        arr (np.ndarray): The result of the tensor product.

    Examples:
        >>>
    """
    arr = arrs[0]
    for item in arrs[1:]:
        arr = np.dot(arr, item)
    return arr


# ======================================================================
def ndot(
        arr,
        dim=-1,
        step=1):
    """
    Cumulative application of `numpy.dot` operation over a given axis.

    Args:
        arr (np.ndarray): The input array.

    Returns:
        prod (np.ndarray): The result of the tensor product.

    Examples:
        >>>
    """
    if dim < 0:
        dim += arr.ndim
    start = 0 if step > 0 else arr.shape[dim] - 1
    stop = arr.shape[dim] if step > 0 else -1
    prod = arr[
        [slice(None) if j != dim else start for j in range(arr.ndim)]]
    for i in range(start, stop, step)[1:]:
        indexes = [slice(None) if j != dim else i for j in range(arr.ndim)]
        prod = np.dot(prod, arr[indexes])
    return prod


# ======================================================================
def commutator(a, b):
    """
    Calculate the commutator of two arrays: [A,B] = AB - BA

    Args:
        a (np.ndarray): The first operand
        b (np.ndarray): The second operand

    Returns:
        c (np.ndarray): The operation result
    """
    return a.dot(b) - b.dot(a)


# ======================================================================
def anticommutator(a, b):
    """
    Calculate the anticommutator of two arrays: [A,B] = AB + BA

    Args:
        a (np.ndarray): The first operand
        b (np.ndarray): The second operand

    Returns:
        c (np.ndarray): The operation result
    """
    return a.dot(b) + b.dot(a)


# ======================================================================
def iwalk2(
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

    Yields:
        result (tuple): The tuple
            contains:
             - path (str): Path to the next object.
             - stats (stat_result): File stats information.

    Returns:
        None.
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
                        next_level = iwalk2(
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
        items (list[tuple]): The list of items.
            Each item contains the tuple with:
             - path (str): Path to the next object.
             - stats (stat_result): File stats information.
    """
    return [item for item in iwalk2(
        base,
        follow_links=follow_links, follow_mounts=follow_mounts,
        allow_special=allow_special, allow_hidden=allow_hidden,
        max_depth=max_depth, on_error=on_error)]


# ======================================================================
def which(args):
    """
    Determine the full path of an executable, if possible.

    It mimics the behavior of the POSIX command `which`.

    Args:
        args (str|list[str]): Command to execute as a list of tokens.
            Optionally can accept a string which will be tokenized.

    Returns:
        args (list[str]): Command to execute as a list of tokens.
            The first item of the list is the full path of the executable.
            If the executable is not found in path, returns the first token of
            the input.
            Other items are identical to input, if the input was a str list.
            Otherwise it will be the tokenized version of the passed string,
            except for the first token.
        is_valid (bool): True if path of executable is found, False otherwise.
    """

    def is_executable(file_path):
        return os.path.isfile(file_path) and os.access(file_path, os.X_OK)

    # ensure args in the correct format
    try:
        args = shlex.split(args)
    except AttributeError:
        pass

    cmd = os.path.expanduser(args[0])
    dirpath, filename = os.path.split(cmd)
    if dirpath:
        is_valid = is_executable(cmd)
    else:
        is_valid = False
        for dirpath in os.environ['PATH'].split(os.pathsep):
            dirpath = dirpath.strip('"')
            tmp = os.path.join(dirpath, cmd)
            is_valid = is_executable(tmp)
            if is_valid:
                cmd = tmp
                break
    return [cmd] + args[1:], is_valid


# ======================================================================
def execute(
        args,
        in_pipe=None,
        mode='call',
        timeout=None,
        encoding='utf-8',
        log=None,
        dry=False,
        verbose=D_VERB_LVL):
    """
    Execute command and retrieve/print output at the end of execution.

    Args:
        args (str|list[str]): Command to execute as a list of tokens.
            Optionally can accept a string.
        in_pipe (str|None): Input data to be used as stdin of the process.
        mode (str): Set the execution mode (affects the return values).
            Allowed modes:
             - 'spawn': Spawn a new process. stdout and stderr will be lost.
             - 'call': Call new process and wait for execution.
                Once completed, obtain the return code, stdout, and stderr.
             - 'flush': Call new process and get stdout+stderr immediately.
                Once completed, obtain the return code.
                Unfortunately, there is no easy
        timeout (float): Timeout of the process in seconds.
        encoding (str): The encoding to use.
        log (str): The template filename to be used for logs.
            If None, no logs are produced.
        dry (bool): Print rather than execute the command (dry run).
        verbose (int): Set level of verbosity.

    Returns:
        ret_code (int|None): if mode not `spawn`, return code of the process.
        p_stdout (str|None): if mode not `spawn`, the stdout of the process.
        p_stderr (str|None): if mode is `call`, the stderr of the process.
    """
    ret_code, p_stdout, p_stderr = None, None, None

    args, is_valid = which(args)
    if is_valid:
        msg('{} {}'.format('$$' if dry else '>>', ' '.join(args)),
            verbose, D_VERB_LVL if dry else VERB_LVL['medium'])
    else:
        msg('W: `{}` is not in available in $PATH.'.format(args[0]))

    if not dry and is_valid:
        if in_pipe is not None:
            msg('< {}'.format(in_pipe),
                verbose, VERB_LVL['highest'])

        proc = subprocess.Popen(
            args,
            stdin=subprocess.PIPE if in_pipe and not mode == 'flush' else None,
            stdout=subprocess.PIPE if mode != 'spawn' else None,
            stderr=subprocess.PIPE if mode == 'call' else subprocess.STDOUT,
            shell=False)

        # handle stdout nd stderr
        if mode == 'flush' and not in_pipe:
            p_stdout = ''
            while proc.poll() is None:
                out_buff = proc.stdout.readline().decode(encoding)
                p_stdout += out_buff
                msg(out_buff, fmt='', end='')
                sys.stdout.flush()
            ret_code = proc.wait()
        elif mode == 'call':
            # try:
            p_stdout, p_stderr = proc.communicate(
                in_pipe.encode(encoding) if in_pipe else None)
            # except subprocess.TimeoutExpired:
            #     proc.kill()
            #     p_stdout, p_stderr = proc.communicate()
            p_stdout = p_stdout.decode(encoding)
            p_stderr = p_stderr.decode(encoding)
            if p_stdout:
                msg(p_stdout, verbose, VERB_LVL['high'], fmt='')
            if p_stderr:
                msg(p_stderr, verbose, VERB_LVL['high'], fmt='')
            ret_code = proc.wait()
        else:
            proc.kill()
            msg('E: mode `{}` and `in_pipe` not supported.'.format(mode))

        if log:
            name = os.path.basename(args[0])
            pid = proc.pid
            for stream, source in ((p_stdout, 'out'), (p_stderr, 'err')):
                if stream:
                    log_filepath = log.format_map(locals())
                    with open(log_filepath, 'wb') as fileobj:
                        fileobj.write(stream.encode(encoding))
    return ret_code, p_stdout, p_stderr


# ======================================================================
def grouping(
        items,
        num_elems):
    """
    Generate a list of lists from a source list and grouping specifications

    Args:
        items (iterable): The source list.
        num_elems (iterable[int]): number of elements that each group contains.

    Returns:
        groups (list[list]): Grouped elements from the source list.

    Examples:
        >>> l = list(range(10))
        >>> grouping(l, 4)
        [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9]]
        >>> grouping(l, (2, 3))
        [[0, 1], [2, 3, 4], [5, 6, 7, 8, 9]]
        >>> grouping(l, (2, 4, 1))
        [[0, 1], [2, 3, 4, 5], [6], [7, 8, 9]]
        >>> grouping(l, (2, 4, 1, 20))
        [[0, 1], [2, 3, 4, 5], [6], [7, 8, 9]]
    """
    if isinstance(num_elems, int):
        num_elems = auto_repeat(num_elems, len(items) // num_elems)
    group, groups = [], []
    j = 0
    count = num_elems[j] if j < len(num_elems) else len(items) + 1
    for i, item in enumerate(items):
        if i >= count:
            loop = True
            while loop:
                groups.append(group)
                group = []
                j += 1
                add = num_elems[j] if j < len(num_elems) else len(items) + 1
                if add < 0:
                    add = len(items) + 1
                count += add
                if add == 0:
                    loop = True
                else:
                    loop = False
        group.append(item)
    groups.append(group)
    return groups


# ======================================================================
def realpath(path):
    """
    Get the expanded absolute path from its short or relative counterpart.

    Args:
        path (str): The path to expand.

    Returns:
        new_path (str): the expanded path.

    Raises:
        OSError: if the expanded path does not exists.
    """
    new_path = os.path.abspath(os.path.realpath(os.path.expanduser(path)))
    if not os.path.exists(new_path):
        raise OSError
    return new_path


# ======================================================================
def listdir(
        path,
        file_ext='',
        full_path=True,
        is_sorted=True,
        verbose=D_VERB_LVL):
    """
    Retrieve a sorted list of files matching specified extension and pattern.

    Args:
        path (str): Path to search.
        file_ext (str|None): File extension. Empty string for all files.
            None for directories.
        full_path (bool): Include the full path.
        is_sorted (bool): Sort results alphabetically.
        verbose (int): Set level of verbosity.

    Returns:
        list[str]: List of file names/paths
    """
    if file_ext is None:
        msg('Scanning for dirs on:\n{}'.format(path),
            verbose, VERB_LVL['debug'])
        filepaths = [
            os.path.join(path, filename) if full_path else filename
            for filename in os.listdir(path)
            if os.path.isdir(os.path.join(path, filename))]
    else:
        msg('Scanning for {} on:\n{}'.format(
            ('`' + file_ext + '`') if file_ext else 'files', path),
            verbose, VERB_LVL['debug'])
        # extracts only those ending with specific file_ext
        filepaths = [
            os.path.join(path, filename) if full_path else filename
            for filename in os.listdir(path)
            if filename.lower().endswith(file_ext.lower())]
    if is_sorted:
        filepaths = sorted(filepaths)
    return filepaths


# ======================================================================
def iflistdir(
        dirpath,
        patterns='*',
        unix_style=True,
        re_kws=None,
        walk_kws=None):
    """
    Recursively list the content of a directory matching the pattern(s).

    Args:
        dirpath (str): The base directory.
        patterns (str|iterable[str]): The pattern(s) to match.
            These must be either a Unix-style pattern or a regular expression,
            depending on the value of `unix_style`.
        unix_style (bool): Interpret the patterns as Unix-style.
            This is achieved by using `fnmatch`.
        re_kws (dict|None): Keyword arguments passed to `re.compile()`.
        walk_kws (dict|None): Keyword arguments passed to `os.walk()`.

    Yields:
        filepath (str): The next matched filepath.
    """
    if isinstance(patterns, str):
        patterns = (patterns,)
    if re_kws is None:
        re_kws = dict()
    if walk_kws is None:
        walk_kws = dict()
    for pattern in patterns:
        if unix_style:
            pattern = fnmatch.translate(pattern)
        re_obj = re.compile(pattern, **re_kws)
        for root, dirs, files in os.walk(dirpath, **walk_kws):
            for base in (dirs + files):
                filepath = os.path.join(root, base)
                if re_obj.match(filepath):
                    yield filepath


# ======================================================================
def flistdir(
        dirpath,
        patterns='*',
        unix_style=True,
        re_kws=None,
        walk_kws=None):
    """
    Recursively list the content of a directory matching the pattern(s).

    Args:
        dirpath (str): The base directory.
        patterns (str|iterable[str]): The pattern(s) to match.
            These must be either a Unix-style pattern or a regular expression,
            depending on the value of `unix_style`.
        unix_style (bool): Interpret the patterns as Unix-style.
            This is achieved by using `fnmatch`.
        re_kws (dict|None): Keyword arguments passed to `re.compile()`.
        walk_kws (dict|None): Keyword arguments passed to `os.walk()`.

    Returns:
        filepaths (list[str]): The matched filepaths.
    """
    return [item for item in iflistdir(
        dirpath,
        patterns=patterns, unix_style=unix_style, re_kws=re_kws,
        walk_kws=walk_kws)]


# ======================================================================
def add_extsep(ext):
    """
    Add a extsep char to a filename extension, if it does not have one.

    Args:
        ext (str): Filename extension to which the dot has to be added.

    Returns:
        ext (str): Filename extension with a prepending dot.

    Examples:
        >>> add_extsep('txt')
        '.txt'
        >>> add_extsep('.txt')
        '.txt'
        >>> add_extsep('')
        '.'
    """
    if not ext:
        ext = ''
    ext = ('' if ext.startswith(os.path.extsep) else os.path.extsep) + ext
    return ext


# ======================================================================
def split_ext(
        filepath,
        ext=None,
        case_sensitive=False,
        auto_multi_ext=True):
    """
    Split the filepath into a pair (root, ext), so that: root + ext == path.
    root is everything that preceeds the first extension separator.
    ext is the extension (including the separator).

    It can automatically detect multiple extensions.
    Since `os.path.extsep` is often '.', a `os.path.extsep` between digits is
    not considered to be generating and extension.

    Args:
        filepath (str): The input filepath.
        ext (str|None): The expected extension (with or without the dot).
            If None, it will be obtained automatically.
            If empty, no split is performed.
        case_sensitive (bool): Case-sensitive match of old extension.
            If `ext` is None or empty, it has no effect.
        auto_multi_ext (bool): Automatically detect multiple extensions.
            If True, include multiple extensions.
            If False, only the last extension is detected.
            If `ext` is not None or empty, it has no effect.

    Returns:
        result (tuple): The tuple
            contains:
             - root (str): The filepath without the extension.
             - ext (str): The extension including the separator.

    Examples:
        >>> split_ext('test.txt', '.txt')
        ('test', '.txt')
        >>> split_ext('test.txt')
        ('test', '.txt')
        >>> split_ext('test.txt.gz')
        ('test', '.txt.gz')
        >>> split_ext('test_1.0.txt')
        ('test_1.0', '.txt')
        >>> split_ext('test.0.txt')
        ('test', '.0.txt')
        >>> split_ext('test.txt', '')
        ('test.txt', '')
    """
    root = filepath
    if ext is not None:
        ext = add_extsep(ext)
        has_ext = filepath.lower().endswith(ext.lower()) \
            if not case_sensitive else filepath.endswith(ext)
        if has_ext:
            root = filepath[:-len(ext)]
        else:
            ext = ''
    else:
        if auto_multi_ext:
            ext = ''
            is_valid = True
            while is_valid:
                tmp_filepath_noext, tmp_ext = os.path.splitext(root)
                if tmp_filepath_noext and tmp_ext:
                    is_valid = not (tmp_ext[1].isdigit() and
                                    tmp_filepath_noext[-1].isdigit())
                    if is_valid:
                        root = tmp_filepath_noext
                        ext = tmp_ext + ext
                else:
                    is_valid = False
        else:
            root, ext = os.path.splitext(filepath)
    return root, ext


# ======================================================================
def split_path(
        filepath,
        auto_multi_ext=True):
    """
    Split the filepath into (root, base, ext).

    Note that: root + os.path.sep + base + ext == path.
    (and therfore: root + base + ext != path).

    root is everything that preceeds the last path separator.
    base is everything between the last path separator and the first
    extension separator.
    ext is the extension (including the separator).

    Note that this separation is performed only on the string and it is not
    aware of the filepath actually existing, being a file, a directory,
    or similar aspects.

    Args:
        filepath (str): The input filepath.
        auto_multi_ext (bool): Automatically detect multiple extensions.
            Refer to `split_ext()` for more details.

    Returns:
        result (tuple): The tuple
            contains:
             - root (str): The filepath without the last item.
             - base (str): The file name without the extension.
             - ext (str): The extension including the extension separator.

    Examples:
        >>> split_path('/path/to/file.txt')
        ('/path/to', 'file', '.txt')
        >>> split_path('/path/to/file.tar.gz')
        ('/path/to', 'file', '.tar.gz')
        >>> split_path('file.tar.gz')
        ('', 'file', '.tar.gz')
        >>> split_path('/path/to/file')
        ('/path/to', 'file', '')

    See Also:
        utils.join_path(), utils.multi_split_path()
    """
    root, base_ext = os.path.split(filepath)
    base, ext = split_ext(base_ext, auto_multi_ext=auto_multi_ext)
    return root, base, ext


# ======================================================================
def multi_split_path(
        filepath,
        auto_multi_ext=True):
    """
    Split the filepath into (root, base, ext).

    Note that: os.path.sep.join(*dirs, base) + ext == path.
    (and therfore: ''.join(dirs) + base + ext != path).

    root is everything that preceeds the last path separator.
    base is everything between the last path separator and the first
    extension separator.
    ext is the extension (including the separator).

    Note that this separation is performed only on the string and it is not
    aware of the filepath actually existing, being a file, a directory,
    or similar aspects.

    Args:
        filepath (str): The input filepath.
        auto_multi_ext (bool): Automatically detect multiple extensions.
            Refer to `split_ext()` for more details.

    Returns:
        result (tuple[str]): The parts of the file.
            If the first char of the filepath is `os.path.sep`, then
            the first item is set to `os.path.sep`.
            The first (n - 2) items are subdirectories, the penultimate item
            is the file name without the extension, the last item is the
            extension including the extension separator.

    Examples:
        >>> multi_split_path('/path/to/file.txt')
        ('/', 'path', 'to', 'file', '.txt')
        >>> multi_split_path('/path/to/file.tar.gz')
        ('/', 'path', 'to', 'file', '.tar.gz')
        >>> multi_split_path('file.tar.gz')
        ('file', '.tar.gz')
        >>> multi_split_path('/path/to/file')
        ('/', 'path', 'to', 'file', '')

    See Also:
        utils.join_path(), utils.split_path()
    """
    root, base_ext = os.path.split(filepath)
    base, ext = split_ext(base_ext, auto_multi_ext=auto_multi_ext)
    if root:
        dirs = root.split(os.path.sep)
        if dirs[0] == '':
            dirs[0] = os.path.sep
    else:
        dirs = ()
    return tuple(dirs) + (base, ext)


# ======================================================================
def join_path(*args):
    """
    Join a list of items into a filepath.

    The last item is treated as the file extension.
    Path and extension separators do not need to be manually included.

    Note that this is the inverse of `split_path()`.

    Args:
        *args (iterable[str]): The path elements to be concatenated.
            The last item is treated as the file extension.

    Returns:
        filepath (str): The output filepath.

    Examples:
        >>> join_path('/path/to', 'file', '.txt')
        '/path/to/file.txt'
        >>> join_path('/path/to', 'file', '.tar.gz')
        '/path/to/file.tar.gz'
        >>> join_path('', 'file', '.tar.gz')
        'file.tar.gz'
        >>> join_path('path/to', 'file', '')
        'path/to/file'
        >>> paths = [
        ...     '/path/to/file.txt', '/path/to/file.tar.gz', 'file.tar.gz']
        >>> all([path == join_path(*split_path(path)) for path in paths])
        True
        >>> paths = [
        ...     '/path/to/file.txt', '/path/to/file.tar.gz', 'file.tar.gz']
        >>> all([path == join_path(*multi_split_path(path)) for path in paths])
        True

    See Also:
        utils.split_path(), utils.multi_split_path()
    """
    return ((os.path.join(*args[:-1]) if args[:-1] else '') +
            (add_extsep(args[-1]) if args[-1] else ''))


# ======================================================================
def basename(
        filepath,
        ext=None,
        case_sensitive=False,
        auto_multi_ext=True):
    """
    Remove path AND the extension from a filepath.

    Args:
        filepath (str): The input filepath.
        ext (str|None): The expected extension (with or without the dot).
            Refer to `split_ext()` for more details.
        case_sensitive (bool): Case-sensitive match of expected extension.
            Refer to `split_ext()` for more details.
        auto_multi_ext (bool): Automatically detect multiple extensions.
            Refer to `split_ext()` for more details.

    Returns:
         root (str): The file name without path and extension.

    Examples:
        >>> basename('/path/to/file/test.txt', '.txt')
        'test'
        >>> basename('/path/to/file/test.txt.gz')
        'test'
    """
    filepath = os.path.basename(filepath)
    root, ext = split_ext(
        filepath, ext, case_sensitive, auto_multi_ext)
    return root


# ======================================================================
def change_ext(
        root,
        new_ext,
        ext=None,
        case_sensitive=False,
        auto_multi_ext=True):
    """
    Substitute the old extension with a new one in a filepath.

    Args:
        filepath (str): The input filepath.
        new_ext (str): The new extension (with or without the dot).
        ext (str|None): The expected extension (with or without the dot).
            Refer to `split_ext()` for more details.
        case_sensitive (bool): Case-sensitive match of expected extension.
            Refer to `split_ext()` for more details.
        auto_multi_ext (bool): Automatically detect multiple extensions.
            Refer to `split_ext()` for more details.

    Returns:
        filepath (str): Output filepath

    Examples:
        >>> change_ext('test.txt', 'dat', 'txt')
        'test.dat'
        >>> change_ext('test.txt', '.dat', 'txt')
        'test.dat'
        >>> change_ext('test.txt', '.dat', '.txt')
        'test.dat'
        >>> change_ext('test.txt', 'dat', '.txt')
        'test.dat'
        >>> change_ext('test.txt', 'dat', 'TXT', False)
        'test.dat'
        >>> change_ext('test.txt', 'dat', 'TXT', True)
        'test.txt.dat'
        >>> change_ext('test.tar.gz', 'tgz')
        'test.tgz'
        >>> change_ext('test.tar.gz', 'tgz', 'tar.gz')
        'test.tgz'
        >>> change_ext('test.tar.gz', 'tgz', auto_multi_ext=False)
        'test.tar.tgz'
        >>> change_ext('test.tar', 'gz', '')
        'test.tar.gz'
        >>> change_ext('test.tar', 'gz', None)
        'test.gz'
        >>> change_ext('test.tar', '')
        'test'
    """
    root, ext = split_ext(
        root, ext, case_sensitive, auto_multi_ext)
    filepath = root + (add_extsep(new_ext) if new_ext else '')
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

    Examples:
        >>> compact_num_str(100.0, 3)
        '100'
        >>> compact_num_str(100.042, 6)
        '100.04'
        >>> compact_num_str(100.042, 9)
        '100.04200'
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
        warnings.warn('Could not convert value `{}` to float'.format(val))
        val_str = 'NaN'
    return val_str


# ======================================================================
def has_decorator(
        text,
        pre_decor='"',
        post_decor='"'):
    """
    Determine if a string is delimited by some characters (decorators).

    Args:
        text (str): The text input string.
        pre_decor (str): initial string decorator.
        post_decor (str): final string decorator.

    Returns:
        has_decorator (bool): True if text is delimited by the specified chars.

    Examples:
        >>> has_decorator('"test"')
        True
        >>> has_decorator('"test')
        False
        >>> has_decorator('<test>', '<', '>')
        True
    """
    return text.startswith(pre_decor) and text.endswith(post_decor)


# ======================================================================
def strip_decorator(
        text,
        pre_decor='"',
        post_decor='"'):
    """
    Strip initial and final character sequences (decorators) from a string.

    Args:
        text (str): The text input string.
        pre_decor (str): initial string decorator.
        post_decor (str): final string decorator.

    Returns:
        text (str): the text without the specified decorators.

    Examples:
        >>> strip_decorator('"test"')
        'test'
        >>> strip_decorator('"test')
        'test'
        >>> strip_decorator('<test>', '<', '>')
        'test'
    """
    begin = len(pre_decor) if text.startswith(pre_decor) else None
    end = -len(post_decor) if text.endswith(post_decor) else None
    return text[begin:end]


# ======================================================================
def auto_convert(
        text,
        pre_decor=None,
        post_decor=None):
    """
    Convert value to numeric if possible, or strip delimiters from string.

    Args:
        text (str|int|float|complex): The text input string.
        pre_decor (str): initial string decorator.
        post_decor (str): final string decorator.

    Returns:
        val (int|float|complex): The numeric value of the string.

    Examples:
        >>> auto_convert('<100>', '<', '>')
        100
        >>> auto_convert('<100.0>', '<', '>')
        100.0
        >>> auto_convert('100.0+50j')
        (100+50j)
        >>> auto_convert('1e3')
        1000.0
        >>> auto_convert(1000)
        1000
        >>> auto_convert(1000.0)
        1000.0
    """
    if isinstance(text, str):
        if pre_decor and post_decor and \
                has_decorator(text, pre_decor, post_decor):
            text = strip_decorator(text, pre_decor, post_decor)
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
    else:
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

    Examples:
        >>> is_number('<100.0>')
        False
        >>> is_number('100.0+50j')
        True
        >>> is_number('1e3')
        True
    """
    try:
        complex(var)
    except (TypeError, ValueError):
        result = False
    else:
        result = True
    return result


# ======================================================================
def guess_decimals(
        val,
        n_max=16,
        base=10,
        fp=16):
    """
    Guess the number of decimals in a given float number.

    Args:
        val ():
        n_max (int): Maximum number of guessed decimals.
        base (int): The base used for the number representation.
        fp (int): The floating point maximum precision.
            A number with precision is approximated by the underlying platform.
            The default value corresponds to the limit of the IEEE-754 floating
            point arithmetic, i.e. 53 bits of precision: log10(2 ** 53) = 16
            approximately. This value should not be changed unless the
            underlying platform follows a different floating point arithmetic.

    Returns:
        prec (int): the guessed number of decimals.

    Examples:
        >>> guess_decimals(10)
        0
        >>> guess_decimals(1)
        0
        >>> guess_decimals(0.1)
        1
        >>> guess_decimals(0.01)
        2
        >>> guess_decimals(0.000001)
        6
        >>> guess_decimals(-0.72)
        2
        >>> guess_decimals(0.9567)
        4
        >>> guess_decimals(0.12345678)
        8
        >>> guess_decimals(0.9999999999999)
        13
        >>> guess_decimals(0.1234567890123456)
        16
        >>> guess_decimals(0.9999999999999999)
        16
        >>> guess_decimals(0.1234567890123456, 6)
        6
        >>> guess_decimals(0.54235, 10)
        5
        >>> guess_decimals(0x654321 / 0x10000, 16, 16)
        4
    """
    offset = 2
    prec = 0
    tol = 10 ** -fp
    x = (val - int(val)) * base
    while base - abs(x) > tol and abs(x % tol) < tol < abs(x) and prec < n_max:
        x = (x - int(x)) * base
        tol = 10 ** -(fp - prec - offset)
        prec += 1
    return prec


# ======================================================================
def significant_figures(
        val,
        num):
    """
    Format a number with the correct number of significant figures.

    Args:
        val (str|float|int): The numeric value to be correctly formatted.
        num (str|int): The number of significant figures to be displayed.

    Returns:
        val (str): String containing the properly formatted number.

    Examples:
        >>> significant_figures(1.2345, 1)
        '1'
        >>> significant_figures(1.2345, 4)
        '1.234'
        >>> significant_figures(1.234e3, 2)
        '1.2e+03'
        >>> significant_figures(-1.234e3, 3)
        '-1.23e+03'
        >>> significant_figures(12345678, 4)
        '1.235e+07'

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

    Examples:
        >>> format_value_error(1234.5, 6.7)
        ('1234.5', '6.7')
        >>> format_value_error(123.45, 6.7, 1)
        ('123', '7')
        >>> format_value_error(12345.6, 7.89, 2)
        ('12345.6', '7.9')
        >>> format_value_error(12345.6, 78.9, 2)
        ('12346', '79')
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

    Escaping and quotes are not supported.
    Dictionary name is always a string.

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

    Examples:
        >>> d = str2dict('{a=10,b=20,c=test}')
        >>> for k in sorted(d.keys()): print(k, ':', d[k])  # display dict
        a : 10
        b : 20
        c : test

    See Also:
        dict2str
    """
    if has_decorator(in_str, pre_decor, post_decor):
        in_str = strip_decorator(in_str, pre_decor, post_decor)
    entries = in_str.split(entry_sep)
    out_dict = {}
    for entry in entries:
        # fetch entry
        key_val = entry.split(key_val_sep)
        # parse entry
        if len(key_val) == 1:
            key, val = key_val[0], None
        elif len(key_val) == 2:
            key, val = key_val
            val = val.strip(strip_val_str)
        elif len(key_val) > 2:
            key, val = key_val[0], key_val[1:]
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
        sorting (callable): Sorting function passed to 'sorted' via `key` arg.
            Used for sorting the dictionary keys.

    Returns:
        out_str (str): The output string generated from the dictionary.

    Examples:
        >>> dict2str({'a': 10, 'b': 20, 'c': 'test'})
        '{a=10,b=20,c=test}'

    See Also:
        str2dict
    """
    keys = sorted(in_dict.keys(), key=sorting)
    out_list = []
    for key in keys:
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

    Examples:
        >>> string_between('roses are red violets are blue', 'ses', 'lets')
        ' are red vio'
        >>> string_between('roses are red, or not?', 'a', 'd')
        're re'
        >>> string_between('roses are red, or not?', ' ', ' ')
        'are red, or'
        >>> string_between('roses are red, or not?', ' ', ' ', greedy=False)
        'are'
        >>> string_between('roses are red, or not?', 'r', 'r')
        'oses are red, o'
        >>> string_between('roses are red, or not?', 'r', 'r', greedy=False)
        'oses a'
        >>> string_between('roses are red, or not?', 'r', 's', True, False)
        'rose'
        >>> string_between('roses are red violets are blue', 'x', 'y')
        ''
    """
    incl_begin = len(begin_str) if not incl_begin else 0
    incl_end = len(end_str) if incl_end else 0
    if begin_str in text and end_str in text:
        if greedy:
            begin = text.find(begin_str) + incl_begin
            end = text.rfind(end_str) + incl_end
        else:
            begin = text.find(begin_str) + incl_begin
            end = text[begin:].find(end_str) + incl_end + begin
        text = text[begin:end]
    else:
        text = ''
    return text


# ======================================================================
def check_redo(
        in_filepaths,
        out_filepaths,
        force=False,
        make_out_dirpaths=False,
        no_empty_input=False):
    """
    Check if input files are newer than output files, to force calculation.

    Args:
        in_filepaths (iterable[str]|None): Input filepaths for computation.
        out_filepaths (iterable[str]): Output filepaths for computation.
        force (bool): Force computation to be re-done.
        make_out_dirpaths (bool): Create output dirpaths if not existing.
        no_empty_input (bool): Check if the input filepath list is empty.

    Returns:
        force (bool): True if the computation is to be re-done.

    Raises:
        IndexError: If the input filepath list is empty.
            Only if `no_empty_input` is True.
        IOError: If any of the input files do not exist.
    """
    # check if output exists
    if not force:
        for out_filepath in out_filepaths:
            if out_filepath and not os.path.exists(out_filepath):
                force = True
                break

    # create output directories
    if force and make_out_dirpaths:
        for out_filepath in out_filepaths:
            out_dirpath = os.path.dirname(out_filepath)
            if not os.path.isdir(out_dirpath):
                os.makedirs(out_dirpath)

    # check if input is older than output
    if not force:
        # check if input is not empty
        if in_filepaths:
            # check if input exists
            for in_filepath in in_filepaths:
                if not os.path.exists(in_filepath):
                    raise IOError('Input file does not exists.')

            for in_filepath, out_filepath in \
                    itertools.product(in_filepaths, out_filepaths):
                if os.path.getmtime(in_filepath) > os.path.getmtime(
                        out_filepath):
                    force = True
                    break
        elif no_empty_input:
            raise IOError('Input file list is empty.')
    return force


# ======================================================================
def bijective_part(arr, invert=False):
    """
    Determine the largest bijective part of an array.

    Args:
        arr (np.ndarray): The input 1D-array.
        invert (bool): Invert the selection order for equally large parts.
            The behavior of `numpy.argmax` is the default.

    Returns:
        slice (slice): The largest bijective portion of arr.
            If two equivalent parts are found, uses the `numpy.argmax` default.

    Examples:
        >>> x = np.linspace(-1 / np.pi, 1 / np.pi, 5000)
        >>> arr = np.sin(1 / x)
        >>> bijective_part(x)
        slice(None, None, None)
        >>> bijective_part(arr)
        slice(None, 833, None)
        >>> bijective_part(arr, True)
        slice(4166, None, None)
    """
    local_mins = sp.signal.argrelmin(arr.ravel())[0]
    local_maxs = sp.signal.argrelmax(arr.ravel())[0]
    # boundaries are considered pseudo-local maxima and minima
    # but are not included in local_mins / local_maxs
    # therefore they are added manually
    extrema = np.zeros((len(local_mins) + len(local_maxs)) + 2, dtype=np.int)
    extrema[-1] = len(arr) - 1
    if len(local_mins) > 0 and len(local_maxs) > 0:
        # start with smallest maxima or minima
        if np.min(local_mins) < np.min(local_maxs):
            extrema[1:-1:2] = local_mins
            extrema[2:-1:2] = local_maxs
        else:
            extrema[1:-1:2] = local_maxs
            extrema[2:-1:2] = local_mins
    elif len(local_mins) == 1 and len(local_maxs) == 0:
        extrema[1] = local_mins
    elif len(local_mins) == 0 and len(local_maxs) == 1:
        extrema[1] = local_maxs
    elif len(local_maxs) == len(local_mins) == 0:
        pass
    else:
        raise ValueError('Failed to determine maxima and/or minima.')

    part_sizes = np.diff(extrema)
    if any(part_sizes) < 0:
        raise ValueError('Failed to determine orders of maxima and minima.')
    if not invert:
        largest = np.argmax(part_sizes)
    else:
        largest = len(part_sizes) - np.argmax(part_sizes[::-1]) - 1
    min_cut, max_cut = extrema[largest:largest + 2]
    return slice(
        min_cut if min_cut > 0 else None,
        max_cut if max_cut < len(arr) - 1 else None)


# =====================================================================
def is_increasing(
        items,
        strict=True):
    """
    Check if items are increasing.

    Args:
        items (iterable): The items to check.
        strict (bool): Check for strict monotonicity.
            If True, consecutive items cannot be equal.
            Otherwise they can be also equal.

    Returns:
        result (bool): True if items are increasing, False otherwise.

    Examples:
        >>> is_increasing([-20, -2, 1, 3, 5, 7, 8, 9])
        True
        >>> is_increasing([1, 3, 5, 7, 8, 9, 9, 10, 400])
        False
        >>> is_increasing([1, 3, 5, 7, 8, 9, 9, 10, 400], False)
        True
        >>> is_increasing([-20, -2, 1, 3, 5, 7, 8, 9])
        True
        >>> is_increasing([-2, -2, 1, 30, 5, 7, 8, 9])
        False
    """
    if strict:
        result = all(x < y for x, y in zip(items, items[1:]))
    else:
        result = all(x <= y for x, y in zip(items, items[1:]))
    return result


# =====================================================================
def is_decreasing(
        items,
        strict=True):
    """
    Check if items are decreasing.

    Args:
        items (iterable): The items to check.
        strict (bool): Check for strict monotonicity.
            If True, consecutive items cannot be equal.
            Otherwise they can be also equal.

    Returns:
        result (bool): True if items are decreasing, False otherwise.

    Examples:
        >>> is_decreasing([312, 54, 53, 7, 3, -5, -100])
        True
        >>> is_decreasing([312, 53, 53, 7, 3, -5, -100])
        False
        >>> is_decreasing([312, 53, 53, 7, 3, -5, -100], False)
        True
        >>> is_decreasing([312, 54, 53, 7, 3, -5, -100])
        True
        >>> is_decreasing([312, 5, 53, 7, 3, -5, -100])
        False
    """
    if strict:
        result = all(x > y for x, y in zip(items, items[1:]))
    else:
        result = all(x >= y for x, y in zip(items, items[1:]))
    return result


# ======================================================================
def is_same_sign(items):
    """
    Determine if all items in an iterable have the same sign.

    Args:
        items (iterable): The items to check.
            The comparison operators '>=' and '<' must be defined.

    Returns:
        same_sign (bool): The result of the comparison.
            True if the items are all positive or all negative.
            False otherwise, i.e. they have mixed signs.

    Examples:
        >>> is_same_sign((0, 1, 2 ,4))
        True
        >>> is_same_sign((-1, -2 , -4))
        True
        >>> is_same_sign((-1, 1))
        False
    """
    return all(item >= 0 for item in items) or all(item < 0 for item in items)


# ======================================================================
def is_in_range(
        arr,
        interval,
        include_extrema=True):
    """
    Determine if the values of an array are within the specified interval.

    Args:
        arr (np.ndarray): The input array.
        interval (tuple[int|float]): The range of values to check.
            A 2-tuple with format (min, max) is expected.

    Returns:
        in_range (bool): The result of the comparison.
            True if all values of the array are within the interval.
            False otherwise.
    """
    if include_extrema:
        in_range = np.min(arr) >= interval[0] and np.max(arr) <= interval[1]
    else:
        in_range = np.min(arr) > interval[0] and np.max(arr) < interval[1]
    return in_range


# ======================================================================
def scale(
        val,
        out_interval=None,
        in_interval=None):
    """
    Linear convert the value from input interval to output interval

    Args:
        val (float|np.ndarray): Value(s) to convert.
        out_interval (float,float): Interval of the output value(s).
            If None, set to: (0, 1).
        in_interval (float,float): Interval of the input value(s).
            If None, and val is iterable, it is calculated as:
            (min(val), max(val)), otherwise set to: (0, 1).

    Returns:
        val (float|np.ndarray): The converted value(s).

    Examples:
        >>> scale(100, (0, 1000), (0, 100))
        1000.0
        >>> scale(50, (0, 1000), (-100, 100))
        750.0
        >>> scale(50, (0, 10), (0, 1))
        500.0
        >>> scale(0.5, (-10, 10))
        0.0
        >>> scale(np.pi / 3, (0, 180), (0, np.pi))
        60.0
        >>> scale(np.arange(5), (0, 1))
        array([ 0.  ,  0.25,  0.5 ,  0.75,  1.  ])
        >>> scale(np.arange(6), (0, 10))
        array([  0.,   2.,   4.,   6.,   8.,  10.])
        >>> scale(np.arange(6), (0, 10), (0, 2))
        array([  0.,   5.,  10.,  15.,  20.,  25.])
    """
    if in_interval:
        in_min, in_max = sorted(in_interval)
    elif isinstance(val, np.ndarray):
        in_min, in_max = minmax(val)
    else:
        in_min, in_max = (0, 1)
    if out_interval:
        out_min, out_max = sorted(out_interval)
    else:
        out_min, out_max = (0, 1)
    return (val - in_min) / (in_max - in_min) * (out_max - out_min) + out_min


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
        operation (str): String with operation to perform.
            Supports the following operations:
                - '+' : addition
                - '-' : subtraction

    Returns:
        new_interval (tuple[float]): Interval resulting from operation

    Examples:
        >>> combine_interval((-1.0, 1.0), (0, 1), '+')
        (-1.0, 2.0)
        >>> combine_interval((-1.0, 1.0), (0, 1), '-')
        (-2.0, 1.0)
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
def midval(arr):
    """
    Calculate the middle value vector.

    Args:
        arr (np.ndarray): The input N-dim array

    Returns:
        arr (np.ndarray): The output (N-1)-dim array

    Examples:
        >>> midval(np.array([0, 1, 2, 3, 4]))
        array([ 0.5,  1.5,  2.5,  3.5])
    """
    return (arr[1:] - arr[:-1]) / 2.0 + arr[:-1]


# ======================================================================
def sgnlog(
        x,
        base=np.e):
    """
    Signed logarithm of x: log(abs(x) * sign(x)

    Args:
        x (float|ndarray): The input value(s)
        base (float): The base of the logarithm.

    Returns:
        The signed logarithm

    Examples:
        >>> sgnlog(-100, 10)
        -2.0
        >>> sgnlog(-64, 2)
        -6.0
        >>> sgnlog(100, 2)
        6.6438561897747253
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

    Examples:
        >>> sgnlogspace(-10, 10, 3)
        array([-10. ,   0.1,  10. ])
        >>> sgnlogspace(-100, -1, 3)
        array([-100.,  -10.,   -1.])
        >>> sgnlogspace(-10, 10, 6)
        array([-10. ,  -1. ,  -0.1,   0.1,   1. ,  10. ])
        >>> sgnlogspace(-10, 10, 5)
        array([-10. ,  -0.1,   0.1,   1. ,  10. ])
        >>> sgnlogspace(2, 10, 4)
        array([  2.        ,   3.41995189,   5.84803548,  10.        ])
    """
    if not is_same_sign((start, stop)):
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
def minmax(arr):
    """
    Calculate the minimum and maximum of an array: (min, max).

    Args:
        arr (np.ndarray): The input array.

    Returns:
        min (float): the minimum value of the array
        max (float): the maximum value of the array

    Examples:
        >>> minmax(np.arange(10))
        (0, 9)
    """
    return np.min(arr), np.max(arr)


# ======================================================================
def subst(
        arr,
        pairs=((np.inf, 0.0), (-np.inf, 0.0), (np.nan, 0.0))):
    """
    Substitute all occurrences of a value in an array.

    Useful to remove specific values, e.g. singularities.

    Args:
        arr (np.ndarray): The input array.
        pairs (tuple[tuple]): The substitution rules.
            Each rule consist of a value to replace and its replacement.
            Each rule is applied sequentially in the order they appear and
            modify the content of the array immediately.

    Returns:
        arr (np.ndarray): The output array.

    Examples:
        >>> a = np.arange(10)
        >>> subst(a, ((1, 100), (7, 700)))
        array([  0, 100,   2,   3,   4,   5,   6, 700,   8,   9])
        >>> a = np.tile(np.arange(4), 3)
        >>> subst(a, ((1, 100), (7, 700)))
        array([  0, 100,   2,   3,   0, 100,   2,   3,   0, 100,   2,   3])
        >>> a = np.tile(np.arange(4), 3)
        >>> subst(a, ((1, 100), (3, 300)))
        array([  0, 100,   2, 300,   0, 100,   2, 300,   0, 100,   2, 300])
        >>> a = np.array([0.0, 1.0, np.inf, -np.inf, np.nan, -np.nan])
        >>> subst(a)
        array([ 0.,  1.,  0.,  0.,  0.,  0.])
        >>> a = np.array([0.0, 1.0, np.inf, 2.0, np.nan])
        >>> subst(a, ((np.inf, 0.0), (0.0, np.inf), (np.nan, 0.0)))
        array([ inf,   1.,  inf,   2.,   0.])
        >>> subst(a, ((np.inf, 0.0), (np.nan, 0.0), (0.0, np.inf)))
        array([ inf,   1.,  inf,   2.,  inf])
    """
    for k, v in pairs:
        if k is np.nan:
            arr[np.isnan(arr)] = v
        else:
            arr[arr == k] = v
    return arr


# ======================================================================
def ravel_clean(
        arr,
        removes=(np.nan, np.inf, -np.inf)):
    """
    Ravel and remove values to an array.

    Args:
        arr (np.ndarray): The input array.
        removes (iterable): Values to remove.
            If empty, no values will be removed.

    Returns:
        arr (np.ndarray): The output array.

    Examples:
        >>> a = np.array([0.0, 1.0, np.inf, -np.inf, np.nan, -np.nan])
        >>> ravel_clean(a)
        array([ 0.,  1.])

    See Also:
        utils.subst
    """
    arr = arr.ravel()
    for val in removes:
        if val is np.nan:
            arr = arr[~np.isnan(arr)]
        if len(arr) > 0:
            arr = arr[arr != val]
    return arr


# ======================================================================
def dftn(arr):
    """
    Discrete Fourier Transform.

    Interface to fftn combined with fftshift.

    Args:
        arr (np.ndarray): Input n-dim array.

    Returns:
        arr (np.ndarray): Output n-dim array.

    Examples:
        >>> a = np.arange(2)
        >>> dftn(a)
        array([-1.+0.j,  1.+0.j])
        >>> print(np.allclose(a, dftn(idftn(a))))
        True

    See Also:
        numpy.fft, scipy.fftpack
    """
    return fftshift(fftn(arr))


# ======================================================================
def idftn(arr):
    """
    Inverse Discrete Fourier transform.

    Interface to ifftn combined with ifftshift.

    Args:
        arr (np.ndarray): Input n-dim array.

    Returns:
        arr (np.ndarray): Output n-dim array.

    Examples:
        >>> a = np.arange(2)
        >>> idftn(a)
        array([ 0.5+0.j,  0.5+0.j])
        >>> print(np.allclose(a, idftn(dftn(a))))
        True

    See Also:
        numpy.fft, scipy.fftpack
    """
    return ifftn(ifftshift(arr))


# ======================================================================
def coord(
        shape,
        position=0.5,
        is_relative=True,
        use_int=True):
    """
    Calculate the coordinate in a given shape for a specified position.

    Args:
        shape (iterable[int]): The shape of the mask in px.
        position (float|iterable[float]): Relative position of the origin.
            Values are in the [0, 1] interval.
        is_relative (bool): Interpret origin as relative.
        use_int (bool): Force interger values for the coordinates.
        
    Returns:
        position (list): The coordinate in the shape.
        
    Examples:
        >>> coord((5, 5))
        (2, 2)
        >>> coord((4, 4))
        (2, 2)
        >>> coord((5, 5), 3, False)
        (3, 3)
    """
    position = auto_repeat(position, len(shape), check=True)
    if is_relative:
        if use_int:
            position = tuple(
                int(scale(x, (0, dim))) for x, dim in zip(position, shape))
        else:
            position = tuple(
                scale(x, (0, dim - 1)) for x, dim in zip(position, shape))
    elif any([not isinstance(x, int) for x in position]) and use_int:
        raise TypeError('Absolute origin must be integer.')
    return position


# ======================================================================
def grid_coord(
        shape,
        position=0.5,
        is_relative=True,
        use_int=True,
        dense=False):
    """
    Calculate the generic x_i coordinates for N-dim operations.

    Args:
        shape (iterable[int]): The shape of the mask in px.
        position (float|iterable[float]): Relative position of the origin.
            Values are in the [0, 1] interval.
        is_relative (bool): Interpret origin as relative.
        dense (bool): Determine the shape of the mesh-grid arrays.
        use_int (bool): Force interger values for the coordinates.

    Returns:
        coord (list[np.ndarray]): mesh-grid ndarrays.
            The shape is identical if dense is True, otherwise only one
            dimension is larger than 1.

    Examples:
        >>> grid_coord((4, 4))
        [array([[-2],
               [-1],
               [ 0],
               [ 1]]), array([[-2, -1,  0,  1]])]
        >>> grid_coord((5, 5))
        [array([[-2],
               [-1],
               [ 0],
               [ 1],
               [ 2]]), array([[-2, -1,  0,  1,  2]])]
        >>> grid_coord((2, 2))
        [array([[-1],
               [ 0]]), array([[-1,  0]])]
        >>> grid_coord((2, 2), dense=True)
        array([[[-1, -1],
                [ 0,  0]],
        <BLANKLINE>
               [[-1,  0],
                [-1,  0]]])
        >>> grid_coord((2, 3), position=(0.0, 0.5))
        [array([[0],
               [1]]), array([[-1,  0,  1]])]
        >>> grid_coord((3, 9), position=(1, 4), is_relative=False)
        [array([[-1],
               [ 0],
               [ 1]]), array([[-4, -3, -2, -1,  0,  1,  2,  3,  4]])]
        >>> grid_coord((3, 9), position=0.2, is_relative=True)
        [array([[0],
               [1],
               [2]]), array([[-1,  0,  1,  2,  3,  4,  5,  6,  7]])]
        >>> grid_coord((4, 4), use_int=False)
        [array([[-1.5],
               [-0.5],
               [ 0.5],
               [ 1.5]]), array([[-1.5, -0.5,  0.5,  1.5]])]
        >>> grid_coord((5, 5), use_int=False)
        [array([[-2.],
               [-1.],
               [ 0.],
               [ 1.],
               [ 2.]]), array([[-2., -1.,  0.,  1.,  2.]])]
        >>> grid_coord((2, 3), position=(0.0, 0.0), use_int=False)
        [array([[ 0.],
               [ 1.]]), array([[ 0.,  1.,  2.]])]
    """
    position = coord(shape, position, is_relative, use_int)
    grid = [slice(-x0, dim - x0) for x0, dim in zip(position, shape)]
    return np.ogrid[grid] if not dense else np.mgrid[grid]


# ======================================================================
def _kk_2(
        shape,
        factors=1):
    """
    Calculate the k^2 kernel to be used for the Laplacian operators.

    Args:
        shape (iterable[int]): The size of the array.
        factors (iterable[int|tuple]): The size conversion factors for each
        dim.

    Returns:
        arr (np.ndarray): The resulting array.

    Examples:
        >>> _kk_2((3, 3, 3))
        array([[[ 3.,  2.,  3.],
                [ 2.,  1.,  2.],
                [ 3.,  2.,  3.]],
        <BLANKLINE>
               [[ 2.,  1.,  2.],
                [ 1.,  0.,  1.],
                [ 2.,  1.,  2.]],
        <BLANKLINE>
               [[ 3.,  2.,  3.],
                [ 2.,  1.,  2.],
                [ 3.,  2.,  3.]]])
        >>> _kk_2((3, 3, 3), np.sqrt(3))
        array([[[ 1.        ,  0.66666667,  1.        ],
                [ 0.66666667,  0.33333333,  0.66666667],
                [ 1.        ,  0.66666667,  1.        ]],
        <BLANKLINE>
               [[ 0.66666667,  0.33333333,  0.66666667],
                [ 0.33333333,  0.        ,  0.33333333],
                [ 0.66666667,  0.33333333,  0.66666667]],
        <BLANKLINE>
               [[ 1.        ,  0.66666667,  1.        ],
                [ 0.66666667,  0.33333333,  0.66666667],
                [ 1.        ,  0.66666667,  1.        ]]])
        >>> _kk_2((2, 2, 2), 0.6)
        array([[[ 8.33333333,  5.55555556],
                [ 5.55555556,  2.77777778]],
        <BLANKLINE>
               [[ 5.55555556,  2.77777778],
                [ 2.77777778,  0.        ]]])
    """
    kk_ = grid_coord(shape)
    if factors and factors != 1:
        factors = auto_repeat(factors, len(shape), check=True)
        kk_ = [k_i / factor for k_i, factor in zip(kk_, factors)]
    kk_2 = np.zeros(shape)
    for k_i, dim in zip(kk_, shape):
        kk_2 += k_i ** 2
    return kk_2


# ======================================================================
def _kk(
        shape,
        factors=1):
    """
    Calculate the k kernel to be used for the gradient operators.

    Args:
        shape (iterable[int]): The size of the array.
        factors (iterable[int|tuple]): The size conversion factors for each
        dim.

    Returns:
        arr (np.ndarray): The resulting array.

    Examples:
        >>> _kk((3, 3, 3))
        array([[[ 1.73205081,  1.41421356,  1.73205081],
                [ 1.41421356,  1.        ,  1.41421356],
                [ 1.73205081,  1.41421356,  1.73205081]],
        <BLANKLINE>
               [[ 1.41421356,  1.        ,  1.41421356],
                [ 1.        ,  0.        ,  1.        ],
                [ 1.41421356,  1.        ,  1.41421356]],
        <BLANKLINE>
               [[ 1.73205081,  1.41421356,  1.73205081],
                [ 1.41421356,  1.        ,  1.41421356],
                [ 1.73205081,  1.41421356,  1.73205081]]])
        >>> _kk((3, 3, 3), np.sqrt(3))
        array([[[ 1.        ,  0.81649658,  1.        ],
                [ 0.81649658,  0.57735027,  0.81649658],
                [ 1.        ,  0.81649658,  1.        ]],
        <BLANKLINE>
               [[ 0.81649658,  0.57735027,  0.81649658],
                [ 0.57735027,  0.        ,  0.57735027],
                [ 0.81649658,  0.57735027,  0.81649658]],
        <BLANKLINE>
               [[ 1.        ,  0.81649658,  1.        ],
                [ 0.81649658,  0.57735027,  0.81649658],
                [ 1.        ,  0.81649658,  1.        ]]])
        >>> _kk((2, 2, 2), 0.6)
        array([[[ 2.88675135,  2.3570226 ],
                [ 2.3570226 ,  1.66666667]],
        <BLANKLINE>
               [[ 2.3570226 ,  1.66666667],
                [ 1.66666667,  0.        ]]])
    """
    kk_ = grid_coord(shape)
    if factors and factors != 1:
        factors = auto_repeat(factors, len(shape), check=True)
        kk_ = [k_i / factor for k_i, factor in zip(kk_, factors)]
    kk = np.zeros(shape)
    for k_i, dim in zip(kk_, shape):
        kk += (k_i ** 2)
    return np.sqrt(kk)


# ======================================================================
def gradient(
        arr,
        ft_factor=(2 * np.pi),
        pad_width=0):
    """
    Apply the gradient operator (in the Fourier domain).

    Args:
        arr (np.ndarray): The input array.
        ft_factor (float): The Fourier factor for the gradient operator.
            Should be either 1 or 2*pi, depending on DFT implementation.
        pad_width (float|int): Size of the border to use.
            This is useful for mitigating border effects.
            If int, it is interpreted as absolute size.
            If float, it is interpreted as relative to the maximum size.

    Returns:
        arr (np.ndarray): The output array.
    """
    if pad_width:
        shape = arr.shape
        pad_width = auto_pad_width(pad_width, shape)
        # mask = [slice(borders, -borders)] * arr.ndim
        mask = [slice(lower, -upper) for (lower, upper) in pad_width]
        arr = np.pad(arr, pad_width, 'constant', constant_values=0)
    else:
        mask = [slice(None)] * arr.ndim
    kk = fftshift(_kk(arr.shape, arr.shape))
    arr = ((-1j * ft_factor) ** 2) * ifftn(kk * fftn(arr))
    return arr[mask]


# ======================================================================
def inv_gradient(
        arr,
        ft_factor=(2 * np.pi),
        pad_width=0,
        singularity=0):
    """
    Apply the inverse gradient operator (in the Fourier domain).

    The singularity in the origin is corrected using the value set in the
    corresponding parameter.


    Args:
        arr (np.ndarray): The input array.
        ft_factor (float): The Fourier factor for the gradient operator.
            Should be either 1 or 2*pi, depending on DFT implementation.
        pad_width (float|int): Size of the border to use.
            This is useful for mitigating border effects.
            If int, it is interpreted as absolute size.
            If float, it is interpreted as relative to the maximum size.

    Returns:
        arr (np.ndarray): The output array.
    """
    if pad_width:
        shape = arr.shape
        pad_width = auto_pad_width(pad_width, shape)
        # mask = [slice(borders, -borders)] * arr.ndim
        mask = [slice(lower, -upper) for (lower, upper) in pad_width]
        arr = np.pad(arr, pad_width, 'constant', constant_values=0)
    else:
        mask = [slice(None)] * arr.ndim
    kk = fftshift(_kk(arr.shape, arr.shape))
    kk[kk != 0] = 1.0 / kk[kk != 0]  # get the inverse
    arr = ((-1j / ft_factor) ** 2) * ifftn(kk * fftn(arr))
    return arr[mask]


# ======================================================================
def auto_pad_width(
        pad_width,
        shape,
        combine=None):
    """
    Ensure pad_width value(s) to be consisting of integer.

    Args:
        pad_width (float|int|iterable[float|int]): Size of the padding to use.
            This is useful for mitigating border effects.
            If iterable, a value for each dim must be specified.
            If not iterable, all dims will have the same value.
            If int, it is interpreted as absolute size.
            If float, it is interpreted as relative to corresponding dim size.
        shape (iterable[int]): The shape to associate to `pad_width`.
        combine (callable|None): The function for combining shape values.
            If None, uses the corresponding dim from the shape.

    Returns:
        pad_width (int|tuple[tuple[int]]): The absolute `pad_width`.
            If input `pad_width` is not iterable, result is not iterable.

    See Also:
        np.pad

    Examples:
        >>> shape = (10, 20, 30)
        >>> auto_pad_width(0.1, shape)
        ((1, 1), (2, 2), (3, 3))
        >>> auto_pad_width(0.1, shape, max)
        ((3, 3), (3, 3), (3, 3))
        >>> shape = (10, 20, 30)
        >>> auto_pad_width(((0.1, 0.5),), shape)
        ((1, 5), (2, 10), (3, 15))
        >>> auto_pad_width(((2, 3),), shape)
        ((2, 3), (2, 3), (2, 3))
        >>> auto_pad_width(((2, 3), (1, 2)), shape)
        Traceback (most recent call last):
            ....
        AssertionError
        >>> auto_pad_width(((0.1, 0.2),), shape, min)
        ((1, 2), (1, 2), (1, 2))
        >>> auto_pad_width(((0.1, 0.2),), shape, max)
        ((3, 6), (3, 6), (3, 6))
    """

    def float_to_int(val, scale):
        return int(val * scale) if isinstance(val, float) else val

    try:
        iter(pad_width)
    except TypeError:
        pad_width = ((pad_width,) * 2,)
    finally:
        combined = combine(shape) if combine else None
        pad_width = list(
            pad_width if len(pad_width) > 1 else pad_width * len(shape))
        assert (len(pad_width) == len(shape))
        for i, (item, dim) in enumerate(zip(pad_width, shape)):
            lower, upper = item
            pad_width[i] = (
                float_to_int(lower, dim if not combine else combined),
                float_to_int(upper, dim if not combine else combined))
        pad_width = tuple(pad_width)
    return pad_width


# ======================================================================
def laplacian(
        arr,
        ft_factor=(2 * np.pi),
        pad_width=0):
    """
    Apply the Laplacian operator (in the Fourier domain).

    Args:
        arr (np.ndarray): The input array.
        ft_factor (float): The Fourier factor for the gradient operator.
            Should be either 1 or 2*pi, depending on DFT implementation.
        pad_width (float|int|iterable[float|int]): Size of the padding to use.
            This is useful for mitigating border effects.
            If iterable, a value for each dim must be specified.
            If not iterable, all dims will have the same value.
            If int, it is interpreted as absolute size.
            If float, it is interpreted as relative to the maximum size.

    Returns:
        arr (np.ndarray): The output array.
    """
    if pad_width:
        shape = arr.shape
        pad_width = auto_pad_width(pad_width, shape)
        mask = [slice(lower, -upper) for (lower, upper) in pad_width]
        arr = np.pad(arr, pad_width, 'constant', constant_values=0)
    else:
        mask = [slice(None)] * arr.ndim
    kk_2 = fftshift(_kk_2(arr.shape, arr.shape))
    arr = ((1j * ft_factor) ** 2) * ifftn(kk_2 * fftn(arr))
    return arr[mask]


# ======================================================================
def inv_laplacian(
        arr,
        ft_factor=(2 * np.pi),
        pad_width=0):
    """
    Apply the inverse Laplacian operator (in the Fourier domain).

    Args:
        arr (np.ndarray): The input array.
        ft_factor (float): The Fourier factor for the gradient operator.
            Should be either 1 or 2*pi, depending on DFT implementation.
        pad_width (float|int): Size of the border to use.
            This is useful for mitigating border effects.
            If int, it is interpreted as absolute size.
            If float, it is interpreted as relative to the maximum size.

    Returns:
        arr (np.ndarray): The output array.
    """
    if pad_width:
        shape = arr.shape
        pad_width = auto_pad_width(pad_width, shape)
        # mask = [slice(borders, -borders)] * arr.ndim
        mask = [slice(lower, -upper) for (lower, upper) in pad_width]
        arr = np.pad(arr, pad_width, 'constant', constant_values=0)
    else:
        mask = [slice(None)] * arr.ndim
    kk_2 = fftshift(_kk_2(arr.shape, arr.shape))
    kk_2[kk_2 != 0] = 1.0 / kk_2[kk_2 != 0]
    arr = ((-1j / ft_factor) ** 2) * ifftn(kk_2 * fftn(arr))
    return arr[mask]


# ======================================================================
def auto_bin(
        arr,
        method='auto'):
    """
    Determine the optimal number of bins for an array.

    Args:
        arr (np.ndarray): The input array.
        method (str|None): The estimation method.
            Available options (with N the array size):
             - 'auto': max('fd', 'sturges')
             - 'sqrt': sqrt(N), simple
             - 'sturges': log_2(N) + 1
             - 'rice': 2 * N^(1/3)
             - 'scott': 3.5 * SD(arr) / N^(1/3)
             - 'fd': (Freedman-Diaconis) 2 * (Q75 - Q25) / N^(1/3)
             - None: N
    Returns:
        num (int): The number of bins.

    Examples:
        >>> arr = np.arange(100)
        >>> auto_bin(arr)
        22
        >>> auto_bin(arr, 'sqrt')
        10
        >>> auto_bin(arr, 'auto')
        22
        >>> auto_bin(arr, 'sturges')
        8
        >>> auto_bin(arr, 'rice')
        10
        >>> auto_bin(arr, 'scott')
        22
        >>> auto_bin(arr, 'fd')
        22
        >>> auto_bin(arr, None)
        100

    References:
         - https://en.wikipedia.org/wiki/Histogram#Number_of_bins_and_width
    """
    if method == 'auto':
        num = max(auto_bin(arr, 'fd'), auto_bin(arr, 'sturges'))
    elif method == 'sqrt':
        num = int(np.ceil(np.sqrt(arr.size)))
    elif method == 'sturges':
        num = int(np.ceil(np.log2(arr.size)) + 1)
    elif method == 'rice':
        num = int(np.ceil(2 * arr.size ** (1 / 3)))
    elif method == 'scott':
        num = int(np.ceil(3.5 * np.std(arr) / arr.size ** (1 / 3)))
    elif method == 'fd':
        q75, q25 = np.percentile(arr, [75, 25])
        num = int(np.ceil(2 * (q75 - q25) / arr.size ** (1 / 3)))
    else:
        num = arr.size
    return num


# ======================================================================
def auto_bins(
        arrs,
        method='auto',
        combine=max):
    """
    Determine the optimal number of bins for a group of arrays.

    Args:
        arrs (iterable[np.ndarray]): The input arrays.
        method (str|iterable[str]|None): The method to use calculating bins.
            If str, the same method is applied to both arrays.
            See `auto_bin()` for available methods.
        combine (callable|None): Combine each bin using the combine function.
            combine(n_bins) -> n_bin
            n_bins is of type iterable[int]

    Returns:
        n_bins (int|tuple[int]): The number of bins.
            If combine is None, returns a tuple of int (one for each input
            array).

    Examples:
        >>> arr1 = np.arange(100)
        >>> arr2 = np.arange(200)
        >>> arr3 = np.arange(300)
        >>> auto_bins((arr1, arr2))
        35
        >>> auto_bins((arr1, arr2, arr3))
        45
        >>> auto_bins((arr1, arr2), ('sqrt', 'sturges'))
        10
        >>> auto_bins((arr1, arr2), combine=None)
        (22, 35)
        >>> auto_bins((arr1, arr2), combine=min)
        22
        >>> auto_bins((arr1, arr2), combine=sum)
        57
        >>> auto_bins((arr1, arr2), combine=lambda x: abs(x[0] - x[1]))
        13
    """
    if isinstance(method, str) or method is None:
        method = (method,) * len(arrs)
    n_bins = []
    for arr, method in zip(arrs, method):
        n_bins.append(auto_bin(arr, method))
    if combine:
        return combine(n_bins)
    else:
        return tuple(n_bins)


# ======================================================================
def entropy(
        hist,
        base=np.e):
    """
    Calculate the simple or joint Shannon entropy H.

    H = -sum(p(x) * log(p(x)))

    p(x) is the probability of x, where x can be N-Dim.

    Args:
        hist (np.ndarray): The probability density function p(x).
            If hist is 1-dim, the Shannon entropy is computed.
            If hist is N-dim, the joint Shannon entropy is computed.
            Zeros are handled correctly.
            The probability density function does not need to be normalized.
        base (int|float): The base units to express the result.
            Should be a number larger than 0.
            If base is 2, the unit is `bits`.
            If base is np.e (Euler's number), the unit is `nats`.

    Returns:
        h (float): The Shannon entropy H = -sum(p(x) * log(p(x)))

    Examples:
        >>>
    """
    # normalize histogram to unity
    hist = hist / np.sum(hist)
    # skip zero values
    mask = hist != 0.0
    log_hist = np.zeros_like(hist)
    log_hist[mask] = np.log(hist[mask]) / np.log(base)
    h = -np.sum(hist * log_hist)
    return h


# ======================================================================
def conditional_entropy(
        hist2,
        hist,
        base=np.e):
    """
    Calculate the conditional probability: H(X|Y)

    Args:
        hist2 (np.ndarray): The joint probability density function.
            Must be the 2D histrogram of X and Y
        hist (np.ndarray): The given probability density function.
            Must be the 1D histogram of Y.
        base (int|float): The base units to express the result.
            Should be a number larger than 0.
            If base is 2, the unit is `bits`.
            If base is np.e (Euler's number), the unit is `nats`.

    Returns:
        hc (float): The conditional entropy H(X|Y)

    Examples:
        >>>
    """
    return entropy(hist2, base) - entropy(hist, base)


# ======================================================================
def variation_information(
        arr1,
        arr2,
        base=np.e,
        bins='sturges'):
    """
    Calculate the variation of information between two arrays.

    Args:
        arr1 (np.ndarray): The first input array.
            Must have same shape as arr2.
        arr2 (np.ndarray): The second input array.
            Must have same shape as arr1.
        base (int|float): The base units to express the result.
            Should be a number larger than 0.
            If base is 2, the unit is `bits`.
            If base is np.e (Euler's number), the unit is `nats`.
        bins (int|str|None): The number of bins to use for the distribution.
            If int, the exact number is used.
            If str, a method accepted by `auto_bin` is expected.
            If None, uses the maximum number of bins (not recommended).
    Returns:
        vi (float): The variation of information.

    Examples:
        >>> np.random.seed(0)
        >>> arr1 = np.zeros(100)
        >>> arr2 = np.arange(100)
        >>> arr3 = np.random.rand(100)
        >>> arr4 = arr3 + np.random.rand(100) / 100
        >>> variation_information(arr1, arr1)
        0.0
        >>> variation_information(arr2, arr2)
        0.0
        >>> variation_information(arr3, arr3)
        0.0
        >>> vi_12 = variation_information(arr1, arr2)
        >>> vi_21 = variation_information(arr2, arr1)
        >>> vi_31 = variation_information(arr3, arr1)
        >>> vi_34 = variation_information(arr3, arr4)
        >>> # print(vi_12, vi_21, vi_31, vi_34)
        >>> np.isclose(vi_12, vi_21)
        True
        >>> vi_34 < vi_31
        True
    """
    if not isinstance(bins, int):
        if bins is not None and not isinstance(bins, str):
            raise ValueError('Invalid value for `bins`')
        bins = auto_bins((arr1, arr2), method=bins, combine=max)

    if not np.array_equal(arr1, arr2):
        hist1, bin_edges1 = np.histogram(arr1, bins)
        hist2, bin_edges2 = np.histogram(arr2, bins)
        hist12, x_edges, y_edges = np.histogram2d(arr1, arr2, bins=bins)
        h12 = entropy(hist12, base)
        h1 = entropy(hist1, base)
        h2 = entropy(hist2, base)
        vi = 2 * h12 - h1 - h2
    else:
        vi = 0.0
    # absolute value to fix rounding errors
    return abs(vi)


# ======================================================================
def mutual_information(
        arr1,
        arr2,
        base=np.e,
        bins='sturges'):
    """
    Calculate the mutual information between two arrays.

    Note that the numerical result depends on the number of bins.

    Args:
        arr1 (np.ndarray): The first input array.
        arr2 (np.ndarray): The second input array.
        base (int|float|None): The base units to express the result.
            Should be a number larger than 0.
            If base is 2, the unit is bits.
            If base is np.e (Euler's number), the unit is `nats`.
            If base is None, the result is normalized to unity.
        bins (int|str|None): The number of bins to use for the distribution.
            If int, the exact number is used.
            If str, a method accepted by `auto_bin` is expected.
            If None, uses the maximum number of bins (not recommended).

    Returns:
        mi (float): The (normalized) mutual information.
            If base is None, the normalized version is returned.
            Otherwise returns the mutual information in the specified base.

    Examples:
        >>> np.random.seed(0)
        >>> arr1 = np.zeros(100)
        >>> arr2 = np.arange(100)
        >>> arr3 = np.random.rand(100)
        >>> arr4 = arr3 + np.random.rand(100) / 100
        >>> mi_11 = mutual_information(arr1, arr1)
        >>> mi_22 = mutual_information(arr2, arr2)
        >>> mi_33 = mutual_information(arr3, arr3)
        >>> mi_44 = mutual_information(arr4, arr4)
        >>> # print(mi_11, mi_22, mi_33, mi_44)
        >>> mi_22 > mi_33 > mi_11
        True
        >>> mi_12 = mutual_information(arr1, arr2)
        >>> mi_21 = mutual_information(arr2, arr1)
        >>> mi_32 = mutual_information(arr3, arr2)
        >>> mi_34 = mutual_information(arr3, arr4)
        >>> # print(mi_12, mi_21, mi_32, mi_34)
        >>> mi_44 > mi_34 and mi_33 > mi_34
        True
        >>> np.isclose(mi_12, mi_21)
        True
        >>> mi_34 > mi_32
        True
        >>> mi_n10 = mutual_information(arr3, arr2, np.e, 10)
        >>> mi_n20 = mutual_information(arr3, arr2, np.e, 20)
        >>> mi_n100 = mutual_information(arr3, arr2, np.e, 100)
        >>> # print(mi_n10, mi_n20, mi_n100)
        >>> mi_n10 < mi_n20 < mi_n100
        True
        >>> mi_be = mutual_information(arr3, arr4, np.e)
        >>> mi_b2 = mutual_information(arr3, arr4, 2)
        >>> mi_b10 = mutual_information(arr3, arr4, 10)
        >>> # print(mi_be, mi_b2, mi_b10)
        >>> mi_b10 < mi_be < mi_b2
        True

    See Also:
        - Cahill, Nathan D. “Normalized Measures of Mutual Information with
          General Definitions of Entropy for Multimodal Image Registration.” In
          International Workshop on Biomedical Image Registration, 258–268.
          Springer, 2010.
          http://link.springer.com/chapter/10.1007/978-3-642-14366-3_23.
    """
    # todo: check implementation speed and consistency
    if not isinstance(bins, int):
        if bins is not None and not isinstance(bins, str):
            raise ValueError('Invalid value for `bins`')
        bins = auto_bins((arr1, arr2), method=bins, combine=max)

    # # scikit.learn implementation
    # hist, x_edges, y_edges = np.histogram2d(arr1, arr2, bins=bins)
    # from sklearn.metrics import mutual_info_score
    # mi = mutual_info_score(None, None, contingency=hist)
    # if base > 0 and base != np.e:
    #     mi /= np.log(base)

    # # alternate implementation
    # hist, x_edges, y_edges = np.histogram2d(arr1, arr2, bins=bins)
    # g, p, dof, expected = scipy.stats.chi2_contingency(
    #     hist + np.finfo(float).eps, lambda_='log-likelihood')
    # mi = g / hist.sum() / 2

    # entropy-based implementation
    hist1, bin_edges1 = np.histogram(arr1, bins)
    hist2, bin_edges2 = np.histogram(arr2, bins)
    hist12, x_edges, y_edges = np.histogram2d(arr1, arr2, bins=bins)
    h12 = entropy(hist12, base)
    h1 = entropy(hist1, base)
    h2 = entropy(hist2, base)
    mi = h1 + h2 - h12

    # absolute value to fix rounding errors
    return abs(mi)


# ======================================================================
def norm_mutual_information(
        arr1,
        arr2,
        bins='sturges'):
    """

    Args:
        arr1 ():
        arr2 ():
        bins ():

    Returns:


    Examples:
        >>> np.random.seed(0)
        >>> arr1 = np.zeros(100)
        >>> arr2 = np.arange(100)
        >>> arr3 = np.random.rand(100)
        >>> arr4 = arr3 + np.random.rand(100) / 100
        >>> mi_11 = norm_mutual_information(arr1, arr1)
        >>> mi_22 = norm_mutual_information(arr2, arr2)
        >>> mi_33 = norm_mutual_information(arr3, arr3)
        >>> mi_44 = norm_mutual_information(arr4, arr4)
        >>> # print(mi_11, mi_22, mi_33, mi_44)
        >>> 1.0 == mi_11 == mi_22 == mi_33 == mi_44
        True
        >>> mi_12 = norm_mutual_information(arr1, arr2)
        >>> mi_21 = norm_mutual_information(arr2, arr1)
        >>> mi_32 = norm_mutual_information(arr3, arr2)
        >>> mi_34 = norm_mutual_information(arr3, arr4)
        >>> # print(mi_12, mi_21, mi_32, mi_34)
        >>> mi_44 > mi_34 and mi_33 > mi_34
        True
        >>> np.isclose(mi_12, mi_21)
        True
        >>> mi_34 > mi_32
        True
        >>> mi_n10 = norm_mutual_information(arr3, arr2, 10)
        >>> mi_n20 = norm_mutual_information(arr3, arr2, 20)
        >>> mi_n100 = norm_mutual_information(arr3, arr2, 100)
        >>> # print(mi_n10, mi_n20, mi_n100)
        >>> mi_n10 < mi_n20 < mi_n100
        True
    """
    if not isinstance(bins, int):
        if bins is not None and not isinstance(bins, str):
            raise ValueError('Invalid value for `bins`')
        bins = auto_bins((arr1, arr2), method=bins, combine=max)
    hist1, bin_edges1 = np.histogram(arr1, bins)
    hist2, bin_edges2 = np.histogram(arr2, bins)
    hist12, x_edges, y_edges = np.histogram2d(arr1, arr2, bins=bins)
    if not np.array_equal(arr1, arr2):
        base = np.e  # results should be independent of the base
        h12 = entropy(hist12, base)
        h1 = entropy(hist1, base)
        h2 = entropy(hist2, base)
        nmi = 1 - (2 * h12 - h1 - h2) / h12
    else:
        nmi = 1.0

    # absolute value to fix rounding errors
    return abs(nmi)


# ======================================================================
def gaussian_nd(
        shape,
        sigmas,
        origin=0.5,
        n_dim=None,
        normalize=True):
    """
    Generate an N-dim Gaussian function.

    Args:
        shape ():
        sigmas ():
        origin ():
        n_dim ():
        normalize ():

    Returns:

    """
    if not n_dim:
        n_dim = max_iter_len((shape, sigmas, origin))

    shape = auto_repeat(shape, n_dim)
    sigmas = auto_repeat(sigmas, n_dim)
    origin = auto_repeat(origin, n_dim)

    xx = grid_coord(shape, origin)
    kernel = np.exp(
        -(
            sum([x_i ** 2 / (2 * sigma ** 2) for x_i, sigma in
                 zip(xx, sigmas)])))
    if normalize:
        kernel /= np.sum(kernel)
    return kernel


# ======================================================================
def moving_average(
        arr,
        weights=1,
        **kws):
    """
    Calculate the moving average (with optional weights).

    The moving average will be applied to the flattened array.
    Unless specified otherwise, the size of the array will be reduced by
    len(weights) - 1
    This is equivalent to passing `mode='valid'` to `scipy.signal.convolve`.
    Please refer to `scipy.signal.convolve` for more options.

    Args:
        arr (np.ndarray): The input array.
        weights (int|iterable): The running weights.
            If int, the number of elements to group in the 'running' axis and
            unity weights are used.
            The size of the weights array len(weights) must be such that
            len(weights) >= 1 and len(weights) <= len(array), otherwise the
            flattened array is returned.
        **kws (dict): Keyword arguments passed to `scipy.signal.convolve`.

    Returns:
        arr (np.ndarray): The output array.

    Example:
        >>> moving_average(np.linspace(1, 9, 9), 1)
        array([ 1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9.])
        >>> moving_average(np.linspace(1, 8, 8), 1)
        array([ 1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.])
        >>> moving_average(np.linspace(1, 9, 9), 2)
        array([ 1.5,  2.5,  3.5,  4.5,  5.5,  6.5,  7.5,  8.5])
        >>> moving_average(np.linspace(1, 8, 8), 2)
        array([ 1.5,  2.5,  3.5,  4.5,  5.5,  6.5,  7.5])
        >>> moving_average(np.linspace(1, 9, 9), 5)
        array([ 3.,  4.,  5.,  6.,  7.])
        >>> moving_average(np.linspace(1, 8, 8), 5)
        array([ 3.,  4.,  5.,  6.])
        >>> moving_average(np.linspace(1, 8, 8), [1, 1, 1])
        array([ 2.,  3.,  4.,  5.,  6.,  7.])
        >>> moving_average(np.linspace(1, 8, 8), [1, 0.2])
        array([ 1.16666667,  2.16666667,  3.16666667,  4.16666667,  5.16666667,
                6.16666667,  7.16666667])
    """
    arr = arr.ravel()
    if isinstance(weights, int):
        weights = np.ones((weights,))
    else:
        # weights order needs to be inverted
        weights = np.array(weights)[::-1]
    num = len(weights) if isinstance(weights, np.ndarray) else 0
    if len(arr) >= num > 1:
        if 'mode' not in kws:
            kws['mode'] = 'valid'
        arr = sp.signal.convolve(arr, weights / len(weights), **kws)
        arr *= len(weights) / np.sum(weights)
    return arr


# ======================================================================
def moving_mean(
        arr,
        num=1):
    """
    Calculate the moving mean.

    The moving average will be applied to the flattened array.
    Unless specified otherwise, the size of the array will be reduced by
    (num - 1).

    Args:
        arr (np.ndarray): The input array.
        num (int|iterable): The running window size.
            The number of elements to group.

    Returns:
        arr (np.ndarray): The output array.

    Example:
        >>> moving_mean(np.linspace(1, 9, 9), 1)
        array([ 1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9.])
        >>> moving_mean(np.linspace(1, 8, 8), 1)
        array([ 1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.])
        >>> moving_mean(np.linspace(1, 9, 9), 2)
        array([ 1.5,  2.5,  3.5,  4.5,  5.5,  6.5,  7.5,  8.5])
        >>> moving_mean(np.linspace(1, 8, 8), 2)
        array([ 1.5,  2.5,  3.5,  4.5,  5.5,  6.5,  7.5])
        >>> moving_mean(np.linspace(1, 9, 9), 5)
        array([ 3.,  4.,  5.,  6.,  7.])
        >>> moving_mean(np.linspace(1, 8, 8), 5)
        array([ 3.,  4.,  5.,  6.])
    """
    arr = arr.ravel()
    arr = np.cumsum(arr)
    arr[num:] = arr[num:] - arr[:-num]
    arr = arr[num - 1:] / num
    return arr


# ======================================================================
def rolling_stat(
        arr,
        weights=1,
        stat_func=np.mean,
        stat_args=None,
        stat_kws=None,
        mode='valid',
        borders=None):
    """
    Calculate the rolling statistics on an array.

    This is calculated by running the specified statistics for each subset of
    the array of given size, including optional weightings.
    The moving average will be applied to the flattened array.

    This function differs from `running_stat` in that it should be faster but
    more memory demanding.
    Also the `stat_func` callable is required to accept an `axis` parameter.

    Args:
        arr (np.ndarray): The input array.
        weights (int|iterable): The running weights.
            If int, the number of elements to group in the 'running' axis and
            unity weights are used.
            The size of the weights array len(weights) must be such that
            len(weights) >= 1 and len(weights) <= len(array), otherwise the
            flattened array is returned.
            Note that these weights are
        stat_func (callable): Function to calculate in the 'running' axis.
            Must accept an `axis` parameter, which will be set to -1 on the
            flattened input.
        stat_args (tuple|list): Positional arguments passed to `stat_func`.
        stat_kws (dict): Keyword arguments passed to `stat_func`.
        mode (str): The output mode.
            Can be one of:
            - 'valid': only values inside the array are used.
            - 'same': must have the same size as the input.
            - 'full': the full output is provided.
        borders (str|complex|iterable[complex]|None): The border parameters.
            If int or float, the value is repeated at the borders.
            If iterable of int, float or complex, the first and last values are
            repeated to generate the head and tail, respectively.
            If str, the following values are accepted:
                - 'same': the array extrema are used to generate head / tail.
                - 'circ': the values are repeated periodically / circularly.
                - 'sym': the values are repeated periodically / symmetrically.

    Returns:
        arr (np.ndarray): The output array.

    Examples:
        >>> num = 8
        >>> arr = np.linspace(1, num, num)
        >>> all([np.allclose(
        ...                  moving_average(arr, n, mode=mode),
        ...                  rolling_stat(arr, n, mode=mode))
        ...      for n in range(num) for mode in ('valid', 'same', 'full')])
        True
        >>> rolling_stat(arr, 4, mode='same', borders=100)
        array([ 50.75,  26.5 ,   2.5 ,   3.5 ,   4.5 ,   5.5 ,   6.5 ,  30.25])
        >>> rolling_stat(arr, 4, mode='full', borders='same')
        array([ 1.  ,  1.25,  1.75,  2.5 ,  3.5 ,  4.5 ,  5.5 ,  6.5 ,  7.25,
                7.75,  8.  ])
        >>> rolling_stat(arr, 4, mode='full', borders='circ')
        array([ 5.5,  4.5,  3.5,  2.5,  3.5,  4.5,  5.5,  6.5,  5.5,  4.5,\
  3.5])
        >>> rolling_stat(arr, 4, mode='full', borders='sym')
        array([ 1.75,  1.5 ,  1.75,  2.5 ,  3.5 ,  4.5 ,  5.5 ,  6.5 ,  7.25,
                7.5 ,  7.25])
        >>> rolling_stat(arr, 4, mode='same', borders='circ')
        array([ 4.5,  3.5,  2.5,  3.5,  4.5,  5.5,  6.5,  5.5])
        >>> rolling_stat(arr, [1, 0.2])
        array([ 1.16666667,  2.16666667,  3.16666667,  4.16666667,  5.16666667,
                6.16666667,  7.16666667])
    """
    arr = arr.ravel()
    if isinstance(weights, int):
        weights = np.ones((weights,))
    else:
        # weights order needs to be inverted
        weights = np.array(weights)[::-1]
    num = len(weights) if isinstance(weights, np.ndarray) else 0
    size = len(arr)
    if size >= num > 1:
        # calculate how to extend the input array
        if borders is None:
            extension = np.zeros((num - 1,))
        elif borders == 'same':
            extension = np.concatenate(
                (np.ones((num - 1,)) * arr[-1],
                 np.ones((num - 1,)) * arr[0]))
        elif borders == 'circ':
            extension = arr
        elif borders == 'sym':
            extension = arr[::-1]
        elif isinstance(borders, (int, float, complex)):
            extension = np.ones((num - 1,)) * borders
        elif isinstance(borders, (tuple, float)):
            extension = np.concatenate(
                (np.ones((num - 1,)) * borders[-1],
                 np.ones((num - 1,)) * borders[0]))
        else:
            raise ValueError(
                '`borders={borders}` not understood'.format_map(locals()))

        # calculate generator for data and weights
        arr = np.concatenate((arr, extension))
        gen = np.zeros((size + num - 1, num))
        for i in range(num):
            gen[:, i] = np.roll(arr, i)[:size + num - 1]
        w_gen = np.stack([weights] * (size + num - 1))

        # calculate the running stats
        arr = stat_func(
            gen * w_gen,
            *(stat_args if stat_args else ()), axis=-1,
            **(stat_kws if stat_kws else {}))
        arr *= len(weights) / np.sum(weights)

        # adjust output according to mode
        if mode == 'valid':
            arr = arr[num - 1:-(num - 1)]
        elif mode == 'same':
            begin = (num - 1) // 2
            arr = arr[begin:begin + size]
    return arr


# ======================================================================
def running_stat(
        arr,
        weights=1,
        stat_func=np.mean,
        stat_args=None,
        stat_kws=None,
        mode='valid',
        borders=None):
    """
    Calculate the running statistics on an array.

    This is calculated by running the specified statistics for each subset of
    the array of given size, including optional weightings.
    The moving average will be applied to the flattened array.

    This function differs from `rolling_stat` in that it should be slower but
    less memory demanding.
    Also the `stat_func` callable is not required to accept an `axis`
    parameter.

    Args:
        arr (np.ndarray): The input array.
        weights (int|iterable): The running weights.
            If int, the number of elements to group in the 'running' axis and
            unity weights are used.
            The size of the weights array len(weights) must be such that
            len(weights) >= 1 and len(weights) <= len(array), otherwise the
            flattened array is returned.
            Note that these weights are
        stat_func (callable): Function to calculate in the 'running' axis.
        stat_args (tuple|list): Positional arguments passed to `stat_func`.
        stat_kws (dict): Keyword arguments passed to `stat_func`.
        mode (str): The output mode.
            Can be one of:
            - 'valid': only values inside the array are used.
            - 'same': must have the same size as the input.
            - 'full': the full output is provided.
        borders (str|complex|None): The border parameters.
            If int, float or complex, the value is repeated at the borders.
            If iterable of int, float or complex, the first and last values are
            repeated to generate the head and tail, respectively.
            If str, the following values are accepted:
                - 'same': the array extrema are used to generate head / tail.
                - 'circ': the values are repeated periodically / circularly.
                - 'sym': the values are repeated periodically / symmetrically.

    Returns:
        arr (np.ndarray): The output array.

    Examples:
        >>> num = 8
        >>> arr = np.linspace(1, num, num)
        >>> all([np.allclose(
        ...                  moving_average(arr, n, mode=mode),
        ...                  running_stat(arr, n, mode=mode))
        ...      for n in range(num) for mode in ('valid', 'same', 'full')])
        True
        >>> running_stat(arr, 4, mode='same', borders=100)
        array([ 50.75,  26.5 ,   2.5 ,   3.5 ,   4.5 ,   5.5 ,   6.5 ,  30.25])
        >>> running_stat(arr, 4, mode='same', borders='circ')
        array([ 4.5,  3.5,  2.5,  3.5,  4.5,  5.5,  6.5,  5.5])
        >>> running_stat(arr, 4, mode='full', borders='circ')
        array([ 5.5,  4.5,  3.5,  2.5,  3.5,  4.5,  5.5,  6.5,  5.5,  4.5,\
  3.5])
        >>> running_stat(arr, [1, 0.2])
        array([ 1.16666667,  2.16666667,  3.16666667,  4.16666667,  5.16666667,
                6.16666667,  7.16666667])
    """
    arr = arr.ravel()
    if isinstance(weights, int):
        weights = np.ones((weights,))
    else:
        weights = np.array(weights)
    num = len(weights) if isinstance(weights, np.ndarray) else 0
    size = len(arr)
    if size >= num > 1:
        # calculate how to extend the input array
        if borders is None:
            head = tail = np.zeros((num - 1,))
        elif borders == 'same':
            head = np.ones((num - 1,)) * arr[0]
            tail = np.ones((num - 1,)) * arr[-1]
        elif borders == 'circ':
            tail = arr[:num - 1]
            head = arr[-num + 1:]
        elif borders == 'sym':
            tail = arr[-num + 1:]
            head = arr[:num - 1]
        elif isinstance(borders, (int, float, complex)):
            head = tail = np.ones((num - 1,)) * borders
        elif isinstance(borders, (tuple, float)):
            head = np.ones((num - 1,)) * borders[0]
            tail = np.ones((num - 1,)) * borders[-1]
        else:
            raise ValueError(
                '`borders={borders}` not understood'.format_map(locals()))

        # calculate generator for data and weights
        gen = np.concatenate((head, arr, tail))
        # print(gen)
        arr = np.zeros((len(gen) - num + 1))
        for i in range(len(arr)):
            arr[i] = stat_func(
                gen[i:i + num] * weights,
                *(stat_args if stat_args else ()),
                **(stat_kws if stat_kws else {}))
        arr *= len(weights) / np.sum(weights)

        # adjust output according to mode
        if mode == 'valid':
            arr = arr[num - 1:-(num - 1)]
        elif mode == 'same':
            begin = (num - 1) // 2
            arr = arr[begin:begin + size]
    return arr


# import timeit
#
# z = timeit.repeat(
#     'moving_mean(np.random.random(10000), 10)',
#     'from __main__ import moving_mean; import numpy as np', number=100)
#
# a = timeit.repeat(
#     'moving_average(np.random.random(10000), 10)',
#     'from __main__ import moving_average; import numpy as np', number=100)
#
# b = timeit.repeat(
#     'rolling_stat(np.random.random(10000), 10)',
#     'from __main__ import rolling_stat; import numpy as np', number=100)
#
# c = timeit.repeat(
#     'running_stat(np.random.random(10000), 10)',
#     'from __main__ import running_stat; import numpy as np', number=100)
#
# print(z, a, b, c)


# ======================================================================
def polar2complex(modulus, phase):
    """
    Calculate complex number from the polar form:
    z = R * exp(i * phi) = R * cos(phi) + i * R * sin(phi).

    Args:
        modulus (float|np.ndarray): The modulus R of the complex number.
        phase (float|np.ndarray): The argument phi of the complex number.

    Returns:
        z (complex|np.ndarray): The complex number z = R * exp(i * phi).
    """
    return modulus * np.exp(1j * phase)


# ======================================================================
def cartesian2complex(real, imag):
    """
    Calculate the complex number from the cartesian form: z = z' + i * z".

    Args:
        real (float|np.ndarray): The real part z' of the complex number.
        imag (float|np.ndarray): The imaginary part z" of the complex number.

    Returns:
        z (complex|np.ndarray): The complex number: z = z' + i * z".
    """
    return real + 1j * imag


# ======================================================================
def complex2cartesian(z):
    """
    Calculate the real and the imaginary part of a complex number.

    Args:
        z (complex|np.ndarray): The complex number or array: z = z' + i * z".

    Returns:
        tuple[float|np.ndarray]:
         - real (float|np.ndarray): The real part z' of the complex number.
         - imag (float|np.ndarray): The imaginary part z" of the complex
         number.
    """
    return np.real(z), np.imag(z)


# ======================================================================
def complex2polar(z):
    """
    Calculate the real and the imaginary part of a complex number.

    Args:
        z (complex|np.ndarray): The complex number or array: z = z' + i * z".

    Returns:
        tuple[float]:
         - modulus (float|np.ndarray): The modulus R of the complex number.
         - phase (float|np.ndarray): The phase phi of the complex number.
    """
    return np.abs(z), np.angle(z)


# ======================================================================
def polar2cartesian(modulus, phase):
    """
    Calculate the real and the imaginary part of a complex number.

    Args:
        modulus (float|np.ndarray): The modulus R of the complex number.
        phase (float|np.ndarray): The phase phi of the complex number.

    Returns:
        tuple[float]:
         - real (float|np.ndarray): The real part z' of the complex number.
         - imag (float|np.ndarray): The imaginary part z" of the complex
         number.
    """
    return modulus * np.cos(phase), modulus * np.sin(phase)


# ======================================================================
def cartesian2polar(real, imag):
    """
    Calculate the real and the imaginary part of a complex number.

    Args:
        real (float): The real part z' of the complex number.
        imag (float): The imaginary part z" of the complex number.

    Returns:
        tuple[float]:
         - modulus (float): The modulus R of the complex number.
         - argument (float): The phase phi of the complex number.
    """
    return np.sqrt(real ** 2 + imag ** 2), np.arctan2(real, imag)


# ======================================================================
def filter_cx(
        arr,
        filter_func,
        filter_args=None,
        filter_kws=None,
        mode='cartesian'):
    """
    Calculate a non-complex function on a complex input array.

    Args:
        arr (np.ndarray): The input array.
        filter_func (callable): The function used to filter the input.
            Requires the first arguments to be an `np.ndarray`.
        filter_args (tuple|None): Positional arguments of `filter_func`.
        filter_kws (dict|None): Keyword arguments of `filter_func`.
        mode (str): Complex calculation mode.
            Available:
             - 'cartesian': apply the n-dim filter to real and imaginary parts.
             - 'polar': apply the n-dim filter to the magnitude and phase.
            If unknown, uses defalt.

    Returns:
        arr (np.ndarray): The filtered complex array.
    """
    if mode:
        mode = mode.lower()
    if not filter_args:
        filter_args = ()
    if not filter_kws:
        filter_kws = {}
    if mode == 'cartesian':
        arr = (
            filter_func(arr.real, *filter_args, **filter_kws) +
            1j * filter_func(arr.imag, *filter_args, **filter_kws))
    elif mode == 'polar':
        arr = (
            filter_func(np.abs(arr), *filter_args, **filter_kws) *
            np.exp(
                1j * filter_func(np.angle(arr), *filter_args, **filter_kws)))
    else:
        warnings.warn(
            'Mode `{}` not known'.format(mode) + ' Using default.')
        arr = filter_cx(arr, filter_func, filter_args, filter_kws)
    return arr


# ======================================================================
def marginal_sep_elbow(items):
    """
    Determine the marginal separation using the elbow method.

    Graphically, this is displayed as an elbow in the plot.
    Mathematically, this is defined as the first item whose (signed) global
    slope is smaller than the (signed) local slope.

    Args:
        items (iterable): The collection of items to inspect.
            The input must be already sorted non-negative values.

    Returns:
        index (int): The position of the marginal separation value.
            If the marginal separation is not found, returns -1.

    Examples:
        >>> items = (100, 90, 70, 60, 50, 30, 20, 5, 4, 3, 2, 1)
        >>> marginal_sep_elbow(items)
        8
        >>> items = (100, 90, 70, 60, 50, 30, 20, 5)
        >>> marginal_sep_elbow(items)
        -1
    """
    if is_increasing(items):
        sign = -1
    elif is_decreasing(items):
        sign = 1
    else:
        sign = None
    if sign:
        index = -1
        for i_, item in enumerate(items[1:]):
            i = i_ + 1
            local_slope = item - items[i_]
            global_slope = item - items[0] / i
            if sign * global_slope < sign * local_slope:
                index = i
                break
    else:
        index = -1
    return index


# ======================================================================
def marginal_sep_quad(items):
    """
    Determine the marginal separation using the quadrature method.

    Mathematically, this is defined as the first item whose value is smaller
    than the sum of the differences of all following items.

    Args:
        items (iterable): The collection of items to inspect.
            The input must be already sorted non-negative values.

    Returns:
        index (int): The position of the marginal separation value.
            If the marginal separation is not found, returns -1.

    Examples:
        >>> items = (100, 90, 70, 50, 30, 20, 5, 2, 1)
        >>> marginal_sep_quad(items)
        5
    """
    if is_increasing(items):
        sign = -1
    elif is_decreasing(items):
        sign = 1
    else:
        sign = None
    if sign:
        index = np.where(
            items[:-1] + sign * np.cumsum(np.diff(items)[::-1]) < 0)[0]
        index = int(index[0]) + 1 if len(index) > 0 else -1
    else:
        index = -1
    return index


# ======================================================================
def marginal_sep_quad_weight(items):
    """
    Determine the marginal separation using the weighted quadrature.

    Mathematically, this is defined as the first item whose value is smaller
    than the sum of the differences of all following items weighted by the
    number of items already considered.

    Args:
        items (iterable): The collection of items to inspect.
            The input must be already sorted non-negative values.

    Returns:
        index (int): The position of the marginal separation value.
            If the marginal separation is not found, returns -1.

    Examples:
        >>> items = (100, 90, 70, 50, 30, 20, 5, 2, 1)
        >>> marginal_sep_quad_weight(items)
        7
    """
    if is_increasing(items):
        sign = -1
    elif is_decreasing(items):
        sign = 1
    else:
        sign = None
    if sign:
        index = np.where(
            items[:-1] + sign * np.cumsum(np.diff(items)[::-1]) /
            np.arange(1, len(items)) < 0)[0]
        index = index[0] + 1 if len(index) else -1
    else:
        index = -1
    return index


# ======================================================================
def marginal_sep_quad_inv_weight(items):
    """
    Determine the marginal separation using the inverse weighted quadrature.

    Mathematically, this is defined as the first item whose value is smaller
    than the sum of the differences of all following items weighted by the
    number of items to be considered.

    Args:
        items (iterable): The collection of items to inspect.
            The input must be already sorted non-negative values.

    Returns:
        index (int): The position of the marginal separation value.
            If the marginal separation is not found, returns -1.

    Examples:
        >>> items = (100, 90, 70, 50, 30, 20, 5, 2, 1)
        >>> marginal_sep_quad_inv_weight(items)
        7
    """
    if is_increasing(items):
        sign = -1
    elif is_decreasing(items):
        sign = 1
    else:
        sign = None
    if sign:
        index = np.where(
            items[:-1] + sign * np.cumsum(np.diff(items)[::-1]) /
            np.arange(len(items), 1, -1) < 0)[0]
        index = index[0] + 1 if len(index) else -1
    else:
        index = -1
    return index


# ======================================================================
def calc_stats(
        arr,
        removes=(np.nan, np.inf, -np.inf),
        val_interval=None,
        save_path=None,
        title=None,
        compact=False):
    """
    Calculate array statistical information (min, max, avg, std, sum, num).

    Args:
        arr (np.ndarray): The array to be investigated.
        removes (iterable): Values to remove.
            If empty, no values will be removed.
        val_interval (tuple): The (min, max) values interval.
        save_path (str|None): The path to which the plot is to be saved.
            If None, no output.
        title (str|None): If title is not None, stats are printed to screen.
        compact (bool): Use a compact format string for displaying results.

    Returns:
        stats_dict (dict): Dictionary of statistical values.
            Statistical parameters calculated:
                - 'min': minimum value
                - 'max': maximum value
                - 'avg': average or mean
                - 'std': standard deviation
                - 'sum': summation
                - 'num': number of elements

    Examples:
        >>> a = np.arange(2)
        >>> d = calc_stats(a)
        >>> tuple(sorted(d.items()))
        (('avg', 0.5), ('max', 1), ('min', 0), ('num', 2), ('std', 0.5),\
 ('sum', 1))
        >>> a = np.arange(200)
        >>> d = calc_stats(a)
        >>> tuple(sorted(d.items()))
        (('avg', 99.5), ('max', 199), ('min', 0), ('num', 200),\
 ('std', 57.734305226615483), ('sum', 19900))
    """
    stats_dict = {
        'avg': None, 'std': None,
        'min': None, 'max': None,
        'sum': None, 'num': None}
    arr = ravel_clean(arr, removes)
    if val_interval is None and len(arr) > 0:
        val_interval = minmax(arr)
    if len(arr) > 0:
        arr = arr[arr >= val_interval[0]]
        arr = arr[arr <= val_interval[1]]
    if len(arr) > 0:
        stats_dict = {
            'avg': np.mean(arr),
            'std': np.std(arr),
            'min': np.min(arr),
            'max': np.max(arr),
            'sum': np.sum(arr),
            'num': np.size(arr), }
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
def ndim_slice(
        arr,
        axes=0,
        indexes=None):
    """
    Slice a M-dim sub-array from an N-dim array (with M < N).

    Args:
        arr (np.ndarray): The input N-dim array
        axes (iterable[int]|int): The slicing axis
        indexes (iterable[int|float|None]|None): The slicing index.
            If None, mid-value is taken.
            Otherwise, its length must match that of axes.
            If an element is None, again the mid-value is taken.
            If an element is a number between 0 and 1, it is interpreted
            as relative to the size of the array for corresponding axis.
            If an element is an integer, it is interpreted as absolute and must
            be smaller than size of the array for the corresponding axis.

    Returns:
        sliced (np.ndarray): The sliced M-dim sub-array

    Raises:
        ValueError: if index is out of bounds

    Examples:
        >>> arr = np.arange(2 * 3 * 4).reshape((2, 3, 4))
        >>> ndim_slice(arr, 2, 1)
        array([[ 1,  5,  9],
               [13, 17, 21]])
        >>> ndim_slice(arr, 1, 2)
        array([[ 8,  9, 10, 11],
               [20, 21, 22, 23]])
        >>> ndim_slice(arr, 0, 0)
        array([[ 0,  1,  2,  3],
               [ 4,  5,  6,  7],
               [ 8,  9, 10, 11]])
        >>> ndim_slice(arr, 0, 1)
        array([[12, 13, 14, 15],
               [16, 17, 18, 19],
               [20, 21, 22, 23]])
        >>> ndim_slice(arr, (0, 1), None)
        array([16, 17, 18, 19])
    """
    # initialize slice index
    slab = [slice(None)] * arr.ndim
    # ensure index is meaningful
    axes = auto_repeat(axes, 1)
    if indexes is None:
        indexes = auto_repeat(None, len(axes))
    else:
        indexes = auto_repeat(indexes, 1)
    indexes = list(indexes)
    for i, (index, axis) in enumerate(zip(indexes, axes)):
        if index is None:
            indexes[i] = index = 0.5
        if isinstance(index, float) and index < 1.0:
            indexes[i] = int(arr.shape[axis] * index)
    # check index
    if any([(index >= arr.shape[axis]) or (index < 0)
            for index, axis in zip(indexes, axes)]):
        raise ValueError('Invalid array index in the specified direction')
    # determine slice index
    for index, axis in zip(indexes, axes):
        slab[axis] = index
    # print(slab)  # debug
    # slice the array
    return arr[slab]


# ======================================================================
def rel_err(
        arr1,
        arr2,
        use_average=False):
    """
    Calculate the element-wise relative error

    Args:
        arr1 (np.ndarray): The input array with the exact values
        arr2 (np.ndarray): The input array with the approximated values
        use_average (bool): Use the input arrays average as the exact values

    Returns:
        array (ndarray): The relative error array

    Examples:
        >>> arr1 = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        >>> arr2 = np.array([1.1, 2.1, 3.1, 4.1, 5.1, 6.1])
        >>> rel_err(arr1, arr2)
        array([ 0.1       ,  0.05      ,  0.03333333,  0.025     ,  0.02      ,
                0.01666667])
        >>> rel_err(arr1, arr2, True)
        array([ 0.0952381 ,  0.04878049,  0.03278689,  0.02469136,  0.01980198,
                0.01652893])
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
    arr *= mask
    arr[mask] = arr[mask] / div[mask]
    return arr


# ======================================================================
def euclid_dist(
        arr1,
        arr2,
        unsigned=True):
    """
    Calculate the element-wise correlation euclidean distance.

    This is the distance D between the identity line and the point of
    coordinates given by intensity:
        \[D = abs(A2 - A1) / sqrt(2)\]

    Args:
        arr1 (ndarray): The first array
        arr2 (ndarray): The second array
        unsigned (bool): Use signed distance

    Returns:
        array (ndarray): The resulting array

    Examples:
        >>> arr1 = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        >>> arr2 = np.array([-1.0, -2.0, -3.0, -4.0, -5.0, -6.0])
        >>> euclid_dist(arr1, arr2)
        array([ 1.41421356,  2.82842712,  4.24264069,  5.65685425,  7.07106781,
                8.48528137])
        >>> euclid_dist(arr1, arr2, False)
        array([-1.41421356, -2.82842712, -4.24264069, -5.65685425, -7.07106781,
               -8.48528137])
    """
    array = (arr2 - arr1) / np.sqrt(2.0)
    if unsigned:
        array = np.abs(array)
    return array


# ======================================================================
if __name__ == '__main__':
    msg(__doc__.strip())
    doctest.testmod()

# ======================================================================
elapsed(os.path.basename(__file__))
