#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pymrt.utils: generic basic utilities.
"""

# ======================================================================
# :: Future Imports
from __future__ import (
    division, absolute_import, print_function, unicode_literals, )

# ======================================================================
# :: Python Standard Library Imports
import os  # Miscellaneous operating system interfaces
import io  # Core tools for working with streams
import sys  # System-specific parameters and functions
import math  # Mathematical functions
import itertools  # Functions creating iterators for efficient looping
import functools  # Higher-order functions and operations on callable objects
import collections  # Container datatypes
import subprocess  # Subprocess management
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
import random  # Generate pseudo-random numbers

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
from pymrt import elapsed, report
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
    'gzip': 'gz',
    'bzip': 'bz2',
    'lzip': 'lz',
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
        in_file,
        dtype,
        mode='@',
        num_blocks=1,
        offset=None,
        whence=io.SEEK_SET):
    """
    Read data from stream.

    Args:
        in_file (str|file): The input file.
            If str, the file is open for reading (as binary).
        offset (int|None): The offset where to start reading.
        dtype (str): The data type to read.
            Accepted values are:
             - 'bool': boolean type (same as: '?', 1B)
             - 'char': signed char type (same as: 'b', 1B)
             - 'uchar': unsigned char type (same as: 'B', 1B)
             - 'short': signed short int type (same as: 'h', 2B)
             - 'ushort': unsigned short int  type (same as: 'H', 2B)
             - 'int': signed int type (same as: 'i', 4B)
             - 'uint': unsigned int type (same as: 'I', 4B)
             - 'long': signed long type (same as: 'l', 4B)
             - 'ulong': unsigned long type (same as: 'L', 4B)
             - 'llong': signed long long type (same as: 'q', 8B)
             - 'ullong': unsigned long long type (same as: 'Q', 8B)
             - 'float': float type (same as: 'f', 4B)
             - 'double': double type (same as: 'd', 8B)
             - 'str': c-str type (same as: 's', 'p')
            See Python's `struct` module for more information.
        num_blocks (int): The number of blocks to read.
        mode (str): Determine the byte order, size and alignment.
            Accepted values are:
             - '@': endianness: native,  size: native,   align: native.
             - '=': endianness:	native,  size: standard, align: none.
             - '<': endianness:	little,  size: standard, align: none.
             - '>': endianness:	big,     size: standard, align: none.
             - '!': endianness:	network, size: standard, align: none.
        whence (int): Where to reference the offset.
            Accepted values are:
             - '0': absolute file positioning.
             - '1': seek relative to the current position.
             - '2': seek relative to the file's end.

    Returns:
        data (tuple): The data read.
    """
    if isinstance(in_file, str):
        file_obj = open(in_file, 'rb')
    else:
        file_obj = in_file
    if offset is not None:
        file_obj.seek(offset, whence)
    fmt = mode + str(num_blocks) + DTYPE_STR[dtype]
    read_size = struct.calcsize(fmt)
    data = struct.unpack_from(fmt, file_obj.read(read_size))
    if isinstance(in_file, str):
        file_obj.close()
    return data


# ======================================================================
def read_cstr(
        in_file,
        offset=None,
        whence=io.SEEK_SET):
    """
    Read a C-type string from file.

    Args:
        in_file (str|file): The input file.
            If str, the file is open for reading (as binary).
        offset (int|None): The offset where to start reading.
        whence (int): Where to reference the offset.
            Accepted values are:
             - '0': absolute file positioning.
             - '1': seek relative to the current position.
             - '2': seek relative to the file's end.

    Returns:
        text (str): The string read.
    """
    if isinstance(in_file, str):
        file_obj = open(in_file, 'rb')
    else:
        file_obj = in_file
    if offset is not None:
        file_obj.seek(offset, whence)
    buffer = []
    while True:
        c = file_obj.read(1).decode('ascii')
        if c is None or c == '\0':
            break
        else:
            buffer.append(c)
    text = ''.join(buffer)
    if isinstance(in_file, str):
        file_obj.close()
    return text


# ======================================================================
def auto_repeat(
        obj,
        n,
        force=False,
        check=False):
    """
    Automatically repeat the specified object n times.

    If the object is not Iterable, a tuple with the specified size is returned.
    If the object is Iterable, the object is left untouched.

    Args:
        obj: The object to operate with.
        n (int): The length of the output object.
        force (bool): Force the repetition, even if the object is Iterable.
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
def flatten(
        items,
        avoid=(str, bytes),
        max_depth=-1):
    """
    Recursively flattens nested Iterables.

    The maximum depth is limited by Python's recursion limit.

    Args:
        items (Iterable[Iterable]): The input items.
        avoid (Iterable): Data types that will not be flattened.
        max_depth (int): Maximum depth to reach. Negative for unlimited.

    Yields:
        item (any): The next non-Iterable item of the flattened items.

    Examples:
        >>> ll = [[1, 2, 3], [4, 5, 6], [7], [8, 9]]
        >>> list(flatten(ll))
        [1, 2, 3, 4, 5, 6, 7, 8, 9]
        >>> list(flatten(ll)) == list(itertools.chain.from_iterable(ll))
        True
        >>> ll = [ll, ll]
        >>> list(flatten(ll))
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        >>> list(flatten([1, 2, 3]))
        [1, 2, 3]
        >>> list(flatten(['best', ['function', 'ever']]))
        ['best', 'function', 'ever']
        >>> ll2 = [[(1, 2, 3), (4, 5)], [(1, 2), (3, 4, 5)], ['1, 2', [6, 7]]]
        >>> list(flatten(ll2, avoid=(tuple, str)))
        [(1, 2, 3), (4, 5), (1, 2), (3, 4, 5), '1, 2', 6, 7]
        >>> list(flatten([['best', 'func'], 'ever'], avoid=None))
        ['b', 'e', 's', 't', 'f', 'u', 'n', 'c', 'e', 'v', 'e', 'r']
    """
    for item in items:
        try:
            no_expand = avoid and isinstance(item, avoid)
            if no_expand or max_depth == 0 or item == next(iter(item)):
                raise TypeError
        except TypeError:
            yield item
        else:
            for i in flatten(item, avoid, max_depth - 1):
                yield i


# ======================================================================
def prod(items):
    """
    Calculate the cumulative product of an arbitrary number of items.

    This is similar to `sum`, but uses product instead of addition.

    Args:
        items (Iterable): The input items.

    Returns:
        result: The cumulative product of `items`.

    Examples:
        >>> prod([2] * 10)
        1024
    """
    return functools.reduce(lambda x, y: x * y, items)


# ======================================================================
def uniques(items):
    """
    Get unique items (keeping order of appearance).

    If the order of appearance is not important, use `set()`.

    Args:
        items (Iterable): The input items.

    Yields:
        item: Unique items.

    Examples:
        >>> items = (5, 4, 3, 2, 1, 1, 2, 3, 4, 5, 3, 2, 4, 2, 4, 1, 1)
        >>> tuple(uniques(items))
        (5, 4, 3, 2, 1)
        >>> tuple(set(items))
        (1, 2, 3, 4, 5)
        >>> sorted(set(items)) == sorted(uniques(items))
        True
    """
    seen = set()
    for item in items:
        if item not in seen and not seen.add(item):
            yield item


# ======================================================================
def replace_iter(
        items,
        condition,
        replace=None,
        cycle=True):
    """
    Replace items matching a specific condition.

    Args:
        items (Iterable):
        condition (callable):
        replace (any|Iterable|callable): The replacement.
            If Iterable, its elements are used for replacement.
            If callable, it is applied to the elements matching `condition`.
            Otherwise,
        cycle (bool): Cycle through the replace.
            If True and `replace` is Iterable, its elements are cycled through.
            Otherwise `items` beyond last replacement are lost.

    Yields:
        item: The next item from items or its replacement.

    Examples:
        >>> ll = list(range(10))
        >>> list(replace_iter(ll, lambda x: x % 2 == 0))
        [None, 1, None, 3, None, 5, None, 7, None, 9]
        >>> list(replace_iter(ll, lambda x: x % 2 == 0, lambda x: x ** 2))
        [0, 1, 4, 3, 16, 5, 36, 7, 64, 9]
        >>> list(replace_iter(ll, lambda x: x % 2 == 0, 100))
        [100, 1, 100, 3, 100, 5, 100, 7, 100, 9]
        >>> list(replace_iter(ll, lambda x: x % 2 == 0, range(10, 0, -1)))
        [10, 1, 9, 3, 8, 5, 7, 7, 6, 9]
        >>> list(replace_iter(ll, lambda x: x % 2 == 0, range(10, 8, -1)))
        [10, 1, 9, 3, 10, 5, 9, 7, 10, 9]
        >>> list(replace_iter(
        ...     ll, lambda x: x % 2 == 0, range(10, 8, -1), False))
        [10, 1, 9, 3]
    """
    if not callable(replace):
        try:
            replace = iter(replace)
        except TypeError:
            replace = (replace,)
            cycle = True
        if cycle:
            replace = itertools.cycle(replace)
    for item in items:
        if not condition(item):
            yield item
        else:
            yield replace(item) if callable(replace) else next(replace)


# ======================================================================
def combine_iter_len(
        items,
        combine=max):
    """
    Determine the maximum length of an item within a collection of items.

    Args:
        items (Iterable): The collection of items to inspect.
        combine (callable): The combination method.

    Returns:
        num (int): The combined length of the collection.
            If none of the items are Iterable, the result is `1`.

    Examples:
        >>> a = list(range(10))
        >>> b = tuple(range(5))
        >>> c = set(range(20))
        >>> combine_iter_len((a, b, c))
        20
        >>> combine_iter_len((a, b, c), min)
        5
        >>> combine_iter_len((1, a))
        10
    """
    num = None
    for val in items:
        try:
            iter(val)
        except TypeError:
            pass
        else:
            if num is None:
                num = len(val)
            else:
                num = combine(len(val), num)
    if num is None:
        num = 1
    return num


# ======================================================================
def grouping(
        items,
        splits):
    """
    Generate a tuple of grouped items.

    Args:
        items (Iterable): The input items.
        splits (int|Iterable[int]): Grouping information.
            If Iterable, each group (except the last) has the number of
            elements specified.
            If int, all groups (except the last, which may have less items)
            has the same number of elements.

    Returns:
        groups (tuple[Iterable]): Grouped items from the source.

    Examples:
        >>> l = list(range(10))
        >>> tuple(grouping(l, 4))
        ([0, 1, 2, 3], [4, 5, 6, 7], [8, 9])
        >>> tuple(grouping(l, (2, 3)))
        ([0, 1], [2, 3, 4], [5, 6, 7, 8, 9])
        >>> tuple(grouping(l, (2, 4, 1)))
        ([0, 1], [2, 3, 4, 5], [6], [7, 8, 9])
        >>> tuple(grouping(l, (2, 4, 1, 20)))
        ([0, 1], [2, 3, 4, 5], [6], [7, 8, 9])
        >>> tuple(grouping(tuple(l), 2))
        ((0, 1), (2, 3), (4, 5), (6, 7), (8, 9))
    """
    if isinstance(splits, int):
        splits = auto_repeat(splits, len(items) // splits)

    num_items = len(items)
    if sum(splits) >= num_items:
        splits = splits[:-1]
    index = (0,) + tuple(itertools.accumulate(splits)) + (num_items,)
    num = len(index) - 1
    for i in range(num):
        yield items[index[i]:index[i + 1]]


# ======================================================================
def chunks(
        items,
        n,
        mode='+',
        balanced=True):
    """
    Yield items into approximately N equally sized chunks.

    If the number of items does not allow chunks of the same size, the chunks
    are determined depending on the values of `balanced`

    Args:
        items (Iterable): The input items.
        n (int): Approximate number of chunks.
            The exact number depends on the value of `mode`.
        mode (str): Determine which approximation to use.
            If str, valid inputs are:
             - 'upper', '+': at most `n` chunks are generated.
             - 'lower', '-': at least `n` chunks are genereated.
             - 'closest', '~': the number of chunks is `n` or `n + 1`
               depending on which gives the most evenly distributed chunks
               sizes.
        balanced (bool): Produce balanced chunks.
            If True, the size of any two chunks is not larger than one.
            Otherwise, the first chunks except the last have the same size.
            This has no effect if the number of items is a multiple of `n`.

    Returns:
        groups (tuple[Iterable]): Grouped items from the source.

    Examples:
        >>> l = list(range(10))
        >>> tuple(chunks(l, 5))
        ([0, 1], [2, 3], [4, 5], [6, 7], [8, 9])
        >>> tuple(chunks(l, 2))
        ([0, 1, 2, 3, 4], [5, 6, 7, 8, 9])
        >>> tuple(chunks(l, 3))
        ([0, 1, 2, 3], [4, 5, 6], [7, 8, 9])
        >>> tuple(chunks(l, 4))
        ([0, 1, 2], [3, 4, 5], [6, 7], [8, 9])
        >>> tuple(chunks(l, -1))
        ([0, 1, 2, 3, 4, 5, 6, 7, 8, 9],)
        >>> tuple(chunks(l, 3, balanced=False))
        ([0, 1, 2, 3], [4, 5, 6, 7], [8, 9])
        >>> tuple(chunks(l, 3, '-'))
        ([0, 1, 2], [3, 4, 5], [6, 7], [8, 9])
        >>> tuple(chunks(l, 3, '-', False))
        ([0, 1, 2], [3, 4, 5], [6, 7, 8], [9])
        >>> tuple(chunks(list(range(10)), 3, '~'))
        ([0, 1, 2], [3, 4, 5], [6, 7], [8, 9])
        >>> tuple(chunks(list(range(10)), 3, '~', False))
        ([0, 1, 2], [3, 4, 5], [6, 7, 8], [9])
    """
    if mode in ('upper', '+'):
        approx = math.ceil
    elif mode in ('lower', '-'):
        approx = math.floor
    elif mode in ('closest', '~'):
        approx = round
    else:
        raise ValueError('Invalid mode `{mode}`'.format(mode=mode))
    n = max(1, n)
    split = int(approx(len(items) / n))
    if balanced and 0 < len(items) % split <= split // 2:
        k = len(items) // split + 1
        q = -len(items) % split
        split = (split,) * (k - q) + (split - 1,) * q
    return grouping(items, split)


# ======================================================================
def partitions(
        items,
        k,
        container=tuple):
    """
    Generate all k-partitions for the items.

    Args:
        items (Iterable): The input items.
        k (int): The number of splitting partitions.
            Each group has exactly `k` elements.
        container (callable): The group container.

    Yields:
        partition (tuple[Iterable]]): The grouped items.
            Each partition contains `k` grouped items from the source.

    Examples:
        >>> tuple(partitions(tuple(range(3)), 2))
        (((0,), (1, 2)), ((0, 1), (2,)))
        >>> tuple(partitions(tuple(range(3)), 3))
        (((0,), (1,), (2,)),)
        >>> tuple(partitions(tuple(range(4)), 3))
        (((0,), (1,), (2, 3)), ((0,), (1, 2), (3,)), ((0, 1), (2,), (3,)))
    """
    num = len(items)
    indexes = tuple(
        (0,) + tuple(index) + (num,)
        for index in itertools.combinations(range(1, num), k - 1))
    for index in indexes:
        yield tuple(
            container(
                items[index[i]:index[i + 1]] for i in range(k)))


# ======================================================================
def random_unique_combinations_k(items, k, pseudo=False):
    """
    Obtain a number of random unique combinations of a sequence of sequences.

    Args:
        items (Sequence[Sequence]): The input sequence of sequences.
        k (int): The number of random unique combinations to obtain.
        pseudo (bool): Generate random combinations somewhat less randomly.
            If True, the memory requirements for intermediate steps will
            be significantly lower (but still all `k` items are required to
            fit in memory).

    Yields:
        combination (Sequence): The next random unique combination.

    Examples:
        >>> import string
        >>> max_lens = [i for i in range(2, 10)]
        >>> items = [string.ascii_lowercase[:max_len] for max_len in max_lens]
        >>> random.seed(0)
        >>> num = 10
        >>> for i in random_unique_combinations_k(items, num):
        ...     print(i)
        ('b', 'a', 'd', 'a', 'd', 'a', 'a', 'f')
        ('a', 'a', 'c', 'c', 'b', 'f', 'd', 'f')
        ('b', 'b', 'b', 'e', 'c', 'b', 'e', 'a')
        ('a', 'b', 'a', 'b', 'd', 'g', 'c', 'd')
        ('b', 'c', 'd', 'd', 'b', 'b', 'f', 'g')
        ('a', 'a', 'b', 'a', 'f', 'd', 'c', 'g')
        ('a', 'c', 'd', 'a', 'f', 'a', 'c', 'f')
        ('b', 'c', 'd', 'a', 'f', 'd', 'h', 'd')
        ('a', 'c', 'b', 'b', 'a', 'e', 'b', 'g')
        ('a', 'c', 'c', 'b', 'e', 'b', 'f', 'e')
        >>> max_lens = [i for i in range(2, 4)]
        >>> items = [string.ascii_uppercase[:max_len] for max_len in max_lens]
        >>> random.seed(0)
        >>> num = 10
        >>> for i in random_unique_combinations_k(items, num):
        ...     print(i)
        ('B', 'B')
        ('B', 'C')
        ('A', 'A')
        ('B', 'A')
        ('A', 'B')
        ('A', 'C')
    """
    if pseudo:
        # randomize generators
        comb_gens = list(items)
        for num, comb_gen in enumerate(comb_gens):
            random.shuffle(list(comb_gens[num]))
        # get the first `k` combinations
        combinations = list(itertools.islice(itertools.product(*comb_gens), k))
        random.shuffle(combinations)
        for combination in itertools.islice(combinations, k):
            yield tuple(combination)
    else:
        max_lens = [len(list(item)) for item in items]
        max_k = prod(max_lens)
        try:
            for num in random.sample(range(max_k), min(k, max_k)):
                indexes = []
                for max_len in max_lens:
                    indexes.append(num % max_len)
                    num = num // max_len
                yield tuple(item[i] for i, item in zip(indexes, items))
        except OverflowError:
            # use `set` to ensure uniqueness
            index_combs = set()
            # make sure that with the chosen number the next loop can exit
            # WARNING: if `k` is too close to the total number of combinations,
            # it may take a while until the next valid combination is found
            while len(index_combs) < min(k, max_k):
                index_combs.add(tuple(
                    random.randint(0, max_len - 1) for max_len in max_lens))
            # make sure their order is shuffled
            # (`set` seems to sort its content)
            index_combs = list(index_combs)
            random.shuffle(index_combs)
            for index_comb in itertools.islice(index_combs, k):
                yield tuple(item[i] for i, item in zip(index_comb, items))


# ======================================================================
def unique_permutations(
        items,
        container=tuple):
    """
    Yield unique permutations of items in an efficient way.

    Args:
        items (Iterable): The input items.
        container (callable)

    Yields:
        items (Iterable): The next unique permutation of the items.

    Examples:
        >>> list(unique_permutations([0, 0, 0]))
        [(0, 0, 0)]
        >>> list(unique_permutations([0, 0, 2]))
        [(0, 0, 2), (0, 2, 0), (2, 0, 0)]
        >>> list(unique_permutations([0, 1, 2]))
        [(0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)]
        >>> p1 = sorted(unique_permutations((0, 1, 2, 3, 4)))
        >>> p2 = sorted(itertools.permutations((0, 1, 2, 3, 4)))
        >>> p1 == p2
        True

    References:
        - Donald Knuth, The Art of Computer Programming, Volume 4, Fascicle
          2: Generating All Permutations.
    """
    indexes = range(len(items) - 1, -1, -1)
    items = sorted(items)
    while True:
        if callable(container):
            yield container(items)
        else:
            yield items.copy()

        for k in indexes[1:]:
            if items[k] < items[k + 1]:
                break
        else:
            return

        k_val = items[k]
        for i in indexes:
            if k_val < items[i]:
                break

        items[k], items[i] = items[i], items[k]
        items[k + 1:] = items[-1:k:-1]


# ======================================================================
def unique_partitions(
        items,
        k):
    """
    Generate all k-partitions for all unique permutations of the items.

    Args:
        items (Iterable): The input items.
        k (int): The number of splitting partitions.
            Each group has exactly `k` elements.

    Yields:
        partitions (tuple[tuple[tuple]]]): The items partitions.
            More precisely, all partitions of size `num` for each unique
            permutations of `items`.

    Examples:
        >>> list(unique_partitions([0, 1], 2))
        [(((0,), (1,)),), (((1,), (0,)),)]

    """
    for permutations in unique_permutations(items):
        yield tuple(partitions(tuple(permutations), k))


# ======================================================================
def shuffle_on_axis(arr, axis=-1):
    """
    Shuffle the elements of the array separately along the specified axis.

    By contrast `numpy.random.shuffle()` shuffle **by** axis and only on the
    first axis.

    Args:
        arr (np.ndarray): The input array.
        axis (int): The axis along which to shuffle.

    Returns:
        result (np.ndarray): The shuffled array.

    Examples:
        >>> np.random.seed(0)
        >>> shape = 2, 3, 4
        >>> arr = np.arange(prod(shape)).reshape(shape)
        >>> shuffle_on_axis(arr.copy())
        array([[[ 1,  0,  2,  3],
                [ 6,  4,  5,  7],
                [10,  8, 11,  9]],
        <BLANKLINE>
               [[12, 15, 13, 14],
                [18, 17, 16, 19],
                [21, 20, 23, 22]]])
        >>> shuffle_on_axis(arr.copy(), 0)
        array([[[ 0, 13,  2, 15],
                [16,  5,  6, 19],
                [ 8,  9, 10, 23]],
        <BLANKLINE>
               [[12,  1, 14,  3],
                [ 4, 17, 18,  7],
                [20, 21, 22, 11]]])
    """
    arr = np.swapaxes(arr, 0, axis)
    shape = arr.shape
    i = np.random.rand(*arr.shape).argsort(0).reshape(shape[0], -1)
    return arr.reshape(shape[0], -1)[i, np.arange(prod(shape[1:]))].reshape(
        shape).swapaxes(axis, 0)


# ======================================================================
def isqrt(num):
    """
    Calculate the integer square root of a number.

    This is defined as the largest integer whose square is smaller then the
    number, i.e. floor(sqrt(n))

    Args:
        num (int): The input number.

    Returns:
        result (int): The integer square root of num.

    Examples:
        >>> isqrt(1024)
        32
        >>> isqrt(1023)
        31
        >>> isqrt(1025)
        32
        >>> isqrt(2 ** 400)
        1606938044258990275541962092341162602522202993782792835301376
    """
    num = abs(num)
    guess = (num >> num.bit_length() // 2) + 1
    result = (guess + num // guess) // 2
    while abs(result - guess) > 1:
        guess = result
        result = (guess + num // guess) // 2
    while result * result > num:
        result -= 1
    return result


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

    See Also:
        - https://en.wikipedia.org/wiki/AKS_primality_test
    """
    # : fastest implementation
    num = abs(num)
    if (num % 2 == 0 and num > 2) or (num % 3 == 0 and num > 3):
        return False
    i = 5
    while i * i <= num:
        if num % i == 0 or num % (i + 2) == 0:
            return False
        else:
            i += 6
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
        numbers (list[int]): The factors of the number.

    Examples:
        >>> factorize(100)
        [2, 2, 5, 5]
        >>> factorize(1234567890)
        [2, 3, 3, 5, 3607, 3803]
        >>> factorize(-65536)
        [-1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
        >>> factorize(0)
        [0]
        >>> factorize(1)
        [1]
        >>> all([n == np.prod(factorize(n)) for n in range(1000)])
        True
    """
    # deal with special numbers: 0, 1, and negative
    if num == 0:
        text = 'Factorization of `0` is undefined.'
        warnings.warn(text)
    factors = [] if num > 1 else [-1] if num < 0 else [num]
    num = abs(num)

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


# ======================================================================
def factorize2(num):
    """
    Find all factors of a number and collect them in an ordered dict.

    Args:
        num (int): The number to factorize.

    Returns:
        factors (collections.OrderedDict): The factors of the number.

    Examples:
        >>> factorize2(100)
        OrderedDict([(2, 2), (5, 2)])
        >>> factorize2(1234567890)
        OrderedDict([(2, 1), (3, 2), (5, 1), (3607, 1), (3803, 1)])
        >>> factorize2(65536)
        OrderedDict([(2, 16)])
    """
    factors = factorize(num)
    return collections.OrderedDict(collections.Counter(factors))


# ======================================================================
def factorize3(
        num,
        exp_sep='^',
        fact_sep=' * '):
    """
    Find all factors of a number and output a human-readable text.

    Args:
        num (int): The number to factorize.
        exp_sep (str): The exponent separator.
        fact_sep (str): The factors separator.

    Returns:
        text (str): The factors of the number.

    Examples:
        >>> factorize3(100)
        '2^2 * 5^2'
        >>> factorize3(1234567890)
        '2 * 3^2 * 5 * 3607 * 3803'
        >>> factorize3(65536)
        '2^16'
    """
    factors = factorize(num)

    text = ''
    last_factor = 1
    exp = 0
    for factor in factors:
        if factor == last_factor:
            exp += 1
        else:
            if exp > 1:
                text += exp_sep + str(exp)
            if last_factor > 1:
                text += fact_sep
            text += str(factor)
            last_factor = factor
            exp = 1
    if exp > 1:
        text += exp_sep + str(exp)
    return text


# =====================================================================
def factorize_k_all(
        num,
        k=2,
        sort=None,
        reverse=False):
    """
    Find all possible factorizations with k factors.

    Ones are not present, unless because there are not enough factors.

    Args:
        num (int): The number of elements to arrange.
        k (int): The number of factors.
        sort (callable): The sorting function.
            This is passed to the `key` arguments of `sorted()`.
        reverse (bool): The sorting direction.
            This is passed to the `reverse` arguments of `sorted()`.
            If False, sorting is ascending.
            Otherwise, sorting is descending.

    Returns:
        factorizations (tuple[tuple[int]]): The possible factorizations.
            Each factorization has exactly `k` items.
            Eventually, `1`s are used to ensure the number of items.

    Examples:
        >>> nums = (32, 41, 46, 60)
        >>> for i in nums:
        ...     factorize_k_all(i, 2)
        ((2, 16), (4, 8), (8, 4), (16, 2))
        ((1, 41), (41, 1))
        ((2, 23), (23, 2))
        ((2, 30), (3, 20), (4, 15), (5, 12), (6, 10), (10, 6), (12, 5),\
 (15, 4), (20, 3), (30, 2))
        >>> for i in nums:
        ...     factorize_k_all(i, 3)
        ((2, 2, 8), (2, 4, 4), (2, 8, 2), (4, 2, 4), (4, 4, 2), (8, 2, 2))
        ((1, 1, 41), (1, 41, 1), (41, 1, 1))
        ((1, 2, 23), (1, 23, 2), (2, 1, 23), (2, 23, 1), (23, 1, 2),\
 (23, 2, 1))
        ((2, 2, 15), (2, 3, 10), (2, 5, 6), (2, 6, 5), (2, 10, 3), (2, 15, 2),\
 (3, 2, 10), (3, 4, 5), (3, 5, 4), (3, 10, 2), (4, 3, 5), (4, 5, 3),\
 (5, 2, 6), (5, 3, 4), (5, 4, 3), (5, 6, 2), (6, 2, 5), (6, 5, 2),\
 (10, 2, 3), (10, 3, 2), (15, 2, 2))
    """
    factors = factorize(num)
    factors = tuple(factors) + (1,) * (k - len(factors))
    factorizations = [
        item
        for subitems in unique_partitions(factors, k)
        for item in subitems]
    factorizations = list(set(factorizations))
    for i in range(len(factorizations)):
        factorizations[i] = tuple(
            functools.reduce(lambda x, y: x * y, j) for j in factorizations[i])
    return tuple(sorted(set(factorizations), key=sort, reverse=reverse))


# =====================================================================
def factorize_k(
        num,
        k=2,
        mode='=',
        balanced=True):
    """
    Generate a factorization of a number with k factors.

    Each factor contains (approximately) the same number of prime factors.

    Args:
        num (int): The number of elements to arrange.
        k (int): The number of factors.
        mode (str): The generation mode.
            This determines the factors order before splitting.
            The splitting itself is obtained with `chunks()`.
            Accepted values are:
             - 'increasing', 'ascending', '+': factors are sorted increasingly
               before splitting;
             - 'decreasing', 'descending', '-': factors are sorted decreasingly
               before splitting;
             - 'random': factors are shuffled before splitting;
             - 'seedX' where 'X' is an int, str or bytes: same as random, but
               'X' is used to initialize the random seed;
             - 'altX' where 'X' is an int: starting from 'X', factors are
               alternated before splitting;
             - 'alt1': factors are alternated before splitting;
             - 'optimal', 'similar', '!', '=': factors have the similar sizes.
        balanced (bool): Balance the number of primes in each factor.
            See `pymrt.utils.chunks()` for more info.

    Returns:
        tuple (int): A listing of `k` factors of `num`.

    Examples:
        >>> [factorize_k(402653184, k) for k in range(3, 6)]
        [(1024, 768, 512), (192, 128, 128, 128), (64, 64, 64, 48, 32)]
        >>> [factorize_k(402653184, k) for k in (2, 12)]
        [(24576, 16384), (8, 8, 8, 8, 6, 4, 4, 4, 4, 4, 4, 4)]
        >>> factorize_k(6, 4)
        (3, 2, 1, 1)
        >>> factorize_k(-12, 4)
        (3, 2, 2, -1)
        >>> factorize_k(0, 4)
        (1, 1, 1, 0)
        >>> factorize_k(720, 4)
        (6, 6, 5, 4)
        >>> factorize_k(720, 4, '+')
        (4, 4, 9, 5)
        >>> factorize_k(720, 3)
        (12, 10, 6)
        >>> factorize_k(720, 3, '+')
        (8, 6, 15)
        >>> factorize_k(720, 3, mode='-')
        (45, 4, 4)
        >>> factorize_k(720, 3, mode='seed0')
        (12, 6, 10)
        >>> factorize_k(720, 3, 'alt')
        (30, 4, 6)
        >>> factorize_k(720, 3, 'alt1')
        (12, 6, 10)
        >>> factorize_k(720, 3, '=')
        (12, 10, 6)
    """
    if k > 1:
        factors = factorize(num)
        if len(factors) < k:
            factors.extend([1] * (k - len(factors)))
        groups = None
        if mode in ('increasing', 'ascending', '+'):
            factors = sorted(factors)
        elif mode in ('decreasing', 'descending', '-'):
            factors = sorted(factors, reverse=True)
        elif mode == 'random':
            random.shuffle(factors)
        elif mode.startswith('seed'):
            seed = auto_convert(mode[len('seed'):])
            random.seed(seed)
            random.shuffle(factors)
        elif mode.startswith('alt'):
            try:
                i = int(mode[len('alt'):]) % (len(factors) - 1)
            except ValueError:
                i = 0
            factors[i::2] = factors[i::2][::-1]
        elif mode in ('optimal', 'similar', '!', '='):
            groups = [[] for _ in itertools.repeat(None, k)]
            # could this algorithm could be improved?
            for factor in sorted(factors, reverse=True):
                groups = sorted(
                    groups, key=lambda x: np.product(x) if len(x) > 0 else 0)
                groups[0].append(factor)
            groups = sorted(groups, key=np.product, reverse=True)
        if not groups:
            groups = chunks(factors, k, mode='+', balanced=True)
        factorization = tuple(
            functools.reduce(lambda x, y: x * y, j) for j in groups)
    else:
        factorization = (num,)

    return factorization


# =====================================================================
def optimal_shape(
        num,
        dims=2,
        sort=lambda x: (sum(x), x[::-1]),
        reverse=False):
    """
    Find the optimal shape for arranging n elements into a rank-k tensor.

    Args:
        num (int): The number of elements to arrange.
        dims (int): The rank of the tensor.
        sort (callable): The function defining optimality.
            The factorization that minimizes (or maximizes)
            This is passed to the `key` arguments of `sorted()`.
        reverse (bool): The sorting direction.
            This is passed to the `reverse` arguments of `sorted()`.
            If False, sorting is ascending and the minimum of the optimization
            function is picked.
            Otherwise, sorting is descending and the maximum of the
            optimization
            function is picked.

    Returns:
        ratios (tuple[int]): The optimal ratio for tensor dims.

    Examples:
        >>> n1, n2 = 40, 46
        >>> [optimal_shape(i) for i in range(n1, n2)]
        [(8, 5), (41, 1), (7, 6), (43, 1), (11, 4), (9, 5)]
        >>> [optimal_shape(i, sort=max) for i in range(n1, n2)]
        [(5, 8), (1, 41), (6, 7), (1, 43), (4, 11), (5, 9)]
        >>> [optimal_shape(i, sort=min) for i in range(n1, n2)]
        [(2, 20), (1, 41), (2, 21), (1, 43), (2, 22), (3, 15)]
        >>> [optimal_shape(i, 3) for i in range(n1, n2)]
        [(5, 4, 2), (41, 1, 1), (7, 3, 2), (43, 1, 1), (11, 2, 2), (5, 3, 3)]
    """
    factorizations = factorize_k_all(num, dims)
    return sorted(factorizations, key=sort, reverse=reverse)[0]


# ======================================================================
def _gcd(a, b):
    """
    Calculate the greatest common divisor (GCD) of a and b.

    Unless b==0, the result will have the same sign as b (so that when
    b is divided by it, the result comes out positive).

    Examples:
        >>> _gcd(123, 45)
        3
        >>> _gcd(45, 123)
        3
        >>> _gcd(0, 1)
        1
        >>> _gcd(-3, 1)
        1
        >>> _gcd(211815584, 211815584)
        211815584

    Note:
        This should never be used as `math.gcd` offers identical functionality,
        but it is faster.
    """
    while b:
        a, b = b, a % b
    return a


# =====================================================================
def gcd(*nums):
    """
    Find the greatest common divisor (GCD) of a list of numbers.

    Args:
        *nums (Iterable[int]): The input numbers.

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
        *numbers (Iterable[int]): The input numbers.

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
        lcm_val = lcm_val * num // math.gcd(lcm_val, num)
    return lcm_val


# ======================================================================
def num_align(
        num,
        align='pow2',
        mode=1):
    """
    Align a number to a specified value, so as to make it multiple of it.

    This calculated

    Args:
        num (int): The input number.
        align (int|str): The number to align to.
            If int, then calculate a multiple of `align` close to `num`.
            If str, possible options are:
             - 'powX' (where X >= 2 must be an int): calculate a power of X
               that is close to `num`.
            The exact number being calculated depends on the value of `mode`.
        mode (int|str): Determine which multiple to convert the number to.
            If str, valid inputs are:
             - 'upper': converts to the smallest multiple larger than `num`.
             - 'lower': converts to the largest multiple smaller than `num`.
             - 'closest': converts to the multiple closest to `num`.
            If int, valid inputs are:
             - '+1' has the same behavior as 'upper'.
             - '-1' has the same behavior as  'lower'.
             - '0' has the same behavior as  'closest'.

    Returns:
        num (int): The aligned number.

    Examples:
        >>> num_align(432)
        512
        >>> num_align(432, mode=-1)
        256
        >>> num_align(432, mode=0)
        512
        >>> num_align(447, 32, mode=+1)
        448
        >>> num_align(447, 32, mode=-1)
        416
        >>> num_align(447, 32, mode=0)
        448
        >>> num_align(45, 90, mode=0)
        0
        >>> num_align(6, 'pow2', mode=0)
        8
        >>> num_align(128, 128, mode=1)
        128
    """
    if mode == 'upper' or mode == +1:
        func = math.ceil
    elif mode == 'lower' or mode == -1:
        func = math.floor
    elif mode == 'closest' or mode == 0:
        func = round
    else:
        raise ValueError('Invalid mode `{mode}`'.format(mode=mode))

    if isinstance(align, str):
        if align.startswith('pow'):
            base = int(align[len('pow'):])
            exp = math.log(num, base)
            num = int(base ** func(exp))
        else:
            raise ValueError('Invalid align `{align}`'.format(align=align))

    elif isinstance(align, int):
        modulus = num % align
        num += func(modulus / align) * align - modulus

    else:
        warnings.warn('Will not align `{num}` to `{align}`.'.format(
            num=num, align=align))

    return num


# ======================================================================
def merge_dicts(*dicts):
    """
    Merge dictionaries into a new dict (new keys overwrite the old ones).

    Args:
        dicts (args[dict]): Dictionaries to be merged together.

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


# =====================================================================
def p_ratio(x, y):
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
        >>> p_ratio(2, 2)
        0.5
        >>> p_ratio(200, 200)
        0.5
        >>> p_ratio(1, 2)
        0.4
        >>> p_ratio(100, 200)
        0.4
        >>> items = 100, 200
        >>> (p_ratio(*items) == p_ratio(*items[::-1]))
        True
    """
    return (x * y) / (x ** 2 + y ** 2)


# =====================================================================
def gen_p_ratio(*items):
    """
    Calculate the generalized pseudo-ratio of x_i: 1 / sum_ij [ x_i / x_j ]

    .. math::
        \\frac{1}{\\sum_{ij} \\frac{x_i}{x_j}}

    Args:
        *items (Iterable[int|float|np.ndarray]): Input values.

    Returns:
        result: 1 / sum_ij [ x_i / x_j ]

    Examples:
        >>> gen_p_ratio(2, 2, 2, 2, 2)
        0.05
        >>> gen_p_ratio(200, 200, 200, 200, 200)
        0.05
        >>> gen_p_ratio(1, 2)
        0.4
        >>> gen_p_ratio(100, 200)
        0.4
        >>> items1 = [x * 10 for x in range(2, 10)]
        >>> items2 = [x * 1000 for x in range(2, 10)]
        >>> np.isclose(gen_p_ratio(*items1), gen_p_ratio(*items2))
        True
        >>> items = list(range(2, 10))
        >>> np.isclose(
        ...     gen_p_ratio(*items), gen_p_ratio(*items[::-1]))
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
    This version works for two Iterables.

    This is known as the `longest common substring` problem, or LCS for short.

    Args:
        seq1 (Iterable): The first input sequence.
            Must be of the same type as seq2.
        seq2 (Iterable): The second input sequence.
            Must be of the same type as seq1.
        sorting (callable): Sorting function passed to 'sorted' via `key` arg.

    Returns:
        commons (list[Iterable]): The longest common subsequence(s).

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
    This version works for an Iterable of Iterables.

    This is known as the `longest common substring` problem, or LCS for short.

    Args:
        seqs (Iterable[Iterable]): The input sequences.
            All the items must be of the same type.
        sorting (callable): Sorting function passed to 'sorted' via `key` arg.

    Returns:
        commons (list[Iterable]): The longest common subsequence(s).

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
def set_func_kws(
        func,
        func_kws):
    """
    Set keyword parameters of a function to specific or default values.

    Args:
        func (callable): The function to be inspected.
        func_kws (dict): The (key, value) pairs to set.
            If a value is None, it will be replaced by the default value.
            To use the names defined locally, use: `locals()`.

    Results:
        kws (dict): A dictionary of the keyword parameters to set.

    See Also:
        inspect, locals, globals.
    """
    try:
        get_argspec = inspect.getfullargspec
    except AttributeError:
        get_argspec = inspect.getargspec
    inspected = get_argspec(func)
    defaults = dict(
        zip(reversed(inspected.args), reversed(inspected.defaults)))
    kws = {}
    for key in inspected.args:
        if key in func_kws:
            kws[key] = func_kws[key]
        elif key in defaults:
            kws[key] = defaults[key]
    return kws


# ======================================================================
def split_func_kws(
        func,
        func_kws):
    """
    Split a set of keywords into accepted and not accepted by some function.

    Args:
        func (callable): The function to be inspected.
        func_kws (dict): The (key, value) pairs to split.

    Results:
        result (tuple): The tuple
            contains:
             - kws (dict): The keywords NOT accepted by `func`.
             - func_kws (dict): The keywords accepted by `func`.

    See Also:
        inspect, locals, globals.
    """
    try:
        get_argspec = inspect.getfullargspec
    except AttributeError:
        get_argspec = inspect.getargspec
    inspected = get_argspec(func)
    kws = {k: v for k, v in func_kws.items() if k not in inspected.args}
    func_kws = {k: v for k, v in func_kws.items() if k in inspected.args}
    return func_kws, kws


# ======================================================================
def unsqueezing(
        source_shape,
        target_shape):
    """
    Generate a broadcasting-compatible shape.

    The resulting shape contains *singletons* (i.e. `1`) for non-matching dims.
    Assumes all elements of the source shape are contained in the target shape
    (excepts for singletons) in the correct order.

    Warning! The generated shape may not be unique if some of the elements
    from the source shape are present multiple times in the target shape.

    Args:
        source_shape (Sequence): The source shape.
        target_shape (Sequence): The target shape.

    Returns:
        shape (tuple): The broadcast-safe shape.

    Raises:
        ValueError: if elements of `source_shape` are not in `target_shape`.

    Examples:
        For non-repeating elements, `unsqueezing()` is always well-defined:

        >>> unsqueezing((2, 3), (2, 3, 4))
        (2, 3, 1)
        >>> unsqueezing((3, 4), (2, 3, 4))
        (1, 3, 4)
        >>> unsqueezing((3, 5), (2, 3, 4, 5, 6))
        (1, 3, 1, 5, 1)
        >>> unsqueezing((1, 3, 5, 1), (2, 3, 4, 5, 6))
        (1, 3, 1, 5, 1)

        If there is nothing to unsqueeze, the `source_shape` is returned:

        >>> unsqueezing((1, 3, 1, 5, 1), (2, 3, 4, 5, 6))
        (1, 3, 1, 5, 1)
        >>> unsqueezing((2, 3), (2, 3))
        (2, 3)

        If some elements in `source_shape` are repeating in `target_shape`,
        a user warning will be issued:

        >>> unsqueezing((2, 2), (2, 2, 2, 2, 2))
        (2, 2, 1, 1, 1)
        >>> unsqueezing((2, 2), (2, 3, 2, 2, 2))
        (2, 1, 2, 1, 1)

        If some elements of `source_shape` are not presente in `target_shape`,
        an error is raised.

        >>> unsqueezing((2, 3), (2, 2, 2, 2, 2))
        Traceback (most recent call last):
          ...
        ValueError: Target shape must contain all source shape elements\
 (in correct order). (2, 3) -> (2, 2, 2, 2, 2)
        >>> unsqueezing((5, 3), (2, 3, 4, 5, 6))
        Traceback (most recent call last):
          ...
        ValueError: Target shape must contain all source shape elements\
 (in correct order). (5, 3) -> (2, 3, 4, 5, 6)

    """
    shape = []
    j = 0
    for i, dim in enumerate(target_shape):
        if j < len(source_shape):
            shape.append(dim if dim == source_shape[j] else 1)
            if i + 1 < len(target_shape) and dim == source_shape[j] \
                    and dim != 1 and dim in target_shape[i + 1:]:
                text = ('Multiple positions (e.g. {} and {})'
                        ' for source shape element {}.'.format(
                    i, target_shape[i + 1:].index(dim) + (i + 1), dim))
                warnings.warn(text)
            if dim == source_shape[j] or source_shape[j] == 1:
                j += 1
        else:
            shape.append(1)
    if j < len(source_shape):
        raise ValueError(
            'Target shape must contain all source shape elements'
            ' (in correct order). {} -> {}'.format(source_shape, target_shape))
    return tuple(shape)


# ======================================================================
def unsqueeze(
        arr,
        axis=None,
        shape=None,
        complement=False):
    """
    Add singletons to the shape of an array to broadcast-match a given shape.

    In some sense, this function implements the inverse of `numpy.squeeze()`.

    Args:
        arr (np.ndarray): The input array.
        axis (int|Iterable|None): Axis or axes in which to operate.
            If None, a valid set axis is generated from `shape` when this is
            defined and the shape can be matched by `unsqueezing()`.
            If int or Iterable, specified how singletons are added.
            This depends on the value of `complement`.
            If `shape` is not None, the `axis` and `shape` parameters must be
            consistent.
            Values must be in the range [-(ndim+1), ndim+1]
            At least one of `axis` and `shape` must be specified.
        shape (int|Iterable|None): The target shape.
            If None, no safety checks are performed.
            If int, this is interpreted as the number of dimensions of the
            output array.
            If Iterable, the result must be broadcastable to an array with the
            specified shape.
            If `axis` is not None, the `axis` and `shape` parameters must be
            consistent.
            At least one of `axis` and `shape` must be specified.
        complement (bool): Interpret `axis` parameter as its complementary.
            If True, the dims of the input array are placed at the positions
            indicated by `axis`, and singletons are placed everywherelse and
            the `axis` length must be equal to the number of dimensions of the
            input array; the `shape` parameter cannot be `None`.
            If False, the singletons are added at the position(s) specified by
            `axis`.
            If `axis` is None, `complement` has no effect.

    Returns:
        arr (np.ndarray): The reshaped array.

    Raises:
        ValueError: if the `arr` shape cannot be reshaped correctly.

    Examples:
        Let's define some input array `arr`:

        >>> arr = np.arange(2 * 3 * 4).reshape((2, 3, 4))
        >>> arr.shape
        (2, 3, 4)

        A call to `unsqueeze()` can be reversed by `np.squeeze()`:

        >>> arr_ = unsqueeze(arr, (0, 2, 4))
        >>> arr_.shape
        (1, 2, 1, 3, 1, 4)
        >>> arr = np.squeeze(arr_, (0, 2, 4))
        >>> arr.shape
        (2, 3, 4)

        The order of the axes does not matter:

        >>> arr_ = unsqueeze(arr, (0, 4, 2))
        >>> arr_.shape
        (1, 2, 1, 3, 1, 4)

        If `shape` is an int, `axis` must be consistent with it:

        >>> arr_ = unsqueeze(arr, (0, 2, 4), 6)
        >>> arr_.shape
        (1, 2, 1, 3, 1, 4)
        >>> arr_ = unsqueeze(arr, (0, 2, 4), 7)
        Traceback (most recent call last):
          ...
        ValueError: Incompatible `[0, 2, 4]` axis and `7` shape for array of\
 shape (2, 3, 4)

        It is possible to complement the meaning to `axis` to add singletons
        everywhere except where specified (but requires `shape` to be defined
        and the length of `axis` must match the array dims):

        >>> arr_ = unsqueeze(arr, (0, 2, 4), 10, True)
        >>> arr_.shape
        (2, 1, 3, 1, 4, 1, 1, 1, 1, 1)
        >>> arr_ = unsqueeze(arr, (0, 2, 4), complement=True)
        Traceback (most recent call last):
          ...
        ValueError: When `complement` is True, `shape` cannot be None.
        >>> arr_ = unsqueeze(arr, (0, 2), 10, True)
        Traceback (most recent call last):
          ...
        ValueError: When `complement` is True, the length of axis (2) must\
 match the num of dims of array (3).

        Axes values must be valid:

        >>> arr_ = unsqueeze(arr, 0)
        >>> arr_.shape
        (1, 2, 3, 4)
        >>> arr_ = unsqueeze(arr, 3)
        >>> arr_.shape
        (2, 3, 4, 1)
        >>> arr_ = unsqueeze(arr, -1)
        >>> arr_.shape
        (2, 3, 4, 1)
        >>> arr_ = unsqueeze(arr, -4)
        >>> arr_.shape
        (1, 2, 3, 4)
        >>> arr_ = unsqueeze(arr, 10)
        Traceback (most recent call last):
          ...
        ValueError: Axis (10,) out of range.

        If `shape` is specified, `axis` can be omitted (USE WITH CARE!) or its
        value is used for addiotional safety checks:

        >>> arr_ = unsqueeze(arr, shape=(2, 3, 4, 5, 6))
        >>> arr_.shape
        (2, 3, 4, 1, 1)
        >>> arr_ = unsqueeze(
        ...     arr, (3, 6, 8), (2, 5, 3, 2, 7, 2, 3, 2, 4, 5, 6), True)
        >>> arr_.shape
        (1, 1, 1, 2, 1, 1, 3, 1, 4, 1, 1)
        >>> arr_ = unsqueeze(
        ...     arr, (3, 7, 8), (2, 5, 3, 2, 7, 2, 3, 2, 4, 5, 6), True)
        Traceback (most recent call last):
          ...
        ValueError: New shape [1, 1, 1, 2, 1, 1, 1, 3, 4, 1, 1] cannot be\
 broadcasted to shape (2, 5, 3, 2, 7, 2, 3, 2, 4, 5, 6)
        >>> arr = unsqueeze(arr, shape=(2, 5, 3, 7, 2, 4, 5, 6))
        >>> arr.shape
        (2, 1, 3, 1, 1, 4, 1, 1)
        >>> arr = np.squeeze(arr)
        >>> arr.shape
        (2, 3, 4)
        >>> arr = unsqueeze(arr, shape=(5, 3, 7, 2, 4, 5, 6))
        Traceback (most recent call last):
          ...
        ValueError: Target shape must contain all source shape elements\
 (in correct order). (2, 3, 4) -> (5, 3, 7, 2, 4, 5, 6)

        The behavior is consistent with other NumPy functions and the
        `keepdims` mechanism:

        >>> axis = (0, 2, 4)
        >>> arr1 = np.arange(2 * 3 * 4 * 5 * 6).reshape((2, 3, 4, 5, 6))
        >>> arr2 = np.sum(arr1, axis, keepdims=True)
        >>> arr2.shape
        (1, 3, 1, 5, 1)
        >>> arr3 = np.sum(arr1, axis)
        >>> arr3.shape
        (3, 5)
        >>> arr3 = unsqueeze(arr3, axis)
        >>> arr3.shape
        (1, 3, 1, 5, 1)
        >>> np.all(arr2 == arr3)
        True
    """
    # calculate `new_shape`
    if axis is None and shape is None:
        raise ValueError(
            'At least one of `axis` and `shape` parameters must be specified.')
    elif axis is None and shape is not None:
        new_shape = unsqueezing(arr.shape, shape)
    elif axis is not None:
        if isinstance(axis, int):
            axis = (axis,)
        # calculate the dim of the result
        if shape is not None:
            if isinstance(shape, int):
                ndim = shape
            else:  # shape is a sequence
                ndim = len(shape)
        elif not complement:
            ndim = len(axis) + arr.ndim
        else:
            raise ValueError(
                'When `complement` is True, `shape` cannot be None.')
        # check that axis is properly constructed
        if any([ax < -ndim - 1 or ax > ndim + 1 for ax in axis]):
            raise ValueError('Axis {} out of range.'.format(axis))
        # normalize axis using `ndim`
        axis = sorted([ax % ndim for ax in axis])
        # manage complement mode
        if complement:
            if len(axis) == arr.ndim:
                axis = [i for i in range(ndim) if i not in axis]
            else:
                raise ValueError(
                    'When `complement` is True, the length of axis ({})'
                    ' must match the num of dims of array ({}).'.format(
                        len(axis), arr.ndim))
        elif len(axis) + arr.ndim != ndim:
            raise ValueError(
                'Incompatible `{}` axis and `{}` shape'
                ' for array of shape {}'.format(axis, shape, arr.shape))
        # generate the new shape from axis, ndim and shape
        new_shape = []
        i, j = 0, 0
        for m in range(ndim):
            if i < len(axis) and m == axis[i] or j >= arr.ndim:
                new_shape.append(1)
                i += 1
            else:
                new_shape.append(arr.shape[j])
                j += 1

    # check that `new_shape` is consistent with `shape`
    if shape is not None:
        if isinstance(shape, int):
            if len(new_shape) != ndim:
                raise ValueError(
                    'Length of new shape {} does not match '
                    'expected length ({}).'.format(len(new_shape), ndim))
        else:
            if not all([new_dim == 1 or new_dim == dim
                        for new_dim, dim in zip(new_shape, shape)]):
                raise ValueError(
                    'New shape {} cannot be broadcasted to shape {}'.format(
                        new_shape, shape))

    return arr.reshape(new_shape)


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
        base (str): Directory where to operate.
        follow_links (bool): Follow links during recursion.
        follow_mounts (bool): Follow mount points during recursion.
        allow_special (bool): Include special files.
        allow_hidden (bool): Include hidden files.
        max_depth (int): Maximum depth to reach. Negative for unlimited.
        on_error (callable): Function to call on error.

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
        base (str): Directory where to operate.
        follow_links (bool): Follow links during recursion.
        follow_mounts (bool): Follow mount points during recursion.
        allow_special (bool): Include special files.
        allow_hidden (bool): Include hidden files.
        max_depth (int): Maximum depth to reach. Negative for unlimited.
        on_error (callable): Function to call on error.

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
        patterns='*',
        dirpath='.',
        unix_style=True,
        re_kws=None,
        walk_kws=None):
    """
    Recursively list the content of a directory matching the pattern(s).

    Args:
        dirpath (str): The base directory.
        patterns (str|Iterable[str]): The pattern(s) to match.
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
        patterns='*',
        dirpath='.',
        unix_style=True,
        re_kws=None,
        walk_kws=None):
    """
    Recursively list the content of a directory matching the pattern(s).

    Args:
        dirpath (str): The base directory.
        patterns (str|Iterable[str]): The pattern(s) to match.
            These must be either a Unix-style pattern or a regular expression,
            depending on the value of `unix_style`.
        unix_style (bool): Interpret the patterns as Unix-style.
            This is achieved by using `fnmatch`.
        re_kws (dict|None): Keyword arguments passed to `re.compile()`.
        walk_kws (dict|None): Keyword arguments passed to `os.walk()`.

    Returns:
        filepaths (list[str]): The matched filepaths.
    """
    return [
        item for item in iflistdir(
            patterns=patterns, dirpath=dirpath, unix_style=unix_style,
            re_kws=re_kws, walk_kws=walk_kws)]


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
        *args (Iterable[str]): The path elements to be concatenated.
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
def next_filepath(
        filepath,
        out_template='{basepath}__{counter}{ext}',
        verbose=D_VERB_LVL):
    """
    Generate a non-existing filepath if current exists.

    Args:
        filepath (str): The input filepath.
        out_template (str): Template for the output filepath.
            The following variables are available for interpolation:
             - `dirpath`: The directory of the input.
             - `base`: The input base file name without extension.
             - `ext`: The input file extension (with leading separator).
             - `basepath`: The input filepath without extension.

    Returns:
        filepath (str)
    """
    if os.path.exists(filepath):
        msg('OLD: {}'.format(filepath), verbose, VERB_LVL['high'])
        dirpath, base, ext = split_path(filepath)
        basepath = os.path.join(dirpath, base)
        counter = 0
        while os.path.exists(filepath):
            counter += 1
            filepath = out_template.format_map(locals())
        msg('NEW: {}'.format(filepath), verbose, VERB_LVL['medium'])
    return filepath


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
        *args (Iterable): Positional arguments passed to `open()`.
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
def zopen(filepath, mode='rb', *args, **kwargs):
    """
    Auto-magically open a gzip-compressed file.

    Note: all compressed files should be opened as binary.
    Opening in text mode is not supported.

    Args:
        filepath (str): The file path.
        mode (str): The mode for file opening.
            See `open()` for more info.
            If the `t` mode is not specified, `b` mode is assumed.
            If `t` mode is specified, the file cannot be compressed.
        *args (Iterable): Positional arguments passed to `open()`.
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
    if 't' not in mode and 'b' not in mode:
        mode += 'b'

    valid_mode = 'b' in mode and 't' not in mode

    # try open file as normal
    file_obj = open(filepath, mode=mode, *args, **kwargs)

    if valid_mode:
        # test if file is gzip using magic type (first 2 bytes)
        try:
            magic = file_obj.read(2)
            by_magic = magic == b'\x1f\x8b'
        except io.UnsupportedOperation:
            by_magic = False
        finally:
            file_obj.seek(0)

        by_ext = split_ext(filepath)[1].endswith(add_extsep(EXT['gzip']))

        if by_magic or by_ext:
            file_obj = gzip.GzipFile(fileobj=file_obj, mode=mode)

    return file_obj


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
        num,
        keep_zeros=4):
    """
    Format a number with the correct number of significant figures.

    Args:
        val (str|float|int): The numeric value to be correctly formatted.
        num (str|int): The number of significant figures to be displayed.
        keep_zeros (int): The number of zeros to keep after the figures.
            This is useful for preventing the use of the scientific notation.

    Returns:
        val (str): String containing the properly formatted number.

    Examples:
        >>> significant_figures(1.2345, 1)
        '1'
        >>> significant_figures(1.2345, 4)
        '1.234'
        >>> significant_figures(1.234e3, 2)
        '1200'
        >>> significant_figures(-1.234e3, 3)
        '-1230'
        >>> significant_figures(12345678, 4)
        '12350000'
        >>> significant_figures(1234567890, 4)
        '1.235e+9'
        >>> significant_figures(-0.1234, 1)
        '-0.1'
        >>> significant_figures(0.0001, 2)
        '1.0e-4'

    See Also:
        The 'decimal' Python standard module.
    """
    val = float(val)
    num = int(num)
    order = int(np.floor(np.log10(abs(val)))) if abs(val) != 0.0 else 0
    prec = num - order - 1
    ofm = ''
    val = round(val, prec)
    if abs(prec) > keep_zeros:
        val = val * 10 ** (-order)
        prec = num - 1
        ofm = 'e{:+d}'.format(order)
    elif prec < 0:
        prec = 0

    # print('val={}, num={}, ord={}, prec={}, ofm={}'.format(
    #     val, num, order, prec, ofm))  # DEBUG
    val = '{val:.{prec}f}{ofm}'.format(val=val, prec=prec, ofm=ofm)
    return val


# ======================================================================
def format_value_error(
        val,
        err,
        num=2,
        keep_zeros=4):
    """
    Outputs correct value/error pairs formatting.

    Args:
        val (str|float|int): The numeric value to be correctly formatted.
        err (str|float|int): The numeric error to be correctly formatted.
        num (str|int): The precision to be used for the error (usually 1 or 2).
        keep_zeros (int): The number of zeros to keep after the figures.
            This is useful for preventing the use of the scientific notation.

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
        >>> format_value_error(12345.6, 0)
        ('12345.6', '0.0')
        >>> format_value_error(12345.6, 0, 0)
        ('12346', '0')
        >>> format_value_error(12345.6, 67)
        ('12346', '67')
        >>> format_value_error(12345.6, 670)
        ('12350', '670')
        >>> format_value_error(1234567890.0, 123456.0)
        ('1234570000', '120000')
        >>> format_value_error(1234567890.0, 1234567.0)
        ('1.2346e+9', '1.2e+6')
        >>> format_value_error(-0.470, 1.722)
        ('-0.5', '1.7')
        >>> format_value_error(0.0025, 0.0001)
        ('2.50e-3', '1.0e-4')
    """
    val = float(val)
    err = float(err)
    num = int(num) if num != 0 else 1
    val_order = int(np.floor(np.log10(np.abs(val)))) if val != 0 else 0
    err_order = int(np.floor(np.log10(np.abs(err)))) if err != 0 else 0
    try:
        # print('val_order={}, err_order={}, num={}'.format(
        #     val_order, err_order, num))  # DEBUG
        val_str = significant_figures(
            val, val_order - err_order + num, keep_zeros)
        err_str = significant_figures(
            err, num, keep_zeros)
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
        verbose=D_VERB_LVL,
        makedirs=False,
        no_empty_input=False):
    """
    Check if input files are newer than output files, to force calculation.

    Args:
        in_filepaths (str|Iterable[str]|None): Input filepaths for computation.
        out_filepaths (str|Iterable[str]): Output filepaths for computation.
        force (bool): Force computation to be re-done.
        verbose (int): Set level of verbosity.
        makedirs (bool): Create output dirpaths if not existing.
        no_empty_input (bool): Check if the input filepath list is empty.

    Returns:
        force (bool): True if the computation is to be re-done.

    Raises:
        IndexError: If the input filepath list is empty.
            Only if `no_empty_input` is True.
        IOError: If any of the input files do not exist.
    """
    # generate singleton list from str argument
    if isinstance(in_filepaths, str):
        in_filepaths = [in_filepaths]
    if isinstance(out_filepaths, str):
        out_filepaths = [out_filepaths]

    # check if output exists
    if not force:
        for out_filepath in out_filepaths:
            if out_filepath and not os.path.exists(out_filepath):
                force = True
                break

    # create output directories
    if force and makedirs:
        for out_filepath in out_filepaths:
            out_dirpath = os.path.dirname(out_filepath)
            if not os.path.isdir(out_dirpath):
                msg('mkdir: {}'.format(out_dirpath),
                    verbose, VERB_LVL['highest'])
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

    if force:
        msg('Calc: {}'.format(out_filepaths), verbose, VERB_LVL['higher'])
        msg('From: {}'.format(in_filepaths), verbose, VERB_LVL['highest'])
    else:
        msg('Skip: {}'.format(out_filepaths), verbose, VERB_LVL['higher'])
        msg('From: {}'.format(in_filepaths), verbose, VERB_LVL['highest'])
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
        items (Iterable): The items to check.
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
        items (Iterable): The items to check.
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
    Determine if all items in an Iterable have the same sign.

    Args:
        items (Iterable): The items to check.
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
            If None, and val is Iterable, it is calculated as:
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
        interval2=None,
        operation='+'):
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
    if interval2 is None:
        interval2 = interval1
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
        removes (Iterable): Values to remove.
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
        shape (Iterable[int]): The shape of the mask in px.
        position (float|Iterable[float]): Relative position of the origin.
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
        shape (Iterable[int]): The shape of the mask in px.
        position (float|Iterable[float]): Relative position of the origin.
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
def laplace_kernel(
        shape,
        factors=1):
    """
    Calculate the kernel to be used for the Laplacian operators.

    This is substantially `k^2`.

    This is in the Fourier domain.
    May require shifting and normalization before using in
    Discrete Fourier Transform (DFT).

    Args:
        shape (Iterable[int]): The size of the array.
        factors (int|float|Iterable[int|float]): The size conversion factors.
            If int or float, the same conversion factor is applied to all dims.
            Otherwise, the Iterable length must match the length of shape.

    Returns:
        kk_2 (np.ndarray): The resulting kernel array.

    Examples:
        >>> laplace_kernel((3, 3, 3))
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
        >>> laplace_kernel((3, 3, 3), np.sqrt(3))
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
        >>> laplace_kernel((2, 2, 2), 0.6)
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
def gradient_kernels(
        shape,
        dims=None,
        factors=1):
    """
    Calculate the kernel to be used for the gradient operators.

    This is substantially: k

    This is in the Fourier domain.
    May require shifting and normalization before using in
    Discrete Fourier Transform (DFT).

    Args:
        shape (Iterable[int]): The size of the array.
        dims (int|Iterable[int]): The direction of the gradient.
            Values must be between `len(shape)` and `-len(shape)`.
        factors (int|float|Iterable[int|float]): The size conversion factors.
            If int or float, the same conversion factor is applied to all dims.
            Otherwise, the Iterable length must match the length of shape.

    Returns:
        kks (tuple(np.ndarray)): The resulting kernel arrays.

    Examples:
        >>> gradient_kernels((2, 2))
        (array([[-1, -1],
               [ 0,  0]]), array([[-1,  0],
               [-1,  0]]))
        >>> gradient_kernels((2, 2, 2))
        (array([[[-1, -1],
                [-1, -1]],
        <BLANKLINE>
               [[ 0,  0],
                [ 0,  0]]]), array([[[-1, -1],
                [ 0,  0]],
        <BLANKLINE>
               [[-1, -1],
                [ 0,  0]]]), array([[[-1,  0],
                [-1,  0]],
        <BLANKLINE>
               [[-1,  0],
                [-1,  0]]]))
        >>> gradient_kernels((2, 2, 2), (1, 2))
        (array([[[-1, -1],
                [ 0,  0]],
        <BLANKLINE>
               [[-1, -1],
                [ 0,  0]]]), array([[[-1,  0],
                [-1,  0]],
        <BLANKLINE>
               [[-1,  0],
                [-1,  0]]]))
        >>> gradient_kernels((2, 2, 2), -1)
        (array([[[-1,  0],
                [-1,  0]],
        <BLANKLINE>
               [[-1,  0],
                [-1,  0]]]),)
        >>> gradient_kernels((2, 2), None, 3)
        (array([[-0.33333333, -0.33333333],
               [ 0.        ,  0.        ]]), array([[-0.33333333,  0.        ],
               [-0.33333333,  0.        ]]))
    """
    kk_ = grid_coord(shape)
    if factors and factors != 1:
        factors = auto_repeat(factors, len(shape), check=True)
        kk_ = [k_i / factor for k_i, factor in zip(kk_, factors)]
    if dims is None:
        dims = range(len(shape))
    else:
        if isinstance(dims, int):
            dims = (dims,)
        dims = tuple(dim % len(shape) for dim in dims)
    kks = tuple(
        np.broadcast_to(k_i, shape)
        for i, (k_i, dim) in enumerate(zip(kk_, shape))
        if i in dims)
    return kks


# ======================================================================
def exp_gradient_kernels(
        shape,
        dims=None,
        factors=1):
    """
    Calculate the kernel to be used for the exponential gradient operators.

    This is substantially: :math:`1 - \\exp(2\\pi\\i k)`

    This is in the Fourier domain.
    May require shifting and normalization before using in
    Discrete Fourier Transform (DFT).

    Args:
        shape (Iterable[int]): The size of the array.
        dims (int|Iterable[int]): The direction of the gradient.
            Values must be between `len(shape)` and `-len(shape)`.
        factors (int|float|Iterable[int|float]): The size conversion factors.
            If int or float, the same conversion factor is applied to all dims.
            Otherwise, the Iterable length must match the length of shape.

    Returns:
        kks (tuple(np.ndarray)): The resulting kernel arrays.

    Examples:
        >>> exp_gradient_kernels((2, 2))
        (array([[ 0. -2.44929360e-16j,  0. -2.44929360e-16j],
               [ 0. +0.00000000e+00j,  0. +0.00000000e+00j]]),\
 array([[ 0. -2.44929360e-16j,  0. +0.00000000e+00j],
               [ 0. -2.44929360e-16j,  0. +0.00000000e+00j]]))
        >>> exp_gradient_kernels((2, 2, 2))
        (array([[[ 0. -2.44929360e-16j,  0. -2.44929360e-16j],
                [ 0. -2.44929360e-16j,  0. -2.44929360e-16j]],
        <BLANKLINE>
               [[ 0. +0.00000000e+00j,  0. +0.00000000e+00j],
                [ 0. +0.00000000e+00j,  0. +0.00000000e+00j]]]), array([[[ 0. -2.44929360e-16j,  0. -2.44929360e-16j],
                [ 0. +0.00000000e+00j,  0. +0.00000000e+00j]],
        <BLANKLINE>
               [[ 0. -2.44929360e-16j,  0. -2.44929360e-16j],
                [ 0. +0.00000000e+00j,  0. +0.00000000e+00j]]]), array([[[ 0. -2.44929360e-16j,  0. +0.00000000e+00j],
                [ 0. -2.44929360e-16j,  0. +0.00000000e+00j]],
        <BLANKLINE>
               [[ 0. -2.44929360e-16j,  0. +0.00000000e+00j],
                [ 0. -2.44929360e-16j,  0. +0.00000000e+00j]]]))
        >>> exp_gradient_kernels((2, 2, 2), (1, 2))
        (array([[[ 0. -2.44929360e-16j,  0. -2.44929360e-16j],
                [ 0. +0.00000000e+00j,  0. +0.00000000e+00j]],
        <BLANKLINE>
               [[ 0. -2.44929360e-16j,  0. -2.44929360e-16j],
                [ 0. +0.00000000e+00j,  0. +0.00000000e+00j]]]), array([[[ 0. -2.44929360e-16j,  0. +0.00000000e+00j],
                [ 0. -2.44929360e-16j,  0. +0.00000000e+00j]],
        <BLANKLINE>
               [[ 0. -2.44929360e-16j,  0. +0.00000000e+00j],
                [ 0. -2.44929360e-16j,  0. +0.00000000e+00j]]]))
        >>> exp_gradient_kernels((2, 2, 2), -1)
        (array([[[ 0. -2.44929360e-16j,  0. +0.00000000e+00j],
                [ 0. -2.44929360e-16j,  0. +0.00000000e+00j]],
        <BLANKLINE>
               [[ 0. -2.44929360e-16j,  0. +0.00000000e+00j],
                [ 0. -2.44929360e-16j,  0. +0.00000000e+00j]]]),)
        >>> exp_gradient_kernels((2, 2), None, 3)
        (array([[ 1.5+0.8660254j,  1.5+0.8660254j],
               [ 0.0+0.j       ,  0.0+0.j       ]]),\
 array([[ 1.5+0.8660254j,  0.0+0.j       ],
               [ 1.5+0.8660254j,  0.0+0.j       ]]))
    """
    kk_ = grid_coord(shape)
    if factors and factors != 1:
        factors = auto_repeat(factors, len(shape), check=True)
        kk_ = [k_i / factor for k_i, factor in zip(kk_, factors)]
    if dims is None:
        dims = range(len(shape))
    else:
        if isinstance(dims, int):
            dims = (dims,)
        dims = tuple(dim % len(shape) for dim in dims)
    kks = tuple(
        np.broadcast_to((1.0 - np.exp(2j * np.pi * k_i)), shape)
        for i, (k_i, dim) in enumerate(zip(kk_, shape))
        if i in dims)
    return kks


# ======================================================================
def auto_pad_width(
        pad_width,
        shape,
        combine=None):
    """
    Ensure pad_width value(s) to be consisting of integer.

    Args:
        pad_width (float|int|Iterable[float|int]): Size of the padding to use.
            This is useful for mitigating border effects.
            If Iterable, a value for each dim must be specified.
            If not Iterable, all dims will have the same value.
            If int, it is interpreted as absolute size.
            If float, it is interpreted as relative to corresponding dim size.
        shape (Iterable[int]): The shape to associate to `pad_width`.
        combine (callable|None): The function for combining shape values.
            If None, uses the corresponding dim from the shape.

    Returns:
        pad_width (int|tuple[tuple[int]]): The absolute `pad_width`.
            If input `pad_width` is not Iterable, result is not Iterable.

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
def gradients(
        arr,
        dims=None,
        ft_factor=(2 * np.pi),
        pad_width=0):
    """
    Apply the gradient operator (in the Fourier domain).

    Args:
        arr (np.ndarray): The input array.
        dims (int|Iterable[int]): The direction of the gradient.
            Values must be between `len(shape)` and `-len(shape)`.
        ft_factor (float): The Fourier factor for the gradient operator.
            Should be either 1 or 2*pi, depending on DFT implementation.
        pad_width (float|int): Size of the border to use.
            This is useful for mitigating border effects.
            If int, it is interpreted as absolute size.
            If float, it is interpreted as relative to the maximum size.

    Returns:
        arrs (np.ndarray): The output array.

    See Also:
        gradient_kernels()
    """
    if pad_width:
        shape = arr.shape
        pad_width = auto_pad_width(pad_width, shape)
        # mask = [slice(borders, -borders)] * arr.ndim
        mask = [slice(lower, -upper) for (lower, upper) in pad_width]
        arr = np.pad(arr, pad_width, 'constant', constant_values=0)
    else:
        mask = [slice(None)] * arr.ndim
    arrs = tuple(
        (((-1j * ft_factor) ** 2) * ifftn(fftshift(kk) * fftn(arr)))[mask]
        for kk in gradient_kernels(arr.shape, dims, arr.shape))
    return arrs


# ======================================================================
def exp_gradients(
        arr,
        dims=None,
        ft_factor=(2 * np.pi),
        pad_width=0):
    """
    Apply the exponential gradient operator (in the Fourier domain).

    Args:
        arr (np.ndarray): The input array.
        dims (int|Iterable[int]): The direction of the gradient.
            Values must be between `len(shape)` and `-len(shape)`.
        ft_factor (float): The Fourier factor for the gradient operator.
            Should be either 1 or 2*pi, depending on DFT implementation.
        pad_width (float|int): Size of the border to use.
            This is useful for mitigating border effects.
            If int, it is interpreted as absolute size.
            If float, it is interpreted as relative to the maximum size.

    Returns:
        arrs (np.ndarray): The output array.

    See Also:
        exp_gradient_kernels()
    """
    if pad_width:
        shape = arr.shape
        pad_width = auto_pad_width(pad_width, shape)
        # mask = [slice(borders, -borders)] * arr.ndim
        mask = [slice(lower, -upper) for (lower, upper) in pad_width]
        arr = np.pad(arr, pad_width, 'constant', constant_values=0)
    else:
        mask = [slice(None)] * arr.ndim
    arrs = tuple(
        (((-1j * ft_factor) ** 2) * ifftn(fftshift(kk) * fftn(arr)))[mask]
        for kk in exp_gradient_kernels(arr.shape, dims, arr.shape))
    return arrs


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
        pad_width (float|int|Iterable[float|int]): Size of the padding to use.
            This is useful for mitigating border effects.
            If Iterable, a value for each dim must be specified.
            If not Iterable, all dims will have the same value.
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
    kk_2 = fftshift(laplace_kernel(arr.shape, arr.shape))
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
    kk_2 = fftshift(laplace_kernel(arr.shape, arr.shape))
    kk_2[kk_2 != 0] = 1.0 / kk_2[kk_2 != 0]
    arr = ((-1j / ft_factor) ** 2) * ifftn(kk_2 * fftn(arr))
    return arr[mask]


# ======================================================================
def auto_bin(
        arr,
        method='auto',
        dim=1):
    """
    Determine the optimal number of bins for histogram of an array.

    Args:
        arr (np.ndarray): The input array.
        method (str|None): The estimation method.
            Accepted values (with: N the array size, D the histogram dim):
             - 'auto': max('fd', 'sturges')
             - 'sqrt': Square-root choice (fast, independent of `dim`)
               n = sqrt(N)
             - 'sturges': Sturges' formula (tends to underestimate)
               n = 1 + log_2(N)
             - 'rice': Rice Rule (fast with `dim` dependence)
               n = 2 * N^(1/(2 + D))
             - 'riced': Modified Rice Rule (fast with strong `dim` dependence)
               n = (1 + D) * N^(1/(2 + D))
             - 'scott': Scott's normal reference rule (depends on data)
               n = N^(1/(2 + D)) *  / (3.5 * SD(arr)
             - 'fd': Freedman–Diaconis' choice (robust variant of 'scott')
               n = N^(1/(2 + D)) * range(arr) / 2 * (Q75 - Q25)
             - 'doane': Doane's formula (correction to Sturges'):
               n = 1 + log_2(N) + log_2(1 + |g1| / sigma_g1)
               where g1 = (|mean|/sigma) ** 3 is the skewness
               and sigma_g1 = sqrt(6 * (N - 2) / ((N + 1) * (N + 3))) is the
               estimated standard deviation on the skewness.
             - None: n = N
        dim (int): The dimension of the histogram.

    Returns:
        num (int): The number of bins.

    Examples:
        >>> arr = np.arange(100)
        >>> auto_bin(arr)
        8
        >>> auto_bin(arr, 'sqrt')
        10
        >>> auto_bin(arr, 'auto')
        8
        >>> auto_bin(arr, 'sturges')
        8
        >>> auto_bin(arr, 'rice')
        10
        >>> auto_bin(arr, 'riced')
        14
        >>> auto_bin(arr, 'scott')
        5
        >>> auto_bin(arr, 'fd')
        5
        >>> auto_bin(arr, None)
        100
        >>> auto_bin(arr, 'sqrt', 2)
        10
        >>> auto_bin(arr, 'auto', 2)
        8
        >>> auto_bin(arr, 'sturges', 2)
        8
        >>> auto_bin(arr, 'rice', 2)
        7
        >>> auto_bin(arr, 'riced', 2)
        13
        >>> auto_bin(arr, 'scott', 2)
        4
        >>> auto_bin(arr, 'fd', 2)
        4
        >>> auto_bin(arr, None, 2)
        100
        >>> np.random.seed(0)
        >>> arr = np.random.random(100) * 1000
        >>> arr /= np.sum(arr)
        >>> auto_bin(arr, 'scott')
        5
        >>> auto_bin(arr, 'fd')
        5
        >>> auto_bin(arr, 'scott', 2)
        4
        >>> auto_bin(arr, 'fd', 2)
        4

    References:
         - https://en.wikipedia.org/wiki/Histogram#Number_of_bins_and_width
    """
    if method == 'auto':
        num = max(auto_bin(arr, 'fd', dim), auto_bin(arr, 'sturges', dim))
    elif method == 'sqrt':
        num = int(np.ceil(np.sqrt(arr.size)))
    elif method == 'sturges':
        num = int(np.ceil(1 + np.log2(arr.size)))
    elif method == 'rice':
        num = int(np.ceil(2 * arr.size ** (1 / (2 + dim))))
    elif method == 'riced':
        num = int(np.ceil((2 + dim) * arr.size ** (1 / (2 + dim))))
    elif method == 'scott':
        h = 3.5 * np.std(arr) / arr.size ** (1 / (2 + dim))
        num = int(np.ceil(np.ptp(arr) / h))
    elif method == 'fd':
        q75, q25 = np.percentile(arr, [75, 25])
        h = 2 * (q75 - q25) / arr.size ** (1 / (2 + dim))
        num = int(np.ceil(np.ptp(arr) / h))
    elif method == 'doane':
        g1 = (np.abs(np.mean(arr)) / np.std(arr)) ** 3
        sigma_g1 = np.sqrt(
            6 * (arr.size - 2) / ((arr.size + 1) * (arr.size + 3)))
        num = int(np.ceil(
            1 + np.log2(arr.size) + np.log2(1 + np.abs(g1) / sigma_g1)))
    else:
        num = arr.size
    return num


# ======================================================================
def auto_bins(
        arrs,
        method='rice',
        dim=None,
        combine=max):
    """
    Determine the optimal number of bins for a histogram of a group of arrays.

    Args:
        arrs (Iterable[np.ndarray]): The input arrays.
        method (str|Iterable[str]|None): The method for calculating bins.
            If str, the same method is applied to both arrays.
            See `pymrt.utils.auto_bin()` for available methods.
        dim (int|None): The dimension of the histogram.
        combine (callable|None): Combine each bin using the combine function.
            combine(n_bins) -> n_bin
            n_bins is of type Iterable[int]

    Returns:
        n_bins (int|tuple[int]): The number of bins.
            If combine is None, returns a tuple of int (one for each input
            array).

    Examples:
        >>> arr1 = np.arange(100)
        >>> arr2 = np.arange(200)
        >>> arr3 = np.arange(300)
        >>> auto_bins((arr1, arr2))
        8
        >>> auto_bins((arr1, arr2, arr3))
        7
        >>> auto_bins((arr1, arr2), ('sqrt', 'sturges'))
        10
        >>> auto_bins((arr1, arr2), combine=None)
        (7, 8)
        >>> auto_bins((arr1, arr2), combine=min)
        7
        >>> auto_bins((arr1, arr2), combine=sum)
        15
        >>> auto_bins((arr1, arr2), combine=lambda x: abs(x[0] - x[1]))
        1
    """
    if isinstance(method, str) or method is None:
        method = (method,) * len(arrs)
    if not dim:
        dim = len(arrs)
    n_bins = []
    for arr, method in zip(arrs, method):
        n_bins.append(auto_bin(arr, method, dim))
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
        bins='rice'):
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
            If str, a method accepted by `auto_bins()` is expected.
            If None, uses the `auto_bins()` default value.
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
def norm_mutual_information(
        arr1,
        arr2,
        bins='rice'):
    """
    Calculate a normalized mutual information between two arrays.

    Note that the numerical result depends on the number of bins.

    Args:
        arr1 (np.ndarray): The first input array.
        arr2 (np.ndarray): The second input array.
        bins (int|str|None): The number of bins to use for the distribution.
            If int, the exact number is used.
            If str, a method accepted by `auto_bin` is expected.
            If None, uses the maximum number of bins (not recommended).

    Returns:
        mi (float): The normalized mutual information.

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
    # todo: check if this is correct
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
def mutual_information(
        arr1,
        arr2,
        base=np.e,
        bins='rice'):
    """
    Calculate the mutual information between two arrays.

    Note that the numerical result depends on the number of bins.

    Args:
        arr1 (np.ndarray): The first input array.
        arr2 (np.ndarray): The second input array.
        base (int|float|None): The base units to express the result.
            Should be a number larger than 1.
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
          General Definitions of Entropy for Multimodal Image Registration.”
          In International Workshop on Biomedical Image Registration,
          258–268. Springer, 2010.
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
    #     hist + np.finfo(np.float).eps, lambda_='log-likelihood')
    # mi = g / hist.sum() / 2

    if base:
        # entropy-based implementation
        hist1, bin_edges1 = np.histogram(arr1, bins)
        hist2, bin_edges2 = np.histogram(arr2, bins)
        hist12, x_edges, y_edges = np.histogram2d(arr1, arr2, bins=bins)
        h12 = entropy(hist12, base)
        h1 = entropy(hist1, base)
        h2 = entropy(hist2, base)
        mi = h1 + h2 - h12
    else:
        norm_mutual_information(arr1, arr2, bins=bins)

    # absolute value to fix rounding errors
    return abs(mi)


# ======================================================================
def gaussian_nd(
        shape,
        sigmas,
        position=0.5,
        n_dim=None,
        norm=np.sum,
        rel_position=True):
    """
    Generate a Gaussian distribution in N dimensions.

    Args:
        shape (int|Iterable[int]): The shape of the array in px.
        sigmas (Iterable[int|float]): The standard deviation in px.
        position (float|Iterable[float]): The position of the center.
            Values are relative to the lowest edge, and scaled by the
            corresponding shape size.
        n_dim (int|None): The number of dimensions.
            If None, the number of dims is guessed from the other parameters.
        norm (callable|None): Normalize using the specified function.
        rel_position (bool): Interpret positions as relative values.
            If True, position values are interpreted as relative,
            i.e. they are scaled for `shape` values.
            Otherwise, they are interpreted as absolute (in px).
            Uses `utils.grid_coord()` internally.

    Returns:
        arr (np.ndarray): The array containing the N-dim Gaussian.

    Examples:
        >>> gaussian_nd(8, 1)
        array([ 0.00087271,  0.01752886,  0.12952176,  0.35207666,  0.35207666,
                0.12952176,  0.01752886,  0.00087271])
        >>> gaussian_nd(9, 2)
        array([ 0.02763055,  0.06628225,  0.12383154,  0.18017382,  0.20416369,
                0.18017382,  0.12383154,  0.06628225,  0.02763055])
        >>> gaussian_nd(3, 1, n_dim=2)
        array([[ 0.07511361,  0.1238414 ,  0.07511361],
               [ 0.1238414 ,  0.20417996,  0.1238414 ],
               [ 0.07511361,  0.1238414 ,  0.07511361]])
        >>> gaussian_nd(7, 2, norm=None)
        array([ 0.32465247,  0.60653066,  0.8824969 ,  1.        ,  0.8824969 ,
                0.60653066,  0.32465247])
        >>> gaussian_nd(4, 2, 1.0, norm=None)
        array([ 0.32465247,  0.60653066,  0.8824969 ,  1.        ])
        >>> gaussian_nd(3, 2, 5.0)
        array([ 0.00982626,  0.10564222,  0.88453152])
        >>> gaussian_nd(3, 2, 5.0, norm=None)
        array([  3.72665317e-06,   4.00652974e-05,   3.35462628e-04])
    """
    if not n_dim:
        n_dim = combine_iter_len((shape, sigmas, position))

    shape = auto_repeat(shape, n_dim)
    sigmas = auto_repeat(sigmas, n_dim)
    position = auto_repeat(position, n_dim)

    position = grid_coord(
        shape, position, is_relative=rel_position, use_int=False)
    arr = np.exp(-(sum([
        x_i ** 2 / (2 * sigma ** 2) for x_i, sigma in zip(position, sigmas)])))
    if callable(norm):
        arr /= norm(arr)
    return arr


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
        weights (int|Iterable): The running weights.
            If int, the number of elements to group in the 'running' axis and
            unity weights are used.
            The size of the weights array len(weights) must be such that
            len(weights) >= 1 and len(weights) <= len(array), otherwise the
            flattened array is returned.
        **kws (dict): Keyword arguments passed to `scipy.signal.convolve`.

    Returns:
        arr (np.ndarray): The output array.

    Examples:
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
        num (int|Iterable): The running window size.
            The number of elements to group.

    Returns:
        arr (np.ndarray): The output array.

    Examples:
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
        weights (int|Iterable): The running weights.
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
        borders (str|complex|Iterable[complex]|None): The border parameters.
            If int or float, the value is repeated at the borders.
            If Iterable of int, float or complex, the first and last values are
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
                (np.full((num - 1,), arr[-1]),
                 np.full((num - 1,), arr[0])))
        elif borders == 'circ':
            extension = arr
        elif borders == 'sym':
            extension = arr[::-1]
        elif isinstance(borders, (int, float, complex)):
            extension = np.full((num - 1,), borders)
        elif isinstance(borders, (tuple, float)):
            extension = np.concatenate(
                (np.full((num - 1,), borders[-1]),
                 np.full((num - 1,), borders[0])))
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
        weights (int|Iterable): The running weights.
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
            If Iterable of int, float or complex, the first and last values are
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
            head = np.full((num - 1,), arr[0])
            tail = np.full((num - 1,), arr[-1])
        elif borders == 'circ':
            tail = arr[:num - 1]
            head = arr[-num + 1:]
        elif borders == 'sym':
            tail = arr[-num + 1:]
            head = arr[:num - 1]
        elif isinstance(borders, (int, float, complex)):
            head = tail = np.full((num - 1,), borders)
        elif isinstance(borders, (tuple, float)):
            head = np.full((num - 1,), borders[0])
            tail = np.full((num - 1,), borders[-1])
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
             - 'cartesian': apply to real and imaginary separately.
             - 'polar': apply to magnitude and phase separately.
             - 'real': apply to real part only.
             - 'imag': apply to imaginary part only.
             - 'mag': apply to magnitude part only.
             - 'phs': apply to phase part only.
            If unknown, uses default.

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
                    1j * filter_func(np.angle(arr), *filter_args,
                        **filter_kws)))
    elif mode == 'real':
        arr = (
                filter_func(arr.real, *filter_args,
                    **filter_kws) + 1j * arr.imag)
    elif mode == 'imag':
        arr = (
                arr.real + 1j * filter_func(arr.imag, *filter_args,
            **filter_kws))
    elif mode == 'mag':
        arr = (
                filter_func(np.abs(arr), *filter_args, **filter_kws) *
                np.exp(1j * np.angle(arr)))
    elif mode == 'phs':
        arr = (
                np.abs(arr) * np.exp(
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
        items (Iterable): The collection of items to inspect.
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
        items (Iterable): The collection of items to inspect.
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
        items (Iterable): The collection of items to inspect.
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
        items (Iterable): The collection of items to inspect.
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
def otsu_threshold(
        items,
        bins='sqrt'):
    """
    Optimal foreground/background threshold value based on Otsu's method.

    Args:
        items (Iterable): The input items.
        bins (int|str|None): Number of bins used to calculate histogram.
            If str or None, this is automatically calculated from the data
            using `utils.auto_bin()` with `method` set to `bins` if str,
            and using the default `utils.auto_bin()` method if set to None.

    Returns:
        threshold (float): The threshold value.

    Raises:
        ValueError: If `arr` only contains a single value.

    Examples:
        >>> num = 1000
        >>> x = np.linspace(-10, 10, num)
        >>> arr = np.sin(x) ** 2
        >>> threshold = otsu_threshold(arr)
        >>> round(threshold, 1)
        0.5

    References:
        - Otsu, N., 1979. A Threshold Selection Method from Gray-Level
          Histograms. IEEE Transactions on Systems, Man, and Cybernetics 9,
          62–66. doi:10.1109/TSMC.1979.4310076
    """
    # ensure items are not identical.
    items = np.array(items)
    if items.min() == items.max():
        warnings.warn('Items are all identical!')
        threshold = items.min()
    else:
        if isinstance(bins, str):
            bins = auto_bin(items, bins)
        elif bins is None:
            bins = auto_bin(items)

        hist, bin_edges = np.histogram(items, bins)
        bin_centers = midval(bin_edges)
        hist = hist.astype(float)

        # class probabilities for all possible thresholds
        weight1 = np.cumsum(hist)
        weight2 = np.cumsum(hist[::-1])[::-1]
        # class means for all possible thresholds
        mean1 = np.cumsum(hist * bin_centers) / weight1
        mean2 = (np.cumsum((hist * bin_centers)[::-1]) / weight2[::-1])[::-1]
        # calculate the variance for all possible thresholds
        variance12 = weight1[:-1] * weight2[1:] * (mean1[:-1] - mean2[1:]) ** 2

        i_max_variance = np.argmax(variance12)
        threshold = bin_centers[:-1][i_max_variance]
    return threshold


# ======================================================================
def auto_num_components(
        k,
        q=None,
        num=None,
        verbose=D_VERB_LVL):
    """
    Calculate the optimal number of principal components.

    Effectively executing a Principal Component Analysis.

    Args:
        k (int|float|str): The number of principal components.
            If int, the exact number is given. It must not exceed the size
            of the `coil_axis` dimension.
            If float, the number is interpreted as relative to the size of
            the `coil_axis` dimension, and values must be in the
            [0.1, 1] interval.
            If str, the number is automatically estimated from the magnitude
            of the eigenvalues using a specific method.
            Accepted values are:
             - 'all': use all components.
             - 'full': same as 'all'.
             - 'elbow': use `utils.marginal_sep_elbow()`.
             - 'quad': use `utils.marginal_sep_quad()`.
             - 'quad_weight': use `utils.marginal_sep_quad_weight()`.
             - 'quad_inv_weight': use `utils.marginal_sep_quad_inv_weight()`.
             - 'otsu': use `segmentation.threshold_otsu()`.
             - 'X%': set the threshold at 'X' percent of the largest eigenval.
        q (Iterable[int|float|complex]|None): The values of the components.
            If None, `num` must be specified.
            If Iterable, `num` must be None.
        num (int|None): The number of components.
            If None, `q` must be specified.
            If
        verbose (int): Set level of verbosity.

    Returns:
        k (int): The optimal number of principal components.

    Examples:
        >>> q = [100, 90, 70, 10, 5, 3, 2, 1]
        >>> auto_num_components('elbow', q)
        4
        >>> auto_num_components('quad_weight', q)
        5
    """
    if (q is None and num is None) or (q is not None and num is not None):
        raise ValueError('At most one of `q` and `num` must not be `None`.')
    elif q is not None and num is None:
        q = np.array(q).ravel()
        msg('q={}'.format(q), verbose, VERB_LVL['debug'])
        num = len(q)

    msg('k={}'.format(k), verbose, VERB_LVL['debug'])
    if isinstance(k, float):
        k = max(1, int(num * min(k, 1.0)))
    elif isinstance(k, str):
        if q is not None:
            k = k.lower()
            if k == 'elbow':
                k = marginal_sep_elbow(np.abs(q / q[0])) % (num + 1)
            elif k == 'quad':
                k = marginal_sep_quad(np.abs(q / q[0])) % (num + 1)
            elif k == 'quad_weight':
                k = marginal_sep_quad_weight(np.abs(q / q[0])) % (num + 1)
            elif k == 'quad_inv_weight':
                k = marginal_sep_quad_inv_weight(np.abs(q / q[0])) % (num + 1)
            elif k.endswith('%') and (100.0 > float(k[:-1]) >= 0.0):
                k = np.abs(q[0]) * float(k[:-1]) / 100.0
                k = np.where(np.abs(q) < k)[0]
                k = k[0] if len(k) else num
            elif k == 'otsu':
                k = otsu_threshold(q)
                k = np.where(q < k)[0]
                k = k[0] if len(k) else num
            elif k == 'all' or k == 'full':
                k = num
            else:
                warnings.warn('`{}`: invalid value for `k`.'.format(k))
                k = num
        else:
            warnings.warn('`{}`: method requires `q`.'.format(k))
            k = num
    if not 0 < k <= num:
        warnings.warn('`{}` is invalid. Using: `{}`.'.format(k, num))
        k = num
    msg('k/num={}/{}'.format(k, num), verbose, VERB_LVL['medium'])
    return k


# ======================================================================
def avg(
        arr,
        axis=None,
        dtype=None,
        out=None,
        keepdims=np._NoValue,
        weights=None,
        removes=(np.inf, -np.inf)):
    """
    Calculate the (weighted) average of the array.

    The weighted average is defined as:

    .. math::
        avg(x, w) = \\frac{\\sum_i w_i x_i}{\\sum_i w_i}

    where :math:`x` is the input N-dim array, :math:`w` is the N-dim array of
    the weights, and :math:`i` runs through the dimension along which to
    compute.

    Args:
        arr (np.ndarray|Iterable): The input data.
        axis (int|Iterable[int]|None): Axis along which to compute.
            See `np.nansum()` for more info.
        dtype (np.dtype|None): The data type of the result.
            See `np.nansum()` for more info.
        out (np.ndarray|None):
            See `np.nansum()` for more info.
        keepdims (bool): Keep reduced axis in the result as dims with size 1.
            See `np.nansum()` for more info.
        weights (np.ndarray|Iterable|None): The weights.
            If np.ndarray or Iterable, the size must match with `arr`.
            If None, all wegiths are set to 1 (equivalent to no weighting).
        removes (Iterable): Values to remove.
            If empty, no values will be removed.

    Returns:
        result (np.ndarray): The computed statistics.
            Its shape depends on the value of axis.

    Examples:
        >>> arr = np.array([0, 0, 1, 0])
        >>> weights = np.array([1, 1, 3, 1])
        >>> avg(arr)
        0.25
        >>> avg(arr, weights=weights)
        0.5
        >>> avg(arr, weights=weights) == avg(np.array([0, 0, 1, 0, 1, 1]))
        True
        >>> np.mean(arr) == avg(arr)
        True
        >>> arr = np.arange(2 * 3 * 4, dtype=float).reshape((2, 3, 4))
        >>> weights = np.arange(4) + 1
        >>> avg(arr, weights=weights, axis=-1)
        array([[  2.,   6.,  10.],
               [ 14.,  18.,  22.]])
        >>> weights = np.arange(2 * 3).reshape((2, 3)) + 1
        >>> avg(arr, weights=weights, axis=(0, 1), removes=(1,))
        array([ 13.33333333,  15.        ,  15.33333333,  16.33333333])

    See Also:
        var(), std()
    """
    arr = np.array(arr)
    if np.issubdtype(arr.dtype, int):
        arr = arr.astype(float)
    if weights is not None:
        weights = np.array(weights, dtype=float)
        if weights.shape != arr.shape:
            weights = unsqueeze(
                weights, axis=axis, shape=arr.shape, complement=True)
            # cannot use `np.broadcast_to()` because we need to write data
            weights = np.zeros_like(arr) + weights
    for val in removes:
        mask = arr == val
        if val in arr:
            arr[mask] = np.nan
            if weights is not None:
                weights[mask] = np.nan
    if weights is None:
        weights = np.ones_like(arr)
    result = np.nansum(
        arr * weights, axis=axis, dtype=dtype, out=out, keepdims=keepdims)
    result /= np.nansum(
        weights, axis=axis, dtype=dtype, out=out, keepdims=keepdims)
    return result


# ======================================================================
def var(
        arr,
        axis=None,
        dtype=None,
        out=None,
        keepdims=np._NoValue,
        weights=None,
        removes=(np.inf, -np.inf)):
    """
    Calculate the (weighted) variance of the array.

    The weighted variance is defined as:

    .. math::
        var(x, w) = \\frac{\\sum_i (w_i x_i - avg(x, w))^2}{\\sum_i w_i}

    where :math:`x` is the input N-dim array, :math:`w` is the N-dim array of
    the weights, and :math:`i` runs through the dimension along which to
    compute.

    Args:
        arr (np.ndarray|Iterable): The input data.
        axis (int|Iterable[int]|None): Axis along which to compute.
            See `np.nansum()` for more info.
        dtype (np.dtype|None): The data type of the result.
            See `np.nansum()` for more info.
        out (np.ndarray|None):
            See `np.nansum()` for more info.
        keepdims (bool): Keep reduced axis in the result as dims with size 1.
            See `np.nansum()` for more info.
        weights (np.ndarray|Iterable|None): The weights.
            If np.ndarray or Iterable, the size must match with `arr`.
            If None, all wegiths are set to 1 (equivalent to no weighting).
        removes (Iterable): Values to remove.
            If empty, no values will be removed.

    Returns:
        result (np.ndarray): The computed statistics.
            Its shape depends on the value of axis.

    See Also:
        avg(), std()

    Examples:
        >>> arr = np.array([0, 0, 1, 0])
        >>> weights = np.array([1, 1, 3, 1])
        >>> var(arr, weights=weights)
        0.25
        >>> var(arr, weights=weights) == var(np.array([0, 0, 1, 0, 1, 1]))
        True
        >>> np.var(arr) == var(arr)
        True
        >>> arr = np.arange(2 * 3 * 4, dtype=float).reshape((2, 3, 4))
        >>> weights = np.arange(4) + 1
        >>> var(arr, weights=weights, axis=-1)
        array([[ 0.8,  0.8,  0.8],
               [ 0.8,  0.8,  0.8]])
        >>> weights = np.arange(2 * 3).reshape((2, 3)) + 1
        >>> var(arr, weights=weights, axis=(0, 1), removes=(1,))
        array([ 28.44444444,  26.15384615,  28.44444444,  28.44444444])
    """
    arr = np.array(arr)
    if weights is not None:
        weights = np.array(weights, dtype=float)
    avg_arr = avg(
        arr, axis=axis, dtype=dtype, out=out, keepdims=True,
        weights=weights, removes=removes)
    result = avg(
        (arr - avg_arr) ** 2, axis=axis, dtype=dtype, out=out,
        keepdims=keepdims,
        weights=weights ** 2 if weights is not None else None, removes=removes)
    return result


# ======================================================================
def std(
        arr,
        axis=None,
        dtype=None,
        out=None,
        keepdims=np._NoValue,
        weights=None,
        removes=(np.inf, -np.inf)):
    """
    Calculate the (weighted) standard deviation of the array.

    The weighted standard deviation is defined as the square root of the
    variance.

    Args:
        arr (np.ndarray|Iterable): The input data.
        axis (int|Iterable[int]|None): Axis along which to compute.
            See `np.nansum()` for more info.
        dtype (np.dtype|None): The data type of the result.
            See `np.nansum()` for more info.
        out (np.ndarray|None):
            See `np.nansum()` for more info.
        keepdims (bool): Keep reduced axis in the result as dims with size 1.
            See `np.nansum()` for more info.
        weights (np.ndarray|Iterable|None): The weights.
            If np.ndarray or Iterable, the size must match with `arr`.
            If None, all wegiths are set to 1 (equivalent to no weighting).
        removes (Iterable): Values to remove.
            If empty, no values will be removed.

    Returns:
        result (np.ndarray): The computed statistics.
            Its shape depends on the value of axis.

    Examples:
        >>> arr = np.array([0, 0, 1, 0])
        >>> weights = np.array([1, 1, 3, 1])
        >>> std(arr, weights=weights)
        0.5
        >>> std(arr, weights=weights) == std(np.array([0, 0, 1, 0, 1, 1]))
        True
        >>> np.std(arr) == std(arr)
        True

    See Also:
        avg(), var()
    """
    return np.sqrt(
        var(arr, axis=axis, dtype=dtype, out=out, keepdims=keepdims,
            weights=weights, removes=removes))


# ======================================================================
def gavg(
        arr,
        axis=None,
        dtype=None,
        out=None,
        keepdims=np._NoValue,
        weights=None,
        removes=(np.inf, -np.inf)):
    """
    Calculate the (weighted) geometric average of the array.

    The weighted geometric average is defined as exponential of the
    weighted average of the logarithm of the absolute value of the array.

    Args:
        arr (np.ndarray|Iterable): The input data.
        axis (int|Iterable[int]|None): Axis along which to compute.
            See `np.nansum()` for more info.
        dtype (np.dtype|None): The data type of the result.
            See `np.nansum()` for more info.
        out (np.ndarray|None):
            See `np.nansum()` for more info.
        keepdims (bool): Keep reduced axis in the result as dims with size 1.
            See `np.nansum()` for more info.
        weights (np.ndarray|Iterable|None): The weights.
            If np.ndarray or Iterable, the size must match with `arr`.
            If None, all wegiths are set to 1 (equivalent to no weighting).
        removes (Iterable): Values to remove.
            If empty, no values will be removed.

    Returns:
        result (np.ndarray): The computed statistics.
            Its shape depends on the value of axis.

    Examples:
        >>> arr = np.array([1, 1, 4, 1])
        >>> weights = np.array([1, 1, 3, 1])
        >>> gavg(arr, weights=weights)
        2.0
        >>> gavg(arr, weights=weights) == gavg(np.array([1, 1, 4, 1, 4, 4]))
        True
        >>> sp.stats.gmean(arr) == gavg(arr)
        True

    See Also:
        avg()
    """
    return np.exp(
        avg(np.log(np.abs(arr)), axis=axis, dtype=dtype, out=out,
            keepdims=keepdims, weights=weights, removes=removes))


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
        removes (Iterable): Values to remove.
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
        axes (Iterable[int]|int): The slicing axis
        indexes (Iterable[int|float|None]|None): The slicing index.
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
