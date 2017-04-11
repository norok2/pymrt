#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PyMRT: read/write files with MATLAB structure.

The module is NumPy-aware.

See: https://github.com/mrirecon/bart
"""

# ======================================================================
# :: Future Imports
from __future__ import (
    division, absolute_import, print_function, unicode_literals)

# ======================================================================
# :: Python Standard Library Imports
import os  # Miscellaneous operating system interfaces

# :: External Imports
import numpy as np  # NumPy (multidimensional numerical arrays library)
import scipy as sp  # SciPy (signal and image processing library)
import h5py  # Read and write HDF5 files from Python

# :: External Imports Submodules
import scipy.io  # SciPy: Input and output

# :: Local Imports
import pymrt as mrt
from pymrt import VERB_LVL, D_VERB_LVL, VERB_LVL_NAMES
from pymrt import msg, dbg


# ======================================================================
# :: additional globals


# ======================================================================
def read(
        filename,
        dirpath='.',
        exclude_keys=lambda s: s.startswith('__'),
        verbose=D_VERB_LVL):
    """
    Read a MATLAB file.

    Args:
        filename (str): The input filename.
            The '.mat' extension can be omitted.
        dirpath (str): The working directory.
        exclude_keys (callable): The function used to exclude unwanted keys.
            Defaults to excluding potential private keys.
        verbose (int): Set level of verbosity.

    Returns:
        data (dict): A dictionary containing the data read.

    Examples:

    """
    if not filename.lower().endswith('.mat'):
        filename += '.mat'
    filepath = os.path.join(dirpath, filename)

    if not exclude_keys:
        def exclude_keys(s):
            return True

    try:
        # load traditional MATLAB files
        data = sp.io.loadmat(filepath)
        data = {
            k: np.array(v) for k, v in data.items() if not exclude_keys(k)}
        msg('Loaded using MATLAB v4 l1, v6, v7 <= v7.2 compatibility layer.',
            VERB_LVL['debug'])
    except NotImplementedError:
        # load new-style MATLAB v7.3+ files (effectively HDF5)
        data = {}
        h5file = h5py.File(filepath)
        for k, v in h5file.items():
            if not exclude_keys(k):
                data[k] = np.array(v)
        msg('Loaded using MATLAB v7.3+ compatibility layer.',
            VERB_LVL['debug'])
    return data


# ======================================================================
def write(
        data,
        filename,
        dirpath='.'):
    """
    Write a MATLAB file.

    Args:
        data (dict): A dictionary of arrays to save.
            The dict key is used for the variable name on load.
        filename (str): The input filename.
            The '.mat' extension can be omitted.
        dirpath (str): The working directory.
        verbose (int): Set level of verbosity.

    Returns:
        None.

    Examples:

    """
    if not filename.lower().endswith('.mat'):
        filename += '.mat'
    filepath = os.path.join(dirpath, filename)

    sp.io.savemat(filepath, data)
    msg('Saved using MATLAB v4 l1, v6, v7 <= v7.2 compatibility layer.',
        VERB_LVL['debug'])


# ======================================================================
if __name__ == '__main__':
    pass
