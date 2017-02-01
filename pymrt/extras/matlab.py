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
from pymrt import msg, dbg


# ======================================================================
# :: additional globals


# ======================================================================
def read(
        filename,
        dirpath='.'):
    """
    Read a MATLAB files.

    Args:
        filename (str): The input filename.
            The '.mat' extension can be omitted.
        dirpath (str): The working directory.

    Returns:
        data (dict): A dictionary containing the data read.

    Examples:

    """
    filepath = os.path.join(dirpath, filename)

    try:
        # load traditional MATLAB files
        data = sp.io.loadmat('file.mat')
    except NotImplementedError:
        # loda new-style MATLAB v7.3+ files (effectively HDF5)
        data = {}
        h5file = h5py.File(filepath)
        for k, v in h5file.items():
            data[k] = np.array(v)
    return data


# ======================================================================
def write(
        data,
        filename,
        dirpath='.'):
    """
    Write a MATLAB files.

    Args:
        data (dict): A dictionary of arrays to save.
            The dict key is used for the variable name on load.
        filename (str): The input filename.
            The '.mat' extension can be omitted.
        dirpath (str): The working directory.

    Returns:
        None.

    Examples:

    """
    filepath = os.path.join(dirpath, filename)

    sp.io.savemat(filepath, data)


# ======================================================================
if __name__ == '__main__':
    pass
