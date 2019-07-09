#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PyMRT: read/write files with BART's CFL structure.

The module is NumPy-aware.

See: https://github.com/mrirecon/bart
"""

# ======================================================================
# :: Future Imports
from __future__ import (
    division, absolute_import, print_function, unicode_literals, )

# ======================================================================
# :: Python Standard Library Imports
import os  # Miscellaneous operating system interfaces

# :: External Imports
import numpy as np  # NumPy (multidimensional numerical arrays library)

# :: Local Imports
import pymrt as mrt
from pymrt import INFO, PATH
from pymrt import VERB_LVL, D_VERB_LVL, VERB_LVL_NAMES
from pymrt import elapsed, report, run_doctests
from pymrt import msg, dbg


# ======================================================================
# :: additional globals


# ======================================================================
def read(
        filepath,
        dirpath='.'):
    """
    Read a CFL header+data pair.

    Args:
        filepath (str): The file path.
            If extension is not set, it will be generated automatically,
            otherwise either of '.hdr' or '.cfl' can be used.
            Corresponding '.hdr' and '.cfl' files must exist.
        dirpath (str): The working directory.

    Returns:
        arr (ndarray): The data read.
    """
    # determine base filepath
    mask = slice(None, -4, None) \
        if filepath.endswith('.hdr') or filepath.endswith('.cfl') \
        else slice(None)
    base_filepath = filepath[mask]

    if dirpath != '.':
        base_filepath = os.path.join(dirpath, base_filepath)

    # load header
    with open(base_filepath + '.hdr', 'r') as header_file:
        header_file.readline()  # skip comment line
        dim_line = header_file.readline()

    # obtain the shape of the image
    shape = [int(i) for i in dim_line.strip().split(' ')]
    # remove trailing singleton dimensions from shape
    while shape[-1] == 1:
        shape.pop(-1)
    # calculate the data size
    data_size = int(np.prod(shape))

    # load data
    with open(base_filepath + '.cfl', 'r') as data_file:
        arr = np.fromfile(
            data_file, dtype=np.complex64, count=data_size)

    # BART uses FORTRAN-style memory allocation
    return arr.reshape(shape, order='F')


# ======================================================================
def write(
        arr,
        filepath,
        dirpath='.',
        default_dims=(1,) * 16):
    """
    Write a CFL header+data pair.

    Args:
        arr (ndarray): The array to save.
        filepath (str): The file path.
            If extension is not set, it will be generated automatically,
            otherwise either of '.hdr' or '.cfl' can be used.
            Corresponding '.hdr' and '.cfl' files will be created/overwritten.
        dirpath (str): The working directory.

    Returns:
        None.
    """
    # determine base filepath
    mask = slice(None, -4, None) \
        if filepath.endswith('.hdr') or filepath.endswith('.cfl') \
        else slice(None)
    base_filepath = filepath[mask]

    if dirpath != '.':
        base_filepath = os.path.join(dirpath, base_filepath)

    # save header
    with open(base_filepath + '.hdr', 'w') as header_file:
        header_file.write(str('# Dimensions\n'))
        dimensions = arr.shape + default_dims[arr.ndim:]
        header_file.write(str(' '.join([str(d) for d in dimensions])) + '\n')

    # save data
    with open(base_filepath + '.cfl', 'w') as data_file:
        # BART uses FORTRAN-style memory allocation with column-major ordering
        arr.T.astype(np.complex64).tofile(data_file)


# ======================================================================
elapsed(__file__[len(PATH['base']) + 1:])

# ======================================================================
if __name__ == '__main__':
    run_doctests(__doc__)
