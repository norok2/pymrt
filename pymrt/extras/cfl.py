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
    division, absolute_import, print_function, unicode_literals)

# ======================================================================
# :: Python Standard Library Imports
import os  # Miscellaneous operating system interfaces

# :: External Imports
import numpy as np  # NumPy (multidimensional numerical arrays library)

# :: Local Imports
import pymrt as mrt
from pymrt import msg, dbg


# ======================================================================
# :: additional globals



# ======================================================================
def read(
        basename,
        dirpath='.'):
    """
    Read a CFL header+data pair.

    Args:
        basename (str): The base filename.
            Corresponding '.hdr' and '.cfl' files must exist.
        dirpath (str): The working directory.

    Returns:
        array (ndarray): The data read.

    Examples:

    """
    filepath = os.path.join(dirpath, basename)

    # load header
    with open(filepath + '.hdr', 'r') as header_file:
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
    with open(filepath + ".cfl", "r") as data_file:
        array = np.fromfile(
            data_file, dtype=np.complex64, count=data_size)

    # note: BART uses FORTRAN-style memory allocation
    return array.reshape(shape, order='F')


# ======================================================================
def write(
        array,
        basename,
        dirpath='.'):
    """
    Write a CFL header+data pair.

    Args:
        array (ndarray): The array to save.
        basename (str): The base filename.
            Corresponding '.hdr' and '.cfl' files will be created/overwritten.
        dirpath (str): The working directory.

    Returns:
        None.
    """
    filepath = os.path.join(dirpath, basename)

    # save header
    with open(filepath + '.hdr', 'w') as header_file:
        header_file.write(str('# Dimensions\n'))
        header_file.write(str(' '.join([str(n) for n in array.shape])) + '\n')

    # save data
    with open(filepath + '.cfl', 'w') as data_file:
        # note: BART uses FORTRAN-style memory allocation
        array.astype(np.complex64, 'F').tofile(data_file)


# ======================================================================
if __name__ == '__main__':
    pass
