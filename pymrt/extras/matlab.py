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
    division, absolute_import, print_function, unicode_literals, )

# ======================================================================
# :: Python Standard Library Imports
import os  # Miscellaneous operating system interfaces
import warnings  # Warning control

# :: External Imports
import numpy as np  # NumPy (multidimensional numerical arrays library)
import scipy as sp  # SciPy (signal and image processing library)
import h5py  # Read and write HDF5 files from Python

# :: External Imports Submodules
import scipy.io  # SciPy: Input and output

# :: Local Imports
import pymrt as mrt
import pymrt.utils
import pymrt.input_output

from pymrt import INFO, DIRS
from pymrt import VERB_LVL, D_VERB_LVL, VERB_LVL_NAMES
from pymrt import elapsed, report
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
            verbose, VERB_LVL['debug'])
    except NotImplementedError:
        # load new-style MATLAB v7.3+ files (effectively HDF5)
        data = {}
        h5file = h5py.File(filepath)
        for k, v in h5file.items():
            if not exclude_keys(k):
                data[k] = np.array(v)
        msg('Loaded using MATLAB v7.3+ compatibility layer.',
            verbose, VERB_LVL['debug'])
    return data


# ======================================================================
def write(
        data,
        filename,
        dirpath='.',
        verbose=D_VERB_LVL):
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
    """
    if not filename.lower().endswith('.mat'):
        filename += '.mat'
    filepath = os.path.join(dirpath, filename)

    sp.io.savemat(filepath, data)
    msg('Saved using MATLAB v4 l1, v6, v7 <= v7.2 compatibility layer.',
        verbose, VERB_LVL['debug'])


# ======================================================================
def auto_convert(
        filepaths,
        out_template='{basepath}__{name}.nii.gz',
        save_kws=None,
        on_exist='unique',
        verbose=D_VERB_LVL):
    """
    Automatically convert MATLAB data to PyMRT's standard format (NIfTI).

    Args:
        filepaths (Iterable[str]): List of filepaths to convert.
            Can be combined with file listing functions from `pymrt.utils`,
            e.g. `iflistdir()`, `flistdir()`, `listdir()` (directly) or
            `iwalk2()`, `walk2()` (indirectly).
        out_template (str): Template for the output filepath.
            The following variables are available for interpolation:
             - `dirpath`: The directory of the input.
             - `base`: The input base file name without extension.
             - `ext`: The input file extension (with leading separator).
             - `basepath`: The input filepath without extension.
             - `name`: The name of the variable as stored in the MATLAB file.
        save_kws (dict|None): Keyword parameters for `input_output.save()`.
        on_exist (str): Determine what to do if output exists.
            Accepted values are:
             - 'unique': Generate a new unique filepath.
             - 'skip': Skip saving the data.
             - 'overwrite': Overwrites the data onto existing filepath!
        verbose (int): Set level of verbosity.

    Returns:
        None.
    """
    if save_kws is None:
        save_kws = {}
    on_exist = on_exist.lower()
    for in_filepath in filepaths:
        if os.path.isfile(in_filepath):
            msg('Loading: {}'.format(in_filepath), verbose, D_VERB_LVL)
            for name, arr in read(in_filepath, verbose=verbose).items():
                dirpath, base, ext = mrt.utils.split_path(in_filepath)
                basepath = os.path.join(dirpath, base)
                out_filepath = out_template.format(**locals())
                do_save = True
                if os.path.exists(out_filepath):
                    if on_exist == 'skip':
                        do_save = False
                    elif on_exist == 'overwrite':
                        pass
                    else:
                        if on_exist != 'unique':
                            text = 'Unknown value `{}` for `on_exist`. ' \
                                   'Default to `unique`.'
                            warnings.warn(text)
                        out_filepath = mrt.utils.next_filepath(out_filepath)

                if do_save:
                    msg('Saving: {}'.format(out_filepath), verbose, D_VERB_LVL)
                    try:
                        mrt.input_output.save(out_filepath, arr, **save_kws)
                    except ValueError:
                        msg('W: Saving failed!'.format(out_filepath), verbose,
                            D_VERB_LVL)
                else:
                    msg('Skip: {}'.format(out_filepath), verbose, D_VERB_LVL)


# ======================================================================
elapsed(__file__[len(DIRS['base']) + 1:])

# ======================================================================
if __name__ == '__main__':
    import doctest  # Test interactive Python examples

    msg(__doc__.strip())
    doctest.testmod()
    msg(report())
