#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PyMRT: read/write files with a JCAMP-DX-like structure.
"""

# ======================================================================
# :: Future Imports
from __future__ import (
    division, absolute_import, print_function, unicode_literals, )

# ======================================================================
# :: Python Standard Library Imports

# :: External Imports
import numpy as np  # NumPy (multidimensional numerical arrays library)

# :: Local Imports
import pymrt as mrt  # Python Magnetic Resonance Tools: the multi-tool of MRI
from pymrt import INFO, PATH
from pymrt import VERB_LVL, D_VERB_LVL, VERB_LVL_NAMES
from pymrt import elapsed, report, run_doctests
from pymrt import msg, dbg, fmt, fmtm


# todo: rewrite using pyparsing

# ======================================================================
def _strip_comments(data, comment_start_str='$$'):
    """
    Strip comments, i.e. lines starting with: $$
    """
    lines = data.split('\n')
    code_lines, comment_lines = [], []
    for line in lines:
        if line.startswith(comment_start_str):
            comment_lines.append(line)
        else:
            code_lines.append(line)
    return '\n'.join(code_lines), '\n'.join(comment_lines)


# ======================================================================
def _auto_convert(val_str):
    """
    Convert value to numeric if possible, or strip '<' and '>' from strings.
    """
    val_str = val_str.strip()
    if _has_str_decorator(val_str):
        val = val_str[1:-1]
    else:
        try:
            val = int(val_str)
        except ValueError:
            try:
                val = float(val_str)
            except ValueError:
                val = val_str
    return val


# ======================================================================
def _has_str_decorator(val):
    """
    Determine if string is delimited by '<' and '>'.
    """
    str_start_id, str_end_id = '<', '>'
    return val.startswith(str_start_id) and val.endswith(str_end_id)


# ======================================================================
def _parse_record(record):
    """
    Parse record to adapt for JCAMP-DX-like specification.
    """
    val = record.strip()
    val = val.replace('\n', '').strip()
    group_start_id, group_end_id = '(', ')'
    list_sep = ', '
    if val.startswith(group_start_id) and val.endswith(group_end_id):
        val_list = val[1:-1].split(list_sep)
        val = [_auto_convert(value) for value in val_list]
    elif val.startswith(group_start_id):
        val_hdr_str = \
            val[val.index(group_start_id) + 1:val.index(group_end_id)]
        val_data_str = val[val.index(group_end_id) + 1:]
        if _has_str_decorator(val_data_str):
            val = _auto_convert(val_data_str)
        else:
            val_hdr = [int(dim)
                       for dim in val_hdr_str.split(',')]
            val_data = [_auto_convert(value)
                        for value in val_data_str.split(' ')]
            val = np.array(val_data).reshape(val_hdr)
    else:
        val = _auto_convert(val)
    return val


# ======================================================================
def read(
        filepath,
        encoding='utf8'):
    """
    Read files with JCAMP-DX-like structure.

    Args:
        filepath (str): Path to file to parse.
        encoding (str): The encoding to use.

    Returns:
        result (tuple): The tuple
            contains:
             - ldr_std (dict): The standard Labelled-Data-Records.
             - ldr_custom (dict): The custom-defined Labelled-Data-Records.
             - comments (str): Comment lines.
    """
    ldr_sep, ldr_custom_sep, ldr_dict_sep = '##', '$', '='
    with open(filepath, 'rb') as in_file:
        ldr_std, ldr_custom = {}, {}
        data = in_file.read().decode(encoding)
        data, comments = _strip_comments(data)
        ldrs = [ldr for ldr in data.split(ldr_sep) if ldr]
        ldr_list = []
        for ldr in ldrs:
            ldr_name, ldr_val = ldr.split(ldr_dict_sep, 1)
            ldr_val = _parse_record(ldr_val)
            if ldr.startswith(ldr_custom_sep):
                ldr_custom[ldr_name.strip(ldr_custom_sep)] = ldr_val
            else:
                ldr_std[ldr_name] = ldr_val
            ldr_list.append(ldr)
    return ldr_std, ldr_custom, comments


# ======================================================================
def my_testing():
    """
    Test module functionalities with files provided in the package.

    Args:
        None.

    Returns:
        None.
    """
    test_filepath_list = [
        '//home/raid1/metere/hd3/sandbox/hmri'
        '/Specimen_170814_1_0_Study_20170814_080054/23/acqp']  #
    # 'test/file1.jcampdx']
    try:
        for test_filepath in test_filepath_list:
            a, b, c = read(test_filepath)
            print(a)
            print(b)
            print(c)
            print(b['GO_raw_data_format'])

    except Exception as e:  # This has to catch all exceptions.
        print(e)
        print('Test not passed.')
    else:
        print('All test were passed successfully.')


# ======================================================================
elapsed(__file__[len(PATH['base']) + 1:])

# ======================================================================
if __name__ == '__main__':
    run_doctests(__doc__)
