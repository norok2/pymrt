#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pymrt/twix: manage Siemens's TWIX (raw) data from MRI scanners.
"""

# ======================================================================
# :: Future Imports
from __future__ import(
    division, absolute_import, print_function, unicode_literals)

# ======================================================================
# :: Python Standard Library Imports
import os  # Miscellaneous operating system interfaces
import io  # Core tools for working with streams
# import shutil  # High-level file operations
# import math  # Mathematical functions
import time  # Time access and conversions
import datetime  # Basic date and time types
# import operator  # Standard operators as functions
# import collections  # Container datatypes
# import itertools  # Functions creating iterators for efficient looping
# import functools  # Higher-order functions and operations on callable objects
# import argparse  # Parser for command-line options, arguments and sub-command
import re  # Regular expression operations
# import subprocess  # Subprocess management
# import multiprocessing  # Process-based parallelism
# import fractions  # Rational numbers
# mport csv  # CSV File Reading and Writing [CSV: Comma-Separated Values]
import json  # JSON encoder and decoder [JSON: JavaScript Object Notation]
import struct  # Interpret strings as packed binary data

# :: External Imports
import numpy as np  # NumPy (multidimensional numerical arrays library)
# import scipy as sp  # SciPy (signal and image processing library)
# import matplotlib as mpl  # Matplotlib (2D/3D plotting library)
# import sympy as sym  # SymPy (symbolic CAS library)
# import PIL  # Python Image Library (image manipulation toolkit)
# import SimpleITK as sitk  # Image ToolKit Wrapper
# import nibabel as nib  # NiBabel (NeuroImaging I/O Library)
# import nipy  # NiPy (NeuroImaging in Python)
# import nipype  # NiPype (NiPy Pipelines and Interfaces)
import pyparsing as pp  # A Python Parsing Module

# :: External Imports Submodules
# import matplotlib.pyplot as plt  # Matplotlib's pyplot: MATLAB-like syntax
# import scipy.optimize  # SciPy: Optimization Algorithms
# import scipy.integrate  # SciPy: Integrations facilities
# import scipy.constants  # SciPy: Mathematal and Physical Constants
# import scipy.ndimage  # SciPy: ND-image Manipulation

# :: Local Imports
import pymrt as mrt
import pymrt.utils

# from pymrt import INFO
# from pymrt import VERB_LVL, D_VERB_LVL
from pymrt import msg, dbg

# ======================================================================
LIMITS = {
    'VD': {
        'max_num_scans': 64,
    },
    'VB': {
        'max_num_scans': 1,
    }
}

PROT = {
    'x_prot_id': '<XProtocol>',
    'prot_decor': (
        '### ASCCONV BEGIN ###',
        '### ASCCONV END ###'
    ),
}


# ======================================================================
def _read_twix(
        file_stream,
        dtype,
        count=None,
        offset=None):
    """

    Args:
        file_stream:
        dtype:
        count:
        offset:

    Returns:

    """
    if count is None:
        count = 1
        mask = 0
    else:
        mask = slice(None)
    data, byte_count = mrt.utils.read_stream(file_stream, dtype, count, '<', offset)
    return data[mask]


# ======================================================================
def _read_protocol(text):
    """
    Parse protocol information and save to a dictionary.
    """
    # todo: improve array support? use pyparsing?
    prot = {}
    for line in text.split('\n'):
        equal_pos = line.find('=')
        # check that lines contain a key=val pair AND is not a comment (#)
        if equal_pos >= 0 and not line.startswith('#'):
            name = line[:equal_pos].strip()
            value = line[equal_pos + 1:].strip()
            # generate proper key
            key = ''
            indexes = []
            for num, field in enumerate(name.split('.')):
                sep1, sep2 = field.find('['), field.find(']')
                is_array = (True if sep1 != sep2 else False)
                if is_array:
                    indexes.append(int(field[sep1 + 1:sep2]))
                key += '.' + (field[:sep1] + '[]' if is_array else field)
            key = key[1:]
            # save data to dict
            if indexes:
                if key in prot:
                    val = prot[key]
                else:
                    val = []
                val.append((indexes, mrt.utils.auto_convert(value, '""', '""')))
            else:
                val = mrt.utils.auto_convert(value, '""', '""')
            if key:
                prot[key] = val
    return prot


# ======================================================================
def _read_x_protocol(text):
    """
    Parse XProtocol (a.k.a. EVP: EValuation Protocol)

    Args:
        text:

    Returns:

    """
    x_prot = {}
    # text = text[len(PROT['x_prot_id']):]
    # text = ' '.join(text.split())
    lbra = pp.Literal('{').suppress()
    rbra = pp.Literal('}').suppress()
    lang = pp.Literal('<').suppress()
    rang = pp.Literal('>').suppress()
    dot = pp.Literal('.')
    cstr = pp.quotedString.addParseAction(pp.removeQuotes)
    number = pp.Regex(r'[+-]?\d+(\.\d*)?').setName('number')
    tag = pp.Combine(
        lang +
        pp.Word(pp.alphanums) +
        pp.Optional(dot + cstr) +
        rang).setName('tag')

    exp = pp.Forward()

    key_value = pp.Group(tag + exp)
    exp <<= (
        key_value |
        number |
        cstr |
        pp.Group(lbra + pp.ZeroOrMore(exp) + rbra)
    )

    # print(x_prot)
    # x_prot = exp.parseString(text)
    # with open('/media/Data/tmp/parsed.json', 'w') as f:
    #     json.dump(
    #         (x_prot.asDict(), x_prot.asList()), f, sort_keys=True, indent=4)
    #
    # quit()
    return x_prot


# ======================================================================
def _guess_version(file_stream):
    file_magic = _read_twix(file_stream, 'uint', 2, 0)
    if (file_magic[0] < 1e4 and
                file_magic[1] <= LIMITS['vd']['max_num_scans']):
        guessed = 'vd?'
    else:
        guessed = 'vb?'
    file_stream.seek(0)
    return guessed


# ======================================================================
def _guess_header_type(text):
    if text.startswith('<XProtocol'):
        header_type = 'x_prot'
    elif mrt.utils.has_decorator(text, *PROT['prot_decor']):
        header_type = 'prot'
    else:
        header_type = None
    return header_type


# ======================================================================
def _parse_vb_header(
        file_stream,
        num_x_prot):
    raw_header = {}
    for i in range(num_x_prot):
        key = mrt.utils.read_cstr(file_stream)
        size = _read_twix(file_stream, 'int')
        val = file_stream.read(size).decode('ascii')
        raw_header[key] = str(val.strip(' \n\t\0'))

    header = {}
    for name, data in sorted(raw_header.items()):
        print(name)
        header_type = _guess_header_type(data)
        if header_type == 'prot':
            header[name] = _read_protocol(data)
        elif header_type == 'x_prot':
            header[name] = _read_x_protocol(data)
        else:
            header[name] = data
        print(name, header_type)

    return header


# ======================================================================
class Twix(object):
    """
    Manages the extraction of data from the disk.
    """

    def __init__(
            self,
            filepath,
            version=None):
        """


        Args:
            filepath:
            version:

        Returns:

        """
        # open file for reading
        self.dirpath, self.filename = os.path.split(filepath)
        self.file_stream = open(filepath, 'rb')

        # perform some magic
        self.version = _guess_version(self.file_stream) \
            if version is None else version.tolower()

        if self.version.startswith('vb'):
            self.byte_count = 0
            self.data_offset = _read_twix(self.file_stream, 'uint')
            self.num_x_prot = _read_twix(self.file_stream, 'uint')
            self.header = _parse_vb_header(self.file_stream, self.num_x_prot)
            self._write_header()

        if self.version.startswith('vd'):
            print('Not supported yet')

    def _write_header(self, filepath=None):
        if filepath is None:
            filepath = os.path.join(self.dirpath, self.filename + '.json')
        with open(filepath, 'w') as file_stream:
            json.dump(self.header, file_stream, sort_keys=True, indent=4)


# ======================================================================
def read(
        filepath,
        version=None):
    """

    Args:
        filepath:

    Returns:
        twix (Twix): object for
    """
    return Twix(filepath, version)


# ======================================================================
def test():
    filepath = '/media/Data/tmp/' + \
               'meas_MID389_gre_qmri_0_6mm_FA30_MTOff_LowRes_FID36111.dat'
    # twix = read_output(filepath)
    # with open('/media/Data/tmp/hdr_Meas.txt', 'r') as f:
    with open('/media/Data/tmp/hdr_Meas.txt', 'r') as f:
        t = f.read()
        _read_x_protocol(t)


# ======================================================================
if __name__ == '__main__':
    msg(__doc__.strip())
    begin_time = datetime.datetime.now()
    test()
    end_time = datetime.datetime.now()
    print('ExecTime: {}'.format(end_time - begin_time))
