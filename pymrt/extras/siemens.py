#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pymrt/twix: manage Siemens's TWIX (raw) data from MRI scanners.
"""

# ======================================================================
# :: Future Imports
from __future__ import (
    division, absolute_import, print_function, unicode_literals, )

# ======================================================================
# :: Python Standard Library Imports
import os  # Miscellaneous operating system interfaces
import json  # JSON encoder and decoder [JSON: JavaScript Object Notation]

# :: External Imports
import numpy as np  # NumPy (multidimensional numerical arrays library)
import pyparsing as pp  # A Python Parsing Module
import flyingcircus as fc  # Everything you always wanted to have in Python.*

# :: External Imports Submodules
import flyingcircus.util  # FlyingCircus: generic basic utilities

# :: Local Imports

from pymrt import INFO, PATH
# from pymrt import VERB_LVL, D_VERB_LVL, VERB_LVL_NAMES
from pymrt import elapsed, report
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
        mask = None
    data = fc.util.read_stream(file_stream, dtype, '<', count, offset)
    return data[mask] if mask is not None else data


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
                val.append((indexes, fc.util.auto_convert(value, '""', '""')))
            else:
                val = fc.util.auto_convert(value, '""', '""')
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
    elif fc.util.has_decorator(text, *PROT['prot_decor']):
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
        key = fc.util.read_cstr(file_stream)
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


# test()

# ======================================================================
elapsed(__file__[len(PATH['base']) + 1:])

# ======================================================================
if __name__ == '__main__':
    import doctest  # Test interactive Python examples

    msg(__doc__.strip())
    doctest.testmod()
    msg(report())
