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

# :: External Imports

# :: External Imports Submodules

# :: Internal Imports
import pymrt as mrt

# :: Local Imports
from pymrt import INFO, PATH
# from pymrt import VERB_LVL, D_VERB_LVL, VERB_LVL_NAMES
from pymrt import elapsed, report
from pymrt import msg, dbg
from flyingcircus.util import EXT

# ======================================================================
# :: Custom defined constants


# ======================================================================
# :: Default values usable in functions
EXT.update({
    'plot': 'png',
    'nii': 'nii',
    'niz': 'nii.gz',
    'text': 'txt',
    'tab': 'csv',
    'data': 'json',
})

# ======================================================================
elapsed(__file__[len(PATH['base']) + 1:])

# ======================================================================
if __name__ == '__main__':
    import doctest  # Test interactive Python examples

    msg(__doc__.strip())
    doctest.testmod()
    msg(report())
