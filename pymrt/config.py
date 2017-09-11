#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PyMRT: configuration file
"""


# ======================================================================
# :: Future Imports
from __future__ import(
    division, absolute_import, print_function, unicode_literals)

# TODO: automagically find them?
EXT_CMD = {
    'fsl/5.0/flirt': 'fsl5.0-flirt',
    'fsl/5.0/bet': 'fsl5.0-bet',
    'fsl/5.0/prelude': 'fsl5.0-prelude',
    'ants/ANTS': '',
}


# ======================================================================
_B0 = 7  # T

CFG = dict(
    use_cache=True,
)
