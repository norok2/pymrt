#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PyMRT - Python Magnetic Resonance Tools: data analysis for quantitative MRI.
"""

# Copyright (c) Riccardo Metere <rick@metere.it>

# ======================================================================
# :: Future Imports
from __future__ import (
    division, absolute_import, print_function, unicode_literals, )

# ======================================================================
# :: Python Standard Library Imports

# ======================================================================
# :: External Imports
# import flyingcircus as fc
from flyingcircus import msg, dbg, elapsed, report, pkg_paths
from flyingcircus import VERB_LVL, VERB_LVL_NAMES, D_VERB_LVL

# ======================================================================
# :: Version
from ._version import __version__

# ======================================================================
# :: Project Details
INFO = {
    'name': 'PyMRT',
    'author': 'PyMRT developers',
    'contrib': (
        'Riccardo Metere <rick@metere.it>',
    ),
    'copyright': 'Copyright (C) 2015-2017',
    'license': 'GNU General Public License version 3 or later (GPLv3+)',
    'notice':
        """
This program is free software and it comes with ABSOLUTELY NO WARRANTY.
It is covered by the GNU General Public License version 3 (GPLv3+).
You are welcome to redistribute it under its terms and conditions.
        """,
    'version': __version__
}

# ======================================================================
# :: quick and dirty timing facility
_EVENTS = []

# ======================================================================
# Greetings
MY_GREETINGS = r"""
 ____        __  __ ____ _____
|  _ \ _   _|  \/  |  _ \_   _|
| |_) | | | | |\/| | |_) || |
|  __/| |_| | |  | |  _ < | |
|_|    \__, |_|  |_|_| \_\|_|
       |___/
"""
# generated with: figlet 'PyMRT' -f standard

# :: Causes the greetings to be printed any time the library is loaded.
print(MY_GREETINGS)

# ======================================================================
PATH = pkg_paths()

# ======================================================================
elapsed()

# ======================================================================
if __name__ == '__main__':
    import doctest  # Test interactive Python examples

    msg(__doc__.strip())
    doctest.testmod()
    msg(report())
