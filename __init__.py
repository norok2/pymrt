#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
mri_tools:
"""


# Copyright (C) 2015 Riccardo Metere <metere@cbs.mpg.de>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.


# ======================================================================
# :: Future Imports
from __future__ import division
from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals


# ======================================================================
# :: Versioning
__version__ = "$Revision$"
# $Source$


# ======================================================================
# :: Project Details
INFO = {
    'authors': (
        'Riccardo Metere <metere@cbs.mpg.de>',
        ),
    'copyright': 'Copyright (C) 2015',
    'license': 'License: GNU General Public License version 3 (GPLv3)',
    'notice':
        """
This program is free software and it comes with ABSOLUTELY NO WARRANTY.
It is covered by the GNU General Public License version 3 (GPLv3).
You are welcome to redistribute it under its terms and conditions.
        """,
    'version': __version__
    }


# ======================================================================
# :: supported verbosity levels (level 4 skipped on purpose)
VERB_LVL = {'none': 0, 'low': 1, 'medium': 2, 'high': 3, 'debug': 5}
D_VERB_LVL = VERB_LVL['low']


# ======================================================================
# :: get first line (useful for documentation)
def _firstline(text):
    """
    Extract the first line from the text

    Parameters
    ==========
    text : str
        The input string.

    Returns
    =======
    text : str
        The first non-empty line.

    """
    return [line for line in __doc__.splitlines() if line][0]


# ======================================================================
# Greetings
MY_GREETINGS = \
"""
                _   _              _
 _ __ ___  _ __(_) | |_ ___   ___ | |___
| '_ ` _ \| '__| | | __/ _ \ / _ \| / __|
| | | | | | |  | | | || (_) | (_) | \__ \\
|_| |_| |_|_|  |_|  \__\___/ \___/|_|___/
"""
# generated with: figlet 'mri tools' -f standard
# note: '\' characters need to be escaped (with another '\')

# :: Causes the greetings to be printed any time the library is loaded.
print(MY_GREETINGS)
