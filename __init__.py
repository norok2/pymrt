#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
mri_tools: collection of software for MRI analysis
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
# :: Python Standard Library Imports

# ======================================================================
# Greetings
MY_GREETINGS = """
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
