#!python
# -*- coding: utf-8 -*-
"""
mri_tools: data analysis for quantitative MRI
"""

# Copyright (c) Riccardo Metere <rick@metere.it>

# ======================================================================
# :: Future Imports
from __future__ import division
from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals

# ======================================================================
# :: Python Standard Library Imports


# ======================================================================
# :: Version
__version__ = '0.2.0.1.dev5+ng0bba04f.d20160223'

# ======================================================================
# :: Project Details
INFO = {
    'authors': (
        'Riccardo Metere <metere@cbs.mpg.de>',
    ),
    'copyright': 'Copyright (C) 2015',
    'license': 'GNU General Public License version 3 or later (GPLv3+)',
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
# :: quick and dirty timing facility
_EVENTS = []

# :: import
# import mri_tools.base as mrb
# import mri_tools.utils as mru
# import mri_tools.geometry as mrg
# import mri_tools.plot as mrp
# import mri_tools.registration as mrr
# import mri_tools.segmentation as mrs
# import mri_tools.computation as mrc
# import mri_tools.correlation as mrl
# import mri_tools.input_output as mrio
# import mri_tools.sequences as mrq
# import mri_tools.extras as mre
# from mri_tools.debug import dbg
# from mri_tools.sequences import mp2rage
# from mri_tools.sequences import matrix_algebra
# from mri_tools.extras import twix
# from mri_tools.extras import jcampdx
# from mri_tools.extras import latex

# ======================================================================
# Greetings
MY_GREETINGS = r"""
                _   _              _
 _ __ ___  _ __(_) | |_ ___   ___ | |___
| '_ ` _ \| '__| | | __/ _ \ / _ \| / __|
| | | | | | |  | | | || (_) | (_) | \__ \
|_| |_| |_|_|  |_|  \__\___/ \___/|_|___/
"""
# generated with: figlet 'mri tools' -f standard

# :: Causes the greetings to be printed any time the library is loaded.
print(MY_GREETINGS)
