#!python
# -*- coding: utf-8 -*-
"""
pymrt: data analysis for quantitative MRI
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
__version__ = '0.0.2.dev0+ng254ac18.d20160410'

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
# import pymrt.base as mrb
# import pymrt.utils as mru
# import pymrt.geometry as mrg
# import pymrt.plot as mrp
# import pymrt.registration as mrr
# import pymrt.segmentation as mrs
# import pymrt.computation as mrc
# import pymrt.correlation as mrl
# import pymrt.input_output as mrio
# import pymrt.sequences as mrq
# import pymrt.extras as mre
# from pymrt.debug import dbg
# from pymrt.sequences import mp2rage
# from pymrt.sequences import matrix_algebra
# from pymrt.extras import twix
# from pymrt.extras import jcampdx
# from pymrt.extras import latex

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
