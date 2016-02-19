#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Quick and dirty script for MT
"""

# ======================================================================
# :: Future Imports
from __future__ import division
from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals

import mri_tools.sequences.matrix_algebra as mrq_ma


# ======================================================================
def mt_seq(
        mt_fa,
        mt_freq,
        t_e=1.7e-3,
        t_r=70.0e-3,
        b1t=1.0,
        w_c=297220696):
    mt_flash = mrq_ma.MtFlash(
        mrq_ma.PulseList([
            mrq_ma.Delay(t_r - ()),
            mrq_ma.Spoiler(0.0),
            mrq_ma.Delay(160.0e-6),
            mrq_ma.PulseExc.shaped(
                20000.0e-6, mt_fa * b1t, 0, '_from_GAUSS5120', {}, 0.0,
                'linear', {'num_samples': 15}),
            mrq_ma.Delay(160.0e-6 + 970.0e-6),
            mrq_ma.Spoiler(1.0),
            mrq_ma.Delay(160.0e-6),
            mrq_ma.PulseExc.shaped(100e-6, 30.0, 1, 'rect', {}),
            mrq_ma.Delay(t_e)],
            w_c=w_c),
        300)
    return mt_flash


# ======================================================================
def main():
    mask_filename = ''

    mt_filenames = [
        '',
        '',
        '',
    ]

    print(mt_flash)


# ======================================================================
if __name__ == '__main__':
    print(__doc__)
    main()
