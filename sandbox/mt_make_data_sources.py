#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json

mt_list = [
    # ('<filename>', <flip_angle/deg>, <frequency/Hz>),

    # from F7T_2010_05_119
    # ('s019__gre_qmri_0.6mm_FA30_MT1100_D2500.nii.gz', 1100, 2500),

    # ('s034__gre_qmri_0.6mm_FA30_MTOff.nii.gz', 0, 0),

    ('s036__gre_qmri_0.6mm_FA30_MT1100.nii.gz', 1100, 316),
    ('s038__gre_qmri_0.6mm_FA30_MT1100.nii.gz', 1100, 562),
    ('s040__gre_qmri_0.6mm_FA30_MT1100.nii.gz', 1100, 1000),
    ('s042__gre_qmri_0.6mm_FA30_MT1100.nii.gz', 1100, 1778),
    ('s044__gre_qmri_0.6mm_FA30_MT1100.nii.gz', 1100, 3162),
    ('s046__gre_qmri_0.6mm_FA30_MT1100.nii.gz', 1100, 5623),
    ('s048__gre_qmri_0.6mm_FA30_MT1100.nii.gz', 1100, 10000),
    ('s050__gre_qmri_0.6mm_FA30_MT1100.nii.gz', 1100, 17783),
    ('s052__gre_qmri_0.6mm_FA30_MT1100.nii.gz', 1100, 31623),
    ('s054__gre_qmri_0.6mm_FA30_MT1100.nii.gz', 1100, 56234),
    ('s056__gre_qmri_0.6mm_FA30_MT1100.nii.gz', 1100, 100000),

    # ('s058__gre_qmri_0.6mm_FA30_MTOff.nii.gz', 0, 0),

    ('s060__gre_qmri_0.6mm_FA30_MT800.nii.gz', 800, 316),
    ('s062__gre_qmri_0.6mm_FA30_MT800.nii.gz', 800, 562),
    ('s064__gre_qmri_0.6mm_FA30_MT800.nii.gz', 800, 1000),
    ('s066__gre_qmri_0.6mm_FA30_MT800.nii.gz', 800, 1778),
    ('s068__gre_qmri_0.6mm_FA30_MT800.nii.gz', 800, 3162),
    ('s070__gre_qmri_0.6mm_FA30_MT800.nii.gz', 800, 5623),
    ('s072__gre_qmri_0.6mm_FA30_MT800.nii.gz', 800, 10000),
    ('s074__gre_qmri_0.6mm_FA30_MT800.nii.gz', 800, 17783),
    ('s076__gre_qmri_0.6mm_FA30_MT800.nii.gz', 800, 31623),
    ('s078__gre_qmri_0.6mm_FA30_MT800.nii.gz', 800, 56234),
    ('s080__gre_qmri_0.6mm_FA30_MT800.nii.gz', 800, 100000),

    # ('s082__gre_qmri_0.6mm_FA30_MTOff.nii.gz', 0, 0),

    ('s084__gre_qmri_0.6mm_FA30_MT500.nii.gz', 500, 316),
    ('s086__gre_qmri_0.6mm_FA30_MT500.nii.gz', 500, 562),
    ('s088__gre_qmri_0.6mm_FA30_MT500.nii.gz', 500, 1000),
    ('s090__gre_qmri_0.6mm_FA30_MT500.nii.gz', 500, 1778),
    ('s092__gre_qmri_0.6mm_FA30_MT500.nii.gz', 500, 3162),
    ('s094__gre_qmri_0.6mm_FA30_MT500.nii.gz', 500, 5623),
    ('s096__gre_qmri_0.6mm_FA30_MT500.nii.gz', 500, 10000),
    ('s098__gre_qmri_0.6mm_FA30_MT500.nii.gz', 500, 17783),
    ('s101__gre_qmri_0.6mm_FA30_MT500.nii.gz', 500, 31623),
    ('s103__gre_qmri_0.6mm_FA30_MT500.nii.gz', 500, 56234),
    ('s105__gre_qmri_0.6mm_FA30_MT500.nii.gz', 500, 100000),

    # ('s107__gre_qmri_0.6mm_FA30_MTOff.nii.gz', 0, 0),

    ('s109__gre_qmri_0.6mm_FA30_MT400.nii.gz', 400, 316),
    ('s111__gre_qmri_0.6mm_FA30_MT400.nii.gz', 400, 562),
    ('s113__gre_qmri_0.6mm_FA30_MT400.nii.gz', 400, 1000),
    ('s115__gre_qmri_0.6mm_FA30_MT400.nii.gz', 400, 1778),
    ('s117__gre_qmri_0.6mm_FA30_MT400.nii.gz', 400, 3162),
    ('s119__gre_qmri_0.6mm_FA30_MT400.nii.gz', 400, 5623),
    ('s121__gre_qmri_0.6mm_FA30_MT400.nii.gz', 400, 10000),
    ('s123__gre_qmri_0.6mm_FA30_MT400.nii.gz', 400, 17783),
    ('s125__gre_qmri_0.6mm_FA30_MT400.nii.gz', 400, 31623),
    ('s127__gre_qmri_0.6mm_FA30_MT400.nii.gz', 400, 56234),

    # ('s129__gre_qmri_0.6mm_FA30_MTOff.nii.gz', 0, 0),

    ('s131__gre_qmri_0.6mm_FA30_MT300.nii.gz', 300, 316),
    ('s133__gre_qmri_0.6mm_FA30_MT300.nii.gz', 300, 562),
    ('s135__gre_qmri_0.6mm_FA30_MT300.nii.gz', 300, 1000),
    ('s137__gre_qmri_0.6mm_FA30_MT300.nii.gz', 300, 1778),
    ('s139__gre_qmri_0.6mm_FA30_MT300.nii.gz', 300, 3162),
    ('s141__gre_qmri_0.6mm_FA30_MT300.nii.gz', 300, 5623),
    ('s143__gre_qmri_0.6mm_FA30_MT300.nii.gz', 300, 10000),
    ('s145__gre_qmri_0.6mm_FA30_MT300.nii.gz', 300, 17783),
    ('s147__gre_qmri_0.6mm_FA30_MT300.nii.gz', 300, 31623),

    ('s149__gre_qmri_0.6mm_FA30_MTOff.nii.gz', 0, 1.0),

    ('s151__gre_qmri_0.6mm_FA30_MT300.nii.gz', 200, 316),
    ('s153__gre_qmri_0.6mm_FA30_MT300.nii.gz', 200, 562),
    ('s155__gre_qmri_0.6mm_FA30_MT300.nii.gz', 200, 1000),
    ('s157__gre_qmri_0.6mm_FA30_MT300.nii.gz', 200, 1778),
    ('s159__gre_qmri_0.6mm_FA30_MT300.nii.gz', 200, 3162),
    ('s161__gre_qmri_0.6mm_FA30_MT300.nii.gz', 200, 5623),
    ('s163__gre_qmri_0.6mm_FA30_MT300.nii.gz', 200, 10000),

    # # from F7T_2010_05_119a
    # ('s007__gre_qmri_0.6mm_FA30_MT1100.nii.gz', 1100, -316),
    # ('s009__gre_qmri_0.6mm_FA30_MT1100.nii.gz', 1100, -562),
    # ('s011__gre_qmri_0.6mm_FA30_MT1100.nii.gz', 1100, -1000),
    # ('s013__gre_qmri_0.6mm_FA30_MT1100.nii.gz', 1100, -1778),
    # ('s015__gre_qmri_0.6mm_FA30_MT1100.nii.gz', 1100, -3162),
    # ('s017__gre_qmri_0.6mm_FA30_MT1100.nii.gz', 1100, -5623),
    # ('s019__gre_qmri_0.6mm_FA30_MT1100.nii.gz', 1100, -10000),
    # ('s021__gre_qmri_0.6mm_FA30_MT1100.nii.gz', 1100, -17783),
    # ('s023__gre_qmri_0.6mm_FA30_MT1100.nii.gz', 1100, -31623),
    # ('s025__gre_qmri_0.6mm_FA30_MT1100.nii.gz', 1100, -56234),
    # ('s027__gre_qmri_0.6mm_FA30_MT1100.nii.gz', 1100, -100000),
    #
    # ('s079__gre_qmri_0.6mm_FA30_MT400.nii.gz', 400, -316),
    # ('s081__gre_qmri_0.6mm_FA30_MT400.nii.gz', 400, -562),
    # ('s083__gre_qmri_0.6mm_FA30_MT400.nii.gz', 400, -1000),
    # ('s085__gre_qmri_0.6mm_FA30_MT400.nii.gz', 400, -1778),
    # # ('s087__gre_qmri_0.6mm_FA30_MT400.nii.gz', 400, -3162),
    #
    # ('s102__gre_qmri_0.6mm_FA30_MT300.nii.gz', 300, -316),
    # ('s104__gre_qmri_0.6mm_FA30_MT300.nii.gz', 300, -562),
    # ('s106__gre_qmri_0.6mm_FA30_MT300.nii.gz', 300, -1000),
    #
    # ('s108__gre_qmri_0.6mm_FA30_MT300.nii.gz', 200, -562),
    #
    # ('s126__gre_qmri_0.6mm_FA30_MT200.nii.gz', 200, -1000),
    # ('s128__gre_qmri_0.6mm_FA30_MT200.nii.gz', 200, -1778),
    # ('s130__gre_qmri_0.6mm_FA30_MT200.nii.gz', 200, -3162),
    # ('s132__gre_qmri_0.6mm_FA30_MT200.nii.gz', 200, -5623),
    # ('s134__gre_qmri_0.6mm_FA30_MT200.nii.gz', 200, -10000),
]

with open('data_sources.json', 'w') as out_file:
    json.dump(mt_list, out_file, sort_keys=True, indent=2)
