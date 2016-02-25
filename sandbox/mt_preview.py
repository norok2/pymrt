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

import os
# import itertools
# import multiprocessing
import hashlib

import numpy as np
# import scipy as sp
import h5py

import scipy.optimize
import matplotlib.pyplot as plt

import mri_tools.base as mrb
import mri_tools.input_output as mrio
from mri_tools.sequences import matrix_algebra as ma
from mri_tools.base import elapsed, print_elapsed


def _extract_mt_mask(d=1, n=96, l=1):
    in_filepath = 'mask.nii.gz'
    out_filepath = 'mt_mask.nii.gz'
    img, aff, hdr = mrio.load(in_filepath, True)
    mask = []
    for i in range(img.ndims):
        mask.append(slice(None) if i != d else slice(n, n + l))
    img[not mask] = 0.0
    mrio.save(out_filepath, img, aff, hdr)


DEBUG = True


# ======================================================================
def main():
    if os.getcwd().find('mt_preview') < 0:
        os.chdir('/media/rick/Q_METERE_IT/FMRIB/mt_preview')

    mt_subdir = 'mt_list'

    test_mask_filename = 'test0.nii.gz'
    mask_filename = 'mask.nii.gz'
    b1t_filename = 'b1t.nii.gz'
    b0_filename = 'b0.nii.gz'
    t1_filename = 't1.nii.gz'
    t2s_filename = 't2s.nii.gz'

    fitted_names = [
        'm0', 'm_a', 'r2_a', 'r2_b', 'k_ab'
    ]

    mt_list = [
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

        # from F7T_2010_05_119a
        ('s007__gre_qmri_0.6mm_FA30_MT1100.nii.gz', 1100, -316),
        ('s009__gre_qmri_0.6mm_FA30_MT1100.nii.gz', 1100, -562),
        ('s011__gre_qmri_0.6mm_FA30_MT1100.nii.gz', 1100, -1000),
        ('s013__gre_qmri_0.6mm_FA30_MT1100.nii.gz', 1100, -1778),
        ('s015__gre_qmri_0.6mm_FA30_MT1100.nii.gz', 1100, -3162),
        ('s017__gre_qmri_0.6mm_FA30_MT1100.nii.gz', 1100, -5623),
        ('s019__gre_qmri_0.6mm_FA30_MT1100.nii.gz', 1100, -10000),
        ('s021__gre_qmri_0.6mm_FA30_MT1100.nii.gz', 1100, -17783),
        ('s023__gre_qmri_0.6mm_FA30_MT1100.nii.gz', 1100, -31623),
        ('s025__gre_qmri_0.6mm_FA30_MT1100.nii.gz', 1100, -56234),
        ('s027__gre_qmri_0.6mm_FA30_MT1100.nii.gz', 1100, -100000),

        ('s079__gre_qmri_0.6mm_FA30_MT400.nii.gz', 400, -316),
        ('s081__gre_qmri_0.6mm_FA30_MT400.nii.gz', 400, -562),
        ('s083__gre_qmri_0.6mm_FA30_MT400.nii.gz', 400, -1000),
        ('s085__gre_qmri_0.6mm_FA30_MT400.nii.gz', 400, -1778),
        # ('s087__gre_qmri_0.6mm_FA30_MT400.nii.gz', 400, -3162),

        ('s102__gre_qmri_0.6mm_FA30_MT300.nii.gz', 300, -316),
        ('s104__gre_qmri_0.6mm_FA30_MT300.nii.gz', 300, -562),
        ('s106__gre_qmri_0.6mm_FA30_MT300.nii.gz', 300, -1000),

        ('s108__gre_qmri_0.6mm_FA30_MT300.nii.gz', 200, -562),

        ('s126__gre_qmri_0.6mm_FA30_MT200.nii.gz', 200, -1000),
        ('s128__gre_qmri_0.6mm_FA30_MT200.nii.gz', 200, -1778),
        ('s130__gre_qmri_0.6mm_FA30_MT200.nii.gz', 200, -3162),
        ('s132__gre_qmri_0.6mm_FA30_MT200.nii.gz', 200, -5623),
        ('s134__gre_qmri_0.6mm_FA30_MT200.nii.gz', 200, -10000),
    ]

    img_mask, aff_mask, hdr_mask = mrio.load(mask_filename, True)
    mask = img_mask.astype(bool)
    try:
        test_mask_filename
    except NameError:
        pass
    else:
        test_mask = mrio.load(test_mask_filename).astype(bool)
        mask = mask * test_mask

    t1 = mrio.load(t1_filename).astype(float)
    t2s = mrio.load(t2s_filename).astype(float)
    r1a_arr = np.zeros_like(t1)
    r2a_arr = np.zeros_like(t2s)

    r1a_arr[t1 != 0.0] = 1e3 / t1[t1 != 0.0]
    r1a_arr = r1a_arr[mask]

    r2a_arr[t2s != 0.0] = 1e3 / t2s[t2s != 0.0]
    r2a_arr = r2a_arr[mask]

    b1t_arr = mrio.load(b1t_filename)[mask]
    b0_arr = mrio.load(b0_filename)[mask]

    tmp_dirpath = 'cache/{}'.format(
        hashlib.md5((str(mt_list) + str(mask)).encode()).hexdigest())
    if not os.path.isdir(tmp_dirpath):
        os.makedirs(tmp_dirpath)

    mt_data_filepath = os.path.join(tmp_dirpath, 'mt_data.h5')
    if os.path.isfile(mt_data_filepath):
        h5f = h5py.File(mt_data_filepath, 'r')
        h5d = h5f['mt_data']
        mt_arr = h5d[:]
        freqs = h5d.attrs['freqs']
        flip_angles = h5d.attrs['flip_angles']
        h5f.close()
    else:
        images, flip_angles, freqs = [], [], []
        for filename, flip_angle, freq in mt_list:
            filepath = os.path.join(mt_subdir, filename)
            if not os.path.isfile(filepath):
                print(filepath, ': not found!')
            else:
                print(': {} {} {}'.format(filename, flip_angle, freq))
            img, aff, hdr = mrio.load(filepath, True)
            images.append(img[mask])
            flip_angles.append(flip_angle)
            freqs.append(freq)
        mt_arr = mrb.ndstack(images)

        h5f = h5py.File(mt_data_filepath, 'w')
        h5d = h5f.create_dataset('mt_data', data=mt_arr)
        h5d.attrs['freqs'] = freqs
        h5d.attrs['flip_angles'] = flip_angles
        h5f.close()

    x_data = np.array(list(zip(flip_angles, freqs)))

    # the starting point for the estimation parameters
    p0 = 12000, 0.9, 60, 11e4, 0.5
    bounds = [
        [1000, 100000], [0, 1], [1, 200], [5e4, 15e4], [0, 1.5]]

    num_par = len(p0)

    mask_size = np.sum(mask)
    p_arr = np.zeros((mask_size, num_par))

    # pool = multiprocessing.Pool(multiprocessing.cpu_count())
    # for i, (par_opt, par_cov) in \
    #         enumerate(pool.imap(mrb.curve_fit, iter_param_list)):
    #         p_arr[i] = par_opt

    data = (mt_arr, r1a_arr, r2a_arr, b1t_arr, b0_arr)

    t_e = 1.7e-3
    t_r = 70.0e-3
    w_c = 297220696

    pos_arr = np.zeros_like(mask, int)
    pos_arr[mask] = np.arange(mask_size) + 1
    # pool = multiprocessing.Pool(multiprocessing.cpu_count() // 2)
    for i, (mt_data, r1a, r2a, b1t, b0) in enumerate(zip(*data)):
        pos = tuple(np.hstack(np.where(pos_arr == i + 1)))
        print('Voxel: {} / {} {}'.format(i + 1, mask_size, pos))

        if DEBUG:
            print(r1a, r2a, b1t, b0)
        if DEBUG:
            # plt.ion()
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.set_xlabel('Pulse Amplitude (flip angle) / deg')
            ax.set_ylabel('Frequency offset / Hz (log10 scale)')
            ax.set_zlabel('Signal Intensity / arb.units')
            ax.scatter(
                flip_angles, mrb.sgnlog(freqs, 10), mt_data, c='b', marker='o')
            plt.show(block=False)

        mt_flash = ma.MtFlash(
            ma.PulseList([
                ma.Delay(
                    t_r - (
                        t_e + 3 * 160.0e-6 + 20000.0e-6 + 970.0e-6 + 100e-6)),
                ma.Spoiler(0.0),
                ma.Delay(160.0e-6),
                ma.PulseExc.shaped(
                    20000.0e-6, 90.0, 0, '_from_GAUSS5120', {}, 0.0,
                    'linear', {'num_samples': 15}),
                ma.Delay(160.0e-6 + 970.0e-6),
                ma.Spoiler(1.0),
                ma.Delay(160.0e-6),
                ma.PulseExc.shaped(100e-6, 30.0 * b1t, 1, 'rect', {}),
                ma.Delay(t_e)],
                w_c=w_c),
            num_repetitions=np.min(r1a_arr.shape) ** 2)

        r2sa = r2a

        def mt_signal(x_arr, s0, mc_a, r2a, r2b, k_ab):
            spin_model = ma.SpinModel(
                s0=s0,
                mc=(mc_a, 1.0 - mc_a),
                # w0=(w_c, w_c * (1 - 3.5e-6)),
                w0=((w_c + b0,) * 2),
                r1=(r1a, 1.0),
                r2=(r2a, r2b),
                k=(k_ab,),
                approx=(None, 'superlorentz_approx'))
            y_arr = np.zeros_like(x_arr[:, 0])
            i = 0
            for fa, freq in x_arr:
                mt_flash.set_flip_angle(fa * b1t)
                mt_flash.set_freq(freq)
                y_arr[i] = mt_flash.signal(spin_model)
                i += 1
            if DEBUG:
                print(s0, mc_a, r2a, r2sa, r2b, k_ab)
                # ax.plot_trisurf(
                #     flip_angles, mrb.sgnlog(freqs, 10), y_arr,
                #     cmap=plt.cm.hot)
                # ax.scatter(
                #     flip_angles, mrb.sgnlog(freqs, 10), y_arr,
                #      c='r', marker='o')
                # plt.show(block=False)
            return y_arr

        if DEBUG:
            ax.scatter(
                flip_angles, mrb.sgnlog(freqs, 10), mt_signal(x_data, *p0),
                 c='r', marker='^')
            plt.show(block=False)

        # def sum_of_squares(params, x_data, m_data):
        #     e_data = mt_signal(x_data, *params)
        #     return np.sum((m_data - e_data) ** 2.0)

        # res = scipy.optimize.minimize(
        #     sum_of_squares, p0, args=(x_data, mt_data), method='SLSQP',
        #     bounds=bounds)

        # res = scipy.optimize.least_squares(
        #     sum_of_squares, p0, args=(x_data, mt_data), method='trf',
        #     bounds=list(zip(*bounds)))

        # success = res.success
        # p_fit = res.x
        # error_msg = res.message

        success = True
        error_msg = 'Fit converged.'
        try:
            p_fit, p_cov = scipy.optimize.curve_fit(
                mt_signal, x_data, mt_data, p0,
                method='trf',
                bounds=list(zip(*bounds))
            )
        except scipy.optimize.OptimizeWarning:
            success = False
            p_fit = np.zeros_like(p0)
            error_msg = 'Fit failed.'

        if DEBUG:
            ax.scatter(
                flip_angles, mrb.sgnlog(freqs, 10), mt_signal(x_data, *p_fit),
                 c='r', marker='o')
            plt.show()

        if success:
            p_arr[i] = p_fit
        else:
            print(error_msg)

    dirpath = mrb.change_ext(test_mask_filename, '', mrb.EXT['img'])
    if not os.path.isdir(dirpath):
        os.makedirs(dirpath)
    images = []
    for fitted_name, image_flat in zip(fitted_names, mrb.ndsplit(p_arr)):
        image = np.zeros_like(mask, float)
        image[mask] = image_flat
        filepath = os.path.join(
            dirpath, mrb.change_ext(fitted_name, mrb.EXT['img']))
        mrio.save(filepath, image, aff_mask)
        images.append(image)
    filepath = os.path.join(
        dirpath, mrb.change_ext('fitted', mrb.EXT['img']))
    mrio.save(filepath, mrb.ndstack(images), aff_mask)


# ======================================================================
if __name__ == '__main__':
    print(__doc__)
    main()
    elapsed('test_fit_spin_model')

    print_elapsed()
