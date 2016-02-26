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
import argparse
# import itertools
# import multiprocessing
import hashlib
import json

import numpy as np
# import scipy as sp
import h5py

import scipy.optimize
import matplotlib.pyplot as plt

import mri_tools.base as mrb
import mri_tools.input_output as mrio
from mri_tools.sequences import matrix_algebra as ma
from mri_tools.base import elapsed, print_elapsed

from mri_tools import INFO
from mri_tools import VERB_LVL
from mri_tools import D_VERB_LVL


# ======================================================================
def _extract_mt_mask(d=1, n=96, l=1):
    in_filepath = 'mask.nii.gz'
    out_filepath = 'mt_mask.nii.gz'
    img, aff, hdr = mrio.load(in_filepath, True)
    mask = []
    for i in range(img.ndims):
        mask.append(slice(None) if i != d else slice(n, n + l))
    img[not mask] = 0.0
    mrio.save(out_filepath, img, aff, hdr)


# ======================================================================
def mt_fit_quick(
        dirpath,
        data_subdir='mt_list',
        filenames=None,
        fitting_names=('m0', 'm_a', 'r2_a', 'r2_b', 'k_ab'),
        verbose=D_VERB_LVL):
    """

    Args:
        dirpath:
        data_subdir:
        config_filename:
        target_filename:
        mask_filename:
        b1t_filename:
        b0_filename:
        t1_filename:
        t2s_filename:
        fitting_names:
        verbose:

    Returns:

    """
    if not filenames:
        filenames = {
            'data_sources': 'data_sources.json',
            'target': 'target.nii.gz',
            'mask': 'mask.nii.gz',
            'b1t': 'b1t.nii.gz',
            'b0': 'b0.nii.gz',
            't1': 't1.nii.gz',
            't2s': 't2s.nii.gz',
        }
    filepaths = {}
    for item, filename in filenames.items():
        filepaths[item] = os.path.realpath(filename)
        if not os.path.exists(filepaths[item]):
            filepaths[item] = os.path.join(dirpath, filename)
        if not os.path.exists(filepaths[item]):
            filepaths[item] = os.path.join(
                os.path.dirname(__file__), filename)

    with open(filepaths['data_sources'], 'r') as src_file:
        data_sources = json.load(src_file)

    img_mask, aff_mask, hdr_mask = mrio.load(filepaths['mask'], True)
    mask = img_mask.astype(bool)
    try:
        filepaths['target']
    except NameError:
        pass
    else:
        target_mask = mrio.load(filepaths['target']).astype(bool)
        mask = mask * target_mask

    b1t_arr = mrio.load(filepaths['b1t'])[mask]
    b0_arr = mrio.load(filepaths['b0'])[mask]

    t1 = mrio.load(filepaths['t1']).astype(float)
    t2s = mrio.load(filepaths['t2s']).astype(float)
    r1a_arr = np.zeros_like(t1)
    r2a_arr = np.zeros_like(t2s)

    r1a_arr[t1 != 0.0] = 1e3 / t1[t1 != 0.0]
    r1a_arr = r1a_arr[mask]

    r2a_arr[t2s != 0.0] = 1e3 / t2s[t2s != 0.0]
    r2a_arr = r2a_arr[mask]

    tmp_name = hashlib.md5(
        (str(data_sources) + str(mask.tostring())).encode()).hexdigest()
    tmp_dirpath = os.path.join(dirpath, 'cache/{}'.format(tmp_name))
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
        p_imgs, flip_angles, freqs = [], [], []
        for filename, flip_angle, freq in data_sources:
            p_filepath = os.path.join(dirpath, data_subdir, filename)
            if not os.path.isfile(p_filepath):
                print(p_filepath, ': not found!')
            else:
                print(': {} {} {}'.format(filename, flip_angle, freq))
            img, aff, hdr = mrio.load(p_filepath, True)
            p_imgs.append(img[mask])
            flip_angles.append(flip_angle)
            freqs.append(freq)
        mt_arr = mrb.ndstack(p_imgs)

        h5f = h5py.File(mt_data_filepath, 'w')
        h5d = h5f.create_dataset('mt_data', data=mt_arr)
        h5d.attrs['freqs'] = freqs
        h5d.attrs['flip_angles'] = flip_angles
        h5f.close()

    x_data = np.array(list(zip(flip_angles, freqs)))

    # the starting point for the estimation parameters
    p0 = 12000, 0.9, 60, 11e4, 0.5
    bounds = [
        [1000, 100000], [0, 1], [1, 200], [5e4, 15e4], [0, 2.0]]

    num_par = len(p0)

    mask_size = np.sum(mask)
    p_arr = np.zeros((mask_size, num_par))
    dp_arr = np.zeros((mask_size, num_par))
    data = (mt_arr, r1a_arr, r2a_arr, b1t_arr, b0_arr)

    # :: define the sequence
    t_e = 1.7e-3
    t_r = 70.0e-3
    w_c = 297220696

    pos_arr = np.zeros_like(mask, int)
    pos_arr[mask] = np.arange(mask_size) + 1
    # pool = multiprocessing.Pool(multiprocessing.cpu_count() // 2)
    for i, (mt_data, r1a, r2a, b1t, b0) in enumerate(zip(*data)):
        if verbose >= VERB_LVL['low']:
            pos = tuple(np.hstack(np.where(pos_arr == i + 1)))
            print('Voxel: {} / {} {}'.format(i + 1, mask_size, pos))

        if verbose >= VERB_LVL['medium']:
            print(r1a, r2a, b1t, b0)
        if verbose >= VERB_LVL['debug']:
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
            if verbose >= VERB_LVL['high']:
                print(s0, mc_a, r2a, r2b, k_ab)
            return y_arr

        if verbose >= VERB_LVL['debug']:
            ax.scatter(
                flip_angles, mrb.sgnlog(freqs, 10), mt_signal(x_data, *p0),
                c='g', marker='s')
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
        # p_opt = res.x
        # error_msg = res.message

        success = True
        error_msg = 'Fit converged.'
        try:
            p_opt, p_cov = scipy.optimize.curve_fit(
                mt_signal, x_data, mt_data, p0,
                method='trf', bounds=list(zip(*bounds)))
            p_err = np.sqrt(np.diag(p_cov))
        except scipy.optimize.OptimizeWarning:
            success = False
            p_opt = np.zeros_like(p0)
            p_err = np.zeros_like(p0)
            error_msg = 'Fit failed.'

        if verbose >= VERB_LVL['debug']:
            ax.scatter(
                flip_angles, mrb.sgnlog(freqs, 10), mt_signal(x_data, *p_opt),
                c='r', marker='o')
            plt.show()

        if success:
            p_arr[i] = p_opt
            dp_arr[i] = p_err
        else:
            print(error_msg)

    dirpath = mrb.change_ext(filepaths['target'], '', mrb.EXT['img'])
    if not os.path.isdir(dirpath):
        os.makedirs(dirpath)
    p_imgs, dp_imgs = [], []
    fit_results = fitting_names, mrb.ndsplit(p_arr), mrb.ndsplit(dp_arr)
    for fitted_name, image_flat, error_flat in zip(*fit_results):
        # save the parameter map
        p_img = np.zeros_like(mask, float)
        p_img[mask] = image_flat
        p_filepath = os.path.join(
            dirpath, mrb.change_ext('p_' + fitted_name, mrb.EXT['img']))
        mrio.save(p_filepath, p_img, aff_mask)
        p_imgs.append(p_img)
        # save the error map
        dp_img = np.zeros_like(mask, float)
        dp_img[mask] = error_flat
        dp_filepath = os.path.join(
            dirpath, mrb.change_ext('dp_' + fitted_name, mrb.EXT['img']))
        mrio.save(dp_filepath, dp_img, aff_mask)
        dp_imgs.append(dp_img)
    # save combined
    p_filepath = os.path.join(
        dirpath, mrb.change_ext('p_all', mrb.EXT['img']))
    mrio.save(p_filepath, mrb.ndstack(p_imgs), aff_mask)
    dp_filepath = os.path.join(
        dirpath, mrb.change_ext('dp_all', mrb.EXT['img']))
    mrio.save(dp_filepath, mrb.ndstack(dp_imgs), aff_mask)


# ======================================================================
def handle_arg():
    """
    Handle command-line application arguments.
    """
    # :: Create Argument Parser
    arg_parser = argparse.ArgumentParser(
        description=__doc__,
        epilog='v.{} - {}\n{}'.format(
            INFO['version'], ', '.join(INFO['authors']), INFO['license']),
        formatter_class=argparse.RawDescriptionHelpFormatter)
    # :: Add POSIX standard arguments
    arg_parser.add_argument(
        '--ver', '--version',
        version='%(prog)s - ver. {}\n{}\n{} {}\n{}'.format(
            INFO['version'],
            next(line for line in __doc__.splitlines() if line),
            INFO['copyright'], ', '.join(INFO['authors']),
            INFO['notice']),
        action='version')
    arg_parser.add_argument(
        '-v', '--verbose',
        action='count', default=D_VERB_LVL,
        help='increase the level of verbosity [%(default)s]')
    # :: Add additional arguments
    # arg_parser.add_argument(
    #     '-f', '--force',
    #     action='store_true',
    #     help='force new processing [%(default)s]')
    arg_parser.add_argument(
        '-d', '--dirpath', metavar='PATH',
        default='.',
        help='set working directory [%(default)s]')
    arg_parser.add_argument(
        '-s', '--data_subdir', metavar='SUBPATH',
        default='mt_list',
        help='set data subpath [%(default)s]')
    arg_parser.add_argument(
        '-a', '--data_sources', metavar='FILE',
        default='data_sources.json',
        help='set data_sources filename [%(default)s]')
    arg_parser.add_argument(
        '-t', '--target', metavar='FILE',
        default='mask.nii.gz',
        help='set target mask filename [%(default)s]')
    arg_parser.add_argument(
        '-m', '--mask', metavar='FILE',
        default='mask.nii.gz',
        help='set extra mask filename [%(default)s]')
    arg_parser.add_argument(
        '-1', '--b1t', metavar='FILE',
        default='b1t.nii.gz',
        help='set B1T map filename [%(default)s]')
    arg_parser.add_argument(
        '-0', '--b0', metavar='FILE',
        default='b0.nii.gz',
        help='set B0 map filename [%(default)s]')
    arg_parser.add_argument(
        '-r', '--t1', metavar='FILE',
        default='t1.nii.gz',
        help='set T1 map filename [%(default)s]')
    arg_parser.add_argument(
        '-l', '--t2s', metavar='FILE',
        default='t2s.nii.gz',
        help='set T2S map filename [%(default)s]')
    arg_parser.add_argument(
        '-n', '--fitting_names', metavar='NAME',
        nargs='+',
        default=('m0', 'm_a', 'r2_a', 'r2_b', 'k_ab'),
        help='set names for fitting parameters [%(default)s]')
    return arg_parser


# ======================================================================
def main():
    # :: handle program parameters
    arg_parser = handle_arg()
    args = arg_parser.parse_args()
    # :: print debug info
    if args.verbose == VERB_LVL['debug']:
        arg_parser.print_help()
        print('II:', 'Parsed Arguments:', args)
    if args.verbose > VERB_LVL['low']:
        print(__doc__)

    args = {
        'dirpath': args.dirpath,
        'data_subdir': args.data_subdir,
        'filenames': {
            'data_sources': args.data_sources,
            'target': args.target,
            'mask': args.mask,
            'b1t': args.b1t,
            'b0': args.b0,
            't1': args.t1,
            't2s': args.t2s,
        },
        'fitting_names': args.fitting_names,
        'verbose': args.verbose
    }
    mt_fit_quick(**args)

    elapsed(os.path.basename(__file__))
    print_elapsed()


# ======================================================================
if __name__ == '__main__':
    main()
