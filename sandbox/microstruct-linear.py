#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Brief single-line description.

Long multi-line description.

Created on Thu Sep 24 09:33:23 2015
@author: metere
"""

#    Copyright (C) 2015 Riccardo Metere <metere@cbs.mpg.de>
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <http://www.gnu.org/licenses/>.

# ======================================================================
# :: Future Imports
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

# ======================================================================
# :: Python Standard Library Imports
import os  # Miscellaneous operating system interfaces
# import math  # Mathematical functions
import time  # Time access and conversions
# import re  # Regular expression operations
# import operator  # Standard operators as functions
# import collections  # High-performance container datatypes
# import argparse  # Parser for command-line options, arguments and
# sub-commands
# import itertools  # Functions creating iterators for efficient looping
# import functools  # Higher-order functions and operations on callable objects
# import subprocess  # Subprocess management
# import multiprocessing  # Process-based parallelism
import json  # JSON encoder and decoder [JSON: JavaScript Object Notation]

# :: External Imports
import numpy as np  # NumPy (multidimensional numerical arrays library)
# import matplotlib as mpl  # Matplotlib (2D/3D plotting library)
import sympy as sym  # SymPy (symbolic CAS library)
# import PIL  # Python Image Library (image manipulation toolkit)
# import nipy  # NiPy (NeuroImaging in Python)
# import nipype  # NiPype (NiPy Pipelines and Interfaces)
# import dicom as pydcm  # PyDicom (Read, modify and write DICOM files.)

# :: External Imports Submodules
# import mayavi.mlab as mlab  # Mayavi's mlab: MATLAB-like syntax
# import scipy.optimize  # SciPy: Optimization Algorithms
# import scipy.integrate  # SciPy: Integrations facilities
# import scipy.constants  # SciPy: Mathematal and Physical Constants

# :: Local Imports
import pymrt.base as mrb
import pymrt.utils as mru
# import pymrt.plot as mrp
# import pymrt.registration as mrr
import pymrt.computation as mrc
# import pymrt.correlation as mrl
import pymrt.input_output as mrio

# from pymrt import VERB_LVL
# from pymrt import D_VERB_LVL
# from pymrt import _EVENTS

# ======================================================================
# :: Paths
PATHS = {
    'acquisition': '/nobackup/isar3/data/siemens/RME**ME-MP2RAGE',
    'cache': '/nobackup/isar2/cache',
}

PATHS['base'] = os.path.join(PATHS['cache'], 'microstruct', 'linear')
PATHS['sources'] = os.path.join(PATHS['base'], 'sources')
PATHS['sandbox'] = os.path.join(PATHS['base'], 'sandbox')
PATHS['templates'] = os.path.join(PATHS['base'], 'templates')
PATHS['masks'] = os.path.join(PATHS['base'], 'masks')

PATHS['prior'] = os.path.join(PATHS['sandbox'], 'prior')
PATHS['model'] = os.path.join(PATHS['sandbox'], 'model')

PATHS['subjects'] = mrb.listdir(PATHS['sources'], None, full_path=False)

# PATHS['masks'] = mrb.listdir(PATHS['masks'], mrn.D_EXT)

# make sure output directory exists
for item in [PATHS['sandbox'], PATHS['prior'], PATHS['model']]:
    if not os.path.isdir(item):
        os.makedirs(item)

mappings = [
    {'type': 'T1',
     'name': 'T_1', 'val_range': (700, 3500), 'val_units': 'ms'},
    {'type': 'T2S',
     'name': 'T_2^*', 'val_range': (1, 80), 'val_units': 'ms'},
    {'type': 'CHI',
     'name': '\\chi', 'val_range': (-99, 99), 'val_units': 'ppb'},
    {'type': 'R1',
     'name': 'R_1', 'val_range': (0.25, 1.5), 'val_units': 'Hz'},
    {'type': 'R2S',
     'name': 'R_2^*', 'val_range': (10, 1000), 'val_units': 'Hz'},
]

models = [
    'MyH-FeH-MyA-FeA',
    'MyH-FeL-MyL-FeH',
    'MyH-FeH-MyA-FeA-My0-Fe0',
    'MyH-FeL-MyL-FeH-My0-Fe0']

tpl_t1w = os.path.join(PATHS['templates'], 'MNI152_T1_1mm.nii.gz')
tpl_t1w_b = os.path.join(PATHS['templates'], 'MNI152_T1_1mm_brain.nii.gz')


# ======================================================================
def get_map(dirpath, type):
    filepaths = mrb.listdir(dirpath, mrb.EXT['niz'])
    for filepath in filepaths:
        if mru.parse_filename(filepath)['type'] == type:
            return filepath


# ======================================================================
def calc_wm_to_brain_ratio(
        dirpath=os.path.join(PATHS['base'], 'estimate_average_myelin')):
    names = {
        'bs': 'BrainStem.nii.gz',
        'cer': 'Cerebellum.nii.gz',
        'lc': 'LeftCerebralCortex.nii.gz',
        'rc': 'RightCerebralCortex.nii.gz',
        'lwm': 'LeftCerebralWhiteMatter.nii.gz',
        'rwm': 'RightCerebralWhiteMatter.nii.gz',
        'b': 'MNI152_T1_1mm_brain.nii.gz',
    }
    img = {}
    for name, filename in names.items():
        img[name] = mrio.load(os.path.join(dirpath, filename))
    wm = img['lwm'] + img['rwm']
    c = img['lc'] + img['rc']
    b = wm + c
    wm_ratio = np.sum(wm) / np.sum(b)
    return wm_ratio


# ======================================================================
def apply_affine(
        i_arrays,
        linear,
        shift):
    shape = i_arrays[0].shape
    i_arrays = np.array([arr.ravel() for arr in i_arrays])
    o_arrays = np.dot(linear, i_arrays) + shift
    o_arrays = [o_array.reshape(shape) for o_array in o_arrays]
    return o_arrays


# ======================================================================
def affine_model(
        r1,
        r2s,
        save_path=None):
    """
    Calculate the affine for the model, given the prior R1 and R2S info.

    Args:
        r1 (dict['My', 'Fe', 'Avg']): R1 in
            Myelin-rich, Iron-rich and Myelin-/Iron-free regions.
        r2s (dict['My', 'Fe', 'Avg']): R1 in
            Myelin-rich, Iron-rich and Myelin-/Iron-free regions.
        model ('full'|'average_only'|'no_average'):
            Model for calculating the affine transform.
        save_path (str): Path to files where to save the calculated parameters.

    Returns:
        (ndarray[2,2], ndarray[2,1]):
            Linear transformation and shift for the affine model.
    """
    # define all symbols
    # X = A * P + C
    relaxants = ('My', 'Fe')
    num = len(relaxants)
    aa = np.array(
        [sym.symbols('A_{}_{}'.format(i, j))
         for i in range(num) for j in range(num)]).reshape((num, num))
    cc = np.array([sym.symbols('C_{}'.format(i)) for i in range(num)])
    # concentrations, 1st index: relaxants, 2nd index: prior information
    xx = np.array(
        [sym.symbols('X_{}_{}'.format(i, j))
         for i in range(num) for j in range(num + 1)]).reshape((num, num + 1))
    # parametric maps, 1st index: parametric map, 2nd index: prior information
    pp = np.array(
        [sym.symbols('P_{}_{}'.format(i, j))
         for i in range(num) for j in range(num + 1)]).reshape((num, num + 1))

    model = list(set(r1.keys()) & set(r2s.keys()))
    print(model)
    try:
        model = model.split('-')
    except AttributeError:
        pass

    check_model = {
        'num_priors': len(model)
    }
    for relaxant in relaxants:
        check_model[relaxant] = 0
        for prior in model:
            if relaxant in prior:
                check_model[relaxant] += 1
    # todo: check that the number of priors is correct


    if 'MyH' in model:
        pass
    # num_eqs = len(model.split('-'))
    # if num_eqs == 4:
    #     eqs = (
    #         sym.Eq(xx[0, 0], aa[0, 0] * pp[0, 0] + aa[0, 1] * pp[1, 0]),
    #         sym.Eq(xx[1, 0], aa[1, 0] * pp[0, 0] + aa[1, 1] * pp[1, 0]),
    #         sym.Eq(xx[0, 1], aa[0, 0] * pp[0, 1] + aa[0, 1] * pp[1, 1]),
    #         sym.Eq(xx[1, 1], aa[1, 0] * pp[0, 1] + aa[1, 1] * pp[1, 1]),
    #     )
    # else:  # num_eqs == 6
    #     eqs = (
    #         sym.Eq(xx[0, 0], aa[0, 0] * pp[0, 0] + aa[0, 1] * pp[1, 0] + cc[0]),
    #         sym.Eq(xx[1, 0], aa[1, 0] * pp[0, 0] + aa[1, 1] * pp[1, 0] + cc[1]),
    #         sym.Eq(xx[0, 1], aa[0, 0] * pp[0, 1] + aa[0, 1] * pp[1, 1] + cc[0]),
    #         sym.Eq(xx[1, 1], aa[1, 0] * pp[0, 1] + aa[1, 1] * pp[1, 1] + cc[1]),
    #         sym.Eq(xx[0, 2], aa[0, 0] * pp[0, 2] + aa[0, 1] * pp[1, 2] + cc[0]),
    #         sym.Eq(xx[1, 2], aa[1, 0] * pp[0, 2] + aa[1, 1] * pp[1, 2] + cc[1]),
    #     )
    # unknowns = list(aa.ravel()) + list(cc)
    # sols = sym.solvers.solve(eqs, *unknowns)
    #
    # arbs, priors = {}, {}
    # if model == 'MyH-FeH-MyA-FeA-My0-Fe0':
    #     arbs = {
    #         xx[0, 0]: 1.0, xx[1, 0]: 1.0,
    #         xx[0, 1]: 0.5, xx[1, 1]: 0.5,
    #         xx[0, 2]: 0.0, xx[1, 2]: 0.1,
    #     }
    #     priors = {
    #         pp[0, 0]: r1['My'], pp[1, 0]: r2s['Fe'],
    #         pp[0, 1]: r1['Avg'], pp[1, 1]: r2s['Avg'],
    #         pp[0, 2]: r1['0'], pp[1, 2]: r2s['0'],
    #     }
    # elif model == 'MyH-FeL-MyL-FeH-My0-Fe0':
    #     arbs = {
    #         xx[0, 0]: 1.0, xx[1, 0]: 0.0,
    #         xx[0, 1]: 0.5, xx[1, 1]: 0.5,
    #         xx[0, 2]: 0.0, xx[1, 2]: 0.1,
    #     }
    #     priors = {
    #         pp[0, 0]: r1['My'], pp[1, 0]: r2s['My'],
    #         pp[0, 1]: r1['Fe'], pp[1, 1]: r2s['Fe'],
    #         pp[0, 2]: r1['0'], pp[1, 2]: r2s['0'],
    #     }
    # elif model == 'MyH-FeL-MyL-FeH':
    #     arbs = {
    #         xx[0, 0]: 1.0, xx[1, 0]: 0.0,
    #         xx[0, 1]: 0.5, xx[1, 1]: 0.5,
    #     }
    #     priors = {
    #         pp[0, 0]: r1['My'], pp[1, 0]: r2s['My'],
    #         pp[0, 1]: r1['Fe'], pp[1, 1]: r2s['Fe'],
    #     }
    # elif model == 'MyH-FeH-MyA-FeA':
    #     arbs = {
    #         xx[0, 0]: 1.0, xx[1, 0]: 0.0,
    #         xx[0, 1]: 0.5, xx[1, 1]: 0.5,
    #         xx[0, 2]: 0.0, xx[1, 2]: 0.1,
    #     }
    #     priors = {
    #         pp[0, 0]: r1['My'], pp[1, 0]: r2s['My'],
    #         pp[0, 1]: r1['Fe'], pp[1, 1]: r2s['Fe'],
    #         pp[0, 2]: r1['0'], pp[1, 2]: r2s['0'],
    #     }
    # elif model == 'MyA-FeA-My0-Fe0':
    #     arbs = {
    #         xx[0, 0]: 1.0, xx[1, 0]: 0.0,
    #         xx[0, 1]: 0.5, xx[1, 1]: 0.5,
    #         xx[0, 2]: 0.0, xx[1, 2]: 0.1,
    #     }
    #     priors = {
    #         pp[0, 0]: r1['My'], pp[1, 0]: r2s['My'],
    #         pp[0, 1]: r1['Fe'], pp[1, 1]: r2s['Fe'],
    #         pp[0, 2]: r1['0'], pp[1, 2]: r2s['0'],
    #     }
    # subst = mrb.merge_dicts(arbs, priors)
    #
    # aa_arr = np.array(
    #     [sols[aa[i, j]].subs(subst)
    #      for i in range(num) for j in range(num)]).reshape((num, num))
    # if num_eqs:
    #     cc_arr = np.array(
    #         [sols[cc[i]].subs(subst) for i in range(num)])
    # else:
    #     cc_arr = np.zeros_like(cc)
    if save_path is not None:
        np.savetxt(save_path, np.concatenate((aa_arr, cc_arr), -1))
    return aa_arr, cc_arr


# ======================================================================
def main():
    # :: calculate R1, R2S and T1W maps
    for subject in PATHS['subjects']:
        dirpath = os.path.join(PATHS['model'], subject)
        time_maps = [
            get_map(os.path.join(PATHS['sources'], subject), 'T1'),
            get_map(os.path.join(PATHS['sources'], subject), 'T2S')]
        rate_maps = [
            os.path.join(dirpath, os.path.basename(filepath))
            for filepath in (
                mru.change_img_type(time_maps[0], 'R1'),
                mru.change_img_type(time_maps[1], 'R2S'))]
        if not os.path.isdir(dirpath):
            os.makedirs(dirpath)
        for time_map, rate_map in zip(time_maps, rate_maps):
            if mrb.check_redo([time_map], [rate_map]):
                mrio.simple_filter_1_1(time_map, rate_map, mrc.time_to_rate)

    # :: calculate prior knowledge
    r1 = {
        'My': 1.00,
        'Fe': 0.95,
        'Avg': 0.70,
        '0': 0.20}
    r2s = {
        'My': 44.0,
        'Fe': 105.0,
        'Avg': 35.0,
        '0': 0.45}

    priors = ('My', 'Fe', 'Avg')
    r1_group = {key: [] for key in priors}
    r2s_group = {key: [] for key in priors}
    for subject in PATHS['subjects'][:-1]:
        dirpath = os.path.join(PATHS['model'], subject)
        if os.path.isdir(dirpath):
            masks = [
                (mru.parse_filename(mask)['type'], mask)
                for mask in mrb.listdir(dirpath, mrb.EXT['niz'])
                if os.path.basename(mask).startswith('mask')]
            r1_arr = mrio.load(
                get_map(os.path.join(PATHS['model'], subject), 'R1'))
            r2s_arr = mrio.load(
                get_map(os.path.join(PATHS['model'], subject), 'R2S'))
            for key, mask_path in masks:
                mask = mrio.load(mask_path).astype(bool)
                if key in r1_group:
                    r1_group[key].append(np.median(r1_arr[mask]))
                if key in r2s_group:
                    r2s_group[key].append(np.median(r2s_arr[mask]))
    print('Group averages: ')
    for key in priors:
        r1[key] = np.median(np.array(r1_group[key]))
        r2s[key] = np.median(np.array(r2s_group[key]))
        print("r1['{}'] = {}".format(key, r1[key]))
        print("r2s['{}'] = {}".format(key, r2s[key]))

    # r1['My'] = 0.997325177474
    # r2s['My'] = 44.1487077718
    # r1['Fe'] = 0.956634244353
    # r2s['Fe'] = 104.423335589
    # r1['Avg'] = 0.688705560823
    # r2s['Avg'] = 35.9445692941

    # :: generate models
    affines = {
        model: affine_model(
            r1, r2s, model, os.path.join(PATHS['model'], model + '.csv'))
        for model in models}

    # :: estimate Fe and My
    force = True
    export = {}
    print(':: Calculating models...')
    for idx, model in enumerate(models):
        print("Model '{}'".format(model))
        export[model] = {}
        for subject in PATHS['subjects']:
            print("Subject '{}'".format(subject))
            export[model][subject] = {}
            dirpath = os.path.join(PATHS['model'], subject)
            subpath = os.path.join(PATHS['model'], subject, model)
            rate_maps = [
                get_map(dirpath, 'R1'),
                get_map(dirpath, 'R2S')]
            conc_maps = [
                os.path.join(subpath, mru.change_param_val(mru.change_img_type(
                    os.path.basename(rate_maps[1]), img_type), 'm', idx))
                for img_type in ('My', 'Fe')]
            if not os.path.isdir(subpath):
                os.makedirs(subpath)
            model_filepath = model + '.csv'
            if mrb.check_redo(rate_maps, conc_maps, force):
                mrio.simple_filter_n_m(
                    rate_maps, conc_maps, apply_affine, *affines[model])
            # :: tentatively validate model
            masks = [
                (mru.parse_filename(mask)['type'], mask)
                for mask in mrb.listdir(dirpath, mrio.D_EXT)
                if os.path.basename(mask).startswith('mask') and
                mru.parse_filename(mask)['type'] not in priors]
            my_arr = mrio.load(conc_maps[0])
            fe_arr = mrio.load(conc_maps[1])
            for key, mask_path in masks:
                export[model][subject][key] = {}
                mask = mrio.load(mask_path).astype(bool)
                export[model][subject][key]['My'] = \
                    np.mean(my_arr[mask]), np.std(my_arr[mask])
                export[model][subject][key]['Fe'] = \
                    np.mean(fe_arr[mask]), np.std(fe_arr[mask])
    save_path = os.path.join(PATHS['model'], 'evaluation.json')
    with open(save_path, 'w') as export_file:
        json.dump(export, export_file, sort_keys=True, indent=4)


# ======================================================================
if __name__ == '__main__':
    begin_time = time.time()

    main()

    mrb.elapsed('microstruct-linear')
    mrb.print_elapsed()



    # for subject in PATHS['subjects']:
    #     dirpath = os.path.join(PATHS['prior'], subject)
    #     inv2m_filepath = get_map(
    #         os.path.join(PATHS['sources'], subject), 'INV2M')
    #     uniform_filepath = get_map(
    #         os.path.join(PATHS['sources'], subject), 'UNIFORM')
    #     aff_filepath = os.path.join(dirpath, 'affine.txt')
    #     warp_filepath = os.path.join(dirpath, 'warpcoef.txt')
    #     bet_filepath = os.path.join(dirpath, 'brain.nii.gz')
    #     flirt_filepath = os.path.join(dirpath, 'flirt.nii.gz')
    #     fnirt_filepath = os.path.join(dirpath, 'fnirt.nii.gz')
    #     in_filepath = os.path.join(
    #         dirpath,
    #         mru.change_img_type(os.path.basename(inv2m_filepath), 'T1W'))
    #     if not os.path.isdir(dirpath):
    #         os.makedirs(dirpath)
    #     mrn.simple_filter_n(
    #         [inv2m_filepath, uniform_filepath], in_filepath,
    #         lambda x:
    #             x[0][..., 0].astype(float) * x[1][..., 0].astype(float) /
    # 1e3)
    #
    #     # brain extract
    #     if mrb.check_redo([in_filepath], [bet_filepath]):
    #         print('bet')
    #         cmd = 'bet {} {} -R'.format(in_filepath, bet_filepath)
    #         mrb.execute(cmd)
    #     # linear registration matrix
    #     if mrb.check_redo(
    #             [bet_filepath, tpl_t1w_b],
    #             [aff_filepath, flirt_filepath]):
    #         print('flirt')
    #         cmd = 'flirt -in {} -ref {} -omat {} -out {}'.format(
    #             tpl_t1w_b, bet_filepath, aff_filepath, flirt_filepath)
    #         mrb.execute(cmd)
    #     # non-linear registration matrix
    #     if mrb.check_redo(
    #             [in_filepath, tpl_t1w, aff_filepath],
    #             [warp_filepath, fnirt_filepath]):
    #         print('fnirt')
    #         cmd = 'fnirt --in={} --ref={} --aff={} --cout={} --iout={
    # }'.format(
    #             tpl_t1w, in_filepath, aff_filepath, warp_filepath,
    #             fnirt_filepath)
    #         mrb.execute(cmd)
    #     # apply registration to masks
    #     for mask in PATHS['masks']:
    #         print('applywarp')
    #         reg_mask = os.path.join(dirpath, os.path.basename(mask))
    #         if mrb.check_redo(
    #                 [mask, tpl_t1w, aff_filepath, warp_filepath],
    #                 [reg_mask]):
    #             cmd = 'applywarp -i {} -o {} -r {} -w {}'.format(
    #                 mask, reg_mask, tpl_t1w, warp_filepath)
    #             mrb.execute(cmd)
