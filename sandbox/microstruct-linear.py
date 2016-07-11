#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Main data analysis for microstructure linear models

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
import collections  # High-performance container datatypes
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

from pymrt import VERB_LVL
from pymrt import D_VERB_LVL
from pymrt import _EVENTS
from pymrt import msg
from pymrt import dbg

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

tpl_t1w = os.path.join(PATHS['templates'], 'MNI152_T1_1mm.nii.gz')
tpl_t1w_b = os.path.join(PATHS['templates'], 'MNI152_T1_1mm_brain.nii.gz')


# ======================================================================
def priors_to_models(priors):
    models = [
        ('My_CC', 'Fe_CC', 'My_SN', 'Fe_SN'),
        ('My_CC', 'Fe_SN', 'My_avg', 'Fe_avg'),
        ('My_CC', 'Fe_SN', 'My_avg', 'Fe_avg', 'My_CSF', 'Fe_CSF'),
        ('My_CC', 'Fe_CC', 'My_SN', 'Fe_SN', 'My_CSF', 'Fe_CSF'),
        # ('My_CC', 'Fe_SN', 'My_CSF', 'Fe_CSF', 'My_avg'),  # the best model?
    ]
    return models


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


def time_to_rate(
        force=False,
        verbose=D_VERB_LVL):
    """
    Calculate R1, R2S maps from the corresponding T1, T2S.
    """
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
            if mrb.check_redo([time_map], [rate_map], force):
                msg('converting `{}` to `{}`...')
                mrio.simple_filter_1_1(time_map, rate_map, mrc.time_to_rate)


def refine_priors(priors):
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
    msg('Group averages: ', fmt='{t.green}{t.bold}')
    for key in priors:
        r1[key] = np.median(np.array(r1_group[key]))
        r2s[key] = np.median(np.array(r2s_group[key]))
        print("r1['{}'] = {}".format(key, r1[key]))
        print("r2s['{}'] = {}".format(key, r2s[key]))


# ======================================================================
def affine_model(
        model,
        priors,
        save_path=None):
    """
    Calculate the affine for the model, given the prior R1 and R2S info.

    Args:
        model (list[str]): Specification of the model as model elements.
            To each model element is associated an equation and an estimate
            of the concentration for the corresponding relaxant.
        priors (dict): Prior information required to calculate the model.
        save_path (str): Path to files where to save the calculated parameters.

    Returns:
        (ndarray[2,2], ndarray[2,1]):
            Linear transformation and shift for the affine model.
    """
    # :: analyze model for relaxants
    relaxants, rois = [], set()
    for elem in model:
        if elem in priors['X']:
            relaxant, roi = elem.split('_')
            if roi in priors['P']:
                rois.add(roi)
                relaxants += [relaxant]
            else:
                raise ValueError('Unknown ROI element `{}`'.format(roi))
        else:
            raise ValueError('Unknown model element `{}`'.format(elem))
    relaxants_counter = collections.Counter(relaxants)
    relaxants = tuple(sorted(relaxants_counter.keys()))
    relaxants_freqs = tuple(relaxants_counter.values())
    # :: analyze model for parameters
    parameters = []
    for roi in rois:
        if roi in priors['P']:  # unnecessary, but safer
            parameters += list(priors['P'][roi].keys())
    parameters_counter = collections.Counter(parameters)
    parameters = tuple(sorted(parameters_counter.keys()))

    # :: frequently used numbers
    n_x = len(relaxants)
    n_p = len(parameters)
    n_eqn = len(model)
    n_eqn_per_x = set(relaxants_freqs).pop()

    # :: consistency check
    # check that the number of equations for each relaxant is the same
    assert (len(set(relaxants_freqs)) == 1)
    assert (n_x * n_eqn_per_x == n_eqn)
    assert (len(set(parameters_counter.values())) == 1)

    # :: define symbols for the eqns: X = A * P + C
    aa = np.array(
        [sym.symbols('A_{}_{}'.format(i, j))
         for i in range(n_x) for j in range(n_p)]).reshape((n_x, n_p))

    ppp = {
        roi: np.array(
            [sym.symbols('{}_{}'.format(parameter, roi))
             for parameter in parameters]).reshape((n_p, 1))
        for roi in rois}
    xx, pp = [], []
    for elem in model:
        relaxant, roi = elem.split('_')
        xx.append(sym.symbols('{}'.format(elem)))
        pp.append(ppp[roi])

    # :: if the information is enough for the determination o
    if n_eqn == n_x * (n_p + 1):
        cc = np.array([sym.symbols('C_{}'.format(i)) for i in range(n_x)])
    else:
        cc = np.zeros((n_p, 1))
    eqs = []
    for i, elem in enumerate(model):
        relaxant, roi = elem.split('_')
        j = relaxants.index(relaxant)
        eqs.append(sym.Eq(xx[i], (np.dot(aa, pp[i])[j] + cc[j])[0]))
    # print(model)
    # print(eqs)
    unknowns = list(aa.ravel()) + list(cc)
    sols = sym.solvers.solve(eqs, *unknowns)
    # print(sols)
    relaxants_priors = {
        sym.symbols(name): value for name, value in priors['X'].items()}
    parameters_priors = {
        sym.symbols('_'.join((name, roi))): priors['P'][roi][name]
        for name in parameters for roi in priors['P'].keys()}
    # print(parameters_priors)
    subst = mrb.merge_dicts(relaxants_priors, parameters_priors)
    # print(subst)
    aa_arr = np.array(
        [sols[aa[i, j]].subs(subst)
         for i in range(n_x) for j in range(n_p)]).reshape((n_x, n_p))
    cc_arr = np.array(
        [(sols[cc[i]].subs(subst) if cc[i] else 0.0)
         for i in range(n_x)])
    # print(aa_arr)
    # print(cc_arr)
    return aa_arr, cc_arr


# ======================================================================
def estimate_relaxants_content(
        models,
        affines,
        force=False,
        verbose=D_VERB_LVL):
    # :: estimate relaxants content
    export = {}
    msg(':: Calculating models...')
    for idx, model in enumerate(models):
        msg('Model `{}`'.format(model))
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
    return


# ======================================================================
def main():
    # todo: get iron values from langkammer

    # Calculate R1, R2S maps from the corresponding T1, T2S.
    # time_to_rate(force)

    # :: define the prior knowledge known
    priors = {
        'P': {
            'CSF': {'R1': 0.20, 'R2S': 0.45},
            'CC': {'R1': 0.997, 'R2S': 44.1},
            'SN': {'R1': 0.957, 'R2S': 150.4},
            'avg': {'R1': 0.689, 'R2S': 35.9},
        },
        'X': {
            'My_CC': 1.0,
            'Fe_CC': 0.5,
            'My_SN': 0.5,
            'Fe_SN': 1.0,
            'My_CSF': 0.0,
            'Fe_CSF': 0.0,
            'My_avg': 0.5,
            'Fe_avg': 0.5,
        },
    }

    # :: refine the priors with current measurements
    refine_priors(priors)

    # :: generate models
    models = priors_to_models(priors['X'])
    affines = {
        model: affine_model(
            model, priors,
            os.path.join(PATHS['model'], '_'.join(model) + '.csv'))
        for model in models}

    # :: apply models to input data
    estimate_relaxants_content(models, affines, force=True)


# ======================================================================
if __name__ == '__main__':
    begin_time = time.time()

    main()

    mrb.elapsed('microstruct-linear')
    mrb.print_elapsed()
