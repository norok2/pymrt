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
# import time  # Time access and conversions
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
# import scipy as sp  # SciPy (signal and image processing library)
# import matplotlib as mpl  # Matplotlib (2D/3D plotting library)
import sympy as sym  # SymPy (symbolic CAS library)
import xarray as xr  # N-D labeled arrays and datasets in Python
# import PIL  # Python Image Library (image manipulation toolkit)
# import nipy  # NiPy (NeuroImaging in Python)
# import nipype  # NiPype (NiPy Pipelines and Interfaces)
# import dicom  # PyDicom (Read, modify and write DICOM files.)

# :: External Imports Submodules
# import mayavi.mlab as mlab  # Mayavi's mlab: MATLAB-like syntax
# import scipy.optimize  # SciPy: Optimization Algorithms
# import scipy.integrate  # SciPy: Integrations facilities
# import scipy.constants  # SciPy: Mathematal and Physical Constants

# :: Local Imports
import pymrt.base as pmb
import pymrt.naming as pmn
# import pymrt.plot as pmp
# import pymrt.registration as pmr
import pymrt.computation as pmc
# import pymrt.correlation as pml
import pymrt.input_output as pmio

from pymrt import INFO
from pymrt import VERB_LVL, D_VERB_LVL
from pymrt import msg, dbg

# ======================================================================
# :: Paths
PATHS = {
    'acquisition': '~/hd3/data/siemens/RME**ME-MP2RAGE',
    'cache': '~/hd2/cache',
}

PATHS['base'] = os.path.join(PATHS['cache'], 'microstruct', 'linear')
PATHS['sources'] = os.path.join(PATHS['base'], 'sources')
PATHS['sandbox'] = os.path.join(PATHS['base'], 'sandbox')
PATHS['templates'] = os.path.join(PATHS['base'], 'templates')
PATHS['masks'] = os.path.join(PATHS['base'], 'masks')

PATHS['prior'] = os.path.join(PATHS['sandbox'], 'prior')
PATHS['model'] = os.path.join(PATHS['sandbox'], 'model')

PATHS['samples'] = pmb.listdir(PATHS['sources'], None, full_path=False)

# PATHS['masks'] = pmb.listdir(PATHS['masks'], pmn.D_EXT)

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
        ('My_aCC', 'Fe_aCC', 'My_SN', 'Fe_SN'),
        ('My_aCC', 'Fe_SN', 'My_avg', 'Fe_avg'),
        ('My_aCC', 'Fe_SN', 'My_avg', 'Fe_avg', 'My_CSF', 'Fe_CSF'),
        ('My_aCC', 'Fe_aCC', 'My_SN', 'Fe_SN', 'My_CSF', 'Fe_CSF'),
        # ('My_CC', 'Fe_SN', 'My_CSF', 'Fe_CSF', 'My_avg'),  # the best model?
    ]
    models = [tuple(sorted(model)) for model in models]
    return models


# ======================================================================
def get_map(dirpath, type):
    filepaths = pmb.listdir(dirpath, pmb.EXT['niz'])
    for filepath in filepaths:
        if pmn.parse_filename(filepath)['type'] == type:
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
        img[name] = pmio.load(os.path.join(dirpath, filename))
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
def time_to_rate(
        sample_subpaths=PATHS['samples'],
        model_dirpath=PATHS['model'],
        sources_dirpath=PATHS['sources'],
        force=False,
        verbose=D_VERB_LVL):
    """
    Calculate R1, R2S maps from the corresponding T1, T2S.
    """
    for sample in sample_subpaths:
        dirpath = os.path.join(model_dirpath, sample)
        time_maps = [
            get_map(os.path.join(sources_dirpath, sample), 'T1'),
            get_map(os.path.join(sources_dirpath, sample), 'T2S')]
        rate_maps = [
            os.path.join(dirpath, os.path.basename(filepath))
            for filepath in (
                pmn.change_img_type(time_maps[0], 'R1'),
                pmn.change_img_type(time_maps[1], 'R2S'))]
        if not os.path.isdir(dirpath):
            os.makedirs(dirpath)
        for time_map, rate_map in zip(time_maps, rate_maps):
            if pmb.check_redo([time_map], [rate_map], force):
                msg('converting `{}` to `{}`...')
                pmio.simple_filter_1_1(time_map, rate_map, pmc.time_to_rate)


# ======================================================================
def get_roi_values(
        dirpath=PATHS['model'],
        out_filename='priors_maps.json',
        force=False,
        verbosity=D_VERB_LVL):
    """
    Refine priors from the measured R1 and R2* maps.
    """
    # todo: reimplement using xarray
    msg(':: Calculating ROI values in maps...')
    stats = {'avg': np.mean, 'std': np.std}
    subdirpaths = [
        item for item in pmb.listdir(dirpath, None) if os.access(item, os.W_OK)]
    inputs = []
    for subdirpath in subdirpaths:
        inputs += [
            mask for mask in pmb.listdir(subdirpath, pmb.EXT['niz'])
            if os.path.basename(mask).startswith('mask')]
    out_filepath = os.path.join(dirpath, out_filename)
    if pmb.check_redo(inputs, [out_filepath], force):
        roi_values = {}
        for subdirpath in subdirpaths:
            subdirname = os.path.basename(subdirpath)
            individual_stat = {}
            masks = [
                (pmn.parse_filename(mask)['type'], mask)
                for mask in pmb.listdir(subdirpath, pmb.EXT['niz'])
                if os.path.basename(mask).startswith('mask')]
            r1_arr = pmio.load(get_map(subdirpath, 'R1'))
            r2s_arr = pmio.load(get_map(subdirpath, 'R2S'))
            msg('Calculating on `{}`'.format(subdirname))
            for key, mask_path in masks:
                mask = pmio.load(mask_path).astype(bool)
                parameters = {'R1': r1_arr, 'R2S': r2s_arr}
                individual_stat[key] = {}
                for name, arr in parameters.items():
                    individual_stat[key][name] = {}
                    for stat, func in stats.items():
                        individual_stat[key][name][stat] = func(arr[mask])
            roi_values[subdirname] = individual_stat
        # group individual results
        msg('Grouping and saving to `{}`'.format(out_filepath))
        grouped = {}
        for individual, roi_value in roi_values.items():
            for roi, roi_vals in roi_value.items():
                if roi not in grouped:
                    grouped[roi] = {}
                for param, param_vals in roi_vals.items():
                    if param not in grouped[roi]:
                        grouped[roi][param] = {}
                    for stat in stats.keys():
                        if stat not in grouped[roi][param]:
                            grouped[roi][param][stat] = []
                        grouped[roi][param][stat].append(param_vals[stat])
        for roi, roi_vals in grouped.items():
            for param, param_vals in roi_vals.items():
                for stat in stats.keys():
                    grouped[roi][param][stat] = np.mean(param_vals[stat])
        roi_values['grouped'] = grouped
        # save to file
        with open(out_filepath, 'w') as out_file:
            json.dump(roi_values, out_file, sort_keys=True, indent=4)
    else:
        # load from file
        with open(out_filepath, 'r') as out_file:
            roi_values = json.load(out_file)
    return roi_values


# ======================================================================
def affine_model(
        model,
        priors,
        dirpath=PATHS['model'],
        filename=None):
    """
    Calculate the affine for the model, given the prior R1 and R2S info.

    Args:
        model (list[str]): Specification of the model as model elements.
            To each model element is associated an equation and an estimate
            of the concentration for the corresponding relaxant.
        priors (dict): Prior information required to calculate the model.
        dirpath (str): Path to directory of model results.
        filename (str|None): File name of model results.
            If None, it is generated from the model.

    Returns:
        (ndarray[2,2], ndarray[2,1]):
            Linear transformation and shift for the affine model.
    """
    msg('Model `{}`'.format(model))
    if not filename:
        filename = '_'.join(model) + '.csv'
    filepath = os.path.join(dirpath, filename)

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

    if not os.path.isfile(filepath):
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
        subst = pmb.merge_dicts(relaxants_priors, parameters_priors)
        # print(subst)
        aa_arr = np.array(
            [sols[aa[i, j]].subs(subst)
             for i in range(n_x) for j in range(n_p)]).reshape((n_x, n_p))
        cc_arr = np.array(
            [(sols[cc[i]].subs(subst) if cc[i] else 0.0)
             for i in range(n_x)]).reshape((n_x, 1))
        combined_arr = np.concatenate((aa_arr, cc_arr), -1)
        np.savetxt(filepath, combined_arr)

    else:
        data = np.loadtxt(filepath)
        aa_arr = data[:, :n_p].reshape((n_x, n_p))
        cc_arr = data[:, n_p].reshape((n_x, 1))
    return aa_arr, cc_arr


# ======================================================================
def estimate_relaxants_content(
        models,
        affines,
        sample_subpaths=PATHS['samples'],
        model_dirpath=PATHS['model'],
        force=False,
        verbose=D_VERB_LVL):
    """
    Produces semi-quantitative estimates of the relaxants content.
    """
    export = {}
    for idx, model in enumerate(models):
        msg('Model `{}`'.format(model))
        model_name = '_'.join(model)
        relaxants = sorted(set([elem.split('_')[0] for elem in model]))
        params = sorted(('R1', 'R2S'))
        export[model] = {}
        for sample in sample_subpaths:
            msg('Sample `{}`'.format(sample))
            export[model][sample] = {}
            dirpath = os.path.join(model_dirpath, sample)
            subpath = os.path.join(model_dirpath, sample, model_name)
            rate_maps = [get_map(dirpath, param) for param in params]
            conc_maps = [
                os.path.join(subpath, pmn.change_param_val(pmn.change_img_type(
                    os.path.basename(rate_maps[1]), img_type), 'm', idx))
                for img_type in relaxants]
            if not os.path.isdir(subpath):
                os.makedirs(subpath)
            if pmb.check_redo(rate_maps, conc_maps, force):
                pmio.simple_filter_n_m(
                    rate_maps, conc_maps, apply_affine, *affines[model])


# ======================================================================
def validate_results(
        models,
        sample_subpaths=PATHS['samples'],
        model_dirpath=PATHS['model'],
        save_filename='evaluation.json',
        force=False,
        verbose=D_VERB_LVL):
    """
    Validate results using information from priors.
    """
    # :: estimate relaxants content
    save_filepath = os.path.join(model_dirpath, save_filename)
    export = {}
    if not os.path.isfile(save_filepath):
        for idx, model in enumerate(models):
            msg('Model `{}`'.format(model))
            model_name = '_'.join(model)
            relaxants = sorted(set([elem.split('_')[0] for elem in model]))
            params = sorted(('R1', 'R2S'))
            export[model_name] = {}
            for sample in sample_subpaths:
                # :: tentatively validate model with roi information
                dirpath = os.path.join(model_dirpath, sample)
                subpath = os.path.join(model_dirpath, sample, model_name)
                rate_maps = [get_map(dirpath, param) for param in params]
                conc_maps = [
                    os.path.join(subpath, pmn.change_param_val(pmn.change_img_type(
                        os.path.basename(rate_maps[1]), img_type), 'm', idx))
                    for img_type in relaxants]
                model_priors = sorted(set([elem.split('_')[1] for elem in model]))
                masks = [
                    (pmn.parse_filename(mask)['type'], mask)
                    for mask in pmb.listdir(dirpath, pmb.EXT['niz'])
                    if os.path.basename(mask).startswith('mask') and
                    pmn.parse_filename(mask)['type'] not in model_priors]
                relaxant_arrs = [pmio.load(filepath) for filepath in conc_maps]
                export[model_name][sample] = {}
                for key, mask_path in masks:
                    export[model_name][sample][key] = {}
                    mask = pmio.load(mask_path).astype(bool)
                    for i, relaxant in enumerate(relaxants):
                        export[model_name][sample][key][relaxant] = \
                            np.mean(relaxant_arrs[i][mask]), \
                            np.std(relaxant_arrs[i][mask])

        with open(save_filepath, 'w') as export_file:
            json.dump(export, export_file, sort_keys=True, indent=4)

    else:
        with open(save_filepath, 'r') as export_file:
            export = json.load(export_file)
    return export

# ======================================================================
def main():
    # todo: get iron values from langkammer

    # Calculate R1, R2S maps from the corresponding T1, T2S.
    # time_to_rate(force)

    # :: define the prior knowledge known
    priors = {
        'P': {
            'CSF': {'R1': 0.20, 'R2S': 0.45},
            # 'aCC': {'R1': 0.997, 'R2S': 44.1},
            # 'SN': {'R1': 0.957, 'R2S': 150.4},
            # 'avg': {'R1': 0.689, 'R2S': 35.9},
        },
        'X': {
            'My_aCC': 1.0,
            'Fe_aCC': 0.5,
            'My_SN': 0.5,
            'Fe_SN': 1.0,
            'My_CSF': 0.0,
            'Fe_CSF': 0.0,
            'My_avg': 0.5,
            'Fe_avg': 0.5,
        },
    }

    # :: refine the priors with current measurements
    roi_values = get_roi_values()
    priors['P'].update(
        {roi: {param: val['avg'] for param, val in values.items()}
         for roi, values in roi_values['grouped'].items()})

    msg('Generating linear models...')
    models = priors_to_models(priors['X'])
    affines = {
        model: affine_model(model, priors)
        for model in models}

    msg(':: Estimating concentration of contrast sources...')
    estimate_relaxants_content(models, affines)

    msg(':: Validating linear models...')
    validate_results(models)


# ======================================================================
if __name__ == '__main__':
    begin_time = datetime.datetime.now()

    main()

    pmb.elapsed('microstruct-linear')
    pmb.print_elapsed()

# IDK what it does
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
#         pmn.change_img_type(os.path.basename(inv2m_filepath), 'T1W'))
#     if not os.path.isdir(dirpath):
#         os.makedirs(dirpath)
#     pmn.simple_filter_n(
#         [inv2m_filepath, uniform_filepath], in_filepath,
#         lambda x:
#         x[0][..., 0].astype(float) * x[1][..., 0].astype(float) / 1e3)
#
#     # brain extract
#     if pmb.check_redo([in_filepath], [bet_filepath]):
#         print('bet')
#         cmd = 'bet {} {} -R'.format(in_filepath, bet_filepath)
#         pmb.execute(cmd)
#     # linear registration matrix
#     if pmb.check_redo(
#             [bet_filepath, tpl_t1w_b],
#             [aff_filepath, flirt_filepath]):
#         print('flirt')
#         cmd = 'flirt -in {} -ref {} -omat {} -out {}'.format(
#             tpl_t1w_b, bet_filepath, aff_filepath, flirt_filepath)
#         pmb.execute(cmd)
#     # non-linear registration matrix
#     if pmb.check_redo(
#             [in_filepath, tpl_t1w, aff_filepath],
#             [warp_filepath, fnirt_filepath]):
#         print('fnirt')
#         cmd = 'fnirt --in={} --ref={} --aff={} --cout={} --iout={
# }'.format(
# tpl_t1w, in_filepath, aff_filepath, warp_filepath,
# fnirt_filepath)
# pmb.execute(cmd)
# # apply registration to masks
# for mask in PATHS['masks']:
#     print('applywarp')
# reg_mask = os.path.join(dirpath, os.path.basename(mask))
# if pmb.check_redo(
#         [mask, tpl_t1w, aff_filepath, warp_filepath],
#         [reg_mask]):
#     cmd = 'applywarp -i {} -o {} -r {} -w {}'.format(
#         mask, reg_mask, tpl_t1w, warp_filepath)
# pmb.execute(cmd)
