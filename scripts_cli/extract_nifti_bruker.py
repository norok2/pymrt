#!python
# -*- coding: utf-8 -*-
"""
Extract NIfTI-1 files from Bruker datasets.

Optimized forimages produced by MT and MGE protocols.

Correction for number of averages is applied.
A tentative receiver gain correction is also performed.
"""

# ======================================================================
# :: Future Imports
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

# ======================================================================
# :: Python Standard Library Imports
import os  # Miscellaneous operating system interfaces
import shutil  # High-level file operations
# import math  # Mathematical functions
import time  # Time access and conversions
import datetime  # Basic date and time types
# import operator  # Standard operators as functions
import argparse  # Parser for command-line options, arguments and subcommands
# import itertools  # Functions creating iterators for efficient looping
import subprocess  # Subprocess management
# import multiprocessing  # Process-based parallelism
# import csv  # CSV File Reading and Writing [CSV: Comma-Separated Values]
# import json  # JSON encoder and decoder [JSON: JavaScript Object Notation]

# :: External Imports
# import numpy as np  # NumPy (multidimensional numerical arrays library)
# import scipy as sp  # SciPy (signal and image processing library)
# import matplotlib as mpl  # Matplotlib (2D/3D plotting library)
# import sympy as sym  # SymPy (symbolic CAS library)
# import PIL  # Python Image Library (image manipulation toolkit)
# import SimpleITK as sitk  # Image ToolKit Wrapper
# import nibabel as nib  # NiBabel (NeuroImaging I/O Library)
# import nipy  # NiPy (NeuroImaging in Python)
# import nipype  # NiPype (NiPy Pipelines and Interfaces)

# :: External Imports Submodules
# import matplotlib.pyplot as plt  # Matplotlib's pyplot: MATLAB-like syntax
# import scipy.optimize  # SciPy: Optimization Algorithms
# import scipy.integrate  # SciPy: Integrations facilities
# import scipy.constants  # SciPy: Mathematal and Physical Constants
# import scipy.ndimage  # SciPy: ND-image Manipulation

# :: Local Imports
import pymrt.utils as pmu
import pymrt.naming as pmn
import pymrt.input_output as pmio
import pymrt.extras as pme
# import pymrt.geometry as pmg
# from pymrt.sequences import mp2rage

from pymrt import INFO
from pymrt import VERB_LVL, D_VERB_LVL
from pymrt import msg, dbg
from pymrt import elapsed, print_elapsed


# ======================================================================
def calc_nii(in_dirpath, compress=True, force=False, verbose=D_VERB_LVL):
    """
    Reconstruct MRI image from Bruker dataset with standard parameters.
    """
    data_filepath = os.path.join(in_dirpath, 'fid')
    in_filename_list = 'method', 'acqp'
    in_filepath_list = [os.path.join(in_dirpath, in_filename)
                        for in_filename in in_filename_list]
    out_dirpath = os.path.join(in_dirpath, 'pdata/1')
    out_filename_list = 'bildabs.nii.gz', 'bildphs.nii.gz'
    out_filepath_list = [os.path.join(out_dirpath, out_filename)
                         for out_filename in out_filename_list]
    if os.path.exists(data_filepath):
        if pmu.check_redo(in_filepath_list, out_filepath_list, force):
            # remove files checked by matlab reco to avoid new folders
            matlab_output = os.path.join(out_dirpath, 'bildabs.mat')
            if os.path.exists(matlab_output):
                os.remove(matlab_output)
            if verbose > VERB_LVL['none']:
                print('Target:\t{}'.format(in_dirpath))
            cmd = [
                'matlab', '-nodesktop', '-nosplash', '-nojvm',
                '-r "calc_nii(\'{}\'); quit;"'.format(in_dirpath)]
            cwd = os.path.expanduser('~')
            subprocess.call(cmd, cwd=cwd)
            if compress:
                for out_filepath in out_filepath_list:
                    uncompressed = out_filepath[:-len('.gz')]
                    if os.path.exists(uncompressed):
                        subprocess.call(
                            ['gzip', '-f', uncompressed])
    return out_filepath_list


# ======================================================================
def postprocess_nii_mag(
        in_filepath, out_filepath, method_ldr_dict, params_ldr_dict,
        force=False, verbose=D_VERB_LVL):
    """
    Additional post-processing of magnitude images.
    """
    # number of averages
    num_avgs = method_ldr_dict['PVM_NAverages']
    # receiver gain of the ADC
    receiver_gain = params_ldr_dict['RG'] / 1000.0  # in Volts
    # correction factor
    factor = num_avgs * receiver_gain
    pmio.simple_filter_1_1(
        in_filepath, out_filepath, (lambda img: img / factor), [])


# ======================================================================
def postprocess_nii_phs(
        in_filepath, out_filepath, method_ldr_dict, params_ldr_dict,
        force=False, verbose=D_VERB_LVL):
    """
    Additional post-processing of phase images
    """
    # simple copy
    if in_filepath != out_filepath:
        shutil.copy(in_filepath, out_filepath)


# ======================================================================
def extract_nii(dirpath, extradir, force, verbose):
    """
    Extract single volumens from calculated NIfTI-1 images.

    TODO: integrate with pmu.check_redo()
    """
    mag_filepath = os.path.join(dirpath, 'pdata/1/bildabs.nii.gz')
    phs_filepath = os.path.join(dirpath, 'pdata/1/bildphs.nii.gz')
    method_filepath = os.path.join(dirpath, 'method')
    params_filepath = os.path.join(dirpath, 'acqp')
    filepath_list = \
        [mag_filepath, phs_filepath, method_filepath, params_filepath]
    if all([os.path.exists(filepath) for filepath in filepath_list]):
        if verbose > VERB_LVL['none']:
            print('Target:\t{}'.format(dirpath))
        # ensure proper destination path
        dest_dirpath, scan_id = os.path.split(dirpath)
        info_dict = pmn.parse_filename('')
        info_dict['scan_num'] = int(scan_id)
        if extradir:
            dest_dirpath = os.path.join(dest_dirpath, extradir)
        if not os.path.exists(dest_dirpath):
            os.makedirs(dest_dirpath)
        # extract info from protocol information files
        method_ldr_std_dict, method_ldr_user_dict, method_comments = \
            pme.jcampdx.read(method_filepath)
        method_ldr_dict = dict(
            list(method_ldr_std_dict.items()) + \
            list(method_ldr_user_dict.items()))
        params_ldr_std_dict, params_ldr_user_dict, params_comments = \
            pme.jcampdx.read(params_filepath)
        params_ldr_dict = dict(
            list(params_ldr_std_dict.items()) + \
            list(params_ldr_user_dict.items()))
        # generate scan number and protocol
        info_dict['protocol'] = params_ldr_dict['ACQ_scan_name']
        if all([method_ldr in method_ldr_dict for method_ldr
                in ['MTyesno', 'MT_superlist_freq', 'MT_superlist_power']]) \
                and method_ldr_dict['MTyesno'] == 'Yes':
            # MT-specific code
            old_mag_filepath_list = pmio.split(mag_filepath)
            old_phs_filepath_list = pmio.split(phs_filepath)
            base_protocol = info_dict['protocol']
            for i, (mt_freq, mt_power, old_mag_filepath, old_phs_filepath) \
                    in enumerate(zip(
                method_ldr_dict['MT_superlist_freq'].ravel().tolist(),
                method_ldr_dict['MT_superlist_power'].ravel().tolist(),
                old_mag_filepath_list, old_phs_filepath_list)):
                param_dict = {
                    'id': i,
                    'mtfreq': float(mt_freq),
                    'mtpower': float(mt_power)}
                info_dict['protocol'] = pmn.to_protocol(
                    base_protocol, param_dict)
                info_dict['img_type'] = pmn.TYPE_ID['mag']
                new_mag_filepath = pmn.to_filename(info_dict, dest_dirpath)
                info_dict['img_type'] = pmn.TYPE_ID['phs']
                new_phs_filepath = pmn.to_filename(info_dict, dest_dirpath)
                postprocess_nii_mag(
                    old_mag_filepath, old_mag_filepath,
                    method_ldr_dict, params_ldr_dict)
                postprocess_nii_phs(
                    old_phs_filepath, old_phs_filepath,
                    method_ldr_dict, params_ldr_dict)
                shutil.move(old_mag_filepath, new_mag_filepath)
                shutil.move(old_phs_filepath, new_phs_filepath)
        elif 'EffectiveTE' in method_ldr_dict:
            # Multi-Echo-specific code
            old_mag_filepath_list = pmio.split(mag_filepath)
            old_phs_filepath_list = pmio.split(phs_filepath)
            for te_val, old_mag_filepath, old_phs_filepath in \
                    zip(method_ldr_dict['EffectiveTE'], old_mag_filepath_list,
                        old_phs_filepath_list):
                info_dict['te_val'] = te_val
                info_dict['img_type'] = pmn.TYPE_ID['mag']
                new_mag_filepath = pmn.to_filename(info_dict, dest_dirpath)
                info_dict['img_type'] = pmn.TYPE_ID['phs']
                new_phs_filepath = pmn.to_filename(info_dict, dest_dirpath)
                postprocess_nii_mag(
                    old_mag_filepath, old_mag_filepath,
                    method_ldr_dict, params_ldr_dict)
                postprocess_nii_phs(
                    old_phs_filepath, old_phs_filepath,
                    method_ldr_dict, params_ldr_dict)
                shutil.move(old_mag_filepath, new_mag_filepath)
                shutil.move(old_phs_filepath, new_phs_filepath)
        else:
            # generic code
            info_dict['img_type'] = pmn.TYPE_ID['mag']
            new_mag_filepath = pmn.to_filename(info_dict, dest_dirpath)
            info_dict['img_type'] = pmn.TYPE_ID['phs']
            new_phs_filepath = pmn.to_filename(info_dict, dest_dirpath)
            old_mag_filepath = mag_filepath[::-1].replace(
                'bildabs'[::-1], 'bildabs-tmp'[::-1], 1)[::-1]
            old_phs_filepath = phs_filepath[::-1].replace(
                'bildphs'[::-1], 'bildphs-tmp'[::-1], 1)[::-1]
            shutil.copy(mag_filepath, old_mag_filepath)
            shutil.copy(phs_filepath, old_phs_filepath)
            postprocess_nii_mag(
                old_mag_filepath, old_mag_filepath,
                method_ldr_dict, params_ldr_dict)
            postprocess_nii_phs(
                old_phs_filepath, old_phs_filepath,
                method_ldr_dict, params_ldr_dict)
            shutil.move(old_mag_filepath, new_mag_filepath)
            shutil.move(old_phs_filepath, new_phs_filepath)


# ======================================================================
def extract_nifti(dirpath, extradir, force, verbose):
    """
    Walk through folders generated by Bruker scanner to extract NIfTI-1 files.
    """
    in_dirpath_list = pmu.listdir(dirpath, None)
    for in_dirpath in in_dirpath_list:
        in_subdirpath_list = pmu.listdir(in_dirpath, None)
        for in_subdirpath in in_subdirpath_list:
            if verbose >= VERB_LVL['medium']:
                print('Folder:\t{}'.format(in_subdirpath))
            calc_nii(in_subdirpath, True, force, verbose)
            extract_nii(in_subdirpath, extradir, force, verbose)
    return dirpath


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
    arg_parser.add_argument(
        '-q', '--quiet',
        action='store_true',
        help='override verbosity settings to suppress output [%(default)s]')
    # :: Add additional arguments
    arg_parser.add_argument(
        '-f', '--force',
        action='store_true',
        help='force new processing [%(default)s]')
    arg_parser.add_argument(
        '-d', '--dir', metavar='DIR',
        default='.',
        help='set working directory [%(default)s]')
    arg_parser.add_argument(
        '-e', '--extradir', metavar='FOLDER',
        default='nifti',
        help='set output subdirectory [%(default)s]')
    return arg_parser


# ======================================================================
def main():
    # :: handle program parameters
    arg_parser = handle_arg()
    args = arg_parser.parse_args()
    # fix verbosity in case of 'quiet'
    if args.quiet:
        args.verbose = VERB_LVL['none']
    # :: print debug info
    if args.verbose >= VERB_LVL['debug']:
        arg_parser.print_help()
        msg('\nARGS: ' + str(vars(args)), args.verbose, VERB_LVL['debug'])
    msg(__doc__.strip())

    extract_nifti(args.dir, args.extradir, args.force, args.verbose)

    elapsed('extract_nifit_bruker')
    print_elapsed()



# ======================================================================
if __name__ == '__main__':
    main()
