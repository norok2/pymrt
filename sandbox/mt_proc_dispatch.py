#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Spawn multiple MT processes for parallel execution
"""

# ======================================================================
# :: Future Imports
from __future__ import division
from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals

import os
import argparse

from math import ceil, log10

import numpy as np

import pymrt.base as mrb
import pymrt.input_output as mrio

from pymrt import INFO
from pymrt import VERB_LVL
from pymrt import D_VERB_LVL


# ======================================================================
def mt_proc_dispatch(
        dirpath,
        target_name,
        chunk_name,
        num_chunks,
        chunk_sizes,
        cmd_dispatch,
        cmd_script,
        run_in_background,
        dry,
        verbose):
    """

    Args:
        dirpath:
        target_name:
        chunk_name:
        num_chunks:
        chunk_sizes:
        cmd_dispatch:
        cmd_script:
        run_in_background:
        dry:
        verbose:

    Returns:

    """
    filenames = {'target': target_name, 'chunk': chunk_name}
    filepaths = {}
    for item, filename in filenames.items():
        filepaths[item] = os.path.realpath(filename)
        if not os.path.exists(filepaths[item]):
            filepaths[item] = os.path.join(dirpath, filename)
    # preprocess input
    img, aff, hdr = mrio.load(filepaths['target'], True)
    img = img.astype(int)
    msk = img[...].astype(bool)
    msk_size = int(sum(img[msk]))
    idx = np.zeros_like(img)
    idx[msk] = np.arange(msk_size) + 1
    # determine num_chunks
    chunk_size = msk_size // num_chunks
    if chunk_size > chunk_sizes[1]:
        old_num_chunks = num_chunks // 1
        num_chunks = msk_size // chunk_sizes[1] + 1
        if verbose >= VERB_LVL['low']:
            print('num chunks adjusted: {} -> {}'.format(
                old_num_chunks, num_chunks))
    if chunk_size < chunk_sizes[0]:
        old_num_chunks = num_chunks // 1
        num_chunks = msk_size // chunk_sizes[0] + 1
        if verbose >= VERB_LVL['low']:
            print('num chunks adjusted: {} -> {}'.format(
                old_num_chunks, num_chunks))
    if verbose >= VERB_LVL['low']:
        print('approx chunk size: {}'.format(msk_size // num_chunks))

    if num_chunks <= 1:
        cmd = (cmd_dispatch + ' ') \
            if cmd_dispatch else ''
        cmd += cmd_script.format(chunk=filepaths['target']) + ' '
        if run_in_background:
            cmd += '&'
        chunk_filepaths = filepaths['target'],
        if verbose >= VERB_LVL['low']:
            print(' > ', cmd)
        if not dry:
            os.system(cmd)

    else:
        chunk_idxs = np.array_split(idx[msk], num_chunks)
        # make sure output filepath exists
        chunk_dirpath = os.path.join(
            os.path.dirname(filepaths['chunk']),
            os.path.dirname(target_name))
        if not os.path.isdir(chunk_dirpath):
            os.makedirs(chunk_dirpath)
        chunk_filepaths = []
        for i, chunk_idx in enumerate(chunk_idxs):
            chunk = np.zeros_like(img)
            chunk_mask = \
                (idx >= np.min(chunk_idx)) * (idx <= np.max(chunk_idx))
            chunk[chunk_mask] = 1
            chunk_filepath = filepaths['chunk'].format(
                name=mrb.change_ext(filenames['target'], '', mrb.EXT['img']),
                id='{:0{len}d}o{:0{len}d}'.format(
                    i + 1, num_chunks, len=int(ceil(log10(num_chunks + 1)))))
            chunk_dirpath = mrb.change_ext(chunk_filepath, '', mrb.EXT['img'])
            if not os.path.isfile(chunk_filepath):
                mrio.save(chunk_filepath, chunk.astype(int), aff)
                chunk_filepaths.append(chunk_filepath)
            if not os.path.isdir(chunk_dirpath):
                cmd = (cmd_dispatch + ' ') \
                    if cmd_dispatch else ''
                cmd += cmd_script.format(chunk=chunk_filepath) + ' '
                if run_in_background:
                    cmd += '&'
                if verbose >= VERB_LVL['low']:
                    print(' > ', cmd)
                if not dry:
                    os.system(cmd)
            else:
                if verbose >= VERB_LVL['low']:
                    print('{}: already processed'.format(
                        os.path.basename(chunk_filepath)))
    return chunk_filepaths


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
    # arg_parser.add_argument(
    #     '-f', '--force',
    #     action='store_true',
    #     help='force new processing [%(default)s]')
    arg_parser.add_argument(
        '-d', '--dirpath', metavar='PATH',
        default='.',
        help='set working directory [%(default)s]')
    arg_parser.add_argument(
        '-t', '--target', metavar='FILE',
        default='target.nii.gz',
        help='set target filename [%(default)s]')
    arg_parser.add_argument(
        '-c', '--chunk', metavar='FILE',
        default='chunks/{name}__{id}.nii.gz',
        help='set chunk filename [%(default)s]')
    arg_parser.add_argument(
        '-n', '--num_chunks', metavar='N',
        type=int, default=30,
        help='set names for fitting parameters [%(default)s]')
    arg_parser.add_argument(
        '-s', '--chunk_sizes', metavar=('MIN', 'MAX'),
        type=int, nargs=2, default=(100, 300),
        help='set names for fitting parameters [%(default)s]')
    arg_parser.add_argument(
        '-p', '--cmd_dispatch', metavar='CMD',
        # default='',
        default='',
        help='set the dispatcher command [%(default)s]')
    arg_parser.add_argument(
        '-x', '--cmd_script', metavar='CMD',
        default='/home/raid1/metere/Documents/workspace/pymrt/sandbox'
                '/mt_fit_quick.py -v '
                '--target {chunk}',
        help='set script to run on each chunk [%(default)s]')
    arg_parser.add_argument(
        '-b', '--bg_run',
        action='store_true',
        help='set to run the script in the background [%(default)s]')
    arg_parser.add_argument(
        '-y', '--dry',
        action='store_true',
        help='set to perform a dry run [%(default)s]')
    return arg_parser


# ======================================================================
def main():
    # :: handle program parameters
    arg_parser = handle_arg()
    args = arg_parser.parse_args()
    if args.quiet:
        args.verbose = VERB_LVL['none']
    # :: print debug info
    if args.verbose == VERB_LVL['debug']:
        arg_parser.print_help()
        print('II:', 'Parsed Arguments:', args)
    if args.verbose > VERB_LVL['low']:
        print(__doc__)

    args = {
        'dirpath': args.dirpath,
        'target_name': args.target,
        'chunk_name': args.chunk,
        'num_chunks': args.num_chunks,
        'chunk_sizes': args.chunk_sizes,
        'cmd_dispatch': args.cmd_dispatch,
        'cmd_script': args.cmd_script,
        'run_in_background': args.bg_run,
        'dry': args.dry,
        'verbose': args.verbose
    }
    mt_proc_dispatch(**args)

    mrb.elapsed(os.path.basename(__file__))
    mrb.print_elapsed()


# ======================================================================
if __name__ == '__main__':
    main()
