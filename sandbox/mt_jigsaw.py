#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Bring back together MT chunks results
"""

# ======================================================================
# :: Future Imports
from __future__ import(
    division, absolute_import, print_function, unicode_literals)

import os
# import shutil
import functools
import argparse

import pymrt.base as pmb
import pymrt.input_output as pmio

from pymrt import INFO
from pymrt import VERB_LVL, D_VERB_LVL
from pymrt import msg, dbg


# ======================================================================
def mt_jigsaw(
        dirpath,
        target,
        sep,
        clean,
        verbose):
    """

    Args:
        dirpath:
        target:
        sep:
        clean:
        verbose:

    Returns:

    """
    # find real dirpath
    dirpath = pmb.realpath(dirpath)
    if verbose >= VERB_LVL['low']:
        print('Dir: ', dirpath)
        print('Target: ', target)
    # autodetect input files
    jigsaw = {}
    for filepath, stats in pmb.walk2(dirpath):
        last_subdir = os.path.basename(os.path.dirname(filepath))
        name = last_subdir.split(sep)[0]
        if name == target:
            key = pmb.change_ext(
                os.path.basename(filepath), '', pmb.EXT['niz'])
            if key in jigsaw:
                jigsaw[key].append(filepath)
            else:
                jigsaw[key] = [filepath]
    # put all the pieces together
    target_dirpath = os.path.join(os.path.dirname(dirpath), target)
    if not os.path.isdir(target_dirpath):
        os.makedirs(target_dirpath)
    for key, in_filepaths in jigsaw.items():
        out_filepath = os.path.join(
            target_dirpath,
            pmb.change_ext(sep.join((target, key)), pmb.EXT['niz']))
        if verbose >= VERB_LVL['low']:
            print('Out: ', out_filepath)
        pmio.simple_filter_n_1(
            in_filepaths, out_filepath,
            lambda imgs: functools.reduce(lambda x, y: x + y, imgs))
        if clean:
            for in_filepath in in_filepaths:
                if os.path.isfile(in_filepath):
                    if verbose >= VERB_LVL['medium']:
                        print('Del: ', in_filepath)
                    os.remove(in_filepath)
    if clean:
        if verbose >= VERB_LVL['low']:
            print('Cleaning up...')
        for filepath, stats in pmb.walk2(dirpath):
            last_subdir = os.path.basename(filepath)
            name = last_subdir.split(sep)[0]
            if name == target:
                if os.path.isdir(filepath):
                    if verbose >= VERB_LVL['medium']:
                        print('Del: ', filepath)
                    os.removedirs(filepath)
                elif os.path.isfile(filepath):
                    if verbose >= VERB_LVL['medium']:
                        print('Del: ', filepath)
                    os.remove(filepath)


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
        default='chunks/',
        help='set working directory [%(default)s]')
    arg_parser.add_argument(
        '-t', '--target', metavar='NAME',
        default='target',
        help='set target filename [%(default)s]')
    arg_parser.add_argument(
        '-s', '--separator', metavar='STR',
        default='__',
        help='set separator used in chunk name [%(default)s]')
    arg_parser.add_argument(
        '-k', '--keep',
        action='store_true',
        help='keep chunk files [%(default)s]')
    return arg_parser


# ======================================================================
def main():
    # :: handle program parameters
    arg_parser = handle_arg()
    args = arg_parser.parse_args()
    if args.quiet:
        args.verbose = VERB_LVL['none']
    # :: print debug info
    if args.verbose >= VERB_LVL['debug']:
        arg_parser.print_help()
        msg('\nARGS: ' + str(vars(args)), args.verbose, VERB_LVL['debug'])
    msg(__doc__.strip())

    args = {
        'dirpath': args.dirpath,
        'target': args.target,
        'sep': args.separator,
        'clean': not args.keep,
        'verbose': args.verbose
    }
    mt_jigsaw(**args)

    pmb.elapsed(os.path.basename(__file__))
    pmb.print_elapsed()

# ======================================================================
if __name__ == '__main__':
    main()
