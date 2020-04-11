#!python
# -*- coding: utf-8 -*-
"""
Combination of multiple coil elements.

Methods not requiring `ref` or `multi_axis` are:
 - 'complex_sum';
 - 'sum_of_squares';
 - 'adaptive';
 - 'block_adaptive';
 - 'adaptive_iter';
 - 'block_adaptive_iter';
Methods requiring `ref` but not `multi_axis` are:
 Not implemented yet.
Methods requiring `multi_axis` but not `ref` are:
 - 'multi_svd';
Methods requiring both `ref` and `multi_axis` are:
 Not implemented yet.

If unsure, use `block_adaptive_iter`.
"""

# ======================================================================
# :: Future Imports
from __future__ import (
    division, absolute_import, print_function, unicode_literals, )

# ======================================================================
# :: Python Standard Library Imports
import datetime  # Basic date and time types
import argparse  # Parser for command-line options, arguments and subcommands
import flyingcircus as fc  # Everything you always wanted to have in Python*

# :: Local Imports
import pymrt as mrt  # Python Magnetic Resonance Tools: the multi-tool of MRI
import pymrt.util
import pymrt.input_output

from pymrt import INFO
from pymrt import VERB_LVL, D_VERB_LVL, VERB_LVL_NAMES
from pymrt import msg, dbg, fmt, fmtm
from pymrt.recipes import coils


# ======================================================================
def coil_combine(
        in_filepaths,
        out_filepaths,
        method='block_adaptive_iter',
        method_kws=None,
        compression='compress_svd',
        compression_kws=None,
        coil_axis=-1,
        split_axis=None,
        q_filepath=None,
        force=False,
        verbose=D_VERB_LVL):
    in_mag_filepath, in_phs_filepath = in_filepaths
    out_mag_filepath, out_phs_filepath = out_filepaths
    msg('Input-MAG: {}'.format(in_mag_filepath))
    msg('Input-PHS: {}'.format(in_phs_filepath))
    msg('Output-MAG: {}'.format(out_mag_filepath))
    msg('Output-PHS: {}'.format(out_phs_filepath))
    msg('Method: {}'.format(method))
    msg('Compression: {}'.format(compression))
    msg('Quality: {}'.format(q_filepath))
    # in_filepaths = [in_mag_filepath, in_phs_filepath]
    # out_filepaths = [out_mag_filepath, out_phs_filepath]
    if fc.check_redo(in_filepaths, out_filepaths, force):
        mag_coils_arr, meta = mrt.input_output.load(in_mag_filepath, meta=True)
        phs_coils_arr = mrt.input_output.load(in_phs_filepath)
        coils_arr = fc.extra.polar2complex(mag_coils_arr, phs_coils_arr)
        del mag_coils_arr, phs_coils_arr
        arr = coils.combine(
            coils_arr, method=method, method_kws=method_kws,
            compression=compression, compression_kws=compression_kws,
            coil_axis=coil_axis, split_axis=split_axis,
            verbose=verbose)
        mag_arr, phs_arr = fc.complex2polar(arr)
        mrt.input_output.save(out_mag_filepath, mag_arr, **meta)
        mrt.input_output.save(out_phs_filepath, phs_arr, **meta)
        if fc.check_redo(out_filepaths, q_filepath):
            q_arr = coils.quality(
                coils_arr, arr, coil_axis=coil_axis, verbose=verbose)
            mrt.input_output.save(q_filepath, q_arr)


# ======================================================================
def handle_arg():
    """
    Handle command-line application arguments.
    """
    # :: Create Argument Parser
    arg_parser = argparse.ArgumentParser(
        description=__doc__,
        epilog=fmtm('v.{version} - {author}\n{license}', INFO),
        formatter_class=argparse.RawDescriptionHelpFormatter)
    # :: Add POSIX standard arguments
    arg_parser.add_argument(
        '--ver', '--version',
        version=fmt(
            '%(prog)s - ver. {version}\n{}\n{copyright} {author}\n{notice}',
            next(line for line in __doc__.splitlines() if line), **INFO),
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
        '-i', '--in_filepaths', metavar=('MAG_FILE', 'PHS_FILE'),
        nargs=2, default=('mag_coils.nii.gz', 'phs_coils.nii.gz'),
        help='set input magnitude and phase filepaths [%(default)s]')
    arg_parser.add_argument(
        '-o', '--out_filepaths', metavar=('MAG_FILE', 'PHS_FILE'),
        nargs=2, default=('mag.nii.gz', 'phs.nii.gz'),
        help='set output magnitude and phase filepaths [%(default)s]')
    arg_parser.add_argument(
        '-p', '--q_filepath', metavar='FILE',
        default=None,
        help='set output quality metric filepath [%(default)s]')
    arg_parser.add_argument(
        '-m', '--method', metavar='STR',
        default='block_adaptive_iter',
        help=' [%(default)s]')
    arg_parser.add_argument(
        '-mk', '--method_kws', metavar='STR',
        default=None,
        help='Additional keyword parameters for the method [%(default)s]')
    arg_parser.add_argument(
        '-c', '--compression', metavar='STR',
        default='compress_svd',
        help=' [%(default)s]')
    arg_parser.add_argument(
        '-ck', '--compression_kws', metavar='STR',
        default=None,
        help='Additional keyword parameters for the compression [%(default)s]')
    arg_parser.add_argument(
        '-a', '--coil_axis', metavar='N',
        default=-1,
        help='Axis along which the single coils are stored [%(default)s]')
    arg_parser.add_argument(
        '-s', '--split_axis', metavar='N',
        default=None,
        help='Axis along which input shall be split [%(default)s]')
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
    else:
        msg(__doc__.strip())
    begin_time = datetime.datetime.now()

    kws = vars(args)
    kws.pop('quiet')
    coil_combine(**kws)

    end_time = datetime.datetime.now()
    if args.verbose > VERB_LVL['low']:
        print('ExecTime: {}'.format(end_time - begin_time))


# ======================================================================
if __name__ == '__main__':
    main()
