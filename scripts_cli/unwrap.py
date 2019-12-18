#!python
# -*- coding: utf-8 -*-
"""
Unwrap of a phase image.

Several algorithms are available:
- sorting-path [Ref: Herráez, Miguel Arevallilo, David R. Burton, Michael
  J. Lalor, and Munther A. Gdeisat. “Fast Two-Dimensional Phase-Unwrapping
  Algorithm Based on Sorting by Reliability Following a Noncontinuous Path.”
  Applied Optics 41, no. 35 (December 10, 2002): 7437.
  doi:10.1364/AO.41.007437;  Abdul-Rahman, Hussein, Munther Gdeisat, David
  Burton, and Michael Lalor. “Fast Three-Dimensional Phase-Unwrapping
  Algorithm Based on Sorting by Reliability Following a Non-Continuous Path,”
  5856:32–40, 2005. doi:10.1117/12.611415.]
- laplacian [Ref: Schofield, Marvin A., and Yimei Zhu. “Fast Phase
  Unwrapping Algorithm for Interferometric Applications.” Optics Letters 28,
  no. 14 (July 15, 2003): 1194–96. doi:10.1364/OL.28.001194.]
"""
# - merge-optim [Ref: Jenkinson, Mark. “Fast, Automated, N-Dimensional
#   Phase-Unwrapping Algorithm.” Magnetic Resonance in Medicine 49, no. 1
#   (January 1, 2003): 193–97. doi:10.1002/mrm.10354.]
# """

# ======================================================================
# :: Future Imports
from __future__ import (
    division, absolute_import, print_function, unicode_literals, )

# ======================================================================
# :: Python Standard Library Imports
import datetime  # Basic date and time types
import argparse  # Parser for command-line options, arguments and subcommands
import json  # JSON encoder and decoder [JSON: JavaScript Object Notation]


import flyingcircus as fc  # Everything you always wanted to have in Python*

# :: Local Imports
import pymrt as mrt  # Python Magnetic Resonance Tools: the multi-tool of MRI.
import pymrt.util
import pymrt.input_output

from pymrt import INFO
from pymrt import VERB_LVL, D_VERB_LVL, VERB_LVL_NAMES
from pymrt import msg, dbg, fmt, fmtm
from pymrt.config import EXT_CMD
from pymrt.recipes import phs

# ======================================================================
METHODS = ('sorting-path', 'laplacian', 'merge-optim')


# ======================================================================
def _unwrap(arr, method, fix_interval, fix_offset, **_kws):
    if fix_interval:
        arr = phs.fix_interval(arr)
    if fix_offset:
        arr = phs.fix_offset(arr)
    return method(arr, **_kws)


# ======================================================================
def unwrap(
        in_filepath,
        out_filepath,
        method,
        fix_interval,
        fix_offset,
        options=None,
        force=False,
        verbose=D_VERB_LVL):
    msg(fmtm('Input:  {in_filepath}'))
    msg(fmtm('Output: {out_filepath}'))
    msg(fmtm('Method: {method}'))
    if fc.base.check_redo([in_filepath], [out_filepath], force):
        if method == 'sorting-path':
            method = phs.unwrap_sorting_path
        elif method == 'laplacian':
            method = phs.unwrap_laplacian
        else:
            msg(fmtm('W: method `{method}` not supported.'))

        if options is not None:
            options = json.loads(options)
        else:
            options = {}

        mrt.input_output.simple_filter_1_1(
            in_filepath, out_filepath,
            _unwrap, *(method, fix_interval, fix_offset), **options)


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
        '-i', '--in_filepath', metavar='FILE',
        default='phs.nii.gz',
        help='set input filepath [%(default)s]')
    arg_parser.add_argument(
        '-o', '--out_filepath', metavar='FILE',
        default='phs_unwrap.nii.gz',
        help='set input filepath [%(default)s]')
    arg_parser.add_argument(
        '-m', '--method', metavar='STR',
        default='sorting-path',
        help=' [%(default)s]')
    arg_parser.add_argument(
        '-n', '--fix_interval',
        action='store_true',
        help='Fix the interval of input values [%(default)s]')
    arg_parser.add_argument(
        '-t', '--fix_offset',
        action='store_true',
        help='Fix the offset of output values [%(default)s]')
    arg_parser.add_argument(
        '-x', '--options', metavar='STR',
        default=None,
        help='Optional parameters to be passed to the algorithm [%(default)s]')
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
    unwrap(**kws)

    end_time = datetime.datetime.now()
    exec_time = datetime.datetime.now() - begin_time
    msg('ExecTime: {}'.format(exec_time), args.verbose, VERB_LVL['debug'])


# ======================================================================
if __name__ == '__main__':
    main()
