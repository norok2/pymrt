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
- merge-optim [Ref: Jenkinson, Mark. “Fast, Automated, N-Dimensional
  Phase-Unwrapping Algorithm.” Magnetic Resonance in Medicine 49, no. 1
  (January 1, 2003): 193–97. doi:10.1002/mrm.10354.]
"""

# ======================================================================
# :: Future Imports
from __future__ import (
    division, absolute_import, print_function, unicode_literals)

# ======================================================================
# :: Python Standard Library Imports
import datetime  # Basic date and time types
import argparse  # Parser for command-line options, arguments and subcommands
import json  # JSON encoder and decoder [JSON: JavaScript Object Notation]

# :: Local Imports
import pymrt.utils as pmu
import pymrt.input_output as pmio

from pymrt import INFO
from pymrt import VERB_LVL, D_VERB_LVL, VERB_LVL_NAMES
from pymrt import msg, dbg
from pymrt.config import EXT_CMD
from pymrt.recipes.unwrap_phase import (
    unwrap_phase_sorting_path, unwrap_phase_laplacian)

# ======================================================================
METHODS = ('sorting-path', 'laplacian', 'merge-optim')


# ======================================================================
def unwrap(
        in_filepath,
        out_filepath,
        method,
        options=None,
        force=False,
        verbose=D_VERB_LVL):
    msg('Input:  {}'.format(in_filepath))
    msg('Output: {}'.format(out_filepath))
    msg('Method: {}'.format(method))
    if pmu.check_redo([in_filepath], [out_filepath], force):
        if method == 'sorting-path':
            if options is not None:
                options = json.loads(options)
            else:
                options = {}
            pmio.simple_filter_1_1(
                in_filepath, out_filepath,
                unwrap_phase_sorting_path, **options)
        elif method == 'laplacian':
            if options is not None:
                options = json.loads(options)
            else:
                options = {}
            pmio.simple_filter_1_1(
                in_filepath, out_filepath,
                unwrap_phase_laplacian, **options)
        elif method == 'merge-optim':
            msg('W: method supported through FSL')
            ext_cmd = EXT_CMD['fsl/5.0/prelude']
            cmd_args = {
                'p': in_filepath,
                'o': out_filepath}
            cmd = ' '.join(
                [ext_cmd] +
                ['-{} {}'.format(k, v) for k, v in cmd_args.items()] +
                [options])
            pmu.execute(cmd, verbose=verbose)


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
    if args.verbose > VERB_LVL['low']:
        print('ExecTime: {}'.format(end_time - begin_time))


# ======================================================================
if __name__ == '__main__':
    main()
