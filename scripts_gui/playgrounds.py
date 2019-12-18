#!python
# -*- coding: utf-8 -*-
"""
PyMRT - Educational Playgrounds

Visualization of effects of parameters on equations of theoretical interest.
This is suitable for educational purposes.
"""

# ======================================================================
# :: Future Imports
from __future__ import (
    division, absolute_import, print_function, unicode_literals, )

# ======================================================================
# :: Python Standard Library Imports
import os  # Miscellaneous operating system interfaces
import argparse  # Argument Parsing
import collections  # Container datatypes
# import datetime  # Basic date and time types
import string  # Common string operations

# :: External Imports
import numpy as np  # NumPy (multidimensional numerical arrays library)
# import scipy as sp  # SciPy (signal and image processing library)
import numex.interactive_tk_mpl

# :: Local Imports
import pymrt as mrt  # Python Magnetic Resonance Tools: the multi-tool of MRI
from pymrt.sequences import mp2rage

from pymrt import INFO, PATH
from pymrt import VERB_LVL, D_VERB_LVL
from pymrt import msg, dbg, fmt, fmtm
from pymrt import elapsed, report

# ======================================================================
from numpy import pi, e, log, log10, log2, \
    sin, cos, tan, sinc, sign, heaviside

# ======================================================================
TITLE_BASE = __doc__.strip().split('\n')[0][:-1]
PREFIX = 'playground_'
TITLE = {}
PARAMS = {}

# ======================================================================
# :: Fourier 1D
TITLE['fourier_1d'] = 'Fourier Transform in 1D'
PARAMS['fourier_1d'] = collections.OrderedDict(
    [('re_f_x', dict(label='Re[f(x)]', default='a*sin(b*x+c)+d', values=())),
     ('im_f_x', dict(label='Im[f(x)]', default='0', values=())), ]
    + [('const_{}'.format(letter), dict(
        label=letter, default=1.0, start=-5, stop=5, step=0.01))
       for letter in string.ascii_lowercase[:20]]
    + [('x_start', dict(
        label='x_start', default=-10, start=-20, stop=20, step=1)),
       ('x_stop', dict(
           label='x_stop', default=10, start=-20, stop=20, step=1)),
       ('x_num', dict(
           label='x_num / #', default=256, start=16, stop=4096, step=16)), ])


# ======================================================================
def playground_fourier_1d(
        fig,
        params=None,
        title=TITLE_BASE):
    axs = fig.subplots(2, 4, squeeze=False)  # , sharex='row', sharey='row')
    x = np.linspace(params['x_start'], params['x_stop'], params['x_num'])

    for k, v in params.items():
        if k.startswith('const_'):
            locals()[k[len('const_'):]] = v
    try:
        re_y = eval(params['re_f_x'])
    except (SyntaxError, NameError, TypeError):
        re_y = 0

    try:
        im_y = eval(params['im_f_x'])
    except (SyntaxError, NameError, TypeError):
        im_y = 0

    y = re_y + 1j * im_y
    ft_y = np.fft.fftshift(np.fft.fftn(y))

    # axs[0, 0].set_axis_off()

    axs[0, 0].set_title(r'$\operatorname{Re}[{\operatorname{f}}(x)]$')
    axs[0, 0].plot(x, np.real(y))

    axs[0, 1].get_shared_x_axes().join(axs[0, 0], axs[0, 1])
    axs[0, 1].set_xticklabels([])
    axs[0, 1].get_shared_y_axes().join(axs[0, 0], axs[0, 1])
    axs[0, 1].set_yticklabels([])
    axs[0, 1].autoscale()
    axs[0, 1].set_title(r'$\operatorname{Im}[{\operatorname{f}}(x)]$')
    axs[0, 1].plot(x, np.imag(y))

    axs[0, 2].get_shared_x_axes().join(axs[0, 0], axs[0, 2])
    axs[0, 2].set_xticklabels([])
    axs[0, 2].autoscale()
    axs[0, 2].set_title(r'$|{\operatorname{f}}(x)|$')
    axs[0, 2].plot(x, np.abs(y))

    axs[0, 3].get_shared_x_axes().join(axs[0, 0], axs[0, 3])
    axs[0, 3].set_xticklabels([])
    axs[0, 3].autoscale()
    axs[0, 3].set_title(r'$\operatorname{\varphi}[{\operatorname{f}}(x)]$')
    axs[0, 3].plot(x, np.angle(y))

    axs[1, 0].set_title(r'$\operatorname{Re}[\hat{\operatorname{f}}(\xi)]$')
    axs[1, 0].plot(x, np.real(ft_y))

    axs[1, 1].get_shared_x_axes().join(axs[1, 0], axs[1, 1])
    axs[1, 1].set_xticklabels([])
    axs[1, 1].get_shared_y_axes().join(axs[1, 0], axs[1, 1])
    axs[1, 1].set_yticklabels([])
    axs[1, 1].autoscale()
    axs[1, 1].set_title(r'$\operatorname{Im}[\hat{\operatorname{f}}(\xi)]$')
    axs[1, 1].plot(x, np.imag(ft_y))

    axs[1, 2].get_shared_x_axes().join(axs[1, 0], axs[1, 2])
    axs[1, 2].set_xticklabels([])
    axs[1, 2].autoscale()
    axs[1, 2].set_title(r'$|\hat{\operatorname{f}}(\xi)|$')
    axs[1, 2].plot(x, np.abs(ft_y))

    axs[1, 3].get_shared_x_axes().join(axs[1, 0], axs[1, 3])
    axs[1, 3].set_xticklabels([])
    axs[1, 3].autoscale()
    axs[1, 3].set_title(
        r'$\operatorname{\varphi}[\hat{\operatorname{f}}(\xi)]$')
    axs[1, 3].plot(x, np.angle(ft_y))


# ======================================================================
# : Fourier 1D
TITLE['fourier_2d'] = 'Fourier Transform in 2D'
PARAMS['fourier_2d'] = collections.OrderedDict(
    [('re_f_x', dict(label='Re[f(x)]', default='a*sin(b*x+c)+d', values=())),
     ('im_f_x', dict(label='Im[f(x)]', default='0', values=())), ]
    + [('const_{}'.format(letter), dict(
        label=letter, default=1.0, start=-5, stop=5, step=0.01))
       for letter in string.ascii_lowercase[:20]]
    + [('x_start', dict(
        label='x_start', default=-10, start=-20, stop=20, step=1)),
       ('x_stop', dict(
           label='x_stop', default=10, start=-20, stop=20, step=1)),
       ('x_num', dict(
           label='x_num / #', default=256, start=16, stop=4096, step=16)), ])


# ======================================================================
def playground_fourier_2d(
        fig,
        params=None,
        title=TITLE_BASE):
    axs = fig.subplots(2, 4, squeeze=False)  # , sharex='row', sharey='row')
    x = np.linspace(params['x_start'], params['x_stop'], params['x_num'])

    for k, v in params.items():
        if k.startswith('const_'):
            locals()[k[len('const_'):]] = v
    try:
        re_y = eval(params['re_f_x'])
    except (SyntaxError, NameError, TypeError):
        re_y = 0

    try:
        im_y = eval(params['im_f_x'])
    except (SyntaxError, NameError, TypeError):
        im_y = 0

    y = re_y + 1j * im_y
    ft_y = np.fft.fftshift(np.fft.fftn(y))

    # axs[0, 0].set_axis_off()

    axs[0, 0].set_title(r'$\operatorname{Re}[{\operatorname{f}}(x)]$')
    axs[0, 0].plot(x, np.real(y))

    axs[0, 1].get_shared_x_axes().join(axs[0, 0], axs[0, 1])
    axs[0, 1].set_xticklabels([])
    axs[0, 1].get_shared_y_axes().join(axs[0, 0], axs[0, 1])
    axs[0, 1].set_yticklabels([])
    axs[0, 1].autoscale()
    axs[0, 1].set_title(r'$\operatorname{Im}[{\operatorname{f}}(x)]$')
    axs[0, 1].plot(x, np.imag(y))

    axs[0, 2].get_shared_x_axes().join(axs[0, 0], axs[0, 2])
    axs[0, 2].set_xticklabels([])
    axs[0, 2].autoscale()
    axs[0, 2].set_title(r'$|{\operatorname{f}}(x)|$')
    axs[0, 2].plot(x, np.abs(y))

    axs[0, 3].get_shared_x_axes().join(axs[0, 0], axs[0, 3])
    axs[0, 3].set_xticklabels([])
    axs[0, 3].autoscale()
    axs[0, 3].set_title(r'$\operatorname{\varphi}[{\operatorname{f}}(x)]$')
    axs[0, 3].plot(x, np.angle(y))

    axs[1, 0].set_title(r'$\operatorname{Re}[\hat{\operatorname{f}}(\xi)]$')
    axs[1, 0].plot(x, np.real(ft_y))

    axs[1, 1].get_shared_x_axes().join(axs[1, 0], axs[1, 1])
    axs[1, 1].set_xticklabels([])
    axs[1, 1].get_shared_y_axes().join(axs[1, 0], axs[1, 1])
    axs[1, 1].set_yticklabels([])
    axs[1, 1].autoscale()
    axs[1, 1].set_title(r'$\operatorname{Im}[\hat{\operatorname{f}}(\xi)]$')
    axs[1, 1].plot(x, np.imag(ft_y))

    axs[1, 2].get_shared_x_axes().join(axs[1, 0], axs[1, 2])
    axs[1, 2].set_xticklabels([])
    axs[1, 2].autoscale()
    axs[1, 2].set_title(r'$|\hat{\operatorname{f}}(\xi)|$')
    axs[1, 2].plot(x, np.abs(ft_y))

    axs[1, 3].get_shared_x_axes().join(axs[1, 0], axs[1, 3])
    axs[1, 3].set_xticklabels([])
    axs[1, 3].autoscale()
    axs[1, 3].set_title(
        r'$\operatorname{\varphi}[\hat{\operatorname{f}}(\xi)]$')
    axs[1, 3].plot(x, np.angle(ft_y))


# ======================================================================
AVAILABLES = [
    k[len(PREFIX):] for k in globals().keys() if k.startswith(PREFIX)]


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
        '-n', '--name', metavar='NAME',
        default=None, choices=AVAILABLES,
        help='playground selection (from: %(choices)s)  [%(default)s]')
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

    if not args.name:
        msg('Choose your playground:')
        msg('Available playgrounds: {AVAILABLES}'.format_map(globals()))
        args.name = input(': ')
    if args.name in AVAILABLES:
        numex.interactive_tk_mpl.plotting(
            globals()[PREFIX + args.name],
            PARAMS[args.name], title=TITLE_BASE + ' - ' + TITLE[args.name],
            about=__doc__)
        elapsed(__file__[len(PATH['base']) + 1:])
        msg(report())
    else:
        msg(fmtm('Plot `{args.name}` not valid.'))


# ======================================================================
if __name__ == '__main__':
    main()
