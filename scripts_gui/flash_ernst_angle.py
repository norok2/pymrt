#!python
# -*- coding: utf-8 -*-
"""
PyMRT - Interactive: FLASH Ernst angle.

Calculate the optimal signal conditions for FLASH sequences.

The optimal signal condition, also known as Ernst angle, defines the
maximum signal obtained from a steady-state gradient-echo (FLASH)
sequence where the repetition time TR and longitudinal relaxation time T1
are known.
"""

# ======================================================================
# :: Future Imports
from __future__ import (
    division, absolute_import, print_function, unicode_literals, )

# ======================================================================
# :: Python Standard Library Imports
import os  # Miscellaneous operating system interfaces
# import math  # Mathematical Functions
import collections  # Container datatypes
import argparse  # Argument Parsing

# :: External Imports
import numpy as np  # NumPy (multidimensional numerical arrays library)
from mpl_toolkits.mplot3d import Axes3D  # required by `projection='3d'`
import numex.interactive_tk_mpl


# :: Local Imports
import pymrt as mrt
from pymrt import INFO, DIRS
from pymrt import VERB_LVL, D_VERB_LVL
from pymrt import msg, dbg
from pymrt import elapsed, report

from pymrt.sequences import flash

# ======================================================================
TITLE = __doc__.strip().split('\n')[0][:-1]
INTERACTIVES = collections.OrderedDict([
    ('t1_start', dict(
        label='T1_start / ms', default=500, start=5, stop=9995, step=5)),
    ('t1_stop', dict(
        label='T1_stop / ms', default=3500, start=5, stop=9995, step=5)),
    ('t1_num', dict(
        label='T1_num / #', default=64, start=16, stop=256, step=16)),

    ('tr_start', dict(
        label='TR_start / ms', default=1, start=5, stop=9995, step=5)),
    ('tr_stop', dict(
        label='TR_stop / ms', default=100, start=5, stop=9995, step=5)),
    ('tr_num', dict(
        label='TR_num / #', default=64, start=16, stop=256, step=16)),

    ('fa_start', dict(
        label='α_start / deg', default=0.1, start=0.1, stop=90, step=0.1)),
    ('fa_stop', dict(
        label='α_stop / deg', default=90, start=0.1, stop=90, step=0.1)),
    ('fa_num', dict(
        label='α_num / #', default=64, start=16, stop=256, step=16)),
])


# ======================================================================
def plot_flash_ernst_angle_t1_tr(
        fig,
        params=None,
        title=TITLE.split(':')[1].strip()):
    ax = fig.gca(projection='3d')

    try:
        t1_arr = np.linspace(
            params['t1_start'], params['t1_stop'], params['t1_num'])
        tr_arr = np.linspace(
            params['tr_start'], params['tr_stop'], params['tr_num'])

        t1_grid, tr_grid = np.meshgrid(t1_arr, tr_arr)
        fa_grid, name, units = flash.ernst_calc(t1=t1_grid, tr=tr_grid)
        ax.plot_surface(
            t1_grid, tr_grid, fa_grid, alpha=0.5)
        ax.contourf(
            t1_grid, tr_grid, fa_grid, zdir='z', alpha=0.5,
            offset=np.min(fa_grid))
        ax.contourf(
            t1_grid, tr_grid, fa_grid, zdir='x', alpha=0.5,
            offset=params['t1_start'])
        ax.contourf(
            t1_grid, tr_grid, fa_grid, zdir='y', alpha=0.5,
            offset=params['tr_stop'])
    except Exception as e:
        print(e)
        ax.set_title('\n'.join(('WARNING! Some plot failed!', title)))
    else:
        ax.set_title(title)
    finally:
        ax.set_xlabel(r'$T_1$ (ms)')
        ax.set_ylabel(r'$T_R$ (ms)')
        ax.set_zlabel(r'$\alpha$ (deg)')
        # ax.legend()


# ======================================================================
def plot_flash_ernst_angle_fa_t1(
        fig,
        params=None,
        title=TITLE.split(':')[1].strip()):
    ax = fig.gca(projection='3d')

    try:
        fa_arr = np.linspace(
            params['fa_start'], params['fa_stop'], params['fa_num'])
        t1_arr = np.linspace(
            params['t1_start'], params['t1_stop'], params['t1_num'])

        fa_grid, t1_grid = np.meshgrid(fa_arr, t1_arr)
        tr_grid, name, units = flash.ernst_calc(t1=t1_grid, fa=fa_grid)
        ax.plot_surface(
            fa_grid, t1_grid, tr_grid, alpha=0.5)
        ax.contourf(
            fa_grid, t1_grid, tr_grid, zdir='z', alpha=0.5,
            offset=np.min(tr_grid))
        ax.contourf(
            fa_grid, t1_grid, tr_grid, zdir='x', alpha=0.5,
            offset=params['fa_start'])
        ax.contourf(
            fa_grid, t1_grid, tr_grid, zdir='y', alpha=0.5,
            offset=params['t1_stop'])
    except Exception as e:
        print(e)
        ax.set_title('\n'.join(('WARNING! Some plot failed!', title)))
    else:
        ax.set_title(title)
    finally:
        ax.set_xlabel(r'$T_1$ (ms)')
        ax.set_ylabel(r'$\alpha$ (deg)')
        ax.set_zlabel(r'$T_R$ (ms)')
        # ax.legend()


# ======================================================================
def plot_flash_ernst_angle_fa_tr(
        fig,
        params=None,
        title=TITLE.split(':')[1].strip()):
    ax = fig.gca(projection='3d')

    try:
        fa_arr = np.linspace(
            params['fa_start'], params['fa_stop'], params['fa_num'])
        tr_arr = np.linspace(
            params['tr_start'], params['tr_stop'], params['tr_num'])

        fa_grid, tr_grid = np.meshgrid(fa_arr, tr_arr)
        t1_grid, name, units = flash.ernst_calc(tr=tr_grid, fa=fa_grid)
        ax.plot_surface(
            fa_grid, tr_grid, t1_grid, alpha=0.5)
        ax.contourf(
            fa_grid, tr_grid, t1_grid, zdir='z', alpha=0.5,
            offset=np.min(t1_grid))
        ax.contourf(
            fa_grid, tr_grid, t1_grid, zdir='x', alpha=0.5,
            offset=params['fa_start'])
        ax.contourf(
            fa_grid, tr_grid, t1_grid, zdir='y', alpha=0.5,
            offset=params['t1_stop'])
    except Exception as e:
        print(e)
        ax.set_title('\n'.join(('WARNING! Some plot failed!', title)))
    else:
        ax.set_title(title)
    finally:
        ax.set_xlabel(r'$T_1$ (ms)')
        ax.set_ylabel(r'$\alpha$ (deg)')
        ax.set_zlabel(r'$T_R$ (ms)')
        # ax.legend()


# ======================================================================
def handle_arg():
    """
    Handle command-line application arguments.
    """
    # :: Create Argument Parser
    arg_parser = argparse.ArgumentParser(
        description=__doc__,
        epilog='v.{} - {}\n{}'.format(
            INFO['version'], INFO['author'], INFO['license']),
        formatter_class=argparse.RawDescriptionHelpFormatter)
    # :: Add POSIX standard arguments
    arg_parser.add_argument(
        '--ver', '--version',
        version='%(prog)s - ver. {}\n{}\n{} {}\n{}'.format(
            INFO['version'],
            next(line for line in __doc__.splitlines() if line),
            INFO['copyright'], INFO['author'], INFO['notice']),
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
    # nothing here yet!
    # avoid mandatory arguments whenever possible
    arg_parser.add_argument(
        '-m', '--mode',
        nargs=2, default=('t1', 'tr'),
        help='set display/calculation mode [%(default)s]')
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

    x_vars = set([x.lower() for x in args.mode])

    filtered_interactives = INTERACTIVES.copy()
    for k in list(filtered_interactives.keys()):
        if k[:2] not in x_vars:
            filtered_interactives.pop(k)

    if x_vars == {'t1', 'tr'}:
        numex.interactive_tk_mpl.plotting(
            plot_flash_ernst_angle_t1_tr,
            filtered_interactives, resources_path=DIRS['resources'],
            title=TITLE, about=__doc__)
    elif x_vars == {'fa', 't1'}:
        numex.interactive_tk_mpl.plotting(
            plot_flash_ernst_angle_fa_t1,
            filtered_interactives, resources_path=DIRS['resources'],
            title=TITLE, about=__doc__)
    elif x_vars == {'fa', 't1'}:
        numex.interactive_tk_mpl.plotting(
            filtered_interactives, resources_path=DIRS['resources'],
            title=TITLE, about=__doc__)

    elapsed(__file__[len(DIRS['base']) + 1:])
    msg(report())


# ======================================================================
if __name__ == '__main__':
    main()
