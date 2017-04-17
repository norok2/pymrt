#!python
# -*- coding: utf-8 -*-
"""
PyMRT - Interactive: MP2RAGE B1T sensitivity.

Simulate the MP2RAGE signal rho as a function of T1.

Calculate the MP2RAGE signal rho as a function of T1, according to given
parameters interactively adjusted.

Two different set of parameters are accepted:
- the sequence parameters, directly related to the pulse sequence diagram.
- the acquisition parameters, directly related to the acquisition protocol.

References:
1) Marques, J.P., Kober, T., Krueger, G., van der Zwaag, W., Van de Moortele, 
   P.-F., Gruetter, R., 2010. MP2RAGE, a self bias-field corrected sequence 
   for 
   improved segmentation and T1-mapping at high field. NeuroImage 49, 
   1271–1281. 
   doi:10.1016/j.neuroimage.2009.10.002
2) Metere, R., Kober, T., Möller, H.E., Schäfer, A., 2017. Simultaneous 
   Quantitative MRI Mapping of T1, T2* and Magnetic Susceptibility with 
   Multi-Echo MP2RAGE. PLOS ONE 12, e0169265. doi:10.1371/journal.pone.0169265
"""

# ======================================================================
# :: Future Imports
from __future__ import (
    division, absolute_import, print_function, unicode_literals)

# ======================================================================
# :: Python Standard Library Imports
import os  # Miscellaneous operating system interfaces
import argparse  # Argument Parsing
import collections  # Container datatypes

# :: External Imports
import numpy as np  # NumPy (multidimensional numerical arrays library)
import scipy as sp  # SciPy (signal and image processing library)
import matplotlib as mpl  # Matplotlib (2D/3D plotting library)
import matplotlib.pyplot as plt  # Matplotlib's pyplot: MATLAB-like syntax
import scipy.optimize
import pytk
import pytk.widgets

import matplotlib.backends.backend_tkagg as tkagg

# :: Local Imports
import pymrt as mrt
from pymrt.sequences import mp2rage
from pymrt.extras import interactive

from pymrt import INFO, DIRS, MY_GREETINGS
from pymrt import VERB_LVL, D_VERB_LVL
from pymrt import msg, dbg
from pymrt import elapsed, print_elapsed

TITLE = __doc__.strip().split('\n')[0][:-1]
SEQ_INTERACTIVES = collections.OrderedDict([
    ('n_gre', dict(
        label='N_GRE / #', default=64, start=1, stop=512, step=1)),

    ('tr_gre', dict(
        label='TR_GRE / ms', default=6, start=1, stop=128, step=0.1)),

    ('ta', dict(
        label='T_A / ms', default=440, start=0, stop=10000, step=10)),
    ('tb', dict(
        label='T_B / ms', default=1180, start=0, stop=10000, step=10)),
    ('tc', dict(
        label='T_C / ms', default=4140, start=0, stop=10000, step=10)),

    ('fa1', dict(
        label='α_1 / deg', default=4.0, start=0.05, stop=22.0, step=0.05)),
    ('fa2', dict(
        label='α_2 / deg', default=5.0, start=0.05, stop=22.0, step=0.05)),

    ('eta_inv', dict(
        label='η_inv / #', default=0.96, start=0, stop=1, step=0.01)),
    ('d_eta_fa', dict(
        label='Δη_α / #', default=0.1, start=0, stop=1, step=0.05)),

    ('t1_start', dict(
        label='T1_start / ms', default=50, start=0, stop=1000, step=10)),
    ('t1_stop', dict(
        label='T1_stop / ms', default=4100, start=2000, stop=6000, step=100)),
    ('t1_step', dict(
        label='T1_step / ms', default=256, start=32, stop=1024, step=16)),
])

ACQ_INTERACTIVES = collections.OrderedDict([
    ('matrix_size_ro', dict(
        label='N_ro / #', default=256, start=16, stop=1024, step=16)),
    ('matrix_size_pe', dict(
        label='N_pe / #', default=256, start=16, stop=1024, step=16)),
    ('matrix_size_sl', dict(
        label='N_sl / #', default=256, start=16, stop=1024, step=16)),

    ('grappa_factor_ro', dict(
        label='GRAPPA_ro / #', default=1, start=1, stop=8, step=1)),
    ('grappa_factor_pe', dict(
        label='GRAPPA_pe / #', default=2, start=1, stop=8, step=1)),
    ('grappa_factor_sl', dict(
        label='GRAPPA_sl / #', default=1, start=1, stop=8, step=1)),

    ('grappa_ref_ro', dict(
        label='GRAPPA_ref_ro / #', default=0, start=0, stop=256, step=1)),
    ('grappa_ref_pe', dict(
        label='GRAPPA_ref_pe / #', default=24, start=0, stop=256, step=1)),
    ('grappa_ref_sl', dict(
        label='GRAPPA_ref_sl / #', default=0, start=0, stop=256, step=1)),

    ('part_fourier_factor_ro', dict(
        label='part.Fourier_ro / #', default=8 / 8, start=4 / 8, stop=8 / 8,
        step=1 / 8)),
    ('part_fourier_factor_pe', dict(
        label='part.Fourier_pe / #', default=8 / 8, start=4 / 8, stop=8 / 8,
        step=1 / 8)),
    ('part_fourier_factor_sl', dict(
        label='part.Fourier_sl / #', default=8 / 8, start=4 / 8, stop=8 / 8,
        step=1 / 8)),

    ('tr_seq', dict(
        label='TR_seq / ms', default=8000, start=0, stop=10000, step=10)),
    ('ti1', dict(
        label='TI_1 / ms', default=900, start=0, stop=10000, step=10)),
    ('ti2', dict(
        label='TI_2 / ms', default=2900, start=0, stop=10000, step=10)),

    ('tr_gre', dict(
        label='TR_GRE / ms', default=6, start=1, stop=128, step=0.1)),

    ('sl_pe_swap', dict(
        label='Swap PE/SL', default=False)),

    ('a1', dict(
        label='α_1 / deg', default=4.0, start=0.05, stop=22.0, step=0.05)),
    ('a2', dict(
        label='α_2 / deg', default=5.0, start=0.05, stop=22.0, step=0.05)),

    ('eta_inv', dict(
        label='η_inv / #', default=0.96, start=0, stop=1, step=0.01)),
    ('d_eta_fa', dict(
        label='Δη_α / #', default=0.1, start=0, stop=1, step=0.05)),

    ('t1_start', dict(
        label='T1_start / ms', default=50, start=0, stop=1000, step=10)),
    ('t1_stop', dict(
        label='T1_stop / ms', default=4100, start=2000, stop=6000, step=100)),
    ('t1_step', dict(
        label='T1_step / ms', default=256, start=32, stop=1024, step=16)),
])


# ======================================================================
def plot_rho_t1_mp2rage_seq(
        fig,
        params=None,
        title=TITLE.split(':')[1].strip()):
    ax = fig.gca()
    t1_arr = np.linspace(
        params['t1_start'], params['t1_stop'], params['t1_step'])
    try:
        kws_names = 'n_gre', 'tr_gre', 'ta', 'tb', 'tc', 'fa1', 'fa2', \
                    'eta_inv'
        seq_kws = {name: params[name] for name in kws_names}

        seq_kws['eta_fa'] = 1
        rho_arr = mp2rage.rho(t1_arr, **seq_kws)
        ax.plot(rho_arr, t1_arr, color='g', label='MP2RAGE')

        seq_kws['eta_fa'] = 1 + params['d_eta_fa']
        rho_arr = mp2rage.rho(t1_arr, **seq_kws)
        ax.plot(
            rho_arr, t1_arr, color='r',
            label='$B_1^+$ +{:.0%}'.format(params['d_eta_fa']))

        seq_kws['eta_fa'] = 1 - params['d_eta_fa']
        rho_arr = mp2rage.rho(t1_arr, **seq_kws)
        ax.plot(
            rho_arr, t1_arr, color='b',
            label='$B_1^+$ -{:.0%}'.format(params['d_eta_fa']))

        seq_kws['eta_fa'] = 1 + 2 * params['d_eta_fa']
        rho_arr = mp2rage.rho(t1_arr, **seq_kws)
        ax.plot(
            rho_arr, t1_arr, color='y',
            label='$B_1^+$ +{:.0%}'.format(2 * params['d_eta_fa']))

        seq_kws['eta_fa'] = 1 - 2 * params['d_eta_fa']
        rho_arr = mp2rage.rho(t1_arr, **seq_kws)
        ax.plot(
            rho_arr, t1_arr, color='c',
            label='$B_1^+$ -{:.0%}'.format(2 * params['d_eta_fa']))
    except Exception as e:
        print(e)
        ax.set_title('\n'.join(('WARNING! Some plot failed!', title)))
    else:
        ax.set_title(title)
    finally:
        ax.set_xlabel(r'$\rho$ (a.u.)')
        ax.set_ylabel(r'$T_1$ (ms)')
        ax.legend()
    return ax


# ======================================================================
def plot_rho_t1_mp2rage_acq(
        fig,
        params=None,
        title=TITLE.split(':')[1].strip()):
    t1_arr = np.linspace(
        params['t1_start'], params['t1_stop'], params['t1_step'])
    ax = fig.gca()
    try:
        seq_kws, extra_info = mp2rage.acq_to_seq_params(
            matrix_sizes=(
                params['matrix_size_ro'],
                params['matrix_size_pe'],
                params['matrix_size_sl'],),
            grappa_factors=(
                params['grappa_factor_ro'],
                params['grappa_factor_pe'],
                params['grappa_factor_sl'],),
            grappa_refs=(
                params['grappa_ref_ro'],
                params['grappa_ref_pe'],
                params['grappa_ref_sl'],),
            part_fourier_factors=(
                params['part_fourier_factor_ro'],
                params['part_fourier_factor_pe'],
                params['part_fourier_factor_sl'],),
            sl_pe_swap=params['sl_pe_swap'],
            tr_seq=params['tr_seq'],
            ti=(params['ti1'], params['ti2']),
            fa=(params['a1'], params['a2']),
            eta_inv=params['eta_inv'],
            tr_gre=params['tr_gre'])

        seq_kws_str = ', '.join(
            sorted(['{}={}'.format(k, v) for k, v in seq_kws.items()]))
        extra_info_str = ', '.join(
            sorted(['{}={}'.format(k, v) for k, v in extra_info.items()]))
        acq_to_seq_info = '\n'.join((seq_kws_str, extra_info_str))

        seq_kws['eta_fa'] = 1
        rho_arr = mp2rage.rho(t1_arr, **seq_kws)
        ax.plot(rho_arr, t1_arr, color='g', label='MP2RAGE')

        seq_kws['eta_fa'] = 1 + params['d_eta_fa']
        rho_arr = mp2rage.rho(t1_arr, **seq_kws)
        ax.plot(
            rho_arr, t1_arr, color='r',
            label='$B_1^+$ +{:.0%}'.format(params['d_eta_fa']))

        seq_kws['eta_fa'] = 1 - params['d_eta_fa']
        rho_arr = mp2rage.rho(t1_arr, **seq_kws)
        ax.plot(
            rho_arr, t1_arr, color='b',
            label='$B_1^+$ -{:.0%}'.format(params['d_eta_fa']))

        seq_kws['eta_fa'] = 1 + 2 * params['d_eta_fa']
        rho_arr = mp2rage.rho(t1_arr, **seq_kws)
        ax.plot(
            rho_arr, t1_arr, color='y',
            label='$B_1^+$ +{:.0%}'.format(2 * params['d_eta_fa']))

        seq_kws['eta_fa'] = 1 - 2 * params['d_eta_fa']
        rho_arr = mp2rage.rho(t1_arr, **seq_kws)
        ax.plot(
            rho_arr, t1_arr, color='c',
            label='$B_1^+$ -{:.0%}'.format(2 * params['d_eta_fa']))
    except:
        ax.set_title('\n'.join(('WARNING! Some plot failed!', title)))
    else:
        ax.set_title('\n'.join((acq_to_seq_info, title)))
    finally:
        ax.set_xlabel(r'$\rho$ (a.u.)')
        ax.set_ylabel(r'$T_1$ (ms)')
        ax.legend()
    return ax


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
    arg_parser.add_argument(
        '-d', '--direct',
        action='store_true',
        help='use sequence params instead of acquisition params [%(default)s]')
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

    if args.direct:
        interactive.mpl_plot(
            plot_rho_t1_mp2rage_seq,
            SEQ_INTERACTIVES, title=TITLE, about=__doc__)
    else:
        interactive.mpl_plot(
            plot_rho_t1_mp2rage_acq,
            ACQ_INTERACTIVES, title=TITLE, about=__doc__)

    elapsed(os.path.basename(__file__))
    print_elapsed()


# ======================================================================
if __name__ == '__main__':
    main()
