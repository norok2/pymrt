#!python
# -*- coding: utf-8 -*-
"""
PyMRT - Interactive: MP2RAGE B1+ -- sensitivity to T1.

Simulate the MP2RAGE signal rho as a function of B1+.

Calculate the MP2RAGE signal rho as a function of B1+, according to given
parameters interactively adjusted.

Two different set of parameters are accepted:
- the sequence parameters, directly related to the pulse sequence diagram.
- the acquisition parameters, directly related to the acquisition protocol.

References:
1) Marques, J.P., Kober, T., Krueger, G., van der Zwaag, W., Van de Moortele, 
   P.-F., Gruetter, R., 2010. MP2RAGE, a self bias-field corrected sequence 
   for improved segmentation and T1-mapping at high field. NeuroImage 49,
   1271–1281. doi:10.1016/j.neuroimage.2009.10.002
2) Metere, R., Kober, T., Möller, H.E., Schäfer, A., 2017. Simultaneous 
   Quantitative MRI Mapping of T1, T2* and Magnetic Susceptibility with 
   Multi-Echo MP2RAGE. PLOS ONE 12, e0169265. doi:10.1371/journal.pone.0169265
3) Eggenschwiler, F., Kober, T., Magill, A.W., Gruetter, R., Marques, J.P.,
   2012. SA2RAGE: A new sequence for fast B1+-mapping. Magnetic Resonance
   Medicine 67, 1609–1619. doi:10.1002/mrm.23145

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
import datetime  # Basic date and time types
import traceback  # Print or retrieve a stack traceback

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

from pymrt import INFO, DIRS
from pymrt import VERB_LVL, D_VERB_LVL
from pymrt import msg, dbg
from pymrt import elapsed, report

TITLE = __doc__.strip().split('\n')[0][:-1]
SEQ_INTERACTIVES = collections.OrderedDict([
    ('mode', dict(
        label='ρ expression', default='p-ratio',
        values=('ratio', 'p-ratio', 'i-ratio'))),

    ('n_gre', dict(
        label='N_GRE / #', default=64, start=1, stop=512, step=1)),

    ('tr_gre', dict(
        label='TR_GRE / ms', default=6.0, start=1, stop=128, step=0.1)),

    ('td0', dict(
        label='T_D0 / ms', default=0, start=0, stop=10000, step=10)),
    ('td1', dict(
        label='T_D1 / ms', default=0, start=0, stop=10000, step=10)),
    ('td2', dict(
        label='T_D2 / ms', default=0, start=0, stop=10000, step=10)),

    ('fa1', dict(
        label='α_1 / deg', default=4.0, start=0.05, stop=22.0, step=0.05)),
    ('fa2', dict(
        label='α_2 / deg', default=20.0, start=0.05, stop=22.0, step=0.05)),

    ('eta_p', dict(
        label='η_p / #', default=0.0, start=0, stop=1, step=0.01)),
    ('fa_p', dict(
        label='α_p / #', default=0, start=-180, stop=180, step=5)),

    ('t1_start', dict(
        label='T1_start / ms', default=500, start=0, stop=1000, step=10)),
    ('t1_stop', dict(
        label='T1_stop / ms', default=4500, start=2000, stop=6000, step=100)),
    ('t1_num', dict(
        label='T1_num / ms', default=9, start=2, stop=32, step=1)),

    ('eta_fa_start', dict(
        label='η_α_start / ms', default=0.05, start=0.0, stop=2, step=0.05)),
    ('eta_fa_stop', dict(
        label='η_α_stop / ms', default=1.8, start=0.0, stop=2, step=0.05)),
    ('eta_fa_num', dict(
        label='η_α_num / #', default=256, start=2, stop=1024, step=16)),
])

ACQ_INTERACTIVES = collections.OrderedDict([
    ('mode', dict(
        label='ρ expression', default='p-ratio',
        values=('ratio', 'p-ratio', 'i-ratio'))),
    
    ('matrix_size_ro', dict(
        label='N_ro / #', default=64, start=1, stop=1024, step=1)),
    ('matrix_size_pe', dict(
        label='N_pe / #', default=64, start=1, stop=1024, step=1)),
    ('matrix_size_sl', dict(
        label='N_sl / #', default=64, start=1, stop=1024, step=1)),

    ('grappa_factor_ro', dict(
        label='GRAPPA_ro / #', default=1, start=1, stop=8, step=1)),
    ('grappa_factor_pe', dict(
        label='GRAPPA_pe / #', default=1, start=1, stop=8, step=1)),
    ('grappa_factor_sl', dict(
        label='GRAPPA_sl / #', default=1, start=1, stop=8, step=1)),

    ('grappa_ref_ro', dict(
        label='GRAPPA_ref_ro / #', default=0, start=0, stop=256, step=1)),
    ('grappa_ref_pe', dict(
        label='GRAPPA_ref_pe / #', default=0, start=0, stop=256, step=1)),
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
        label='TR_seq / ms', default=610, start=0, stop=10000, step=10)),
    ('ti1', dict(
        label='TI_1 / ms', default=80, start=0, stop=10000, step=10)),
    ('ti2', dict(
        label='TI_2 / ms', default=380, start=0, stop=10000, step=10)),

    ('tr_gre', dict(
        label='TR_GRE / ms', default=6, start=1, stop=128, step=0.1)),

    ('sl_pe_swap', dict(
        label='Swap PE/SL', default=False)),

    ('fa1', dict(
        label='α_1 / deg', default=4.0, start=0.05, stop=22.0, step=0.05)),
    ('fa2', dict(
        label='α_2 / deg', default=20.0, start=0.05, stop=22.0, step=0.05)),

    ('eta_p', dict(
        label='η_p / #', default=0.0, start=0, stop=1, step=0.01)),
    ('fa_p', dict(
        label='α_p / #', default=0, start=-180, stop=180, step=5)),

    ('t1_start', dict(
        label='T1_start / ms', default=500, start=0, stop=1000, step=10)),
    ('t1_stop', dict(
        label='T1_stop / ms', default=4500, start=2000, stop=6000, step=100)),
    ('t1_num', dict(
        label='T1_num / ms', default=9, start=2, stop=32, step=1)),

    ('eta_fa_start', dict(
        label='η_α_start / ms', default=0.05, start=0.0, stop=2, step=0.05)),
    ('eta_fa_stop', dict(
        label='η_α_stop / ms', default=1.8, start=0.0, stop=2, step=0.05)),
    ('eta_fa_num', dict(
        label='η_α_num / #', default=256, start=2, stop=1024, step=16)),
])


# ======================================================================
def plot_rho_b1t_mp2rage_seq(
        fig,
        params=None,
        title=TITLE.split(':')[1].strip()):
    ax = fig.gca()
    try:
        eta_fa_arr = np.linspace(
            params['eta_fa_start'], params['eta_fa_stop'],
            params['eta_fa_num'])
        t1_arr = np.linspace(
            params['t1_start'], params['t1_stop'], params['t1_num'])
        kws_names = (
            'mode', 'n_gre', 'tr_gre', 'td0', 'td1', 'td2', 'fa1', 'fa2',
            'eta_p', 'fa_p')
        seq_kws = {name: params[name] for name in kws_names}
        seq_kws['eta_fa'] = eta_fa_arr
        if seq_kws['eta_p'] == 0:
            seq_kws['eta_p'] = None
        for t1 in t1_arr:
            seq_kws['t1'] = t1
            rho_arr = mp2rage.rho(**seq_kws)
            ax.plot(rho_arr, eta_fa_arr, label='T1={:.1f} ms'.format(t1))
    except Exception as e:
        print(traceback.format_exc())
        ax.set_title('\n'.join(('WARNING! Some plot failed!', title)))
    else:
        ax.set_title(title)
    finally:
        ax.set_ylim(params['eta_fa_start'], params['eta_fa_stop'])
        ax.set_ylabel(r'$\eta_\alpha$ / #')
        if params['mode'] == 'p-ratio':
            ax.set_xlim(mp2rage.PSEUDO_RATIO_INTERVAL)
        if params['mode'] == 'p-ratio':
            expression = r'\frac{T_{I,1}T_{I,2}}{T_{I,1}^2+T_{I,2}^2}'
        elif params['mode'] == 'ratio':
            expression = r'\frac{T_{I,1}}{T_{I,2}}'
        elif params['mode'] == 'i-ratio':
            expression = r'\frac{T_{I,2}}{T_{I,1}}'
        else:
            expression = None
        ax.set_xlabel(r'$\rho={}$ / arb.units'.format(expression))
        ax.legend()
    return ax


# ======================================================================
def plot_rho_b1t_mp2rage_acq(
        fig,
        params=None,
        title=TITLE.split(':')[1].strip()):
    ax = fig.gca()
    try:
        if params['use_rho']:
            mp2rage_rho = mp2rage.rho
        else:
            mp2rage_rho = mp2rage.ratio
        eta_fa_arr = np.linspace(
            params['eta_fa_start'], params['eta_fa_stop'],
            params['eta_fa_num'])
        t1_arr = np.linspace(
            params['t1_start'], params['t1_stop'], params['t1_num'])
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
            tr_gre=params['tr_gre'])

        seq_kws_str = ', '.join(
            sorted(['{}={:.2f}'.format(k, v) for k, v in seq_kws.items()]))
        extra_info_str = ', '.join(
            sorted(['{}={:.2f}'.format(k, v) for k, v in extra_info.items()]))
        extra_info_str += ', {!s}'.format(
            datetime.timedelta(seconds=extra_info['t_acq']))
        acq_to_seq_info = '\n'.join((seq_kws_str, extra_info_str))

        seq_kws['eta_fa'] = eta_fa_arr
        kws_names = ('mode', 'fa1', 'fa2', 'eta_p', 'fa_p')
        seq_kws.update({name: params[name] for name in kws_names})
        if seq_kws['eta_p'] == 0:
            seq_kws['eta_p'] = None
        for t1 in t1_arr:
            seq_kws['t1'] = t1
            rho_arr = mp2rage.rho(**seq_kws)
            ax.plot(rho_arr, eta_fa_arr, label='T1={:.1f} ms'.format(t1))
    except Exception as e:
        print(traceback.format_exc())
        ax.set_title('\n'.join(('WARNING! Some plot failed!', title)))
    else:
        ax.set_title('\n'.join((acq_to_seq_info, title)))
    finally:
        ax.set_ylim(params['eta_fa_start'], params['eta_fa_stop'])
        ax.set_ylabel(r'$\eta_\alpha$ / #')
        if params['mode'] == 'p-ratio':
            ax.set_xlim(mp2rage.PSEUDO_RATIO_INTERVAL)
        if params['mode'] == 'p-ratio':
            expression = r'\frac{T_{I,1}T_{I,2}}{T_{I,1}^2+T_{I,2}^2}'
        elif params['mode'] == 'ratio':
            expression = r'\frac{T_{I,1}}{T_{I,2}}'
        elif params['mode'] == 'i-ratio':
            expression = r'\frac{T_{I,2}}{T_{I,1}}'
        else:
            expression = None
        ax.set_xlabel(r'$\rho={}$ / arb.units'.format(expression))
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
        '-i', '--indirect',
        action='store_true',
        help='use acquisition params instead of sequence params [%(default)s]')
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

    if not args.indirect:
        interactive.mpl_plot(
            plot_rho_b1t_mp2rage_seq,
            SEQ_INTERACTIVES, title=TITLE, about=__doc__)
    else:
        interactive.mpl_plot(
            plot_rho_b1t_mp2rage_acq,
            ACQ_INTERACTIVES, title=TITLE, about=__doc__)

    elapsed(os.path.basename(__file__))
    msg(report())


# ======================================================================
if __name__ == '__main__':
    main()
