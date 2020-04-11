#!python
# -*- coding: utf-8 -*-
"""
PyMRT - Interactive: MP2RAGE T1 -- sensitivity to B1+.

Simulate the MP2RAGE signal rho as a function of T1.

Calculate the MP2RAGE signal rho as a function of T1, according to given
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
    division, absolute_import, print_function, unicode_literals, )

# ======================================================================
# :: Python Standard Library Imports
import os  # Miscellaneous operating system interfaces
import argparse  # Argument Parsing
import collections  # Container datatypes
import datetime  # Basic date and time types

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

TITLE = __doc__.strip().split('\n')[0][:-1]
SEQ_INTERACTIVES = collections.OrderedDict([
    ('mode', dict(
        label='ρ expression', default='p-ratio',
        values=('ratio', 'p-ratio', 'i-ratio'))),

    ('n_gre', dict(
        label='N_GRE / #', default=64, start=1, stop=512, step=1)),

    ('tr_gre', dict(
        label='TR_GRE / ms', default=6.0, start=1, stop=128, step=0.1)),
    ('k_gre', dict(
        label='k_GRE / one units',
        default=0.5, start=0.0, stop=1.0, step=0.05)),

    ('td0', dict(
        label='T_D0 / ms', default=0, start=0, stop=10000, step=10)),
    ('td1', dict(
        label='T_D1 / ms', default=0, start=0, stop=10000, step=10)),
    ('td2', dict(
        label='T_D2 / ms', default=0, start=0, stop=10000, step=10)),

    ('fa1', dict(
        label='α_1 / deg', default=4.0, start=0.05, stop=22.0, step=0.05)),
    ('fa2', dict(
        label='α_2 / deg', default=5.0, start=0.05, stop=22.0, step=0.05)),

    ('eta_p', dict(
        label='η_p / one units', default=0.96, start=0, stop=1, step=0.01)),
    ('fa_p', dict(
        label='α_p / deg', default=180, start=-180, stop=180, step=5)),
    ('eta_fa', dict(
        label='η_α / one units', default=1.0, start=0, stop=1, step=0.01)),
    ('d_eta_fa', dict(
        label='Δη_α / one units', default=0.1, start=0, stop=1, step=0.05)),

    ('t1_start', dict(
        label='T1_start / ms', default=50, start=1, stop=1000, step=10)),
    ('t1_stop', dict(
        label='T1_stop / ms', default=4100, start=1, stop=6000, step=100)),
    ('t1_num', dict(
        label='T1_num / #', default=256, start=1, stop=1024, step=16)),
])

ACQ_INTERACTIVES = collections.OrderedDict([
    ('mode', dict(
        label='ρ expression', default='p-ratio',
        values=('ratio', 'p-ratio', 'i-ratio'))),

    ('matrix_size_ro', dict(
        label='N_ro / #', default=256, start=1, stop=1024, step=1)),
    ('matrix_size_pe', dict(
        label='N_pe / #', default=256, start=1, stop=1024, step=1)),
    ('matrix_size_sl', dict(
        label='N_sl / #', default=256, start=1, stop=1024, step=1)),

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
        label='part.Fourier_ro / (#/8)', default=8, start=4, stop=8, step=1)),
    ('part_fourier_factor_pe', dict(
        label='part.Fourier_pe / (#/8)', default=8, start=4, stop=8, step=1)),
    ('part_fourier_factor_sl', dict(
        label='part.Fourier_sl / (#/8)', default=8, start=4, stop=8, step=1)),

    ('k_lines_fix_ro', dict(
        label='k-space Fix Factor RO / one units',
        default=1.0, start=0.5, stop=1.0, step=0.01)),
    ('k_lines_fix_pe', dict(
        label='k-space Fix Factor PE / one units',
        default=1.0, start=0.5, stop=1.0, step=0.01)),
    ('k_lines_fix_sl', dict(
        label='k-space Fix Factor SL / one units',
        default=1.0, start=0.5, stop=1.0, step=0.01)),

    ('tr_seq', dict(
        label='TR_seq / ms', default=8000, start=0, stop=10000, step=10)),
    ('ti1', dict(
        label='TI_1 / ms', default=900, start=0, stop=10000, step=10)),
    ('ti2', dict(
        label='TI_2 / ms', default=2900, start=0, stop=10000, step=10)),

    ('tr_gre', dict(
        label='TR_GRE / ms', default=6.0, start=1, stop=128, step=0.1)),
    ('k_gre', dict(
        label='k_GRE / one units',
        default=0.5, start=0.0, stop=1.0, step=0.05)),

    ('sl_pe_swap', dict(
        label='Swap PE/SL', default=False)),

    ('fa1', dict(
        label='α_1 / deg', default=4.0, start=0.05, stop=22.0, step=0.05)),
    ('fa2', dict(
        label='α_2 / deg', default=5.0, start=0.05, stop=22.0, step=0.05)),

    ('eta_p', dict(
        label='η_p / one units', default=0.96, start=0, stop=1, step=0.01)),
    ('fa_p', dict(
        label='α_p / deg', default=180, start=-180, stop=180, step=5)),
    ('eta_fa', dict(
        label='η_α / one units', default=1.0, start=0, stop=1, step=0.01)),
    ('d_eta_fa', dict(
        label='Δη_α / one units', default=0.1, start=0, stop=1, step=0.05)),

    ('t1_start', dict(
        label='T1_start / ms', default=50, start=0, stop=1000, step=10)),
    ('t1_stop', dict(
        label='T1_stop / ms', default=4100, start=2000, stop=6000, step=100)),
    ('t1_num', dict(
        label='T1_num / ms', default=256, start=32, stop=1024, step=16)),
])


# ======================================================================
def plot_rho_t1_mp2rage_seq(
        fig,
        params=None,
        title=TITLE.split(':')[1].strip()):
    ax = fig.gca()
    t1_arr = np.linspace(
        params['t1_start'], params['t1_stop'], params['t1_num'])
    try:
        kws_names = (
            'mode', 'n_gre', 'tr_gre', 'k_gre',
            'td0', 'td1', 'td2', 'fa1', 'fa2',
            'eta_p', 'fa_p')
        seq_kws = {name: params[name] for name in kws_names}
        seq_kws['t1'] = t1_arr
        if seq_kws['eta_p'] == 0:
            seq_kws['eta_p'] = None

        seq_kws['eta_fa'] = params['eta_fa']
        rho_arr = mp2rage.rho(**seq_kws)
        ax.plot(rho_arr, t1_arr, color='g', label='MP2RAGE')

        seq_kws['eta_fa'] = params['eta_fa'] * (1 + params['d_eta_fa'])
        rho_arr = mp2rage.rho(**seq_kws)
        ax.plot(
            rho_arr, t1_arr, color='#cc3333',
            label='$B_1^+$ +{:.0%}'.format(params['d_eta_fa']))

        seq_kws['eta_fa'] = params['eta_fa'] * (1 - params['d_eta_fa'])
        rho_arr = mp2rage.rho(**seq_kws)
        ax.plot(
            rho_arr, t1_arr, color='#3333cc',
            label='$B_1^+$ −{:.0%}'.format(params['d_eta_fa']))

        seq_kws['eta_fa'] = params['eta_fa'] * (1 + 2 * params['d_eta_fa'])
        rho_arr = mp2rage.rho(**seq_kws)
        ax.plot(
            rho_arr, t1_arr, color='#ff9999',
            label='$B_1^+$ +{:.0%}'.format(2 * params['d_eta_fa']))

        seq_kws['eta_fa'] = params['eta_fa'] * (1 - 2 * params['d_eta_fa'])
        rho_arr = mp2rage.rho(**seq_kws)
        ax.plot(
            rho_arr, t1_arr, color='#9999ff',
            label='$B_1^+$ −{:.0%}'.format(2 * params['d_eta_fa']))
    except Exception as e:
        print(e)
        ax.set_title('\n'.join(('WARNING! Some plot failed!', title)))
    else:
        ax.set_title(title)
    finally:
        ax.set_ylim(params['t1_start'], params['t1_stop'])
        ax.set_ylabel(r'$T_1$ / ms')
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
        ax.set_xlabel(r'$\rho={}$ / arb. units'.format(expression))
        ax.legend()


# ======================================================================
def plot_rho_t1_mp2rage_acq(
        fig,
        params=None,
        title=TITLE.split(':')[1].strip()):
    t1_arr = np.linspace(
        params['t1_start'], params['t1_stop'], params['t1_num'])
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
                params['part_fourier_factor_ro'] / 8,
                params['part_fourier_factor_pe'] / 8,
                params['part_fourier_factor_sl'] / 8,),
            sl_pe_swap=params['sl_pe_swap'],
            k_lines_fix=(
                params['k_lines_fix_ro'],
                params['k_lines_fix_pe'],
                params['k_lines_fix_sl'],),
            tr_seq=params['tr_seq'],
            ti=(params['ti1'], params['ti2']),
            tr_gre=params['tr_gre'],
            k_gre=params['k_gre'])

        seq_kws_str = ', '.join(
            sorted(['{}={:.2f}'.format(k, v) for k, v in seq_kws.items()]))
        extra_info_str = ', '.join(
            sorted(['{}={:.2f}'.format(k, v) for k, v in extra_info.items()]))
        extra_info_str += ', {!s}'.format(
            datetime.timedelta(seconds=extra_info['t_acq']))
        acq_to_seq_info = '\n'.join((seq_kws_str, extra_info_str))

        kws_names = ('mode', 'fa1', 'fa2', 'eta_p', 'fa_p')
        seq_kws.update({name: params[name] for name in kws_names})
        seq_kws['t1'] = t1_arr
        if seq_kws['eta_p'] == 0:
            seq_kws['eta_p'] = None

        seq_kws['eta_fa'] = params['eta_fa']
        rho_arr = mp2rage.rho(**seq_kws)
        ax.plot(rho_arr, t1_arr, color='g', label='MP2RAGE')

        seq_kws['eta_fa'] = params['eta_fa'] * (1 + params['d_eta_fa'])
        rho_arr = mp2rage.rho(**seq_kws)
        ax.plot(
            rho_arr, t1_arr, color='#cc3333',
            label='$B_1^+$ +{:.0%}'.format(params['d_eta_fa']))

        seq_kws['eta_fa'] = params['eta_fa'] * (1 - params['d_eta_fa'])
        rho_arr = mp2rage.rho(**seq_kws)
        ax.plot(
            rho_arr, t1_arr, color='#3333cc',
            label='$B_1^+$ −{:.0%}'.format(params['d_eta_fa']))

        seq_kws['eta_fa'] = params['eta_fa'] * (1 + 2 * params['d_eta_fa'])
        rho_arr = mp2rage.rho(**seq_kws)
        ax.plot(
            rho_arr, t1_arr, color='#ff9999',
            label='$B_1^+$ +{:.0%}'.format(2 * params['d_eta_fa']))

        seq_kws['eta_fa'] = params['eta_fa'] * (1 - 2 * params['d_eta_fa'])
        rho_arr = mp2rage.rho(**seq_kws)
        ax.plot(
            rho_arr, t1_arr, color='#9999ff',
            label='$B_1^+$ −{:.0%}'.format(2 * params['d_eta_fa']))
    except Exception as e:
        print(e)
        ax.set_title('\n'.join(('WARNING! Some plot failed!', title)))
    else:
        ax.set_title('\n'.join((acq_to_seq_info, title)))
    finally:
        ax.set_ylim(params['t1_start'], params['t1_stop'])
        ax.set_ylabel(r'$T_1$ / ms')
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
        ax.set_xlabel(r'$\rho={}$ / arb. units'.format(expression))
        ax.legend()


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
        numex.interactive_tk_mpl.plotting(
            plot_rho_t1_mp2rage_seq,
            SEQ_INTERACTIVES, title=TITLE, about=__doc__)
    else:
        numex.interactive_tk_mpl.plotting(
            plot_rho_t1_mp2rage_acq,
            ACQ_INTERACTIVES, resources_path=PATH['resources'],
            title=TITLE, about=__doc__)

    elapsed(__file__[len(PATH['base']) + 1:])
    msg(report())


# ======================================================================
if __name__ == '__main__':
    main()
