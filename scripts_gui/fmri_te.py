#!python
# -*- coding: utf-8 -*-
"""
PyMRT - Interactive: fMRI experiment -- Echo Time dependence

Simulate the FLASH signal as a function of TE, including MT effects.

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
from pymrt.sequences import flash

from pymrt import INFO, PATH
from pymrt import VERB_LVL, D_VERB_LVL
from pymrt import msg, dbg, fmt, fmtm
from pymrt import elapsed, report

TITLE = __doc__.strip().split('\n')[0][:-1]
INTERACTIVES = collections.OrderedDict([
    ('tr', dict(
        label='TR / ms',
        default=2000, start=0, stop=4000, step=50)),
    ('fa', dict(
        label='Flip Angle / deg',
        default=50, start=0, stop=360, step=1)),
    ('te_start', dict(
        label='TE_start / ms',
        default=0, start=0, stop=250, step=5)),
    ('te_stop', dict(
        label='TE_stop / ms',
        default=120, start=0, stop=250, step=5)),
    ('te_num', dict(
        label='TE_num / #',
        default=256, start=16, stop=1024, step=16)),
    ('mt_suppression', dict(
        label='MT suppression / %',
        default=60, start=0, stop=100, step=1)),

    ('rest_arterial_m0', dict(
        label='Rest Arterial M0 / arb.units',
        default=1.00, start=0, stop=10, step=0.05)),
    ('rest_capillary_m0', dict(
        label='Rest Capillary M0 / arb.units',
        default=1.95, start=0, stop=10, step=0.05)),
    ('rest_venous_m0', dict(
        label='Rest Venous M0 / arb.units',
        default=2.25, start=0, stop=10, step=0.05)),
    ('rest_tissue_m0', dict(
        label='Rest Tissue M0 / arb.units',
        default=94.8, start=30, stop=100, step=0.05)),

    ('rest_arterial_t1', dict(
        label='Rest Arterial T1 / ms',
        default=1650, start=100, stop=4000, step=5)),
    ('rest_capillary_t1', dict(
        label='Rest Capillary T1 / ms',
        default=1650, start=100, stop=4000, step=5)),
    ('rest_venous_t1', dict(
        label='Rest Venous T1 / ms',
        default=1650, start=100, stop=4000, step=5)),
    ('rest_tissue_t1', dict(
        label='Rest Tissue T1 / ms',
        default=1605, start=100, stop=4000, step=5)),

    ('rest_arterial_t2s', dict(
        label='Rest Arterial T2* / ms',
        default=96.7, start=0, stop=120, step=0.1)),
    ('rest_capillary_t2s', dict(
        label='Rest Capillary T2* / ms',
        default=33.3, start=0, stop=120, step=0.1)),
    ('rest_venous_t2s', dict(
        label='Rest Venous T2* / ms',
        default=19.6, start=0, stop=120, step=0.1)),
    ('rest_tissue_t2s', dict(
        label='Rest Tissue T2* / ms',
        default=65.8, start=0, stop=120, step=0.1)),

    ('rest_perfusion', dict(
        label='Rest Perfusion / arb.units',
        default=1.0, start=0, stop=10, step=0.05)),

    ('actv_arterial_m0', dict(
        label='Actv Arterial M0 / arb.units',
        default=2.00, start=0, stop=10, step=0.05)),
    ('actv_capillary_m0', dict(
        label='Actv Capillary M0 / arb.units',
        default=2.35, start=0, stop=10, step=0.05)),
    ('actv_venous_m0', dict(
        label='Actv Venous M0 / arb.units',
        default=2.35, start=0, stop=10, step=0.05)),
    ('actv_tissue_m0', dict(
        label='Actv Tissue M0 / arb.units',
        default=93.3, start=30, stop=100, step=0.05)),

    ('actv_arterial_t1', dict(
        label='Actv Arterial T1 / ms',
        default=1650, start=100, stop=4000, step=5)),
    ('actv_capillary_t1', dict(
        label='Actv Capillary T1 / ms',
        default=1650, start=100, stop=4000, step=5)),
    ('actv_venous_t1', dict(
        label='Actv Venous T1 / ms',
        default=1650, start=100, stop=4000, step=5)),
    ('actv_tissue_t1', dict(
        label='Actv Tissue T1 / ms',
        default=1605, start=100, stop=4000, step=5)),

    ('actv_arterial_t2s', dict(
        label='Actv Arterial T2* / ms',
        default=96.7, start=0, stop=120, step=0.1)),
    ('actv_capillary_t2s', dict(
        label='Actv Capillary T2* / ms',
        default=43.5, start=0, stop=120, step=0.1)),
    ('actv_venous_t2s', dict(
        label='Actv Venous T2* / ms',
        default=27.2, start=0, stop=120, step=0.1)),
    ('actv_tissue_t2s', dict(
        label='Actv Tissue T2* / ms',
        default=67.6, start=0, stop=120, step=0.1)),

    ('actv_perfusion', dict(
        label='Actv Perfusion / arb.units',
        default=1.6, start=0, stop=10, step=0.05)),
])

COMPARTMENTS = 'arterial', 'capillary', 'venous', 'tissue'


# ======================================================================
def signals_perfusion(
        m0,
        t1,
        t2s,
        fa,
        tr,
        te,
        mf=0.0,
        mt=0.0,
        compartments=COMPARTMENTS):
    m0['arterial'] = m0['arterial']  # unchanged
    m0['capillary'] = (m0['capillary'] - 0.5 * mf) + 0.5 * mf * (1 - mt)
    m0['venous'] = (m0['venous'] - 0.5 * mf) + 0.5 * mf * (1 - mt)
    m0['tissue'] = (m0['tissue'] - mf) * (1 - mt) + mf
    result = tuple(
        flash.signal(
            m0[compartment], fa, tr, t1[compartment], te, t2s[compartment])
        for compartment in compartments)
    return np.array(result)


# ======================================================================
def group_params(
        name,
        params):
    data = {}
    for k, v in params.items():
        if k.startswith(f'{name}_'):
            try:
                name, compartment, param = k.split('_')
                if param not in data:
                    data[param] = {}
                data[param][compartment] = v
            except ValueError:
                name, param = k.split('_')
                if param not in data:
                    data[param] = {}
                data[param] = v
    return data


# ======================================================================
def plot(
        fig,
        params=None,
        title=TITLE.split(':')[1].strip()):
    axs = fig.subplots(1, 1, squeeze=False, sharex='row', sharey='row')
    te = params['te'] = np.linspace(
        params['te_start'], params['te_stop'], params['te_num'])
    rest = group_params('rest', params)
    actv = group_params('actv', params)
    ss_rest = signals_perfusion(
        rest['m0'], rest['t1'], rest['t2s'],
        params['fa'], params['tr'], params['te'],
        rest['perfusion'], params['mt_suppression'] / 100)
    ss_actv = signals_perfusion(
        actv['m0'], actv['t1'], actv['t2s'],
        params['fa'], params['tr'], params['te'],
        actv['perfusion'], params['mt_suppression'] / 100)
    ds = ss_actv - ss_rest
    ds_tot = np.sum(ds, axis=0)
    te_i = np.argmax(ds_tot)
    colors = dict(
        arterial='#cc3333',
        capillary='#cc66cc',
        venous='#3333cc',
        tissue='#33cc33',
    )
    ax = axs[0, 0]
    for j, name in enumerate(COMPARTMENTS):
        ax.plot(te, ds[j], label=f'{name.title()}', color=colors[name])
    ax.plot(te, ds_tot, label=f'Total', color='#000000')
    ax.axvline(x=te[te_i], color='#999999', linestyle='dotted')
    ax.legend()
    ax.set_title(f'Optimal $T_{{E}}$ = {te[te_i]:.1f} ms')
    ax.set_ylabel('$\Delta S$ / arb. units')
    ax.set_xlabel('$T_E$ / ms')


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

    numex.interactive_tk_mpl.plotting(
        plot,
        INTERACTIVES, resources_path=PATH['resources'],
        title=TITLE, about=__doc__)

    elapsed(__file__[len(PATH['base']) + 1:])
    msg(report())


# ======================================================================
if __name__ == '__main__':
    main()
