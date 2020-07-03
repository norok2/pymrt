#!python
# -*- coding: utf-8 -*-
"""
PyMRT - Interactive: fMRI experiment -- Echo Time dependence.

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
    ('mode', dict(
        label='Mode', default='Default', values=('Default', 'Difference'))),

    ('show_arterial', dict(
        label='Show Arterial', default=True, values=(False, True))),
    ('show_capillary', dict(
        label='Show Capillary', default=True, values=(False, True))),
    ('show_venous', dict(
        label='Show Venous', default=True, values=(False, True))),
    ('show_tissue', dict(
        label='Show Tissue', default=True, values=(False, True))),
    ('show_total', dict(
        label='Show Total', default=True, values=(False, True))),
    ('show_opt', dict(
        label='Show Optimal', default=True, values=(False, True))),

    ('y_min', dict(
        label='Min. Y', default=0.0, start=-10, stop=10, step=0.01)),
    ('y_max', dict(
        label='Max. Y', default=0.0, start=-10, stop=10, step=0.01)),

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

    ('both_blood_wc', dict(
        label='Water content Blood / #',
        default=0.87, start=0, stop=1, step=0.005)),
    ('both_tissue_wc', dict(
        label='Water content Tissue / #',
        default=0.89, start=0, stop=1, step=0.005)),

    ('both_arterial_mt', dict(
        label='MT suppr. Arterial / #',
        default=0.07, start=0, stop=1, step=0.005)),
    ('both_capillary_mt', dict(
        label='MT suppr. Capillary / #',
        default=0.07, start=0, stop=1, step=0.005)),
    ('both_venous_mt', dict(
        label='MT suppr. Venous / #',
        default=0.07, start=0, stop=1, step=0.005)),
    ('both_tissue_mt', dict(
        label='MT suppr. Tissue / #',
        default=0.60, start=0, stop=1, step=0.005)),

    ('rest_arterial_m0', dict(
        label='Rest Arterial M0 / arb.units',
        default=1.16, start=0, stop=10, step=0.01)),
    ('rest_capillary_m0', dict(
        label='Rest Capillary M0 / arb.units',
        default=1.81, start=0, stop=10, step=0.01)),
    ('rest_venous_m0', dict(
        label='Rest Venous M0 / arb.units',
        default=2.53, start=0, stop=10, step=0.01)),
    # ('rest_tissue_m0', dict(
    #     label='Rest Tissue M0 / arb.units',
    #     default=94.8, start=30, stop=100, step=0.05)),

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
        default=58.1, start=0, stop=120, step=0.1)),
    ('rest_capillary_t2s', dict(
        label='Rest Capillary T2* / ms',
        default=32.5, start=0, stop=120, step=0.1)),
    ('rest_venous_t2s', dict(
        label='Rest Venous T2* / ms',
        default=22.2, start=0, stop=120, step=0.1)),
    ('rest_tissue_t2s', dict(
        label='Rest Tissue T2* / ms',
        default=65.8, start=0, stop=120, step=0.1)),

    ('rest_perfusion', dict(
        label='Rest Perfusion / arb.units',
        default=1.0, start=0, stop=10, step=0.05)),

    ('actv_arterial_m0', dict(
        label='Actv Arterial M0 / arb.units',
        default=2.03, start=0, stop=10, step=0.01)),
    ('actv_capillary_m0', dict(
        label='Actv Capillary M0 / arb.units',
        default=2.24, start=0, stop=10, step=0.01)),
    ('actv_venous_m0', dict(
        label='Actv Venous M0 / arb.units',
        default=2.88, start=0, stop=10, step=0.01)),
    # ('actv_tissue_m0', dict(
    #     label='Actv Tissue M0 / arb.units',
    #     default=93.3, start=30, stop=100, step=0.05)),

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
        default=1600, start=100, stop=4000, step=5)),

    ('actv_arterial_t2s', dict(
        label='Actv Arterial T2* / ms',
        default=58.1, start=0, stop=120, step=0.1)),
    ('actv_capillary_t2s', dict(
        label='Actv Capillary T2* / ms',
        default=47.6, start=0, stop=120, step=0.1)),
    ('actv_venous_t2s', dict(
        label='Actv Venous T2* / ms',
        default=32.3, start=0, stop=120, step=0.1)),
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
        mf,
        mt,
        wc,
        compartments=COMPARTMENTS):
    m0 = m0.copy()
    m0['arterial'] = wc['blood'] * m0['arterial'] * (1 - mt['arterial'])
    m0['capillary'] = wc['blood'] * (
            (m0['capillary'] - 0.5 * mf) * (1 - mt['capillary'])
            + 0.5 * mf * (1 - mt['tissue']))
    m0['venous'] = wc['blood'] * (
            (m0['venous'] - 0.5 * mf) * (1 - mt['venous'])
            + 0.5 * mf * (1 - mt['tissue']))
    m0['tissue'] = wc['tissue'] * (
            (m0['tissue'] - mf) * (1 - mt['tissue']) \
            + mf * (1 - mt['arterial']))
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
def _normalize_qty(params, prefix, suffix, value=100):
    return value - sum(
        params[x] for x in params
        if x.startswith(prefix) and x.endswith(suffix))


# ======================================================================
def plot(
        fig,
        params=None,
        title=TITLE.split(':')[1].strip()):
    axs = fig.subplots(1, 1, squeeze=False, sharex='row', sharey='row')
    te = params['te'] = np.linspace(
        params['te_start'], params['te_stop'], params['te_num'])
    params['rest_tissue_m0'] = _normalize_qty(params, 'rest_', '_m0')
    params['actv_tissue_m0'] = _normalize_qty(params, 'actv_', '_m0')

    rest = group_params('rest', params)
    actv = group_params('actv', params)
    both = group_params('both', params)
    ss_rest = signals_perfusion(
        rest['m0'], rest['t1'], rest['t2s'],
        params['fa'], params['tr'], params['te'],
        rest['perfusion'], both['mt'], both['wc'])
    ss_actv = signals_perfusion(
        actv['m0'], actv['t1'], actv['t2s'],
        params['fa'], params['tr'], params['te'],
        actv['perfusion'], both['mt'], both['wc'])
    ds = ss_actv - ss_rest
    if params['mode'] == 'Difference':
        mtoff = dict(arterial=0.0, capillary=0.0, venous=0.0, tissue=0.0)
        ss_rest_mtoff = signals_perfusion(
            rest['m0'], rest['t1'], rest['t2s'],
            params['fa'], params['tr'], params['te'],
            rest['perfusion'], mtoff, both['wc'])
        ss_actv_mtoff = signals_perfusion(
            actv['m0'], actv['t1'], actv['t2s'],
            params['fa'], params['tr'], params['te'],
            actv['perfusion'], mtoff, both['wc'])
        ds_mton = ds
        ds_mtoff = ss_actv_mtoff - ss_rest_mtoff
        ds = ds_mtoff - ds_mton
    ds_tot = np.sum(ds, axis=0)
    if params['mode'] != 'Default':
        te_i = np.argmin(np.abs(ds_tot))
        te_name = 'Zero Crossing'
        title = 'Difference Signal Variation upon Activation'
    else:
        te_i = np.argmax(ds_tot)
        te_name = 'Optimal'
        title = 'Signal Variation upon Activation'

    colors = dict(
        arterial='#cc3333',
        capillary='#cc66cc',
        venous='#3333cc',
        tissue='#33cc33',
    )
    ax = axs[0, 0]
    for j, name in enumerate(COMPARTMENTS):
        if params['show_' + name]:
            ax.plot(te, ds[j], label=name.title(), color=colors[name])
    if params['show_total']:
        ax.plot(te, ds_tot, label='Total', color='#000000')
    if params['show_opt']:
        ax.axvline(x=te[te_i], color='#999999', linestyle='dotted')
        title += fmt(
            '\n' + te_name + '  $T_{{E}} = {:.1f}\\pm{:.1f}\\;\\mathrm{{ms}}$',
            te[te_i], te[1] - te[0])
    ax.set_title(title)
    ax.legend()
    ax.set_ylabel('$\\Delta S$ / arb. units')
    ax.set_xlabel('$T_E$ / ms')
    if params['y_min'] != params['y_max']:
        ax.set_ylim([params['y_min'], params['y_max']])


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
