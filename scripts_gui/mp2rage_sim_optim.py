#!python
# -*- coding: utf-8 -*-
"""
Simulate the MP2RAGE signal as a function of T1.

Calculate the MP2RAGE signal as a function of T1, according to given
parameters (adjustable at on-the-fly).
Two different set of parameters (direct and indirect) are accepted.
Direct:
- eff : efficiency eff of the adiabatic inversion pulse
- n : number of slices
- TR_GRE : repetition time of GRE pulses in ms
- TA : time between inversion pulse and first GRE block in ms
- TB : time between first and second GRE blocks in ms
- TC : time after second GRE block in ms
- A1 : flip angle of the first GRE block in deg
- A2 : flip angle a2 of the second GRE block in deg
Indirect:
- eff : efficiency eff of the adiabatic inversion pulse
- n : number of slices
- TR_GRE : repetition time of GRE pulses in ms
- TR_SEQ : total time of the MP2RAGE sequence in ms
- TI1 : inversion time (at center of k-space) of the first GRE blocks in ms
- TI2 : inversion time (at center of k-space) of the second GRE blocks in ms
- A1 : flip angle of the first GRE block in deg
- A2 : flip angle a2 of the second GRE block in deg
Warning: when using indirect parameters, remember to check that TA, TB and TC
timing parameters are positive.

[ref: J. P. Marques at al., NeuroImage 49 (2010) 1271-1281]
"""

# ======================================================================
# :: Future Imports
from __future__ import(
    division, absolute_import, print_function, unicode_literals)


# ======================================================================
# :: Python Standard Library Imports
import os  # Operating System facilities
import argparse  # Argument Parsing


# :: External Imports
import numpy as np  # NumPy (multidimensional numerical arrays library)
import scipy as sp  # SciPy (signal and image processing library)
import matplotlib as mpl  # Matplotlib (2D/3D plotting library)
import matplotlib.pyplot as plt  # Matplotlib's pyplot: MATLAB-like syntax
import scipy.optimize

# :: Local Imports
import pymrt.base as pmb
import pymrt.sequences.mp2rage as mp2rage

from pymrt import INFO
from pymrt import VERB_LVL, D_VERB_LVL
from pymrt import msg, dbg
from pymrt import elapsed, print_elapsed

# ======================================================================
# :: sequence default parameters (MARQUES TR8)
D_EFF = 0.96  # %
D_N_GRE = 160  # #
D_TR_GRE = 7.0  # ms
D_A1 = 4.0  # deg
D_A2 = 5.0  # deg
D_TR_SEQ = 8000.0  # ms
D_TI1 = 1100.0  # ms
D_TI2 = 3300.0  # ms
D_TA = 440.0  # ms
D_TB = 1180.0  # ms
D_TC = 4140.0 # ms
# :: GUI constants
T1_NUM = 256.0
T1_INTERVAL = (100.0, 5000.0)
T1_LINSPACE = T1_INTERVAL + (T1_NUM,)
EFF_SLIDER = (0.0, 1.0, D_EFF)  # %
N_SLIDER = (32, 512, D_N_GRE)  # #
TR_GRE_SLIDER = (0.1, 50.0, D_TR_GRE)  # ms
A1_SLIDER = (0.1, 90.0 / 4, D_A1)  # deg
A2_SLIDER = (0.1, 90.0 / 4, D_A2)  # deg
B1T_PLUS_SLIDER = (0.0 + 0.001, 0.5 - 0.001, 0.2)  # %
B1T_MINUS_SLIDER = (0.0 + 0.001, 0.5 - 0.001, 0.2)  # %
# only in indirect mode
TR_SEQ_SLIDER = (1000.0, 10000.0, D_TR_SEQ)  # ms
TI1_SLIDER = (10.0, 5000.0, D_TI1)  # ms
TI2_SLIDER = (10.0, 10000.0, D_TI2)  # ms
# only in direct mode
TA_SLIDER = (1.0, 10000.0, D_TA)  # ms
TB_SLIDER = (1.0, 10000.0, D_TB)  # ms
TC_SLIDER = (1.0, 10000.0, D_TC)  # ms

# optimization parameters
OPTIM_T1_INTERVAL = (1.0, 4000.0)
OPTIM_S_INTERVAL = (-0.4, 0.4)
OPTIM_A1_INTERVAL = (1.5, 90.0 / 5)


# Turn OFF interactive mode in MatPlotLib
# plt.ion()


# ======================================================================
# :: Create the GUI plot
def ui_plot(
        t1_linspace, is_direct, use_dicom_INTERVAL, optim_a1,
        optim_t1_INTERVAL):
    """
    User-interface and plot.
    """
    a1_idx = 6

    # :: define helper functions
    def get_params():
        """Determine parameters from"""
        # MP2RAGE parameters
        par = []
        for slider in sld_params_list[0:num_params]:
            par.append(slider.val)
        # update B1T tune
        b1t_tune = []
        for slider in sld_params_list[num_params:num_sliders]:
            b1t_tune.append(slider.val)
        return par, b1t_tune

    def adjust_params(par, b1t_tune):
        """Adjust parameters according to B1+ tuning."""
        # adjust parameters for B1T tunes
        par_p = [param * (1.0 + b1t_tune[0]) for param in par[-2:]]
        par_m = [param * (1.0 - b1t_tune[1]) for param in par[-2:]]
        par_p2 = [param * (1.0 + 2.0 * b1t_tune[0])
                  for param in par[-2:]]
        par_m2 = [param * (1.0 - 2.0 * b1t_tune[1])
                  for param in par[-2:]]
        return par_p, par_m, par_p2, par_m2

    def calc_mp2rage(par, par_p, par_m, par_p2, par_m2):
        # update plots
        ii_val = mp2rage_ii(t1_val, *par)
        ii_p_val = mp2rage_ii(t1_val, *(par[:-2] + par_p))
        ii_m_val = mp2rage_ii(t1_val, *(par[:-2] + par_m))
        ii_p2_val = mp2rage_ii(t1_val, *(par[:-2] + par_p2))
        ii_m2_val = mp2rage_ii(t1_val, *(par[:-2] + par_m2))
        if use_dicom_INTERVAL:
            ii_val, ii_p_val, ii_m_val, ii_p2_val, ii_m2_val = [
                pmb.scale(
                    ii, mp2rage.STD_INTERVAL, mp2rage.DICOM_INTERVAL)
                for ii in (ii_val, ii_p_val, ii_m_val, ii_p2_val, ii_m2_val)]
        return ii_val, ii_p_val, ii_m_val, ii_p2_val, ii_m2_val

    def ui_update(event=None):
        """Update plot from slider values"""
        par, b1t_tune = get_params()
        par_p, par_m, par_p2, par_m2 = adjust_params(par, b1t_tune)
        ii_val, ii_p_val, ii_m_val, ii_p2_val, ii_m2_val = \
            calc_mp2rage(par, par_p, par_m, par_p2, par_m2)
        plt_base.set_xdata(ii_val)
        plt_p.set_xdata(ii_p_val)
        plt_m.set_xdata(ii_m_val)
        plt_p2.set_xdata(ii_p2_val)
        plt_m2.set_xdata(ii_m2_val)
        # update label
        plt_p.set_label('B1+ +{:.0f}%'.format(b1t_tune[0] * 100))
        plt_m.set_label('B1+ -{:.0f}%'.format(b1t_tune[1] * 100))
        plt_p2.set_label('B1+ +{:.0f}%'.format(2 * b1t_tune[0] * 100))
        plt_m2.set_label('B1+ -{:.0f}%'.format(2 * b1t_tune[1] * 100))
        # update title
        if is_direct:
            tr_t = mp2rage._calc_tr_seq(*par)
            ti1 = mp2rage._calc_ti1(*par)
            ti2 = mp2rage._calc_ti2(*par)
            ax_main.set_title('MP2RAGE: ' +
                              'TR_t={:.1f} ms, TI1={:.1f} ms, TI2={:.1f} ms'
                              .format(tr_t, ti1, ti2))
        else:
            t_a = mp2rage._calc_ta(*par)
            t_b = mp2rage._calc_tb(*par)
            t_c = mp2rage._calc_tc(*par)
            ax_main.set_title('MP2RAGE: ' +
                              'TA={:.1f} ms, TB={:.1f} ms, TC={:.1f} ms'
                              .format(t_a, t_b, t_c))
        # issue a redraw
        ax_main.legend()
        fig.canvas.draw_idle()

    def ui_reset(event=None):
        """Reset slider to original values"""
        for slider in sld_params_list:
            slider.reset()

    def mp2rage_b1t_spread(a1):
        """Calculate B1T-related spread"""
        par, b1t_tune = get_params()
        par[a1_idx] = a1
        par_p, par_m, par_p2, par_m2 = adjust_params(par, b1t_tune)
        ii_val, ii_p_val, ii_m_val, ii_p2_val, ii_m2_val = \
            calc_mp2rage(par, par_p, par_m, par_p2, par_m2)
        ii_val, ii_p_val, ii_m_val, ii_p2_val, ii_m2_val = \
            calc_mp2rage(par, par_p, par_m, par_p2, par_m2)
        # determine mask
        mask = np.ones_like(t1_val)
        mask *= t1_val > optim_t1_INTERVAL[0]
        mask *= t1_val < optim_t1_INTERVAL[1]
        mask = mask.astype(np.bool)
        # calculate the spread integral
        return np.sum(np.abs(ii_p_val[mask] - ii_m_val[mask]))

    def ui_optim(event):
        """Optimize acquisition parameters"""
        # :: DEBUG
        #        a1 = sld_params_list[a1_idx].val
        #        print("Optimizing: a1={:<8.2f}, sum={:<8.2f}".format(
        #            a1, mp2rage_b1t_spread(a1)))
        a1 = sp.optimize.fminbound(mp2rage_b1t_spread, *OPTIM_A1_INTERVAL)
        print("Optimized:  a1={:<8.2f}, sum={:<8.2f}".format(
            a1, mp2rage_b1t_spread(a1)))
        sld_params_list[a1_idx].set_val(a1)
        return  # TODO: improve optimization?

    # define parameter list
    if is_direct:
        mp2rage_ii = mp2rage.signal
        param_list = [
            ['eff / #', EFF_SLIDER],
            ['n_GRE / #', N_SLIDER],
            ['TR_GRE / ms', TR_GRE_SLIDER],
            ['TA / ms', TA_SLIDER],
            ['TB / ms', TB_SLIDER],
            ['TC / ms', TC_SLIDER],
            ['a1 / deg', A1_SLIDER],
            ['a2 / deg', A2_SLIDER]]
    else:
        mp2rage_ii = mp2rage._signal2
        param_list = [
            ['$\\eta$ / #', EFF_SLIDER],
            ['$n_{GRE}$ / #', N_SLIDER],
            ['$T_{R,GRE}$ / ms', TR_GRE_SLIDER],
            ['$T_{R,seq}$ / ms', TR_SEQ_SLIDER],
            ['$T_{I,1}$ / ms', TI1_SLIDER],
            ['$T_{I,2}$ / ms', TI2_SLIDER],
            ['$\\alpha_1$ / deg', A1_SLIDER],
            ['$\\alpha_2$ / deg', A2_SLIDER]]
    num_params = len(param_list)
    b1t_tune_list = [
        ['$B_1^+$ +', B1T_PLUS_SLIDER],
        ['$B_1^+$ -', B1T_MINUS_SLIDER]]
    slider_list = param_list + b1t_tune_list
    num_sliders = len(slider_list)
    # :: Calculate rhoting values
    t1_val = np.linspace(*t1_linspace)
    # :: figure to plot
    fig, ax_main = plt.subplots()
    # set axes
    ax_main.set_xlabel('MP2RAGE signal (a.u.)')
    ax_main.set_ylabel('T1 (ms)')
    # set xy-ranges
    plt.ylim(T1_INTERVAL)
    if use_dicom_INTERVAL:
        plt.xlim(mp2rage.DICOM_INTERVAL)
    else:
        plt.xlim(mp2rage.STD_INTERVAL)
    # adjust subplot to include sliders
    ui_plt_main_b = 0.125
    ui_sld_params_h = 0.25
    plt.subplots_adjust(bottom=ui_plt_main_b + ui_sld_params_h)
    # plot graph (with dummy values)
    plt_base, = plt.plot(t1_val, t1_val, color='g', label='MP2RAGE')
    plt_p, = plt.plot(t1_val, t1_val, color='r', label='$B_1^+$ +x%')
    plt_m, = plt.plot(t1_val, t1_val, color='b', label='$B_1^+$ -x%')
    plt_p2, = plt.plot(t1_val, t1_val, color='y', label='$B_1^+$ +2x%')
    plt_m2, = plt.plot(t1_val, t1_val, color='c', label='$B_1^+$ -2x%')
    # plot optimized goal
    plt.plot(OPTIM_T1_INTERVAL, OPTIM_S_INTERVAL, '-k', label='optim')
    ax_main.legend()
    # :: create a slider for each MP2RAGE parameters and B1T tuning
    ax_params_list = []
    sld_params_list = []
    ui_slider_l = 0.1875  # 3 / 16
    ui_slider_w = 0.625  # 10 / 16
    ui_sld_params_b = 0.015625  # 1 / 64
    ui_slider_h = ui_sld_params_h / num_sliders
    for title, sliderval in slider_list:
        idx = slider_list.index([title, sliderval])
        ui_slider_b = ui_slider_h * (num_sliders - idx - 1.0) + ui_sld_params_b
        axes = plt.axes([ui_slider_l, ui_slider_b, ui_slider_w, ui_slider_h])
        slider = mpl.widgets.Slider(axes, title, *sliderval)
        ax_params_list.append(axes)
        sld_params_list.append(slider)
    # append on-change trigger to sliders
    for slider in sld_params_list:
        slider.on_changed(ui_update)
    # :: create a reset button
    ui_btn_reset_w = 0.08
    ui_btn_reset_h = 0.04
    ui_btn_reset_l = 1.0 - ui_btn_reset_w
    ui_btn_reset_b = 1.0 - ui_btn_reset_h
    ax_reset = plt.axes(
        [ui_btn_reset_l, ui_btn_reset_b, ui_btn_reset_w, ui_btn_reset_h])
    btn_reset = mpl.widgets.Button(ax_reset, 'Reset')
    btn_reset.on_clicked(ui_reset)
    # :: create an optimize button
    if optim_a1:
        ui_btn_optim_w = 0.12
        ui_btn_optim_h = 0.04
        ui_btn_optim_l = 1.0 - ui_btn_optim_w - ui_btn_reset_w
        ui_btn_optim_b = 1.0 - ui_btn_optim_h
        ax_optim = plt.axes(
            [ui_btn_optim_l, ui_btn_optim_b, ui_btn_optim_w, ui_btn_optim_h])
        btn_optim = mpl.widgets.Button(ax_optim, 'Optim a1')
        btn_optim.on_clicked(ui_optim)
    # :: show the graph
    ui_update()
    plt.show()


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
        '--t1', metavar=('MIN', 'MAX', 'N_PT'),
        type=int, nargs=3, default=T1_LINSPACE,
        help='set parameters for the values of T1 in ms [%(default)s]')
    arg_parser.add_argument(
        '-d', '--direct',
        action='store_true',
        help='use TA, TB, TC direct parameters instead of TI1, TI2, TR_tot')
    arg_parser.add_argument(
        '--dicom_INTERVAL',
        action='store_true',
        help='use {} intensity interval instead of {}.'. \
            format(mp2rage.DICOM_INTERVAL, mp2rage.STD_INTERVAL))
    arg_parser.add_argument(
        '--no_optim_a1',
        action='store_false',
        help='disable a1 optimization [%(default)s]')
    arg_parser.add_argument(
        '--ot1', metavar=('MIN', 'MAX'),
        type=int, nargs=2, default=OPTIM_T1_INTERVAL,
        help='set the optimization range for T1 in ms [%(default)s]')
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

    ui_plot(args.t1, args.direct, args.dicom_INTERVAL, args.no_optim_a1,
            args.ot1)

    elapsed(os.path.basename(__file__))
    print_elapsed()



# ======================================================================
if __name__ == '__main__':
    main()
