#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PyMRT: quick tests for matrix algebra
"""

import itertools
import datetime

import numpy as np
import sympy as sym
import matplotlib as mpl
import seaborn as sns

import matplotlib.pyplot as plt

import pymrt as mrt
import pymrt.utils

from pymrt import VERB_LVL, D_VERB_LVL, VERB_LVL_NAMES
from pymrt import msg, dbg
from pymrt import elapsed, report

from pymrt.sequences.matrix_algebra import (
    GAMMA, GAMMA_BAR,
    dynamics_operator, SpinModel, PulseSequence, Pulse, Delay, Spoiler,
    PulseExc, ReadOut, MagnetizationPreparation, )
from pymrt.recipes.qmt import (
    MultiMtSteadyState, MultiMtSteadyState2, MultiMtVarMGESS, )


# ======================================================================
def check_dynamics_operator_symbolic():
    """
    Notes: import pi, sin and cos from sympy

    Returns:
        None
    """
    # todo: make it flexible and working
    w_c, w1 = sym.symbols('w_c w1')
    mc = [sym.symbols('m0{}'.format())]

    # 2-pool model
    spin_model = SpinModel(
        s0=100,
        mc=(1.0, 0.152),
        w0=((w_c,) * 2),
        r1=(1.8, 1.0),
        r2=(32.2581, 8.4746e4),
        k=(0.05,),
        approx=(None, 'superlorentz_approx'))

    print(spin_model)
    print(spin_model.m_eq)
    print(spin_model._k_op)
    print(spin_model._l_op)
    print(dynamics_operator(spin_model, w_c + 10.0, w1))

    # 3-pool model
    spin_model = SpinModel(
        s0=100,
        mc=(1.0, 0.152),
        w0=((w_c,) * 3),
        r1=(1.8, 1.0, 1.2),
        r2=(32.2581, 8.4746e4, 30.0),
        k=(0.05, 0.5, 0.1),
        approx=(None, 'superlorentz_approx', None))

    print(spin_model)
    print(spin_model.m_eq)
    print(spin_model._k_op)
    print(spin_model._l_op)
    print(dynamics_operator(spin_model, w_c + 10.0, w1))

    # 4-pool model
    spin_model = SpinModel(
        s0=100,
        mc=(1.0, 0.152),
        w0=((w_c,) * 4),
        r1=(1.8, 1.0, 1.2, 2.0),
        r2=(32.2581, 8.4746e4, 30.0, 60.0),
        k=(0.05, 0.5, 0.1, 0.001, 0.4, 0.2),
        approx=(None, 'superlorentz_approx', None, 'gauss'))

    print(spin_model)
    print(spin_model.m_eq)
    print(spin_model._k_op)
    print(spin_model._l_op)
    print(dynamics_operator(spin_model, w_c + 10.0, w1))


# ======================================================================
def check_dynamics_operator():
    """
    Notes: import pi, sin and cos from numpy

    Returns:

    """
    w_c = GAMMA['1H'] * B0
    w1 = 1.0

    # 2-pool model
    spin_model = SpinModel(
        s0=100,
        mc=(1.0, 0.152),
        w0=((w_c,) * 2),
        r1=(1.8, 1.0),
        r2=(32.2581, 8.4746e4),
        k=(0.3456,),
        approx=(None, 'superlorentz_approx'))

    print(spin_model)
    print(spin_model.m_eq)
    print(spin_model._k_op)
    print(spin_model._l_op)
    print(dynamics_operator(spin_model, w_c + 10.0, w1))

    # 3-pool model
    spin_model = SpinModel(
        m0=[v * 100.0 for v in (1.0, 0.152, 0.3)],
        w0=((w_c,) * 3),
        r1=(1.8, 1.0, 1.2),
        r2=(32.2581, 8.4746e4, 30.0),
        k=(0.05, 0.5, 0.1),
        approx=(None, 'superlorentz_approx', None))

    print(spin_model)
    print(spin_model.m_eq)
    print(spin_model._k_op)
    print(spin_model._l_op)
    print(dynamics_operator(spin_model, w_c + 10.0, w1))

    # 4-pool model
    spin_model = SpinModel(
        m0=[v * 100.0 for v in (1.0, 0.152, 0.3, 0.01)],
        w0=((w_c,) * 4),
        r1=(1.8, 1.0, 1.2, 2.0),
        r2=(32.2581, 8.4746e4, 30.0, 60.0),
        k=(0.05, 0.5, 0.1, 0.001, 0.4, 0.2),
        approx=(None, 'superlorentz_approx', None, 'gauss'))

    print(spin_model)
    print(spin_model.m_eq)
    print(spin_model._k_op)
    print(spin_model._l_op)
    print(dynamics_operator(spin_model, w_c + 10.0, w1))


# ======================================================================
def check_mt_sequence():
    """
    Test for the MT sequence.
    """
    w_c = GAMMA['1H'] * B0

    spin_model = SpinModel(
        s0=100,
        mc=(1.0, 0.152),
        w0=((w_c,) * 2),
        r1=(1.8, 1.0),
        r2=(32.2581, 8.4746e4),
        k=(0.3456,),
        approx=(None, 'superlorentz_approx'))

    num_repetitions = 300

    mt_flash_kernel = PulseSequence([
        Delay(10.0e-3),
        Spoiler(1.0),
        Pulse.shaped(40.0e-3, 220.0, 4000, 'gauss', None,
            w_c + 50.0, 'poly', {'fit_order': 5}),
        Delay(20.0e-3),
        Spoiler(1.0),
        Pulse.shaped(10.0e-6, 90.0, 1, 'rect', None),
        Delay(30.0e-3)],
        b0=3.0)
    mt_flash = PulseSequenceRepeated(mt_flash_kernel, num_repetitions)

    signal = mt_flash.signal(spin_model)

    print(mt_flash)
    print(mt_flash.propagator(spin_model))
    print(signal)


# ======================================================================
def check_approx_propagator(
        spin_model=SpinModel(
            s0=100,
            mc=(0.8681, 0.1319),
            w0=((GAMMA['1H'] * 7.0,) * 2),
            r1=(1.8, 1.0),
            r2=(32.2581, 8.4746e4),
            k=(0.3456,),
            approx=(None, 'superlorentz_approx')),
        flip_angle=90.0):
    """
    Test the approximation of propagators - for speeding up.

    Args:
        spin_model (SpinModel):
        flip_angles (float):
    """
    w_c = spin_model.w0[0]

    modes = ['exact']
    modes.extend(['linear', 'reduced'])
    # modes.extend(['sum_simple', 'sum_order1', 'sum_sep', 'reduced'])
    modes.extend(['poly_{}'.format(order) for order in range(4, 5)])
    modes.extend(['interp_{}_{}'.format(mode, num_samples)
                  for mode in ['linear', 'cubic']
                  for num_samples in range(4, 5)])
    modes = {
        'linear': {
            'num_samples': tuple(range(10, 20, 5))},
        'interp': {
            'method': ('linear', 'cubic'),
            'num_samples': tuple(range(10, 20, 3))},
        'reduced': {
            'num_resamples': tuple(range(10, 20, 5))},
        'poly': {
            'fit_order': tuple(range(3, 6))}
    }

    shapes = {
        'gauss': {},
        'lorentz': {},
        'sinc': {},
        'fermi': {},
        # 'random': {},
        'cos_sin': {},
    }
    exact_p_ops = {}
    for shape, shape_kwargs in shapes.items():
        pulse = Pulse.shaped(
            40.0e-3, flip_angle, 4000, shape, shape_kwargs, w_c, 'exact', {})
        exact_p_ops[shape] = pulse.propagator(spin_model)

    for shape, shape_kwargs in shapes.items():
        for mode, mode_params in modes.items():
            kwargs_items = [{}]
            names = mode_params.keys()
            for values in itertools.product(*[mode_params[i] for i in names]):
                kwargs_items.append(dict(zip(names, values)))
            for kwargs in kwargs_items:
                pulse = Pulse.shaped(
                    40.0e-3, flip_angle, 4000, shape, shape_kwargs, w_c,
                    mode, kwargs)
                begin_time = datetime.datetime.now()
                p_op = pulse.propagator(spin_model)
                elapsed = datetime.timedelta(
                    datetime.datetime.now() - begin_time)
                rel_error = np.sum(np.abs(exact_p_ops[shape] - p_op)) / \
                            np.sum(np.abs(exact_p_ops[shape]))
                print('{:>8s}, {:>8s}, {:>48s},\t{:.3e}, {}'.format(
                    shape, mode, str(kwargs), rel_error, elapsed))


# ======================================================================
def check_z_spectrum(
        spin_model=SpinModel(
            s0=1e8,
            mc=(0.8681, 0.1319),
            w0=((GAMMA['1H'] * 7.0,) * 2),
            r1=(1.8, 1.0),
            r2=(32.2581, 8.4746e4),
            k=(0.3456,),
            approx=(None, 'superlorentz_approx')),
        frequencies=np.round(np.geomspace(50, 10000, 32)),
        amplitudes=np.round(np.linspace(1, 5000, 24)),
        plot_data=True,
        save_file=None):
    """
    Test calculation of z-spectra

    Args:

        spin_model (SpinModel):
        frequencies (ndarray[float]):
        amplitudes (ndarray[float]):
        plot_data (bool):
        save_file (string):

    Returns:
        freq

    """
    print('Checking Z-spectrum')
    w_c = spin_model.w0[0]

    flip_angles = amplitudes * 11.799 / 50.0

    my_seq = MultiMtSteadyState(
        pulses=[
            MagnetizationPreparation.shaped(
                10.0e-3, 90.0, 4000, 'gauss', {}, w_c, 'poly',
                {'fit_order': 3}),
            Delay(1.0e-3),
            Spoiler(1.0),
            PulseExc.shaped(2.1e-3, 15.0, 1, 'rect', {}),
            ReadOut(),
            Spoiler(1.0), ],
        te=5.0e-3,
        tr=70.0e-3,
        n_r=300,
        w_c=w_c,
        preps=[(df, mfa) for df in frequencies for mfa in flip_angles])
    data = my_seq.signal(spin_model).reshape(
        (len(frequencies), len(flip_angles)))

    # plot results
    if plot_data:
        sns.set_style('whitegrid')
        X, Y = np.meshgrid(flip_angles, np.log10(frequencies))
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlabel('Pulse Amplitude (flip angle) / deg')
        ax.set_ylabel('Frequency offset / Hz (log10 scale)')
        ax.set_zlabel('Signal Intensity / arb. units')
        ax.plot_surface(
            X, Y, data, cmap=mpl.cm.plasma,
            rstride=1, cstride=1, linewidth=0.01, antialiased=False)
    if save_file:
        np.savez(save_file, frequencies, amplitudes, data)
    return data, frequencies, flip_angles


# ======================================================================
def check_z_spectrum2(
        spin_model=SpinModel(
            s0=1e8,
            mc=(0.8681, 0.1319),
            w0=((GAMMA['1H'] * 7.0,) * 2),
            r1=(1.8, 1.0),
            r2=(32.2581, 8.4746e4),
            k=(0.3456,),
            approx=(None, 'superlorentz_approx')),
        frequencies=np.round(np.geomspace(50, 10000, 32)),
        amplitudes=np.round(np.linspace(1, 5000, 24)),
        plot_data=True,
        save_file=None):
    """
    Test calculation of z-spectra

    Args:

        spin_model (SpinModel):
        frequencies (ndarray[float]):
        amplitudes (ndarray[float]):
        plot_data (bool):
        save_file (string):

    Returns:
        freq

    """
    print('Checking Z-spectrum (2)')
    w_c = spin_model.w0[0]

    flip_angles = amplitudes * 11.799 / 50.0

    my_seq = MultiMtSteadyState2(
        pulses=[
            MagnetizationPreparation.shaped(
                10.0e-3, 90.0, 4000, 'gauss', {}, w_c, 'poly',
                {'fit_order': 3}),
            Delay(1.0e-3),
            Spoiler(1.0),
            PulseExc.shaped(2.1e-3, 15.0, 1, 'rect', {}),
            ReadOut(),
            Spoiler(1.0), ],
        te=5.0e-3,
        tr=70.0e-3,
        n_r=300,
        w_c=w_c,
        freqs=frequencies,
        mfas=flip_angles)
    data = my_seq.signal(spin_model)

    # plot results
    if plot_data:
        sns.set_style('whitegrid')
        X, Y = np.meshgrid(flip_angles, np.log10(frequencies))
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlabel('Pulse Amplitude (flip angle) / deg')
        ax.set_ylabel('Frequency offset / Hz (log10 scale)')
        ax.set_zlabel('Signal Intensity / arb. units')
        ax.plot_surface(
            X, Y, data, cmap=mpl.cm.plasma,
            rstride=1, cstride=1, linewidth=0.01, antialiased=False)
    if save_file:
        np.savez(save_file, frequencies, amplitudes, data)
    return data, frequencies, flip_angles


# ======================================================================
def check_z_spectrum_sparse(
        spin_model=SpinModel(
            s0=1e8,
            mc=(0.8681, 0.1319),
            w0=((GAMMA['1H'] * 7.0,) * 2),
            r1=(1.8, 1.0),
            r2=(32.2581, 8.4746e4),
            k=(0.3456,),
            approx=(None, 'superlorentz_approx')),
        frequencies=np.round(np.geomspace(50, 10000, 32)),
        amplitudes=np.round(np.linspace(1, 5000, 24)),
        plot_data=True,
        save_file=None):
    """
    Test calculation of z-spectra

    Args:

        spin_model (SpinModel):
        frequencies (ndarray[float]):
        amplitudes (ndarray[float]):
        plot_data (bool):
        save_file (string):

    Returns:
        freq

    """
    print('Checking Z-spectrum (sparse)')
    w_c = spin_model.w0[0]

    flip_angles = amplitudes * 11.799 / 50.0

    my_seq = MultiMtVarMGESS(
        pulses=[
            MagnetizationPreparation.shaped(
                10.0e-3, 90.0, 4000, 'gauss', {}, w_c, 'poly',
                {'fit_order': 3}),
            Delay(1.0e-3),
            Spoiler(1.0),
            PulseExc.shaped(2.1e-3, 15.0, 1, 'rect', {}),
            ReadOut(),
            Spoiler(1.0), ],
        tes=5.0e-3,
        tr=70.0e-3,
        n_r=300,
        w_c=w_c,
        preps=[(df, mfa,) for df in frequencies for mfa in flip_angles])
    data = my_seq.signal(spin_model).reshape(
        (len(frequencies), len(flip_angles)))

    # plot results
    if plot_data:
        sns.set_style('whitegrid')
        X, Y = np.meshgrid(flip_angles, np.log10(frequencies))
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlabel('Pulse Amplitude (flip angle) / deg')
        ax.set_ylabel('Frequency offset / Hz (log10 scale)')
        ax.set_zlabel('Signal Intensity / arb. units')
        ax.plot_surface(
            X, Y, data, cmap=mpl.cm.plasma,
            rstride=1, cstride=1, linewidth=0.01, antialiased=False)
    if save_file:
        np.savez(save_file, frequencies, amplitudes, data)
    return data, frequencies, flip_angles


# ======================================================================
def check_fit_spin_model(
        snr_level=20,
        plot_data=True):
    """
    Test calculation of z-spectra

    Args:
        snr_level (float):
        plot_data (bool):

    Returns:
        None
    """
    w_c = GAMMA['1H'] * B0

    # mt_flash = MtFlash(
    #     PulseList([
    #         Delay(16610.0e-6),
    #         Spoiler(0.0),
    #         Delay(160.0e-6),
    #         PulseExc.shaped(10000.0e-6, 90.0, 0, '_from_GAUSS5120', {},
    #                         0.0, 'poly', {'fit_order': 3}),
    #         Delay(160.0e-6 + 970.0e-6),
    #         Spoiler(1.0),
    #         Delay(160.0e-6),
    #         PulseExc.shaped(100e-6, 11.0, 1, 'rect', {}),
    #         Delay(4900.0e-6)],
    #         w_c=w_c),
    #     300)

    t_e = 1.7e-3
    t_r = 70.0e-3
    w_c = 297220696
    mt_flash = MultiMtSteadyState2(
        PulseSequence([
            Delay(
                t_r - (t_e + 3 * 160.0e-6 + 20000.0e-6 + 970.0e-6 + 100e-6)),
            Spoiler(0.0),
            Delay(160.0e-6),
            Pulse.shaped(
                20000.0e-6, 90.0, 0, '_from_GAUSS5120', {}, 0.0,
                'linear', {'num_samples': 15}),
            Delay(160.0e-6 + 970.0e-6),
            Spoiler(1.0),
            Delay(160.0e-6),
            Pulse.shaped(100e-6, 30.0, 1, 'rect', {}),
            Delay(t_e)],
            w_c=w_c),
        num_repetitions=100 * 100)

    def mt_signal(x_arr, s0, mc_a, r1a, r2a, r2b, k_ab):
        spin_model = SpinModel(
            s0=s0,
            mc=(mc_a, 1.0 - mc_a),
            w0=(w_c, w_c * (1 - 3.5e-6)),
            # w0=((w_c,) * 2),
            r1=(r1a, 1.0),
            r2=(r2a, r2b),
            k=(k_ab,),
            approx=(None, 'superlorentz_approx'))
        y_arr = np.zeros_like(x_arr[:, 0])
        i = 0
        for freq, flip_angle in x_arr:
            mt_flash.set_flip_angle(flip_angle)
            mt_flash.set_freq(freq)
            y_arr[i] = mt_flash.signal(spin_model)
            i += 1
        return y_arr

    # simulate a measurement
    freqs = np.geomspace(100.0, 300.0e3, 32)
    flip_angles = np.linspace(1.0, 1100.0, 32)

    x_data = np.array(tuple(itertools.product(freqs, flip_angles)))

    # see: mt_signal
    p_e = 100, 0.8681, 2.0, 32.2581, 8.4746e4, 0.3456
    exact = mt_signal(x_data, *p_e).reshape((len(freqs), len(flip_angles)))
    # num = len(freqs) * len(flip_angles)
    # noise = (np.random.rand(*exact.shape) - 0.5) * np.max(exact) / snr_level
    # measured = exact + noise
    #
    # p0 = 100, 0.5, 5.0, 20.0, 5e4, 0.5
    # bounds = [[50, 1000], [0, 1], [0.1, 10], [10, 50], [1e4, 1e5], [0, 1]]
    # y_data = measured.ravel()
    #
    # def sum_of_squares(params, x_data, m_data):
    #     e_data = mt_signal(x_data, *params)
    #     return np.sum((m_data - e_data) ** 2.0)
    #
    # res = scipy.optimize.minimize(
    #     sum_of_squares, p0, args=(x_data, y_data), method='L-BFGS-B',
    #     bounds=bounds, options={'gtol': 1e-05, 'ftol': 2e-09})
    # print(res.x, res.success, res.message)
    #
    # fitted = mt_signal(x_data, *res.x).reshape(measured.shape)

    # # faked fitted
    # fitted = mt_signal(x_data, *p0).reshape(measured.shape)

    if plot_data:
        X, Y = np.meshgrid(flip_angles, fc.util.sgnlog(freqs, 10.0))
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlabel('Pulse Amplitude (flip angle) / deg')
        ax.set_ylabel('Frequency offset / Hz (log10 scale)')
        ax.set_zlabel('Signal Intensity / arb. units')
        ax.plot_surface(
            X, Y, exact, cmap=plt.cm.hot,
            rstride=1, cstride=1, linewidth=0.005, antialiased=False)
        # ax.plot_surface(
        #     X, Y, measured, cmap=plt.cm.ocean,
        #     rstride=1, cstride=1, linewidth=0.01, antialiased=False)
        # ax.plot_surface(
        #     X, Y, fitted, cmap=plt.cm.bone,
        #     rstride=1, cstride=1, linewidth=0.01, antialiased=False)


# ======================================================================
if __name__ == '__main__':
    msg(__doc__.strip())
    # check_dynamics_operator_symbolic()
    # fc.util.elapsed('check_symbolic')

    # check_dynamics_operator()
    # fc.util.elapsed'check_dynamics_operator')

    # check_mt_sequence()
    # fc.util.elapsed'check_mt_sequence')

    # check_approx_propagator()
    # fc.util.elapsed'check_approx_propagator')

    # check_z_spectrum(
    #     SpinModel(100.0, (0.5, 0.3, 0.1, 0.1), (GAMMA['1H'] * B0,) * 4,
    #               (0.25, 0.8, 0.001, 1.0), (20.0, 60.0, 8e4, 5e4),
    #               (1.0, 0.3, 0.0, 1.0, 0.5, 1.0),
    #               (None, None, 'superlorenz_approx', 'superlorenz_approx')))
    x1 = check_z_spectrum()
    elapsed('check_z_spectrum')
    x2 = check_z_spectrum2()
    elapsed('check_z_spectrum2')
    x3 = check_z_spectrum_sparse()
    elapsed('check_z_spectrum_sparse')
    # print(x2[0].ravel() / x1[0].ravel())
    # print(x3[0].ravel() / x1[0].ravel())

    # check_fit_spin_model()
    # fc.util.elapsed('check_fit_spin_model')

    msg(report())
    # profile.run('check_z_spectrum()', sort=1)
    plt.show()
