#!/usr/bin/env python

# todo: fix me

# ======================================================================
def test_dynamics_operator_symbolic():
    """
    Notes: import pi, sin and cos from sympy

    Returns:
        None
    """
    # todo: make it flexible and working
    w_c, w1 = sym.symbols('w_c w1')
    m0 = [sym.symbols('m0{}'.format())]

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
    print(spin_model.k_op)
    print(spin_model.l_op)
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
    print(spin_model.k_op)
    print(spin_model.l_op)
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
    print(spin_model.k_op)
    print(spin_model.l_op)
    print(dynamics_operator(spin_model, w_c + 10.0, w1))


# ======================================================================
def test_dynamics_operator():
    """
    Notes: import pi, sin and cos from numpy

    Returns:

    """
    w_c = GAMMA * B0
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
    print(spin_model.k_op)
    print(spin_model.l_op)
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
    print(spin_model.k_op)
    print(spin_model.l_op)
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
    print(spin_model.k_op)
    print(spin_model.l_op)
    print(dynamics_operator(spin_model, w_c + 10.0, w1))


# ======================================================================
def test_mt_sequence():
    """
    Test for the MT sequence.
    """
    w_c = GAMMA * B0

    spin_model = SpinModel(
        s0=100,
        mc=(1.0, 0.152),
        w0=((w_c,) * 2),
        r1=(1.8, 1.0),
        r2=(32.2581, 8.4746e4),
        k=(0.3456,),
        approx=(None, 'superlorentz_approx'))

    num_repetitions = 300

    mt_flash_kernel = PulseList([
        Delay(10.0e-3),
        Spoiler(1.0),
        PulseExc.shaped(40.0e-3, 220.0, 4000, 'gauss', None,
                        w_c + 50.0, 'poly', {'fit_order': 5}),
        Delay(20.0e-3),
        Spoiler(1.0),
        PulseExc.shaped(10.0e-6, 90.0, 1, 'rect', None),
        Delay(30.0e-3)],
        b0=3.0)
    mt_flash = PulseTrain(mt_flash_kernel, num_repetitions)

    signal = mt_flash.signal(spin_model)

    print(mt_flash)
    print(mt_flash.propagator(spin_model))
    print(signal)


# ======================================================================
def test_approx_propagator(
        spin_model=SpinModel(
            s0=100,
            mc=(0.8681, 0.1319),
            w0=((GAMMA * B0,) * 2),
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
    w_c = GAMMA * B0

    modes = ['exact']
    modes += ['linear', 'reduced']
    # modes += ['sum_simple', 'sum_order1', 'sum_sep', 'reduced']
    modes += ['poly_{}'.format(order) for order in range(4, 5)]
    modes += ['interp_{}_{}'.format(mode, num_samples)
              for mode in ['linear', 'cubic'] for num_samples in range(4, 5)]
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
        # 'fermi': {},
        # 'random': {},
        'cos_sin': {},
    }
    exact_p_ops = {}
    for shape, shape_kwargs in shapes.items():
        pulse = PulseExc.shaped(
            40.0e-3, flip_angle, 4000, shape, shape_kwargs, w_c, 'exact', {})
        exact_p_ops[shape] = pulse.propagator(spin_model)

    for shape, shape_kwargs in shapes.items():
        for mode, mode_params in modes.items():
            kwargs_items = [{}]
            names = mode_params.keys()
            for values in itertools.product(*[mode_params[i] for i in names]):
                kwargs_items.append({k: v for k, v in zip(names, values)})
            for kwargs in kwargs_items:
                pulse = PulseExc.shaped(
                    40.0e-3, flip_angle, 4000, shape, shape_kwargs, w_c,
                    mode, kwargs)
                begin_time = datetime.datetime.now()
                p_op = pulse.propagator(spin_model)
                elapsed = datetime.timedelta(0, time.time() - begin_time)
                rel_error = np.sum(np.abs(exact_p_ops[shape] - p_op)) / \
                            np.sum(np.abs(exact_p_ops[shape]))
                print('{:>8s}, {:>8s}, {:>48s},\t{:.3e}, {}'.format(
                    shape, mode, str(kwargs), rel_error, elapsed))


# ======================================================================
def test_z_spectrum(
        spin_model=SpinModel(
            s0=100,
            mc=(0.8681, 0.1319),
            w0=((GAMMA * B0,) * 2),
            r1=(1.8, 1.0),
            r2=(32.2581, 8.4746e4),
            k=(0.3456,),
            approx=(None, 'superlorentz_approx')),
        freqs=np.round(mrb.sgnlogspace(50, 50000, 16)),
        amplitudes=np.round(mrb.sgnlogspace(50, 5000, 16)),
        plot_data=True,
        save_file=None):
    """
    Test calculation of z-spectra

    Args:

        spin_model (SpinModel):
        freqs (ndarray[float]):
        amplitudes (ndarray[float]):
        plot_data (bool):
        save_file (string):

    Returns:
        freq

    """
    w_c = spin_model.w0[0]

    flip_angles = amplitudes * 11.799 / 50.0

    mt_flash = MtFlash(
        PulseList([
            Delay(10.0e-3),
            Spoiler(1.0),
            PulseExc.shaped(10.0e-3, 90.0, 4000, 'gauss', {},
                            0.0, 'poly', {'fit_order': 3}),
            Delay(10.0e-3),
            Spoiler(1.0),
            PulseExc.shaped(2.1e-3, 11.0, 1, 'rect', {})],
            w_c=w_c),
        300)

    data = np.zeros((len(freqs), len(amplitudes)))
    for j, freq in enumerate(freqs):
        for i, flip_angle in enumerate(flip_angles):
            mt_flash.set_flip_angle(flip_angle)
            mt_flash.set_freq(freq)
            data[j, i] = mt_flash.signal(spin_model)

    # plot results
    if plot_data:
        X, Y = np.meshgrid(amplitudes, np.log10(freqs))
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(
            X, Y, data, cmap=plt.cm.hot,
            rstride=1, cstride=1, linewidth=0.01, antialiased=False)
    if save_file:
        np.savez(save_file, freqs, amplitudes, data)
    return data, freqs, flip_angles


# ======================================================================
def test_fit_spin_model(
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
    w_c = GAMMA * B0

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
    mt_flash = MtFlash(
        PulseList([
            Delay(
                t_r - (t_e + 3 * 160.0e-6 + 20000.0e-6 + 970.0e-6 + 100e-6)),
            Spoiler(0.0),
            Delay(160.0e-6),
            PulseExc.shaped(
                20000.0e-6, 90.0, 0, '_from_GAUSS5120', {}, 0.0,
                'linear', {'num_samples': 15}),
            Delay(160.0e-6 + 970.0e-6),
            Spoiler(1.0),
            Delay(160.0e-6),
            PulseExc.shaped(100e-6, 30.0, 1, 'rect', {}),
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
    freqs = mrb.sgnlogspace(100.0, 300.0e3, 32)
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
        X, Y = np.meshgrid(flip_angles, mrb.sgnlog(freqs, 10.0))
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlabel('Pulse Amplitude (flip angle) / deg')
        ax.set_ylabel('Frequency offset / Hz (log10 scale)')
        ax.set_zlabel('Signal Intensity / arb.units')
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
    # test_dynamics_operator_symbolic()
    # mrb.elapsed'test_symbolic')
    # test_dynamics_operator()
    # mrb.elapsed'test_dynamics_operator')
    # test_mt_sequence()
    # mrb.elapsed'test_mt_sequence')
    # test_approx_propagator()
    # mrb.elapsed'test_approx_propagator')
    # test_z_spectrum(
    #     SpinModel(100.0, (0.5, 0.3, 0.1, 0.1), (GAMMA * B0,) * 4,
    #               (0.25, 0.8, 0.001, 1.0), (20.0, 60.0, 8e4, 5e4),
    #               (1.0, 0.3, 0.0, 1.0, 0.5, 1.0),
    #               (None, None, 'superlorenz_approx', 'superlorenz_approx')))
    test_z_spectrum()
    elapsed('test_z_spectrum')
    # test_fit_spin_model()
    # mrb.elapsed('test_fit_spin_model')

    print_elapsed()
    # profile.run('test_z_spectrum()', sort=1)
    plt.show()
