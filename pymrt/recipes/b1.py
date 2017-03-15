# ======================================================================
def calc_afi(
        images,
        affines,
        params,
        ti_label,
        fa_label,
        img_types):
    """
    Fit monoexponential decay to images using the log-linear method.
    """
    y_arr = np.stack(images, -1).astype(float)

    s_arr = pmu.polar2complex(y_arr[..., 0], fix_phase_interval(y_arr[..., 1]))
    # s_arr = images[0]
    t_r = params[ti_label]
    nominal_fa = params[fa_label]

    mask = s_arr[..., 0] != 0.0
    r = np.zeros_like(s_arr[..., 1])
    r[mask] = s_arr[..., 0][mask] / s_arr[..., 1][mask]
    n = t_r[1] / t_r[0]  # usually: t_r[1] > t_r[0]
    fa = np.rad2deg(np.real(np.arccos((r * n - 1) / (n - r))))

    img_list = [fa / nominal_fa]
    aff_list = _simple_affines(affines)
    type_list = ['eff']
    img_type_list = tuple(img_types[key] for key in type_list)
    params_list = ({},) * len(img_list)
    return img_list, aff_list, img_type_list, params_list
