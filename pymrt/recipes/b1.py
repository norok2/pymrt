import numpy as np


# ======================================================================
def actual_flip_angle(
        arr,
        repetition_times,
        nominal_flip_angle,
        zero_cutoff=np.spacing(1)):
    """
    Calculate the flip angle efficiency from `Actual Flip Angle` (AFI) data.

    The efficiency factor :math:`\eta_{\\alpha}` is defined by:

    .. math:
        \eta_{\\alpha} =
        \\frac{\\alpha_{\mathrm{meas.}}}{\\alpha_{\mathrm{nom.}}}

    Args:
        arr (np.ndarray): The input array in arb.units.
            The repetition time T_R varies in the last dimension.
            Only the first two T_R images are used.
            If no phase data is provided, then the maximum measured flip angle
            must be 90°, otherwise a flip angle efficiency above 1 cannot be
            measured.
        repetition_times (iterable[int|float]): The repetition times T_R in ms.
            The number of points must match the last shape size of the data
            mag_arr (and phs_arr if not None).
            Only the first two T_R values are used.
            The measuring units are irrelevant as only the ratio is used.
            It assumes that values are sorted increasingly.
        nominal_flip_angle (int|float): The nominal flip angle in deg.
        zero_cutoff (float|None): The threshold value for masking zero values.

    Returns
        results (dict): The calculated information.
            `fa` contains the measured flip angle map.
            `eff` contains the flip angle efficiency factor.
    """
    valid = np.abs(arr[..., 0]) > zero_cutoff
    valid *= np.abs(arr[..., 1]) > zero_cutoff

    ratio_arr = np.zeros_like(arr[..., 1])
    ratio_arr[valid] = arr[..., 0][valid] / arr[..., 1][valid]

    tr_ratio = repetition_times[1] / repetition_times[0]

    with np.errstate(divide='ignore', invalid='ignore'):
        flip_angle_arr = np.rad2deg(np.real(np.arccos(
            (ratio_arr * tr_ratio - 1) / (tr_ratio - ratio_arr))))

    result = {
        'fa': flip_angle_arr,
        'eff': flip_angle_arr / nominal_flip_angle}

    return result
