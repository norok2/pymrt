import numpy as np

import pymrt.utils as pmu
from pymrt.computation import fix_phase_interval



# ======================================================================
def rho_mp2rage(
        inv1m_arr,
        inv1p_arr,
        inv2m_arr,
        inv2p_arr,
        regularization=np.spacing(1),
        values_interval=None):
    """
    Calculate the rho image from an MP2RAGE acquisition.

    Args:
        inv1m_arr (float|np.ndarray): Magnitude of the first inversion image.
        inv1p_arr (float|np.ndarray): Phase of the first inversion image.
        inv2m_arr (float|np.ndarray): Magnitude of the second inversion image.
        inv2p_arr (float|np.ndarray): Phase of the second inversion image.
        regularization (float|int): Parameter for the regularization.
            This parameter is added to the denominator of the rho expression
            for normalization purposes, therefore should be much smaller than
            the average of the magnitude images.
            Larger values of this parameter will have the side effect of
            denoising the background.
        values_interval (tuple[float|int]|None): The output values interval.
            The standard values are linearly converted to this range.
            If None, the natural [-0.5, 0.5] interval will be used.

    Returns:
        rho_arr (float|np.ndarray): The calculated rho image from
        MP2RAGE.
    """
    if not regularization:
        regularization = 0
    inv1m_arr = inv1m_arr.astype(float)
    inv2m_arr = inv2m_arr.astype(float)
    inv1p_arr = fix_phase_interval(inv1p_arr)
    inv2p_arr = fix_phase_interval(inv2p_arr)
    inv1_arr = pmu.polar2complex(inv1m_arr, inv1p_arr)
    inv2_arr = pmu.polar2complex(inv2m_arr, inv2p_arr)
    rho_arr = np.real(inv1_arr.conj() * inv2_arr /
                      (inv1m_arr ** 2 + inv2m_arr ** 2 + regularization))
    if values_interval:
        rho_arr = pmu.scale(rho_arr, values_interval, (-0.5, 0.5))
    return rho_arr


# ======================================================================
def rho_to_t1_mp2rage(
        rho_arr,
        eff_arr=None,
        t1_values_range=(100, 5000),
        t1_num=512,
        eff_num=32,
        **acq_params_kws):
    """
    Calculate the T1 map from an MP2RAGE acquisition.

    Args:
        rho_arr (float|np.ndarray): Magnitude of the first inversion image.
        eff_arr (float|np.array|None): Efficiency of the RF pulse excitation.
            This is equivalent to the normalized B1T field.
            Note that this must have the same spatial dimensions as the images
            acquired with MP2RAGE.
            If None, no correction for the RF efficiency is performed.
        t1_values_range (tuple[float]): The T1 value range to consider.
            The format is (min, max) where min < max.
            Values should be positive.
        t1_num (int): The base number of sampling points of T1.
            The actual number of sampling points is usually smaller, because of
            the removal of non-bijective branches.
            This affects the precision of the MP2RAGE estimation.
        eff_num (int): The base number of sampling points for the RF efficiency.
            This affects the precision of the RF efficiency correction.
        **acq_params_kws (dict): The acquisition parameters.
            This should match the signature of:  `mp2rage.acq_to_seq_params`.

    Returns:
        t1_arr (float|np.ndarray): The calculated T1 map for MP2RAGE.
    """
    from pymrt.sequences import mp2rage
    if eff_arr:
        # todo: implement B1T correction
        raise NotImplementedError('B1T correction is not yet implemented')
    else:
        # determine the rho expression
        t1 = np.linspace(t1_values_range[0], t1_values_range[1], t1_num)
        rho = mp2rage.rho(
            t1, **mp2rage.acq_to_seq_params(**acq_params_kws)[0])
        # remove non-bijective branches
        bijective_slice = pmu.bijective_part(rho)
        t1 = t1[bijective_slice]
        rho = rho[bijective_slice]
        if rho[0] > rho[-1]:
            rho = rho[::-1]
            t1 = t1[::-1]
        # check that rho values are strictly increasing
        if not np.all(np.diff(rho) > 0):
            raise ValueError('MP2RAGE look-up table was not properly prepared.')

        # fix values range for rho
        if not pmu.is_in_range(rho_arr, mp2rage.RHO_INTERVAL):
            rho_arr = pmu.scale(rho_arr, mp2rage.RHO_INTERVAL)

        t1_arr = np.interp(rho_arr, rho, t1)
    return t1_arr


# ======================================================================
def t1_mp2rage(
        inv1m_arr,
        inv1p_arr,
        inv2m_arr,
        inv2p_arr,
        regularization=np.spacing(1),
        eff_arr=None,
        t1_values_range=(100, 5000),
        t1_num=512,
        eff_num=32,
        **acq_param_kws):
    """
    Calculate the T1 map from an MP2RAGE acquisition.

    Args:
        inv1m_arr (float|np.ndarray): Magnitude of the first inversion image.
        inv1p_arr (float|np.ndarray): Phase of the first inversion image.
        inv2m_arr (float|np.ndarray): Magnitude of the second inversion image.
        inv2p_arr (float|np.ndarray): Phase of the second inversion image.
        regularization (float|int): Parameter for the regularization.
            This parameter is added to the denominator of the rho expression
            for normalization purposes, therefore should be much smaller than
            the average of the magnitude images.
            Larger values of this parameter will have the side effect of
            denoising the background.
        eff_arr (float|np.array|None): Efficiency of the RF pulse excitation.
            This is equivalent to the normalized B1T field.
            Note that this must have the same spatial dimensions as the images
            acquired with MP2RAGE.
            If None, no correction for the RF efficiency is performed.
        t1_values_range (tuple[float]): The T1 value range to consider.
            The format is (min, max) where min < max.
            Values should be positive.
        t1_num (int): The base number of sampling points of T1.
            The actual number of sampling points is usually smaller, because of
            the removal of non-bijective branches.
            This affects the precision of the MP2RAGE estimation.
        eff_num (int): The base number of sampling points for the RF efficiency.
            This affects the precision of the RF efficiency correction.
        **acq_param_kws (dict): The acquisition parameters.
            This should match the signature of:  `mp2rage.acq_to_seq_params`.

    Returns:
        t1_arr (float|np.ndarray): The calculated T1 map for MP2RAGE.
    """
    rho_arr = rho_mp2rage(
        inv1m_arr, inv1p_arr, inv2m_arr, inv2p_arr, regularization,
        values_interval=None)
    t1_arr = rho_to_t1_mp2rage(
        rho_arr, eff_arr, t1_values_range, t1_num, eff_num, **acq_param_kws)
    return t1_arr
