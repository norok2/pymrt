#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
pymrt.recipes.chi: generic basic utilities.
"""

import numpy as np

import pymrt.utils as pmu
import pymrt.geometry as pmg

from pymrt.constants import GAMMA, GAMMA_BAR


# ======================================================================
def dipole_kernel(
        shape,
        origin=0.5,
        theta=0.0,
        phi=0.0):
    """
    Generate the 3D dipole kernel in the Fourier domain.

    .. math::

        C=\\frac{1}{3}-\\frac{k_z \\cos(\\theta)\\cos(\\phi)
        -k_y\\sin(\\theta) \\cos(\\phi)+k_x\\sin(\\phi))^2}
        {k_x^2 + k_y^2 + k_z^2}

    Args:
        shape (tuple[int]): 3D-shape of the dipole kernel array.
            If not a 3D array, the function fails.
        origin (float|tuple[float]): Relative position of the origin.
            Values are in the [0, 1] interval.
        theta (int|float): Angle of 1st rotation (along x-axis) in deg.
            Equivalent to the projection in the yz-plane of the angle between
            the main magnetic field B0 (i.e. the principal axis of the dipole
            kernel) and the z-axis. If phi is 0, the projection simplifies to
            the identity.
        phi (int|float): Angle of 2nd rotation (along y-axis) in deg.
            Equivalent to the projection in the xz-plane of the angle between
            the main magnetic field B0 (i.e. the principal axis of the dipole
            kernel) and the z-axis. If theta is 0, the projection simplifies to
            the identity.

    Returns:
        dk_arr (np.ndarray): The dipole kernel in the Fourier domain.
    """
    #     / 1   (kz cos(th) cos(ph) - ky sin(th) cos(ph) + kx sin(ph))^2 \
    # C = | - - -------------------------------------------------------- |
    #     \ 3                      kx^2 + ky^2 + kz^2                    /
    # convert input angles to radians
    theta = np.deg2rad(theta)
    phi = np.deg2rad(phi)
    # generate the dipole kernel
    assert (len(shape) == 3)
    kk_x, kk_y, kk_z = pmu.coord(shape, origin)
    dk_arr = (1.0 / 3.0 - (
        kk_z * np.cos(theta) * np.cos(phi) - kk_y * np.sin(theta) * np.cos(phi)
        + kk_x * np.sin(phi)) ** 2 / (kk_x ** 2 + kk_y ** 2 + kk_z ** 2))
    # fix singularity at |k|^2 == 0 in the denominator
    singularity = np.isnan(dk_arr)
    dk_arr[singularity] = 1.0 / 3.0
    return dk_arr


# ======================================================================
def phase_to_field(
        phs_arr,
        b0=3.0,
        te=20.0,
        units='ppb'):
    """
    Convert a phase array to relative magnetic field variations.

    Args:
        phs_arr (np.ndarray): The input unwrapped phase array.
        b0 (float): Main Magnetic Field B0 Strength in T
        te (float): Echo Time in ms
        units (str|float): Relative units of the field array.
            Supported string values are:
                - 'ppm': parts-per-million: 1e-6
                - 'ppb': parts-per-billion: 1e-9
            Otherwise, a numerical factor can be used.

    Returns:
        db0_arr (np.ndarray): The relative field array in pure numbers.
            A scale factor is used, depending on the value of `units`.
    """
    if units == 'ppm':
        units_factor = 1.0e-6
    elif units == 'ppb':
        units_factor = 1.0e-9
    elif isinstance(units, int) or isinstance(units, float):
        units_factor = units
    else:
        raise ValueError(units + ': unknown output units')
    db0_arr = phs_arr / (GAMMA['1H'] * b0 * (te * 1e-3) * units_factor)
    return db0_arr


# ======================================================================
def qsm_sharp(
        phs_arr,
        mask,
        threshold,
        radius):
    """
    Filter out the non-harmonic components of the phase.

    Args:
        phs_arr (np.ndarray): The unwrapped phase to deconvolve.
        mask (np.ndarray): The tissue mask.
        threshold (float): The deconvolution threshold.
        radius (float): The radius of the kernel sphere.

    Returns:
        arr (np.ndarray):
    """
    # % generate spherical mask in real space
    # dims=size(mask);
    # [sk,cc] = getsphere(dims,0,0,0,rad);
    # sk = sk/(sum(sk(:)));
    # sk2 = sk;
    # sk2(cc(1),cc(2),cc(3))=sk2(cc(1),cc(2),cc(3))-1;
    # sk2 = -sk2;
    spheric_mask = pmg.sphere(phs_arr.shape, 0.5, )

    # %generate k-space convolution and deconvolution filters
    # smk = fftn(sk);
    # dek = fftn(sk2);
    #
    # % convolve phase with sphere
    # conv_f = fftshift(real(ifftn(fftn(phase).*smk)));
    # clear smk
    #
    # % subtract off smv
    # conv_fm = (phase-conv_f).*mask;
    # clear phase
    #
    # %deconvolve
    # quotient = 1./dek;
    # clear dek
    # quotient(abs(quotient)>(1/TSVD_TH))=0;
    # pfilt = (fftshift(real(ifftn(fftn(conv_fm).*(quotient))))).*mask;


# ======================================================================
def field_to_source(
        db0_arr,
        mask_arr=None,
        threshold=1e-2,
        mode='absolute',
        theta=0.0,
        phi=0.0):
    """

    Args:
        db0_arr (np.ndarray): The B0 field array in ppb.
        mask_arr (np.ndarray): The boolean mask array.
        threshold (float): The threshold for excluding the dipole kernel zeros.
        mode (str): Determines how to interpret / process the threshold value.
            Available values are:
             - 'absolute': use the absolute value
             - 'relative': use a value relative to values interval
             - 'percentile': use the value obtained from the percentiles
        theta (int|float): Angle of 1st rotation (along x-axis) in deg.
            Equivalent to the projection in the yz-plane of the angle between
            the main magnetic field B0 (i.e. the principal axis of the dipole
            kernel) and the z-axis. If phi is 0, the projection simplifies to
            the identity.
        phi (int|float): Angle of 2nd rotation (along y-axis) in deg.
            Equivalent to the projection in the xz-plane of the angle between
            the main magnetic field B0 (i.e. the principal axis of the dipole
            kernel) and the z-axis. If theta is 0, the projection simplifies to
            the identity.

    Returns:

    """
    if mask_arr is not None:
        db0_arr[~mask_arr] = 0.0
    theta = np.deg2rad(theta)
    phi = np.deg2rad(phi)
    dk = dipole_kernel(db0_arr.shape, theta=theta, phi=phi)
    db0_k_arr = pmu.dftn(db0_arr)
    # threshold the zeros of the dipole kernel
    if threshold:
        mask = np.abs(dk) < threshold
        dk[mask] = threshold
    chi_k_arr = db0_k_arr / dk
    # remove singularity of susceptibility
    chi_k_arr = pmu.subst(chi_k_arr)
    chi_arr = np.real(pmu.idftn(chi_k_arr))
    return chi_arr


# ======================================================================
def source_to_field(
        chi_arr,
        threshold=1e-2,
        theta=0.0,
        phi=0.0):
    """

    Args:
        chi_arr (np.ndarray): The magnetic susceptibility array in ppb.
        mask_arr (np.ndarray): The boolean mask array.
        threshold (float): The threshold for excluding the dipole kernel zeros.
            See the values range for the dipole kernel.
        theta (int|float): Angle of 1st rotation (along x-axis) in deg.
            Equivalent to the projection in the yz-plane of the angle between
            the main magnetic field B0 (i.e. the principal axis of the dipole
            kernel) and the z-axis. If phi is 0, the projection simplifies to
            the identity.
        phi (int|float): Angle of 2nd rotation (along y-axis) in deg.
            Equivalent to the projection in the xz-plane of the angle between
            the main magnetic field B0 (i.e. the principal axis of the dipole
            kernel) and the z-axis. If theta is 0, the projection simplifies to
            the identity.

    Returns:

    """
    theta = np.deg2rad(theta)
    phi = np.deg2rad(phi)
    dk = dipole_kernel(chi_arr.shape, theta=theta, phi=phi)
    chi_k_arr = pmu.dftn(chi_arr)
    db0_k_arr = chi_k_arr * dk
    db0_arr = np.real(pmu.idftn(db0_k_arr))
    return db0_arr


'''
    # one-step QSM.. need for bias field removal?
    # convert input angles to radians
    theta = np.deg2rad(theta)
    phi = np.deg2rad(phi)

    k_x, k_y, k_z = coord(arr_cx.shape)
    k_2 = (k_x ** 2 + k_y ** 2 + k_z ** 2)
    cc = (k_z * np.cos(theta) * np.cos(phi) -
          k_y * np.sin(theta) * np.cos(phi) +
          k_x * np.sin(phi)) ** 2
    dd = 1 / (k_2 - cc)
    dd = subst(dd)
    chi_arr = np.abs(idftn(3 * k_2 * dd * dftn(phs_arr)))
'''
