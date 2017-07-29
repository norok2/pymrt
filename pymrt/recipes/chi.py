#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
pymrt.recipes.chi: chi (relative) magnetic susceptibility computation.
"""

# ======================================================================
# :: Future Imports
from __future__ import (
    division, absolute_import, print_function, unicode_literals)

# ======================================================================
# :: Python Standard Library Imports
# import itertools  # Functions creating iterators for efficient looping
import warnings  # Warning control
import collections  # Container datatypes

# :: External Imports
import numpy as np  # NumPy (multidimensional numerical arrays library)

# :: Local Imports
import pymrt as mrt

# from pymrt import VERB_LVL, D_VERB_LVL, VERB_LVL_NAMES
from pymrt import elapsed, print_elapsed
from pymrt import msg, dbg

import pymrt.utils
import pymrt.geometry
import pymrt.segmentation

from pymrt.config import _B0
from pymrt.recipes import db0, phs, generic


# ======================================================================
def dipole_kernel(
        shape,
        origin=0.5,
        b0_direction=(0, 0, 1),
        theta=0.0,
        phi=0.0):
    """
    Generate the 3D dipole kernel in the Fourier domain.

    .. math::
    
        C=\\frac{1}{3}-\\frac{(\\vec{k} \\cdot \\vec{B}_0)^2}{\\vec{k}^2}

    or:

    .. math::

        C=\\frac{1}{3}-\\frac{k_z \\cos(\\theta)\\cos(\\phi)
        -k_y\\sin(\\theta) \\cos(\\phi)+k_x\\sin(\\phi))^2}
        {k_x^2 + k_y^2 + k_z^2}

    Args:
        shape (iterable[int]): 3D-shape of the dipole kernel array.
            If not a 3D array, the function fails.
        origin (float|tuple[float]): Relative position of the origin.
            Values are in the [0, 1] interval.
        b0_direction (np.ndarray|None): The direction of the magnetic field B0.
            Must be either a 3D unit vector (versor) or None.
            If None, angular parameters `theta` and `phi` are used.
        theta (int|float): Angle of 1st rotation (along x-axis) in deg.
            If `b0_direction` is not None, this parameter is ignored.
            Equivalent to the projection in the yz-plane of the angle between
            the main magnetic field B0 (i.e. the principal axis of the dipole
            kernel) and the z-axis. If phi is 0, the projection simplifies to
            the identity.
        phi (int|float): Angle of 2nd rotation (along y-axis) in deg.
            If `b0_direction` is not None, this parameter is ignored.
            Equivalent to the projection in the xz-plane of the angle between
            the main magnetic field B0 (i.e. the principal axis of the dipole
            kernel) and the z-axis. If theta is 0, the projection simplifies to
            the identity.

    Returns:
        dk_arr (np.ndarray): The dipole kernel in the Fourier domain.
            Values are in the (1/3, -2/3) range.

    Examples:
        >>> dk1 = dipole_kernel([5, 5, 5], 0.5, np.array([0, 0, 1]))
        >>> dk2 = dipole_kernel([5, 5, 5], 0.5, None, 0.0, 0.0)
        >>> np.isclose(np.sum(np.abs(dk1 - dk2)), 0)
        True
        >>> dk1 = dipole_kernel([5, 5, 5], 0.5, np.array([0, 1, 0]))
        >>> dk2 = dipole_kernel([5, 5, 5], 0.5, None, 90.0, 0.0)
        >>> np.isclose(np.sum(np.abs(dk1 - dk2)), 0)
        True
        >>> dk1 = dipole_kernel([5, 5, 5], 0.5, np.array([1, 0, 0]))
        >>> dk2 = dipole_kernel([5, 5, 5], 0.5, None, 0.0, 90.0)
        >>> np.isclose(np.sum(np.abs(dk1 - dk2)), 0)
        True
    """
    #     / 1   (k . B0)^2 \
    # C = | - - ---------- |
    #     \ 3      k^2     /
    # generate the dipole kernel
    assert (len(shape) == 3)
    kk = np.array(mrt.utils.grid_coord(shape, origin))
    if b0_direction is None:
        theta, phi = [np.deg2rad(angle) for angle in (theta, phi)]
        b0_direction = [
            np.sin(phi),
            -np.sin(theta) * np.cos(phi),
            np.cos(theta) * np.cos(phi)]
    b0_direction = np.array(b0_direction) / np.sum(b0_direction)
    with np.errstate(divide='ignore', invalid='ignore'):
        dk_arr = (1.0 / 3.0 - (np.dot(kk, b0_direction)) ** 2 / np.dot(kk, kk))
    # fix singularity at |k|^2 == 0 in the denominator
    singularity = np.isnan(dk_arr)
    dk_arr[singularity] = 1.0 / 3.0
    return dk_arr


# ======================================================================
def ppm_to_ppb(arr):
    """
    Convert input array from ppm to ppb.
    
    Args:
        arr (np.ndarray): The input array.

    Returns:
        arr (np.ndarray): The output array.
        
    Examples:
        >>> arr = np.arange(8)
        >>> ppm_to_ppb(arr)
        array([    0.,  1000.,  2000.,  3000.,  4000.,  5000.,  6000.,  7000.])
        >>> all(arr == ppm_to_ppb(ppb_to_ppm(arr)))
        True
    """
    return arr * 1e3


# ======================================================================
def ppb_to_ppm(arr):
    """
    Convert input array from ppb to ppm.
    
    Args:
        arr (np.ndarray): The input array.

    Returns:
        arr (np.ndarray): The output array.
        
    Examples:
        >>> arr = np.arange(8)
        >>> ppb_to_ppm(arr)
        array([ 0.   ,  0.001,  0.002,  0.003,  0.004,  0.005,  0.006,  0.007])
        >>> all(arr == ppb_to_ppm(ppm_to_ppb(arr)))
        True
    """
    return arr * 1e-3


# ======================================================================
def chi_to_db0(
        chi_arr,
        b0_direction=(0, 0, 1),
        theta=0.0,
        phi=0.0):
    """
    Convert magnetic susceptibility to magnetic field variation.

    Args:
        chi_arr (np.ndarray): The magnetic susceptibility in ppb.
        b0_direction (np.ndarray|None): The direction of the magnetic field B0.
            Must be either a 3D unit vector (versor) or None.
            If None, angular parameters `theta` and `phi` are used.
        theta (int|float): Angle of 1st rotation (along x-axis) in deg.
            If `b0_direction` is not None, this parameter is ignored.
            Equivalent to the projection in the yz-plane of the angle between
            the main magnetic field B0 (i.e. the principal axis of the dipole
            kernel) and the z-axis. If phi is 0, the projection simplifies to
            the identity.
        phi (int|float): Angle of 2nd rotation (along y-axis) in deg.
            If `b0_direction` is not None, this parameter is ignored.
            Equivalent to the projection in the xz-plane of the angle between
            the main magnetic field B0 (i.e. the principal axis of the dipole
            kernel) and the z-axis. If theta is 0, the projection simplifies to
            the identity.

    Returns:
        db0_arr (np.ndarray): The magnetic field variation in ppb.
    """
    dk = dipole_kernel(
        chi_arr.shape, b0_direction=b0_direction, theta=theta, phi=phi)
    return np.real(mrt.utils.idftn(mrt.utils.dftn(chi_arr) * dk))


# ======================================================================
def db0_to_chi(
        db0_arr,
        mask_arr=None,
        threshold=np.spacing(1),
        b0_direction=(0, 0, 1),
        theta=0.0,
        phi=0.0):
    """
    Convert magnetic field variation to magnetic susceptibility.

    This implements the so-called Threshold-based K-space Division (TKD).
    This requires that magnetic susceptibility sources are within the region
    where magnetic field variation is known.

    Args:
        db0_arr (np.ndarray): The magnetic field variation in ppb.
        mask_arr (np.ndarray): The boolean mask array.
        threshold (float): The deconvolution threshold.
            Effectively excludes the dipole kernel zeros from the division.
            The dipole kernel values range is (1/3, -2/3).
        b0_direction (np.ndarray|None): The direction of the magnetic field B0.
            Must be either a 3D unit vector (versor) or None.
            If None, angular parameters `theta` and `phi` are used.
        theta (int|float): Angle of 1st rotation (along x-axis) in deg.
            If `b0_direction` is not None, this parameter is ignored.
            Equivalent to the projection in the yz-plane of the angle between
            the main magnetic field B0 (i.e. the principal axis of the dipole
            kernel) and the z-axis. If phi is 0, the projection simplifies to
            the identity.
        phi (int|float): Angle of 2nd rotation (along y-axis) in deg.
            If `b0_direction` is not None, this parameter is ignored.
            Equivalent to the projection in the xz-plane of the angle between
            the main magnetic field B0 (i.e. the principal axis of the dipole
            kernel) and the z-axis. If theta is 0, the projection simplifies to
            the identity.

    Returns:
        chi_arr (np.ndarray): The magnetic susceptibility in ppb.
    """
    if mask_arr is not None:
        db0_arr[~mask_arr] = 0.0
    dk = dipole_kernel(
        db0_arr.shape, b0_direction=b0_direction, theta=theta, phi=phi)
    # threshold the zeros of the dipole kernel
    if threshold:
        mask = np.abs(dk) < threshold
        dk[mask] = threshold
    chi_k_arr = mrt.utils.dftn(db0_arr) / dk
    # remove singularity of susceptibility
    chi_k_arr = mrt.utils.subst(chi_k_arr)
    chi_arr = np.real(mrt.utils.idftn(chi_k_arr))
    return chi_arr


# ======================================================================
def qsm_sharp(
        dphs_arr,
        mask_arr,
        radius=0.01,
        threshold=np.spacing(1)):
    """
    Filter out the non-harmonic components of the phase.

    Args:
        dphs_arr (np.ndarray): The input unwrapped phase in rad.
        mask_arr (np.ndarray): The inner-volume mask.
        radius (float): The radius of the kernel sphere.
        threshold (float): The deconvolution threshold.

    Returns:
        phs_arr (np.ndarray): The SHARP filtered phase.
        
    See Also:
        - Schweser, F., Deistung, A., Lehr, B.W., Reichenbach, J.R., 
          2011. Quantitative imaging of intrinsic magnetic tissue properties 
          using MRI signal phase: An approach to in vivo brain iron metabolism? 
          NeuroImage 54, 2789–2807. doi:10.1016/j.neuroimage.2010.10.070
        - Schweser, F., Robinson, S.D., de Rochefort, L., Li, W., Bredies, K., 
          2016. An illustrated comparison of processing methods for phase MRI 
          and QSM: removal of background field contributions from sources 
          outside the region of interest. NMR Biomed. n/a-n/a. 
          doi:10.1002/nbm.3604

    """
    # todo: if radius is float -> relative, if int -> absolute

    sphere = mrt.geometry.sphere(dphs_arr.shape, 0.5, radius).astype(float)
    sphere /= np.sum(sphere)

    dphs_arr = mask_arr * \
               (dphs_arr - mrt.utils.idftn(
                   mrt.utils.dftn(dphs_arr) * mrt.utils.dftn(sphere)))

    sphere_dirac_k = mrt.utils.dftn(
        mrt.geometry.dirac_delta(dphs_arr.shape, 0.5, 1) - sphere)
    mask = np.abs(sphere_dirac_k) > threshold
    sphere_dirac_k[mask] = 1 / sphere_dirac_k[mask]
    sphere_dirac_k[~mask] = 0

    dphs_arr = mask_arr * \
               (dphs_arr - mrt.utils.idftn(
                   mrt.utils.dftn(dphs_arr) * sphere_dirac_k))

    return dphs_arr


# ======================================================================
def qsm_v_sharp(
        db0_arr,
        mask_arr,
        radius=0.01,
        threshold=np.spacing(1)):
    """
    Filter out the non-harmonic components of the magnetic field variation.

    Args:
        db0_arr (np.ndarray): The input unwrapped phase in rad.
        mask_arr (np.ndarray): The inner-volume mask.
        radius (float): The radius of the kernel sphere.
        threshold (float): The deconvolution threshold.        

    Returns:
        phs_arr (np.ndarray): The SHARP filtered phase.
    """
    # todo: implement V-SHARP
    sphere = mrt.geometry.sphere(db0_arr.shape, 0.5, radius).astype(float)
    sphere /= np.sum(sphere)

    db0_arr = mask_arr * \
              (db0_arr - mrt.utils.idftn(
                  mrt.utils.dftn(db0_arr) * mrt.utils.dftn(sphere)))

    inv_sphere_dirac_k = \
        1 / mrt.utils.dftn(mrt.geometry.dirac_delta(db0_arr.shape, 0.5, 1) - sphere)
    inv_sphere_dirac_k[np.abs(inv_sphere_dirac_k) < (1 / threshold)] = 0.0

    db0_arr = mask_arr * \
              (db0_arr - mrt.utils.idftn(
                  mrt.utils.dftn(db0_arr) * inv_sphere_dirac_k))

    return db0_arr


# ======================================================================
def qsm_sdi_from_dphs(
        dphs_arr,
        mask_arr,
        sharp_radius,
        sharp_threshold,
        tkd_threshold,
        b0_direction=(0, 0, 1),
        theta=0.0,
        phi=0.0,
        b0=_B0):
    chi_arr = db0_to_chi(
        qsm_sharp(
            db0.dphs_to_db0(dphs_arr, b0=b0),
            mask_arr=mask_arr, radius=sharp_radius, threshold=sharp_threshold),
        mask_arr=mask_arr, threshold=tkd_threshold, b0_direction=b0_direction,
        theta=theta, phi=phi)
    return chi_arr


# ======================================================================
def qsm_preprocess(mag_arr, phs_arr, echo_times, echo_times_mask=None):
    echo_times = np.array(mrt.utils.auto_repeat(echo_times, 1))
    if len(echo_times) > 1:
        dphs_arr = phs.phs_to_dphs(
            phs_arr, tis=echo_times, tis_mask=echo_times_mask)
        mag_arr = mag_arr[..., 0]
    else:
        dphs_arr = phs.single_phs_to_dphs(phs_arr, echo_times[0])
    mask_arr = mrt.segmentation.mask_threshold_compact(mag_arr)
    return dphs_arr, mask_arr


# ======================================================================
def qsm_sdi():
    pass


# ======================================================================
def _qsm_test(
        phs_arr,
        mask_arr,
        threshold,
        theta,
        phi,
        radius=0.01):
    # one-step QSM.. need for bias field removal?
    # convert input angles to radians
    # theta = np.deg2rad(theta)
    # phi = np.deg2rad(phi)
    #
    # k_x, k_y, k_z = coord(arr_cx.shape)
    # k_2 = (k_x ** 2 + k_y ** 2 + k_z ** 2)
    # cc = (k_z * np.cos(theta) * np.cos(phi) -
    #       k_y * np.sin(theta) * np.cos(phi) +
    #       k_x * np.sin(phi)) ** 2
    # dd = 1 / (k_2 - cc)
    # dd = subst(dd)
    # chi_arr = np.abs(idftn(3 * k_2 * dd * dftn(phs_arr)))
    pass


# ======================================================================
# : aliases for common QSM nomenclature
qsm_tkd = db0_to_chi


# ======================================================================
def _test(use_cache=True):
    import pymrt.utils
    import os
    base_dir = mrt.utils.realpath('~/hd1/TEMP')
    filepath = os.path.join(base_dir, 'tau_arr.npz')

    # test dipole kernel
    k1 = dipole_kernel([5, 5, 5], 0.5, (1, 0, 0))
    k2 = dipole_kernel_angle([5, 5, 5], 0.5, 0, 90)
    print(np.sum(np.abs(k1 - k2)))

    print_elapsed()


_test()

# ======================================================================
if __name__ == '__main__':
    import doctest

    msg(__doc__.strip())
    doctest.testmod()

else:
    elapsed()
