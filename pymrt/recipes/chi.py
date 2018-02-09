#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pymrt.recipes.chi: chi (relative) magnetic susceptibility computation.
"""

# ======================================================================
# :: Future Imports
from __future__ import (
    division, absolute_import, print_function, unicode_literals, )

# ======================================================================
# :: Python Standard Library Imports
# import itertools  # Functions creating iterators for efficient looping
import warnings  # Warning control
import collections  # Container datatypes

# :: External Imports
import numpy as np  # NumPy (multidimensional numerical arrays library)
import scipy as sp  # SciPy (signal and image processing library)

import scipy.sparse  # SciPy: Sparse Matrices
import scipy.sparse.linalg  # SciPy: Sparse Matrices - Linear Algebra

# :: Local Imports
import pymrt as mrt

# from pymrt import VERB_LVL, D_VERB_LVL, VERB_LVL_NAMES
from pymrt import elapsed, report
from pymrt import msg, dbg

import pymrt.utils
import pymrt.geometry
import pymrt.segmentation

from pymrt.recipes import db0, phs, generic

from pymrt.constants import CHI_V

from pymrt import VERB_LVL, D_VERB_LVL, VERB_LVL_NAMES
from pymrt import elapsed, report
from pymrt import msg, dbg


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
    
        C=\\frac{1}{3}-\\frac{(\\vec{k} \\cdot \\hat{B}_0)^2}{\\vec{k}^2}

    where :math:`\\hat{B}_0` is the unit vector identifying the direction of
    :math:`\\vec{B}_0`.

    or, in Cartesian coordinates:

    .. math::

        C=\\frac{1}{3}-\\frac{k_z \\cos(\\theta)\\cos(\\phi)
        -k_y\\sin(\\theta) \\cos(\\phi)+k_x\\sin(\\phi))^2}
        {k_x^2 + k_y^2 + k_z^2}

    Args:
        shape (Iterable[int]): 3D-shape of the dipole kernel array.
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
    #     / 1   (k . B0^)^2 \
    # C = | - - ----------- |
    #     \ 3       k^2     /

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
    # generate the dipole kernel
    dk = dipole_kernel(
        chi_arr.shape, b0_direction=b0_direction, theta=theta, phi=phi)
    dk = np.fft.fftshift(dk)

    return np.real(np.fft.ifftn(np.fft.fftn(chi_arr) * dk))


# ======================================================================
def db0_to_chi(
        db0_arr,
        mask_arr=None,
        threshold=2.0e-1,
        b0_direction=(0, 0, 1),
        theta=0.0,
        phi=0.0):
    """
    Convert magnetic field variation to magnetic susceptibility.

    This implements a variation of Threshold-based K-space Division (TKD),
    where the singularities of the dipole kernel are substituted with the
    threshold value rather than set to 0.
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
        db0_arr *= mask_arr

    # generate the dipole kernel
    dk = dipole_kernel(
        db0_arr.shape, b0_direction=b0_direction, theta=theta, phi=phi)
    dk = np.fft.fftshift(dk)

    # threshold the zeros of the dipole kernel
    if threshold:
        mask = np.abs(dk) < threshold
        dk[mask] = threshold

    # divide the magnetic field variation by the dipole kernel
    chi_k_arr = np.fft.fftn(db0_arr) / dk

    # remove singularity of susceptibility
    chi_k_arr = mrt.utils.subst(chi_k_arr)

    # perform the inverse Fourier transform
    chi_arr = np.real(np.fft.ifftn(chi_k_arr))

    if mask_arr is not None:
        chi_arr *= mask_arr

    return chi_arr


# ======================================================================
def qsm_remove_background_milf(
        db0_arr,
        mask_arr,
        threshold=np.spacing(1),
        pad_width=0):
    """
    Filter out the background component of the phase using MILF.

    MILF is the Masked Inverse Laplacian Filtering.

    Assumes that no sources are close to the boundary.

    Args:
        db0_arr (np.ndarray): The magnetic field variation in ppb.
        mask_arr (np.ndarray): The inner-volume mask.
        threshold (float): The deconvolution threshold.
        pad_width (float|int|Iterable[float|int]): Size of the padding to use.
            This is useful for mitigating border effects.
            If Iterable, a value for each dim must be specified.
            If not Iterable, all dims will have the same value.
            If int, it is interpreted as absolute size.
            If float, it is interpreted as relative to the maximum size.

    Returns:
        db0i_arr (np.ndarray): The internal magnetic field variation in ppb.

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
    if pad_width:
        shape = db0_arr.shape
        pad_width = mrt.utils.auto_pad_width(pad_width, shape)
        mask = [slice(lower, -upper) for (lower, upper) in pad_width]
        db0_arr = np.pad(db0_arr, pad_width, 'constant', constant_values=0)
        mask_arr = np.pad(mask_arr, pad_width, 'constant', constant_values=0)
    else:
        mask = [slice(None)] * db0_arr.ndim

    kernel_k = np.fft.fftshift(mrt.utils.laplace_kernel(db0_arr.shape))

    kernel_mask = np.abs(kernel_k) > threshold
    kernel_k_inv = kernel_mask.astype(complex)
    kernel_k_inv[kernel_mask] = 1.0 / kernel_k[kernel_mask]

    # ft_factor = (2 * np.pi)  # can be neglected because it cancels out
    db0_arr = mask_arr * (
        # ((1j * ft_factor) ** 2) *
        np.fft.ifftn(np.fft.fftn(db0_arr) * kernel_k))
    db0_arr = np.real(
        # ((1j / ft_factor) ** 2) *
        np.fft.ifftn(np.fft.fftn(db0_arr) * kernel_k_inv))

    return db0_arr[mask]


# ======================================================================
def qsm_remove_background_sharp(
        db0_arr,
        mask_arr,
        radius=range(11, 3, -2),
        threshold=np.spacing(1),
        pad_width=0.3,
        rel_radius=True):
    """
    Filter out the background component of the phase using SHARP.

    SHARP is the Sophisticated Harmonic Artifact Reduction for Phase.

    Assumes that no sources are close to the boundary of the mask.

    Args:
        db0_arr (np.ndarray): The magnetic field variation in ppb.
        mask_arr (np.ndarray): The inner-volume mask.
        radius (int|float): The radius of the kernel sphere.
            If `rel_radius` is False, the radius is in px.
            If `rel_radius` is True, the radius is relative to the largest
            dimension of `db0_arr`.
        threshold (float): The deconvolution threshold.
        pad_width (float|int|Iterable[float|int]): Size of the padding to use.
            This is useful for mitigating border effects.
            If Iterable, a value for each dim must be specified.
            If not Iterable, all dims will have the same value.
            If int, it is interpreted as absolute size.
            If float, it is interpreted as relative to the maximum size.
        rel_radius (bool): Interpret the radius as relative to dims.

    Returns:
        db0i_arr (np.ndarray): The internal magnetic field variation in ppb.

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
    if pad_width:
        shape = db0_arr.shape
        pad_width = mrt.utils.auto_pad_width(pad_width, shape)
        mask = [slice(lower, -upper) for (lower, upper) in pad_width]
        db0_arr = np.pad(db0_arr, pad_width, 'constant', constant_values=0)
        mask_arr = np.pad(mask_arr, pad_width, 'constant', constant_values=0)
    else:
        mask = [slice(None)] * db0_arr.ndim

    if rel_radius:
        radius = mrt.geometry.rel2abs(max(db0_arr.shape), radius)

    # # generate the spherical kernel
    sphere = mrt.geometry.sphere(db0_arr.shape, radius).astype(complex)
    sphere /= np.sum(sphere)
    dirac_delta = mrt.geometry.dirac_delta(db0_arr.shape, 0.5, 1.0)
    kernel_k = np.fft.fftn(np.fft.ifftshift(dirac_delta - sphere))

    kernel_mask = np.abs(kernel_k) > threshold
    kernel_k_inv = kernel_mask.astype(complex)
    kernel_k_inv[kernel_mask] = 1.0 / kernel_k[kernel_mask]

    # ft_factor = (2 * np.pi)  # can be neglected because it cancels out
    db0_arr = mask_arr * (
        # ((1j * ft_factor) ** 2) *
        np.fft.ifftn(np.fft.fftn(db0_arr) * kernel_k))
    db0_arr = np.real(
        # ((1j / ft_factor) ** 2) *
        np.fft.ifftn(np.fft.fftn(db0_arr) * kernel_k_inv))

    return db0_arr[mask]


# ======================================================================
def qsm_remove_background_pdf(
        uphs_arr,
        mask_arr,
        radius=0.01,
        threshold=np.spacing(1)):
    """
    Filter out the non-harmonic components of the magnetic field variation.

    Args:
        uphs_arr (np.ndarray): The input unwrapped phase in rad.
        mask_arr (np.ndarray): The inner-volume mask.
        radius (float): The radius of the kernel sphere.
        threshold (float): The deconvolution threshold.

    Returns:
        phs_arr (np.ndarray): The SHARP filtered phase.
    """
    # todo: implement
    raise NotImplementedError


# ======================================================================
def qsm_field2source_tkd(
        db0i_arr,
        mask_arr=None,
        threshold=1.0e-1,
        b0_direction=(0, 0, 1),
        theta=0.0,
        phi=0.0):
    """
    Convert magnetic field variation to magnetic susceptibility.

    This implements the so-called Threshold-based K-space Division (TKD).
    This requires that magnetic susceptibility sources are within the region
    where magnetic field variation is known.

    Args:
        db0i_arr (np.ndarray): The magnetic field variation in relative units.
            The contribution from sources external to the `mask_arr`
            must be minimal.
            A spatial 3D array is expected.
        mask_arr (np.ndarray|None): The boolean mask array.
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
        chi_arr (np.ndarray): The magnetic susceptibility in relative units.
            It has the same scaling units as `db0_arr` (e.g. ppm or ppb).
            Values are defined up to an aribitrary offset.
    """
    if mask_arr is None:
        mask_arr = 1

    db0i_arr *= mask_arr

    # generate the dipole kernel
    dk = dipole_kernel(
        db0i_arr.shape, b0_direction=b0_direction, theta=theta, phi=phi)
    dk = np.fft.fftshift(dk)

    # threshold the zeros of the dipole kernel
    if threshold:
        dk_mask = np.abs(dk) > threshold
    else:
        dk_mask = np.ones_like(dk, dtype=bool)
    dk_inv = dk_mask.astype(complex)
    dk_inv[dk_mask] = (1.0 / dk[dk_mask])

    # divide the magnetic field variation by the dipole kernel
    chi_k_arr = dk_inv * np.fft.fftn(db0i_arr)

    # remove singularity of susceptibility
    chi_k_arr = mrt.utils.subst(chi_k_arr)

    # perform the inverse Fourier transform
    chi_arr = np.real(np.fft.ifftn(chi_k_arr))

    if mask_arr is not None:
        chi_arr *= mask_arr

    return chi_arr


# ======================================================================
def qsm_field2source_l2_closed_form(
        db0i_arr,
        mask_arr=None,
        grad_regularization=0.09,
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
        db0i_arr (np.ndarray): The magnetic field variation in relative units.
            The contribution from sources external to the `mask_arr`
            must be minimal.
            A spatial 3D array is expected.
        mask_arr (np.ndarray|None): The boolean mask array.
        grad_regularization (float): The regularization parameter.
            This is used to weight the L2 constrain penalty.
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
        chi_arr (np.ndarray): The magnetic susceptibility in relative units.
            It has the same scaling units as `db0_arr` (e.g. ppm or ppb).
            Values are defined up to an aribitrary offset.
    """
    if mask_arr is None:
        mask_arr = 1

    db0i_arr *= mask_arr

    # generate the dipole kernel
    dk = dipole_kernel(
        db0i_arr.shape, b0_direction=b0_direction, theta=theta, phi=phi)
    dk = np.fft.fftshift(dk)

    # threshold the zeros of the dipole kernel
    if threshold:
        mask = np.abs(dk) < threshold
        dk[mask] = threshold

    # compute the gradient operators along all dims
    exp_2_k = sum(
        mrt.utils.exp_gradient_kernels(db0i_arr.shape, None, db0i_arr.shape))
    exp_2_k = np.fft.fftshift(exp_2_k)

    # perform the inverse Fourier transform
    chi_arr = mask_arr * np.real(np.fft.ifftn(
        np.conj(dk) * np.fft.fftn(db0i_arr) / (
                dk ** 2 + grad_regularization ** 2 * exp_2_k)))

    return chi_arr


# ======================================================================
def qsm_field2source_l2_iter(
        db0i_arr,
        mask_arr=None,
        weight_arr=None,
        grad_regularization=1.0e-1,
        preconditioner=False,
        threshold=1.0e-1,
        b0_direction=(0, 0, 1),
        theta=0.0,
        phi=0.0,
        linsolve_iter_kws=(
                ('method', 'gmres'),
                ('max_iter', 512)),
        verbose=D_VERB_LVL):
    """
    Convert magnetic field variation to magnetic susceptibility.

    This implements the so-called Threshold-based K-space Division (TKD).
    This requires that magnetic susceptibility sources are within the region
    where magnetic field variation is known.

    Args:
        db0i_arr (np.ndarray): The magnetic field variation in relative units.
            The contribution from sources external to the `mask_arr`
            must be minimal.
            A spatial 3D array is expected.
        mask_arr (np.ndarray|None): The boolean mask array.
            Must have the same shape as `db0_arr`.
        weight_arr (np.ndarray|None): The weight array.
            Must have the same shape as `db0_arr`.
        grad_regularization (float): The regularization parameter.
            This is used to weight the L2 constrain penalty.
        preconditioner (bool): Use a preconditioner for the linear solver.
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
        linsolve_iter_kws (dict|tuple|None): Additional keyword arguments.
            These are passed to `pymrt.recipes.generic.linsolve_iter()`.
        verbose (int): Set level of verbosity.

    Returns:
        chi_arr (np.ndarray): The magnetic susceptibility in relative units.
            It has the same scaling units as `db0_arr` (e.g. ppm or ppb).
            Values are defined up to an aribitrary offset.
    """
    linsolve_iter_kws = {} \
        if linsolve_iter_kws is None else dict(linsolve_iter_kws)
    if mask_arr is None:
        mask_arr = 1
    if weight_arr is None:
        weight_arr = 1

    db0i_arr *= mask_arr

    # generate the dipole kernel
    dk = dipole_kernel(
        db0i_arr.shape, b0_direction=b0_direction, theta=theta, phi=phi)
    dk = np.fft.fftshift(dk)

    # generate the inverse dipole kernel
    if threshold:
        dk_mask = np.abs(dk) > threshold
    else:
        dk_mask = np.ones_like(dk)
    dk_inv = dk_mask.astype(complex)
    dk_inv[dk_mask] = (1.0 / dk[dk_mask])

    # compute the gradient operators along all dims
    exp_ks = mrt.utils.exp_gradient_kernels(db0i_arr.shape, None, db0i_arr.shape)
    exp_k_invs = []
    for kernel_k in exp_ks:
        if threshold:
            kernel_mask = np.abs(kernel_k) > threshold
        else:
            kernel_mask = np.ones_like(kernel_k, dtype=bool)
        kernel_k_inv = kernel_mask.astype(complex)
        kernel_k_inv[kernel_mask] = (1.0 / kernel_k[kernel_mask])
        exp_k_invs.append(kernel_k_inv)
    x_shape = tuple(db0i_arr.shape) + (4,)

    # -----------------------------------
    def _ax(
            arr,
            weight_arr=weight_arr,
            mask_arr=mask_arr,
            dk=dk,
            exp_ks=exp_ks):
        """Computes the linear operator."""
        o_arr = arr.reshape(x_shape)[..., 0]
        ax_arrs = [
            weight_arr *
            np.real(np.fft.ifftn(dk * np.fft.fftn(o_arr)))]
        ax_arrs.extend(
            [grad_regularization *
             mask_arr *
             np.real(np.fft.ifftn(exp_k * np.fft.fftn(o_arr)))
             for exp_k in exp_ks])
        return np.ravel(np.stack(ax_arrs, -1))

    # -----------------------------------
    def _ahb(arr, mask_arr=mask_arr, dk=dk, exp_ks=exp_ks):
        """Computes the transposed linear operator."""
        o_arr = arr.reshape(x_shape)[..., 0]
        ahb_arrs = [
            weight_arr *
            np.real(np.fft.ifftn(np.conj(dk) * np.fft.fftn(o_arr)))]
        ahb_arrs.extend(
            [grad_regularization *
             mask_arr *
             np.real(np.fft.ifftn(np.conj(exp_k) * np.fft.fftn(o_arr)))
             for exp_k in exp_ks])
        return np.ravel(np.stack(ahb_arrs, -1))

    # -----------------------------------
    def _aib(arr, mask_arr=mask_arr, dk_inv=dk_inv, exp_k_invs=exp_k_invs):
        """Computes the inverse linear operator."""
        o_arr = arr.reshape(x_shape)[..., 0]
        aib_arrs = [
            weight_arr *
            np.real(np.fft.ifftn(dk_inv * np.fft.fftn(o_arr)))]
        aib_arrs.extend(
            [1.0 / grad_regularization *
             mask_arr *
             np.real(np.fft.ifftn(exp_k_inv * np.fft.fftn(aib_arrs[0])))
             for exp_k_inv in exp_k_invs])
        return np.ravel(np.stack(aib_arrs, -1))

    const_term = np.ravel(
        np.stack((db0i_arr,) + 3 * (np.zeros_like(db0i_arr),), -1))
    x0_arr = _aib(const_term)

    linear_operator = sp.sparse.linalg.LinearOperator(
        (len(const_term),) * 2, matvec=_ax, rmatvec=_ahb)
    if preconditioner:
        preconditioner = sp.sparse.linalg.LinearOperator(
            (len(const_term),) * 2, matvec=_aib, rmatvec=_ax)
    else:
        preconditioner = None
    chi_arr = generic.linsolve_iter(
        linear_operator, const_term,
        x0_arr=x0_arr, preconditioner=preconditioner,
        verbose=verbose, **linsolve_iter_kws)

    # select the chi part of the solution and reshape
    chi_arr = chi_arr.reshape(x_shape)[..., 0]
    chi_arr *= mask_arr

    return chi_arr


# ======================================================================
def qsm_single_step(
        db0_arr):
    raise NotImplementedError


# ======================================================================
def qsm_total_field_inversion(
        db0_arr,
        weight_arr=None,
        mask_arr=None,
        precond_arr=None,
        grad_regularization=1.0e-1,
        threshold=1.0e-1,
        b0_direction=(0, 0, 1),
        theta=0.0,
        phi=0.0,
        linsolve_iter_kws=(
                ('method', 'gmres'),
                ('max_iter', 512)),
        verbose=D_VERB_LVL):
    """
    Convert magnetic field variation to magnetic susceptibility.

    This implements the so-called Threshold-based K-space Division (TKD).
    This requires that magnetic susceptibility sources are within the region
    where magnetic field variation is known.

    Args:
        db0_arr (np.ndarray): The magnetic field variation in relative units.
            The contribution from sources external to the `mask_arr`
            are not required to be minimal.
            A spatial 3D array is expected.
        mask_arr (np.ndarray|None): The boolean mask array.
            Must have the same shape as `db0_arr`.
        weight_arr (np.ndarray|None): The weight array.
            Must have the same shape as `db0_arr`.
        precond_arr (np.ndarray|None): The precondition array.
            Must have the same shape as `db0_arr`.
        grad_regularization (float): The regularization parameter.
            This is used to weight the L2 constrain penalty.
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
        linsolve_iter_kws (dict|tuple|None): Additional keyword arguments.
            These are passed to `pymrt.recipes.generic.linsolve_iter()`.
        verbose (int): Set level of verbosity.

    Returns:
        chi_arr (np.ndarray): The magnetic susceptibility in relative units.
            It has the same scaling units as `db0_arr` (e.g. ppm or ppb).
            Values are defined up to an aribitrary offset.
    """
    linsolve_iter_kws = {} \
        if linsolve_iter_kws is None else dict(linsolve_iter_kws)

    if weight_arr is None:
        weight_arr = 1
    if mask_arr is None:
        mask_arr = 1

    # generate the dipole kernel
    dk = dipole_kernel(
        db0_arr.shape, b0_direction=b0_direction, theta=theta, phi=phi)
    dk = np.fft.fftshift(dk)

    # generate the inverse dipole kernel
    if threshold:
        dk_mask = np.abs(dk) > threshold
    else:
        dk_mask = np.ones_like(dk)
    dk_inv = dk_mask.astype(complex)
    dk_inv[dk_mask] = (1.0 / dk[dk_mask])

    # compute the gradient operators along all dims
    exp_ks = mrt.utils.exp_gradient_kernels(db0_arr.shape, None, db0_arr.shape)
    exp_k_invs = []
    for kernel_k in exp_ks:
        if threshold:
            kernel_mask = np.abs(kernel_k) > threshold
        else:
            kernel_mask = np.ones_like(kernel_k, dtype=bool)
        kernel_k_inv = kernel_mask.astype(complex)
        kernel_k_inv[kernel_mask] = (1.0 / kernel_k[kernel_mask])
        exp_k_invs.append(kernel_k_inv)
    x_shape = tuple(db0_arr.shape) + (4,)

    # -----------------------------------
    def _ax(
            arr,
            weight_arr=weight_arr,
            mask_arr=mask_arr,
            dk=dk,
            exp_ks=exp_ks):
        """Computes the linear operator."""
        o_arr = arr.reshape(x_shape)[..., 0]
        ax_arrs = [
            weight_arr *
            np.real(np.fft.ifftn(dk * np.fft.fftn(o_arr)))]
        ax_arrs.extend(
            [grad_regularization *
             mask_arr *
             np.real(np.fft.ifftn(exp_k * np.fft.fftn(o_arr)))
             for exp_k in exp_ks])
        return np.ravel(np.stack(ax_arrs, -1))

    # -----------------------------------
    def _ahb(arr, mask_arr=mask_arr, dk=dk, exp_ks=exp_ks):
        """Computes the transposed linear operator."""
        o_arr = arr.reshape(x_shape)[..., 0]
        ahb_arrs = [
            weight_arr *
            np.real(np.fft.ifftn(np.conj(dk) * np.fft.fftn(o_arr)))]
        ahb_arrs.extend(
            [grad_regularization *
             mask_arr *
             np.real(np.fft.ifftn(np.conj(exp_k) * np.fft.fftn(o_arr)))
             for exp_k in exp_ks])
        return np.ravel(np.stack(ahb_arrs, -1))

    const_term = np.ravel(
        np.stack((db0_arr,) + 3 * (np.zeros_like(db0_arr),), -1))
    x0_arr = None  # TODO: compute approximate solution

    linear_operator = sp.sparse.linalg.LinearOperator(
        (len(const_term),) * 2, matvec=_ax, rmatvec=_ahb)
    if precond_arr is None:
        # preconditioner = sp.sparse.eye(len(const_term))
        preconditioner = None
    else:
        ext_precond_arr = np.ravel(
            np.stack((precond_arr,) + 3 * (np.ones_like(db0_arr),), -1))
        preconditioner = sp.sparse.linalg.LinearOperator(
            (len(const_term),) * 2,
            matvec=lambda x: ext_precond_arr * x,
            rmatvec=lambda x: ext_precond_arr * x)

    chi_arr = generic.linsolve_iter(
        linear_operator, const_term,
        x0_arr=x0_arr, preconditioner=preconditioner,
        verbose=verbose, **linsolve_iter_kws)

    # select the chi part of the solution and reshape
    chi_arr = chi_arr.reshape(x_shape)[..., 0]

    return chi_arr


# ======================================================================
def qsm_superfast_dipole_inversion(
        dphs_arr,
        mask_arr,
        sharp_radius,
        sharp_threshold,
        tkd_threshold,
        b0,
        b0_direction=(0, 0, 1),
        theta=0.0,
        phi=0.0):
    chi_arr = db0_to_chi(
        qsm_remove_background_sharp(
            db0.dphs_to_db0(dphs_arr, b0=b0),
            mask_arr=mask_arr, radius=sharp_radius, threshold=sharp_threshold),
        mask_arr=mask_arr, threshold=tkd_threshold, b0_direction=b0_direction,
        theta=theta, phi=phi)
    # return chi_arr
    raise NotImplementedError


# ======================================================================
def qsm_preprocess(
        mag_arr,
        phs_arr,
        echo_times,
        echo_times_mask=None):
    """


    Args:
        mag_arr ():
        phs_arr ():
        echo_times ():
        echo_times_mask ():

    Returns:

    """
    echo_times = np.array(mrt.utils.auto_repeat(echo_times, 1))
    if len(echo_times) > 1:
        dphs_arr = phs.phs_to_dphs(
            phs_arr, tis=echo_times, tis_mask=echo_times_mask)
        mag_arr = mag_arr[..., 0]
    else:
        dphs_arr = phs.single_phs_to_dphs(phs_arr, echo_times[0])
    mask_arr = mrt.segmentation.mask_threshold_compact(mag_arr)
    raise NotImplementedError
    # return dphs_arr, mask_arr


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
    raise NotImplementedError


# ======================================================================
def wip():
    import os
    import pymrt.input_output

    force = False

    base_path = '/home/raid1/metere/hd3/cache/qsm_coil_reco/COIL_RECO/' \
                'HJJT161103_nifti/0091_as_gre_nifti_TE17ms'

    mag_filepath = os.path.join(base_path, 'bai_mag.nii.gz')
    phs_filepath = os.path.join(base_path, 'bai_phs.nii.gz')
    msk_filepath = os.path.join(base_path, 'mask.nii.gz')

    mag_arr, meta = mrt.input_output.load(mag_filepath, meta=True)
    phs_arr = mrt.input_output.load(phs_filepath)
    msk_arr = mrt.input_output.load(msk_filepath).astype(bool)

    uphs_filepath = os.path.join(base_path, 'unwrap_bai_phs.nii.gz')
    if mrt.utils.check_redo(phs_filepath, uphs_filepath, force):
        from pymrt.recipes import phs

        uphs_arr = phs.unwrapping(phs_arr)
        mrt.input_output.save(uphs_filepath, uphs_arr)
    else:
        uphs_arr = mrt.input_output.load(uphs_filepath)

    dphs_filepath = os.path.join(base_path, 'dphs.nii.gz')
    if mrt.utils.check_redo(phs_filepath, dphs_filepath, force):
        from pymrt.recipes import phs

        dphs_arr = phs.phs_to_dphs(phs_arr)
        mrt.input_output.save(dphs_filepath, uphs_arr)
    else:
        dphs_arr = mrt.input_output.load(dphs_filepath)

    db0_filepath = os.path.join(base_path, 'db0.nii.gz')
    if mrt.utils.check_redo(dphs_filepath, db0_filepath, force):
        from pymrt.recipes import db0

        db0_arr = db0.dphs_to_db0(dphs_arr, b0=2.89362)
        mrt.input_output.save(db0_filepath, db0_arr)
    else:
        db0_arr = mrt.input_output.load(db0_filepath)

    # milf_filepath = os.path.join(base_path, 'milf_bai_phs.nii.gz')
    # if mrt.utils.check_redo(uphs_filepath, milf_filepath, force):
    #     from pymrt.recipes import phs
    #
    #     mask_arr = sp.ndimage.binary_erosion(msk_arr, iterations=6)
    #     milf_arr = qsm_remove_background_milf(uphs_arr, mask_arr)
    #     mrt.input_output.save(milf_filepath, milf_arr)
    #     msg('MILF')
    # else:
    #     milf_arr = mrt.input_output.load(milf_filepath)
    #
    # sharp_filepath = os.path.join(base_path, 'sharp_bai_phs.nii.gz')
    # if mrt.utils.check_redo(uphs_filepath, sharp_filepath, force):
    #     from pymrt.recipes import phs
    #     import scipy.ndimage
    #
    #     radius = 5
    #     mask_arr = sp.ndimage.binary_erosion(msk_arr, iterations=radius * 4)
    #     sharp_arr = qsm_remove_background_sharp(
    #         uphs_arr, mask_arr, radius, rel_radius=False)
    #     mrt.input_output.save(sharp_filepath, sharp_arr)
    #     msg('SHARP')
    # else:
    #     sharp_arr = mrt.input_output.load(sharp_filepath)


# ======================================================================
if __name__ == '__main__':
    # import doctest
    #
    # msg(__doc__.strip())
    # doctest.testmod()
    wip()

else:
    elapsed()
