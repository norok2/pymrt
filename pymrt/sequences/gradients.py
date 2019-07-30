#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pymrt.sequences.trajectories: create and manipulate 2D, 3D and N-D trajectories

This is useful for pulse sequences evaluation.
"""

# ======================================================================
# :: Future Imports
from __future__ import (
    division, absolute_import, print_function, unicode_literals, )

# ======================================================================
# :: Python Standard Library Imports
# import os  # Miscellaneous operating system interfaces
import itertools  # Functions creating iterators for efficient looping
import functools  # Higher-order functions and operations on callable objects
import warnings  # Warning control
import random  # Generate pseudo-random numbers
import doctest  # Test interactive Python examples

# :: External Imports
import numpy as np  # NumPy (multidimensional numerical arrays library)
import scipy as sp  # SciPy (signal and image processing library)
import flyingcircus as fc  # Everything you always wanted to have in Python.*

# :: External Imports Submodules
import scipy.signal  # SciPy: Signal Processing
import scipy.ndimage  # SciPy: ND-image Manipulation

# :: Local Imports
import pymrt as mrt
import pymrt.util
import pymrt.constants
import pymrt.plot

from pymrt import INFO, PATH
from pymrt import VERB_LVL, D_VERB_LVL, VERB_LVL_NAMES
from pymrt import elapsed, report, run_doctests
from pymrt import msg, dbg, fmt, fmtm

# ======================================================================
GRAD_SPEC = {
    # 'vendor.model_version': dict(G=[T/m], SR=[T/m/s])
    'mrcoils.insert_20180501': dict(G=200e-3, SR=2600),
    'mrcoils.insert_20180529': dict(G=150e-3, SR=None),
    'mrcoils.insert_20181001': dict(G=45e-3, SR=None),
    'siemens.prisma': dict(G=45e-3, SR=200),
    'siemens.skyra': dict(G=45e-3, SR=200),
}


# ======================================================================
def moment_to_k_value(
        moment,
        species='1H'):
    """
    Compute the k-space value associated with a gradient moment.

    Args:
        moment (int|float|np.ndarray): The moment of the gradient in T/m*s.
        species (str): The species to consider for the gyromagnetic ratio.
            See `pymrt.constants.GAMMA_BAR` for the available species.

    Returns:
        k_value (int|float|np.ndarray): The k-space value in 1/m.
    """
    if species in mrt.constants.GAMMA_BAR:
        return mrt.constants.GAMMA_BAR[species] * moment
    else:
        raise KeyError('Unknown species for `pymrt.constants.GAMMA_BAR`')


# ======================================================================
def k_value_to_moment(
        k_value,
        species='1H'):
    """
    Compute the gradient moment associated with a k-space value.

    Args:
        k_value (int|float|np.ndarray): The k-space value in 1/m.
        species (str): The species to consider for the gyromagnetic ratio.
            See `pymrt.constants.GAMMA_BAR` for the available species.

    Returns:
        moment (int|float|np.ndarray): The moment of the gradient in T/m*s.
    """
    if species in mrt.constants.GAMMA_BAR:
        return k_value / mrt.constants.GAMMA_BAR[species]
    else:
        raise KeyError('Unknown species for `pymrt.constants.GAMMA_BAR`')


# ======================================================================
def k_value_to_length(k_value):
    """
    Compute the spatial length associated to a k-space value.

    The spatial length could be either a resolution or a field-of-view (FOV).

    Args:
        k_value (int|float|np.ndarray): The k-space value in 1/m.

    Returns:
        length (int|float|np.ndarray): The spatial length in m.
    """
    return 1 / k_value


# ======================================================================
def length_to_k_value(length):
    """
    Compute the k-space value associated to a spatial length.

    The spatial length could be either a resolution or a field-of-view (FOV).

    Args:
        length (int|float|np.ndarray): The spatial length in m.

    Returns:
        k_value (int|float|np.ndarray): The k-space value in 1/m.
    """
    return 1 / length


# ======================================================================
def moment_to_length(
        moment,
        species='1H'):
    """
    Compute the spatial length associated with a gradient moment.

    Args:
        moment (int|float|np.ndarray): The moment of the gradient in T/m*s.
        species (str): The species to consider for the gyromagnetic ratio.
            See `pymrt.constants.GAMMA_BAR` for the available species.

    Returns:
        length (int|float|np.ndarray): The spatial length in m.
    """
    return k_value_to_length(moment_to_k_value(moment, species))


# ======================================================================
def length_to_moment(
        length,
        species='1H'):
    """
    Compute the gradient moment associated with a spatial length.

    Args:
        length (int|float|np.ndarray): The spatial length in m.
        species (str): The species to consider for the gyromagnetic ratio.
            See `pymrt.constants.GAMMA_BAR` for the available species.

    Returns:
        moment (int|float|np.ndarray): The moment of the gradient in T/m*s.
    """
    return k_value_to_moment(length_to_k_value(length), species)


# ======================================================================
def calc_constant(
        duration=None,
        amplitude=None,
        moment=None):
    """
    Compute the missing parameter for a constant gradient.
    
    This assumes the relationship: moment = duration * amplitude.
    The slew rate is assumed to be infinity.

    Args:
        duration (int|float|np.ndarray|None): Gradient duration in s.
        amplitude (int|float|np.ndarray|None): Gradient amplitude in T/m.
        moment (int|float|np.ndarray|None): Gradient moment in T/m*s.

    Returns:
        result (tuple): The tuple
            contains:
             - duration (int|float|np.ndarray): Gradient duration in s.
             - amplitude (int|float|np.ndarray): Gradient amplitude in T/m.
             - slew_rate (int|float|np.ndarray): Gradient slew rate in T/m/s.
             - moment (int|float|np.ndarray): Gradient moment in T/m*s.
    
    Examples:
        >>> calc_constant(100, 50)
        (100, 50, inf, 5000)

        >>> results = [
        ...     calc_constant(100.0, 50.0, None),
        ...     calc_constant(100.0, None, 5000.0),
        ...     calc_constant(None, 50.0, 5000.0)]
        >>> np.all([np.isclose(result, results[0]) for result in results])
        True
    """
    slew_rate = np.inf
    if duration is not None and amplitude is not None and moment is None:
        moment = duration * amplitude
    elif duration is not None and moment is not None and amplitude is None:
        amplitude = moment / duration
    elif moment is not None and amplitude is not None and duration is None:
        duration = moment / amplitude
    else:
        text = 'Exactly two of `duration`, `amplitude` and `moment` ' \
               'must be different from None.'
        raise ValueError(text)
    return duration, amplitude, slew_rate, moment


# ======================================================================
def calc_triangular(
        duration=None,
        amplitude=None,
        slew_rate=None,
        moment=None):
    """
    Perform gradient computations based on triangular shape.

    Args:
        duration (int|float|np.ndarray|None): Gradient duration in s.
        amplitude (int|float|np.ndarray|None): Gradient amplitude in T/m.
        slew_rate (int|float|np.ndarray|None): Gradient slew rate in T/m/s.
        moment (int|float|np.ndarray|None): Gradient moment in T/m*s.

    Returns:
        result (tuple): The tuple
            contains:
             - duration (int|float|np.ndarray): Gradient duration in s.
             - amplitude (int|float|np.ndarray): Gradient amplitude in T/m.
             - slew_rate (int|float|np.ndarray): Gradient slew rate in T/m/s.
             - moment (int|float|np.ndarray): Gradient moment in T/m*s.

    Examples:
        >>> [round(x, 6) for x in calc_triangular(5.0, 10.0, None, None)]
        [5.0, 10.0, 4.0, 25.0]

        >>> results = [
        ...     calc_triangular(5.0, 10.0, None, None),
        ...     calc_triangular(5.0, None, 4.0, None),
        ...     calc_triangular(5.0, None, None, 25.0),
        ...     calc_triangular(None, 10.0, 4.0, None),
        ...     calc_triangular(None, 10.0, None, 25.0),
        ...     calc_triangular(None, None, 4.0, 25.0)]
        >>> np.all([np.isclose(result, results[0]) for result in results])
        True

        >>> # overconditioned
        >>> calc_triangular(5.0, 10.0, 4.0, None)
        Traceback (most recent call last):
            ...
        ValueError: Exactly two of `duration`, `amplitude`, `slew_rate`,\
 `moment` must be different from None.

        >>> # underconditioned
        >>> calc_triangular(5.0, None, None, None)
        Traceback (most recent call last):
            ...
        ValueError: Exactly two of `duration`, `amplitude`, `slew_rate`,\
 `moment` must be different from None.
    """

    d, a, s, m = duration, amplitude, slew_rate, moment

    if d is not None and a is not None and s is None and m is None:
        slew_rate = 2 * amplitude / duration
        moment = duration * amplitude / 2
    elif d is not None and a is None and s is not None and m is None:
        amplitude = slew_rate * duration / 2
        moment = slew_rate * duration ** 2 / 4
    elif d is not None and a is None and s is None and m is not None:
        amplitude = 2 * moment / duration
        slew_rate = 4 * moment / duration ** 2
    elif d is None and a is not None and s is not None and m is None:
        duration = 2 * amplitude / slew_rate
        moment = amplitude ** 2 / slew_rate
    elif d is None and a is not None and s is None and m is not None:
        duration = 2 * moment / amplitude
        slew_rate = amplitude ** 2 / moment
    elif d is None and a is None and s is not None and m is not None:
        duration = 2 * (moment / slew_rate) ** 0.5
        amplitude = (moment * slew_rate) ** 0.5
    else:
        text = 'Exactly two of `duration`, `amplitude`,' \
               ' `slew_rate`, `moment` must be different from None.'
        raise ValueError(text)
    return duration, amplitude, slew_rate, moment


# ======================================================================
def calc_trapezoidal(
        duration=None,
        amplitude=None,
        slew_rate=None,
        moment=None,
        plateau=None):
    """
    Perform gradient computations based on trapezoidal shape.

    The duration refers to the full trapezium, including the rise up and down.

    Args:
        duration (int|float|np.ndarray|None): Gradient duration in s.
        amplitude (int|float|np.ndarray|None): Gradient amplitude in T/m.
        slew_rate (int|float|np.ndarray|None): Gradient slew rate in T/m/s.
        moment (int|float|np.ndarray|None): Gradient moment in T/m*s.
        plateau (int|float|np.ndarray|None): Gradient plateau in s.

    Returns:
        result (tuple): The tuple
            contains:
             - duration (int|float|np.ndarray): Gradient duration in s.
             - amplitude (int|float|np.ndarray): Gradient amplitude in T/m.
             - slew_rate (int|float|np.ndarray): Gradient slew rate in T/m/s.
             - moment (int|float|np.ndarray): Gradient moment in T/m*s.
             - plateau (int|float|np.ndarray|None): Gradient plateau in s.

    Examples:
        >>> [round(x, 6) for x in calc_trapezoidal(5.0, 10.0, 4.0)]
        [5.0, 10.0, 12.566371, 15.915494, 0.2]

        >>> results = [
        ...     calc_sinusoidal(5.0, 10.0, None, None),
        ...     calc_sinusoidal(5.0, None, 12.566371, None),
        ...     calc_sinusoidal(5.0, None, None, 15.915494),
        ...     calc_sinusoidal(None, 10.0, 12.566371, None),
        ...     calc_sinusoidal(None, 10.0, None, 15.915494),
        ...     calc_sinusoidal(None, None, 12.566371, 15.915494)]
        >>> np.all([np.isclose(result, results[0]) for result in results])
        True

    #todo: fix issues with plateau/duty
    """
    d, a, s, m, p = duration, amplitude, slew_rate, moment, plateau

    if all(x is not None for x in (d, a, s, p)):
        min_rise_time = amplitude / slew_rate
        plateau = duration - 2 * min_rise_time

    return duration, amplitude, slew_rate, moment, plateau


# ======================================================================
def calc_sinusoidal(
        duration=None,
        amplitude=None,
        slew_rate=None,
        moment=None,
        frequency=None):
    """
    Perform gradient computations based on sinusoidal shape.

    The moment computation is based on half the period.
    The duration and the frequency refer to the full sinusoidal shape.

    Args:
        duration (int|float|np.ndarray|None): Gradient duration in s.
        amplitude (int|float|np.ndarray|None): Gradient amplitude in T/m.
        slew_rate (int|float|np.ndarray|None): Gradient slew rate in T/m/s.
        moment (int|float|np.ndarray|None): Gradient moment in T/m*s.
        frequency (int|float|np.ndarray|None): Gradient frequency in Hz.

    Returns:
        result (tuple): The tuple
            contains:
             - duration (int|float|np.ndarray): Gradient duration in s.
             - amplitude (int|float|np.ndarray): Gradient amplitude in T/m.
             - slew_rate (int|float|np.ndarray): Gradient slew rate in T/m/s.
             - moment (int|float|np.ndarray): Gradient moment in T/m*s.
             - frequency (int|float|np.ndarray): Gradient frequency in Hz.

    Examples:
        >>> [round(x, 6) for x in calc_sinusoidal(5.0, 10.0, None, None)]
        [5.0, 10.0, 12.566371, 15.915494, 0.2]

        >>> results = [
        ...     calc_sinusoidal(5.0, 10.0, None, None),
        ...     calc_sinusoidal(5.0, None, 12.566371, None),
        ...     calc_sinusoidal(5.0, None, None, 15.915494),
        ...     calc_sinusoidal(None, 10.0, 12.566371, None),
        ...     calc_sinusoidal(None, 10.0, None, 15.915494),
        ...     calc_sinusoidal(None, None, 12.566371, 15.915494)]
        >>> np.all([np.isclose(result, results[0]) for result in results])
        True

        >>> # overconditioned
        >>> calc_sinusoidal(5.0, 10.0, 4.0, None)
        Traceback (most recent call last):
            ...
        ValueError: Exactly two of `duration` (or `frequency`), `amplitude`,\
 `slew_rate`, `moment` must be different from None.

        >>> # underconditioned
        >>> calc_sinusoidal(5.0, None, None, None)
        Traceback (most recent call last):
            ...
        ValueError: Exactly two of `duration` (or `frequency`), `amplitude`,\
 `slew_rate`, `moment` must be different from None.

        >>> [round(x, 6) for x in calc_sinusoidal(None, 10.0, None, None, 0.2)]
        [5.0, 10.0, 12.566371, 15.915494, 0.2]

        >>> # redundant frequency
        >>> calc_sinusoidal(5.0, 10.0, None, None, 0.2)
        Traceback (most recent call last):
            ...
        ValueError: Parameters `duration` and `frequency` are mutually\
 exclusive.
    """
    # integrate(G * sin(2*%pi*f*x), x, 0, 1 / 2 / f);
    if duration is None and frequency is not None:
        duration = 1 / frequency
    elif duration is not None and frequency is not None:
        text = 'Parameters `duration` and `frequency` are mutually exclusive.'
        raise ValueError(text)

    d, a, s, m = duration, amplitude, slew_rate, moment

    # todo: uniform syntax
    if d is not None and a is not None and s is None and m is None:
        moment = amplitude * duration / np.pi
        slew_rate = 2 * np.pi * amplitude / duration
    elif d is None and a is not None and s is None and m is not None:
        duration = moment / amplitude * np.pi
        slew_rate = 2 * amplitude ** 2 / moment
    elif a is not None and d is None and m is None and s is not None:
        duration = 2 * np.pi * amplitude / slew_rate
        moment = amplitude * duration / np.pi
    elif a is None and d is not None and m is not None and s is None:
        amplitude = np.pi * moment / duration
        slew_rate = 2 * np.pi ** 2 * moment / duration ** 2
    elif a is None and d is not None and m is None and s is not None:
        amplitude = duration * slew_rate / (2 * np.pi)
        moment = duration ** 2 * slew_rate / (2 * np.pi ** 2)
    elif a is None and d is None and m is not None and s is not None:
        amplitude = (moment * slew_rate / 2) ** 0.5
        duration = np.pi * (2 * moment / slew_rate) ** 0.5
    else:
        text = 'Exactly two of `duration` (or `frequency`), `amplitude`,' \
               ' `slew_rate`, `moment` must be different from None.'
        raise ValueError(text)
    if frequency is None:
        frequency = 1 / duration
    return duration, amplitude, slew_rate, moment, frequency


# ======================================================================
def calc_trapz(
        grad,
        slew_rate,
        duration,
        duty=0.5,
        raster=10e-6,
        verbose=D_VERB_LVL):
    original_duration = duration
    if not duty:
        duty = 0.0
    has_plateau = True if duty > 0.0 else False
    min_rise_time = fc.base.align(grad / slew_rate, raster)

    if fc.base.align(duration * (1.0 - duty), raster) <= 2 * min_rise_time:
        max_grad = grad
        rise_time = fc.base.align(
            duration * (1.0 - duty) / 2, raster, 'closest')
        if not rise_time:
            rise_time = raster
        duration = (
                fc.base.align(duration - 2 * rise_time, raster)
                + 2 * rise_time)
        grad = min(rise_time * slew_rate, max_grad)
        msg(
            'Will use a maximum gradient of {} T/m (was: {})'.format(
                grad, max_grad),
            verbose, VERB_LVL['medium'])
    else:
        rise_time = min_rise_time
    if not has_plateau:
        msg('No plateau, assume sampling during ramps!',
            verbose, VERB_LVL['high'])
        moment = 2 * slew_rate * rise_time ** 2
        raster_duration = 2 * rise_time
    else:
        plateau = fc.base.align(
            duration - 2 * rise_time, raster, 'closest')
        if plateau < 0.0:
            plateau = 0.0
        moment = slew_rate * rise_time ** 2 + plateau * grad
        raster_duration = 2 * rise_time + plateau
    if (not np.isclose(raster_duration, original_duration, atol=raster / 2)
            and raster_duration > original_duration):
        msg(
            'Duration must be increased to: {} s (was: {})'.format(
                raster_duration, original_duration),
            verbose, VERB_LVL['medium'])
    duty = 1 - (2 * rise_time) / raster_duration
    return moment, raster_duration, duty


# ======================================================================
elapsed(__file__[len(PATH['base']) + 1:])

# ======================================================================
if __name__ == '__main__':
    run_doctests(__doc__)
