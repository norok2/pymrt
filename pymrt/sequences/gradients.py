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
import flyingcircus.util  # FlyingCircus: generic basic utilities
import flyingcircus.num  # FlyingCircus: generic numerical utilities

# :: Local Imports
import pymrt as mrt
import pymrt.utils
import pymrt.constants
import pymrt.plot

from pymrt import INFO, PATH
from pymrt import VERB_LVL, D_VERB_LVL, VERB_LVL_NAMES
from pymrt import elapsed, report
from pymrt import msg, dbg

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
def freq2duration(freq):
    return 1.0 / freq / 2.0


# ======================================================================
def duration2freq(duration):
    return 1.0 / duration / 2.0


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
        moment (int|float|np.ndarray|None): Gradient moment in T/m*s.
        amplitude (int|float|np.ndarray|None): Gradient amplitude in T/m.

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
        >>> calc_constant(None, 50, 5000)
        (100.0, 50, inf, 5000)
        >>> calc_constant(100, None, 5000)
        (100, 50.0, inf, 5000)
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
def calc_trapz(
        duration=None,
        amplitude=None,
        slew_rate=None,
        moment=None,
        duty=0.3,
        raster=10e-6,
        verbose=D_VERB_LVL):
    """
    Perform gradient computations based on trapezoidal shape.

    #todo: fix issues with duty
    """
    original_duration = duration
    if not duty:
        duty = 0.0
    has_plateau = True if duty > 0.0 else False
    min_rise_time = fc.util.num_align(grad / slew_rate, raster)

    if fc.util.num_align(duration * (1.0 - duty), raster) <= 2 * min_rise_time:
        max_grad = grad
        rise_time = fc.util.num_align(
            duration * (1.0 - duty) / 2, raster, 'closest')
        if not rise_time:
            rise_time = raster
        duration = (
                fc.util.num_align(duration - 2 * rise_time, raster)
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
        plateau = fc.util.num_align(
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
    # return duration, amplitude, slew_rate, moment, duty


# ======================================================================
def calc_sinusoidal(
        duration=None,
        amplitude=None,
        slew_rate=None,
        moment=None,
        frequency=None):
    """
    Perform gradient computations based on sinusoidal shape.

    Args:

    Returns:

    Examples:
        >>> [np.round(x, 3) for x in calc_sinusoidal(5, 10, None, None, None)]
        [10, 5, 3.142, 15.915, 0.05]
    """
    # integrate(G * sin(2*%pi*f*x), x, 0, 1 / 2 / f);
    if frequency is None and duration is not None:
        frequency = duration2freq(duration)
    elif duration is None and frequency is not None:
        duration = freq2duration(frequency)
    elif frequency is None and duration is None:
        text = 'At least two of `duration`, `frequency` ' \
               'must be different from None.'
        raise ValueError(text)
    else:
        assert (np.isclose(frequency, duration2freq(duration)))

    a, d, m, s = amplitude, duration, moment, slew_rate

    if a is not None and d is not None and m is None and s is None:
        moment = amplitude * duration / np.pi
        slew_rate = 2 * np.pi * amplitude / duration
    elif a is not None and d is None and m is not None and s is None:
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
        duration = np.pi * (2.0 * moment / slew_rate)
    return duration, amplitude, slew_rate, moment, frequency
