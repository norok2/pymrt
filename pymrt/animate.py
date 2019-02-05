#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PyMRT: useful basic utilities.
"""

# ======================================================================
# :: Future Imports
from __future__ import (
    division, absolute_import, print_function, unicode_literals, )

# ======================================================================
# :: Python Standard Library Imports
# import os  # Miscellaneous operating system interfaces
# import sys  # System-specific parameters and functions
# import shutil  # High-level file operations
# import math  # Mathematical functions
import time  # Time access and conversions
# import datetime  # Basic date and time types
# import operator  # Standard operators as functions
# import collections  # Container datatypes
import itertools  # Functions creating iterators for efficient looping
# import functools  # Higher-order functions and operations on callable objects
# import argparse  # Parser for command-line options, arguments and sub-command
# import subprocess  # Subprocess management
# import multiprocessing  # Process-based parallelism
# import fractions  # Rational numbers
# import csv  # CSV File Reading and Writing [CSV: Comma-Separated Values]
# import json  # JSON encoder and decoder [JSON: JavaScript Object Notation]
import warnings  # Warning control
import string  # Common string operations

# :: External Imports
import numpy as np  # NumPy (multidimensional numerical arrays library)
# import scipy as sp  # SciPy (signal and image processing library)
import matplotlib as mpl  # Matplotlib (2D/3D plotting library)
# import sympy as sym  # SymPy (symbolic CAS library)
# import PIL  # Python Image Library (image manipulation toolkit)
# import SimpleITK as sitk  # Image ToolKit Wrapper
# import nibabel as nib  # NiBabel (NeuroImaging I/O Library)
# import nipy  # NiPy (NeuroImaging in Python)
# import nipype  # NiPype (NiPy Pipelines and Interfaces)
import numeral  # Support for various integer-to-numeral (and back) conversion
import seaborn as sns  # Seaborn: statistical data visualization
import flyingcircus as fc  # Everything you always wanted to have in Python.*

# :: External Imports Submodules
import matplotlib.pyplot as plt  # Matplotlib's pyplot: MATLAB-like syntax
import matplotlib.cm  # Matplotlib's colormaps
import matplotlib.colors  # Matplotlib's colors
import matplotlib.gridspec  # Matplotlib's grid specifications
import matplotlib.animation as anim  # Matplotlib's animations
from mpl_toolkits.mplot3d.axes3d import Axes3D
# import scipy.optimize  # SciPy: Optimization Algorithms
# import scipy.integrate  # SciPy: Integrations facilities
# import scipy.constants  # SciPy: Mathematal and Physical Constants
# import scipy.ndimage  # SciPy: ND-image Manipulation
import scipy.stats  # SciPy: Statistical functions
import flyingcircus.util  # FlyingCircus: generic basic utilities
import flyingcircus.num  # FlyingCircus: generic numerical utilities

# :: Local Imports
import pymrt as mrt
import pymrt.utils
import pymrt.plot

from pymrt import INFO, PATH
from pymrt import VERB_LVL, D_VERB_LVL, VERB_LVL_NAMES
from pymrt import elapsed, report
from pymrt import msg, dbg

# ======================================================================
MSEC_IN_SEC = 1000


# ======================================================================
def sample2d(
        arr,
        axis=None,
        start=None,
        stop=None,
        step=None,
        duration=10000,
        title=None,
        array_interval=None,
        ticks_limit=None,
        orientation=None,
        flip_ud=False,
        flip_lr=False,
        cmap=None,
        cbar_kws=None,
        cbar_txt=None,
        text_color=None,
        resolution=None,
        size_info=None,
        more_texts=None,
        more_elements=None,
        ax=None,
        save_filepath=None,
        save_kws=None,
        force=False,
        verbose=D_VERB_LVL):
    """
    Plot a 2D sample image of a 3D array.

    Parameters
    ==========
    arr : ndarray
        The original 3D array.
    axis : int (optional)
        The slicing axis. If None, use the shortest one.
    step : int (optional)
        The slicing index. Must be 1 or more.
    title : str (optional)
        The title of the plot.
    array_interval : 2-tuple (optional)
        The (min, max) values interval.
    cmap : MatPlotLib ColorMap (optional)
        The colormap to be used for displaying the histogram.
    use_new_figure : bool (optional)
        Plot the histogram in a new figure.
    close_figure : bool (optional)
        Close the figure after saving (useful for batch processing).
    save_filepath : str (optional)
        The path to which the plot is to be saved. If unset, no output.

    Returns
    =======
    sample : ndarray
        The sliced (N-1)D-array.
    plot : matplotlib.pyplot.Figure
        The figure object containing the plot.

    """
    if arr.ndim != 3:
        raise IndexError('3D array required')

    fig, ax = mrt.plot._ensure_fig_ax(ax)

    n_frames = arr.shape[axis]

    if start is None:
        start = 0
    if stop is None:
        stop = n_frames
    if step is None:
        step = 1

    # prepare data
    sample = fc.num.ndim_slice(arr, axis, start)

    if title:
        ax.set_title(title)

    if array_interval is None:
        array_interval = fc.num.minmax(arr)

    if not text_color:
        text_color = 'k'

    ax.set_aspect('equal')

    mrt.plot._manage_ticks_limit(ticks_limit, ax)

    plots = []
    datas = []
    for i in range(start, stop, step):
        data = mrt.plot._reorient_2d(
            fc.num.ndim_slice(arr, axis, i), orientation, flip_ud, flip_lr)
        pax = ax.imshow(
            data, cmap=cmap,
            vmin=array_interval[0], vmax=array_interval[1], animated=True)
        # include additional text
        if more_texts is not None:
            for text_kws in more_texts:
                ax.text(**dict(text_kws))
        datas.append(data)
        if len(plots) <= 0:
            mrt.plot._manage_colorbar(cbar_kws, cbar_txt, ax, pax)
        plots.append([pax])

        # print resolution information and draw a ruler
        mrt.plot._manage_resolution_info(
            size_info, resolution, data.shape, text_color, ax)

        mrt.plot._more_texts(more_texts, ax)
        mrt.plot._more_elements(more_elements, ax)

    mov = mpl.animation.ArtistAnimation(fig, plots, blit=False)
    if save_filepath and fc.util.check_redo(None, [save_filepath], force):
        fig.tight_layout()
        save_kwargs = {'fps': n_frames / step / duration / MSEC_IN_SEC}
        if save_kws is None:
            save_kws = {}
        save_kwargs.update(save_kws)
        mov.save(save_filepath, **dict(save_kws))
        msg('Anim: {}'.format(save_filepath, verbose, VERB_LVL['medium']))
        plt.close(fig)
    return datas, fig, mov


# ======================================================================
def trajectory_2d(
        trajectory,
        duration=10000,
        last_frame_duration=3000,
        support_intervals=None,
        plot_kws=(('marker', 'o'), ('linewidth', 1)),
        ticks_limit=None,
        title=None,
        more_texts=None,
        more_elements=None,
        ax=None,
        save_filepath=None,
        save_kws=None,
        force=False,
        verbose=D_VERB_LVL):
    n_dims, n_points = trajectory.shape
    if n_dims != 2:
        raise IndexError('2D trajectory required')
    if ax is None:
        fig = plt.figure()
        ax = fig.gca()
    else:
        fig = plt.gcf()
    if title:
        ax.set_title(title)
    if support_intervals is None:
        support_intervals = (
            fc.num.minmax(trajectory[0]), fc.num.minmax(trajectory[1]))

    n_frames = int(n_points * (1 + last_frame_duration / duration))
    data = trajectory

    if plot_kws is None:
        plot_kws = {}

    line, = ax.plot([], [], **dict(plot_kws))
    # points = ax.scatter([], [])
    ax.grid()
    x_data, y_data = [], []

    def data_gen():
        for i in range(n_points):
            yield trajectory[0, i], trajectory[1, i]
        for i in range(n_frames - n_points):
            yield None, None

    def init():
        xlim_size = np.ptp(support_intervals[0])
        ylim_size = np.ptp(support_intervals[1])
        ax.set_xlim(
            (support_intervals[0][0] - 0.1 * xlim_size,
             support_intervals[0][1] + 0.1 * xlim_size))
        ax.set_ylim(
            (support_intervals[1][0] - 0.1 * ylim_size,
             support_intervals[1][1] + 0.1 * ylim_size))
        del x_data[:]
        del y_data[:]
        line.set_data(x_data, y_data)
        # points.set_offsets(np.c_[x_data, y_data])
        return line,  # points

    def run(data_generator):
        # update the data
        x, y = data_generator
        x_data.append(x)
        y_data.append(y)
        line.set_data(x_data, y_data)
        # points.set_offsets(np.c_[x_data, y_data])
        return line,  # points

    mrt.plot._more_texts(more_texts, ax)
    mrt.plot._more_elements(more_elements, ax)
    mov = mpl.animation.FuncAnimation(
        fig, run, data_gen, init_func=init, save_count=n_frames,
        blit=False, repeat=False, repeat_delay=None,
        interval=duration / n_frames)

    if save_filepath and fc.util.check_redo(None, [save_filepath], force):
        fig.tight_layout()
        save_kwargs = dict(
            fps=n_frames / duration / MSEC_IN_SEC, save_count=n_frames)
        if save_kws is None:
            save_kws = {}
        save_kwargs.update(save_kws)
        mov.save(save_filepath, **dict(save_kws))
        msg('Anim: {}'.format(save_filepath, verbose, VERB_LVL['medium']))
        # plt.close(fig)
    return trajectory, fig, mov


# ======================================================================
elapsed(__file__[len(PATH['base']) + 1:])

# ======================================================================
if __name__ == '__main__':
    import doctest  # Test interactive Python examples

    msg(__doc__.strip())
    doctest.testmod()
    msg(report())
