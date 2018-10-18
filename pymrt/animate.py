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
        step=1,
        duration=10000,
        title=None,
        array_interval=None,
        ticks_limit=None,
        orientation=None,
        cmap=None,
        cbar_kws=None,
        cbar_txt=None,
        text_color=None,
        resolution=None,
        size_info=None,
        more_texts=None,
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
    if ax is None:
        fig = plt.figure()
        ax = fig.gca()
    else:
        fig = plt.gcf()
    if axis is None:
        axis = np.argmin(arr.shape)
    sample = fc.num.ndim_slice(arr, axis, 0)
    if title:
        ax.set_title(title)
    if array_interval is None:
        array_interval = fc.num.minmax(arr)
    if not cmap:
        if not fc.util.is_same_sign(array_interval):
            cmap = mpl.cm.get_cmap('RdBu_r')
        else:
            cmap = mpl.cm.get_cmap('gray_r')
    if not text_color:
        if not fc.util.is_same_sign(array_interval):
            text_color = 'k'
        else:
            text_color = 'k'
    ax.set_aspect('equal')
    if (orientation == 'portrait' and sample.shape[0] < sample.shape[1]) or \
            (orientation == 'landscape' and sample.shape[0] > sample.shape[1]):
        sample = sample.transpose()
    mrt.plot._manage_ticks_limit(ticks_limit, ax)

    # print resolution information and draw a ruler
    if size_info is not None and resolution is not None:
        if size_info >= 0.0:
            # print resolution information
            if resolution[0] == resolution[1] == resolution[2]:
                res_str = '{} {} iso.'.format(resolution[0], 'mm')
            else:
                res_str = 'x'.join(
                    [str(x) for x in resolution[0:3]]) + ' ' + 'mm'
            ax.text(
                0.975, 0.975, res_str, rotation=0, color=text_color,
                horizontalalignment='right', verticalalignment='top',
                transform=ax.transAxes)
        if size_info != 0.0:
            res = resolution[1]
            size_info_size = round(abs(size_info) * (sample.shape[1] * res),
                                   -1)
            size_info_str = '{} {}'.format(size_info_size, 'mm')
            size_info_px = size_info_size / res
            ax.text(
                0.025, 0.050, size_info_str, rotation=0, color=text_color,
                horizontalalignment='left', verticalalignment='bottom',
                transform=ax.transAxes)
            ax.plot(
                (sample.shape[1] * 0.025,
                 sample.shape[1] * 0.025 + size_info_px),
                (sample.shape[0] * 0.965, sample.shape[0] * 0.965),
                color=text_color, linewidth=2.5)

    n_frames = arr.shape[axis]
    plots = []
    data = []
    for i in range(0, n_frames, step):
        sample = fc.num.ndim_slice(arr, axis, i)
        pax = ax.imshow(
            sample, cmap=cmap,
            vmin=array_interval[0], vmax=array_interval[1], animated=True)
        # include additional text
        if more_texts is not None:
            for text_kws in more_texts:
                ax.text(**dict(text_kws))
        data.append(sample)
        if len(plots) <= 0:
            if cbar_kws is not None:
                cbar = ax.figure.colorbar(pax, ax=ax, **dict(cbar_kws))
                if cbar_txt is not None:
                    only_extremes = \
                        'ticks' in cbar_kws and len(cbar_kws['ticks']) == 2
                    if only_extremes:
                        cbar.ax.text(
                            2.0, 0.5, cbar_txt, fontsize='small', rotation=90)
                    else:
                        cbar.set_label(cbar_txt)
        plots.append([pax])
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
    return data, fig, mov


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

    mov = mpl.animation.FuncAnimation(
        fig, run, data_gen, init_func=init, save_count=n_frames,
        blit=False, repeat=False, repeat_delay=None,
        interval=duration / n_frames)

    if save_filepath and fc.util.check_redo(None, [save_filepath], force):
        fig.tight_layout()
        save_kwargs = dict(
            fps=n_frames / duration / MSEC_IN_SEC, writer='mencoder',
            codec='libx264', save_count=n_frames)
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
