#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
mri_tools: useful basic utilities.
"""


# ======================================================================
# :: Future Imports
from __future__ import division
from __future__ import absolute_import
from __future__ import print_function
# from __future__ import unicode_literals


# ======================================================================
# :: Python Standard Library Imports
# import os  # Miscellaneous operating system interfaces
# import sys  # System-specific parameters and functions
# import shutil  # High-level file operations
# import math  # Mathematical functions
# import time  # Time access and conversions
# import datetime  # Basic date and time types
# import operator  # Standard operators as functions
# import collections  # High-performance container datatypes
import itertools  # Functions creating iterators for efficient looping
# import functools  # Higher-order functions and operations on callable objects
# import argparse  # Parser for command-line options, arguments and sub-command
# import subprocess  # Subprocess management
# import multiprocessing  # Process-based parallelism
# import fractions  # Rational numbers
# import csv  # CSV File Reading and Writing [CSV: Comma-Separated Values]
# import json  # JSON encoder and decoder [JSON: JavaScript Object Notation]

# :: External Imports
import numpy as np  # NumPy (multidimensional numerical arrays library)
# import scipy as sp  # SciPy (signal and image processing library)
# import matplotlib as mpl  # Matplotlib (2D/3D plotting library)
# import sympy as sym  # SymPy (symbolic CAS library)
# import PIL  # Python Image Library (image manipulation toolkit)
# import SimpleITK as sitk  # Image ToolKit Wrapper
# import nibabel as nib  # NiBabel (NeuroImaging I/O Library)
# import nipy  # NiPy (NeuroImaging in Python)
# import nipype  # NiPype (NiPy Pipelines and Interfaces)

# :: External Imports Submodules
import matplotlib.pyplot as plt  # Matplotlib's pyplot: MATLAB-like syntax
# import mpl_toolkits.mplot3d as mpl3  # Matplotlib's 3D support
import mayavi.mlab as mlab  # Mayavi's mlab: MATLAB-like syntax
# import scipy.optimize  # SciPy: Optimization Algorithms
# import scipy.integrate  # SciPy: Integrations facilities
# import scipy.constants  # SciPy: Mathematal and Physical Constants
# import scipy.ndimage  # SciPy: ND-image Manipulation

# :: Local Imports
import mri_tools.base as mrb
# from mri_tools import INFO
# from mri_tools import VERB_LVL
# from mri_tools import D_VERB_LVL
# from mri_tools import get_first_line

# ======================================================================
# :: MatPlotLib-related constants
# standard plot resolutions
D_PLOT_DPI = 72
# colors and linestyles
PLOT_COLORS = ('r', 'g', 'b', 'c', 'm', 'y')
PLOT_LINESTYLES = ('-', '--', '-.', ':')


# ======================================================================
# def plot_with_adjusting_parameters()
# TODO: make a plot with possibility to adjust params


# ======================================================================
def quick(array):
    """
    Quickly plot an array in 2D or 3D.

    Parameters
    ==========
    mask : ndarray
        The mask to plot.

    Returns
    =======
    None

    """

    if array.ndim == 1:
        # using Matplotlib
        plt.figure()
        plt.plot(np.arange(len(array)), array.astype(float))
        plt.draw()
        plt.show()
    elif array.ndim == 2:
        # # using Matplotlib
        # fig = plt.subplots()
        # plt.imshow(array.astype(float), cmap=plt.cm.binary)
        # plt.draw()

        # using Mayavi2
        mlab.figure()
        mlab.imshow(array.astype(float))
        mlab.draw()
        mlab.show()
    elif array.ndim == 3:
        # # using Matplotlib
        # fig = plt.subplots()
        # ax = mpl3.Axes3D(fig)
        # fig.colorbar(plot)

        # using Mayavi2
        mlab.figure()
        mlab.contour3d(array.astype(float))
        mlab.draw()
        mlab.show()
    else:
        print('W: cannot plot more than 3 dimension.')


# ======================================================================
def sample2d(
        array,
        axis=0,
        index=None,
        title=None,
        val_range=None,
        cmap=None,
        use_new_figure=True,
        close_figure=False,
        save_path=None):
    """
    Plot a 2D sample image of a 3D array.

    Parameters
    ==========
    array : ndarray
        The original 3D array.
    axis : int (optional)
        The slicing axis.
    index : int (optional)
        The slicing index. If None, mid-value is taken.
    title : str (optional)
        The title of the plot.
    val_range : 2-tuple (optional)
        The (min, max) values range.
    cmap : MatPlotLib ColorMap (optional)
        The colormap to be used for displaying the histogram.
    use_new_figure : bool (optional)
        Plot the histogram in a new figure.
    close_figure : bool (optional)
        Close the figure after saving (useful for batch processing).
    save_path : str (optional)
        The path to which the plot is to be saved. If unset, no output.

    Returns
    =======
    sample : ndarray
        The sliced (N-1)D-array.

    """
    if array.ndim != 3:
        raise IndexError('3D array required')
    sample = mrb.slice_array(array, axis, index)
    if use_new_figure:
        plot = plt.figure()
    if title:
        plt.title(title)
    if val_range is None:
        val_range = mrb.range_array(array)
    plt.imshow(sample, cmap=cmap, vmin=val_range[0], vmax=val_range[1])
    plt.colorbar(use_gridspec=True)
    if save_path is not None:
        plt.tight_layout()
        plt.savefig(save_path, dpi=D_PLOT_DPI)
    if close_figure:
        plt.close()
    return sample, plot


# ======================================================================
def histogram1d(
        array,
        bin_size=1,
        hist_range=(0.0, 1.0),
        bins=None,
        array_range=None,
        scale='linear',
        title='Histogram',
        labels=('Value', 'Value Frequency'),
        style='-k',
        use_new_figure=True,
        close_figure=False,
        save_path=None):
    """
    Plot 1D histogram of array with MatPlotLib.

    Parameters
    ==========
    array : nd-array
        The array for which histogram is to be plotted.
    bin_size : int or float or float (optional)
        The size of the bins.
    hist_range : float 2-tuple (optional)
        The range of the histogram to display in percentage.
    bins : int (optional)
        The number of bins to use. If set, it overrides bin_size parameter.
    array_range : float 2-tuple (optional)
        Theoretical range of values for the array. If unset, uses min and max.
    scale : ['linear'|'log'|'log10'|'normed'] string (optional)
        The frequency value scaling.
    title : str (optional)
        The title of the plot.
    labels : str 2-tuple (optional)
        A 2-tuple of strings containing x-labels and y-labels.
    style : str (optional)
        Plotting style string (as accepted by MatPlotLib).
    use_new_figure : bool (optional)
        Plot the histogram in a new figure.
    close_figure : bool (optional)
        Close the figure after saving (useful for batch processing).
    save_path : str (optional)
        The path to which the plot is to be saved. If unset, no output.

    Returns
    =======
    hist : array
        The calculated histogram.

    """
    # setup array range
    if not array_range:
        array_range = (np.nanmin(array), np.nanmax(array))
    # setup bins
    if not bins:
        bins = int(mrb.range_size(array_range) / bin_size + 1)
    # setup histogram reange
    hist_range = tuple([mrb.to_range(val, out_range=array_range)
                        for val in hist_range])
    # calculate histogram
    if scale == 'normed':
        is_normed = True
    else:
        is_normed = False
    # prepare figure
    if use_new_figure:
        plot = plt.figure()
    # create histogram
    hist, bin_edges = np.histogram(
        array, bins=bins, range=hist_range, normed=is_normed)
    # adjust scale
    if scale == 'log':
        hist[hist != 0.0] = np.log(hist[hist != 0.0])
    elif scale == 'log10':
        hist[hist != 0.0] = np.log10(hist[hist != 0.0])
    # plot figure
    plt.plot(mrb.mid_val_array(bin_edges), hist, style)
    # setup title and labels
    if title:
        plt.title(title)
    if labels[0]:
        plt.xlabel(labels[0])
    if labels[1]:
        plt.ylabel(labels[1] + ' ({})'.format(scale))
    else:
        plt.ylabel('{}'.format(scale))
    # save figure to file
    if save_path is not None:
        plt.tight_layout()
        plt.savefig(save_path, dpi=D_PLOT_DPI)
    # closing figure
    if close_figure:
        plt.close()
    return hist, bin_edges, plot


# ======================================================================
def histogram1d_list(
        array_list,
        bin_size=1,
        hist_range=(0.0, 1.0),
        bins=None,
        array_range=None,
        scale='linear',
        title='Histogram',
        labels=('Value', 'Value Frequency'),
        legends=None,
        styles=None,
        use_new_figure=True,
        close_figure=False,
        save_path=None):
    """
    Plot 1D histograms of multiple arrays with MatPlotLib.

    Parameters
    ==========
    array : nd-array
        The array for which histogram is to be plotted.
    bin_size : int or float (optional)
        The size of the bins.
    hist_range : float 2-tuple (optional)
        The range of the histogram to display in percentage.
    bins : int (optional)
        The number of bins to use. If set, it overrides bin_size parameter.
    array_range : float 2-tuple (optional)
        Theoretical range of values for the array. If unset, uses min and max.
    scale : ['linear'|'log'|'log10'|'normed'] string (optional)
        The frequency value scaling.
    title : str (optional)
        The title of the plot.
    labels : str 2-tuple (optional)
        The strings for x- and y-labels.
    styles : str list (optional)
        MatPlotLib's plotting style strings. If None, uses color cycling.
    legends : str list (optional)
        Legend for each histogram. If None, no legend will be displayed.
    use_new_figure : bool (optional)
        Plot the histogram in a new figure.
    close_figure : bool (optional)
        Close the figure after saving (useful for batch processing).
    save_path : str (optional)
        The path to which the plot is to be saved. If unset, no output.

    Returns
    =======
    hist : array
        The calculated histogram.
    bin_edges : array
        The bin edges of the calculated histogram.
    plot : array
        The plot for further manipulation of the figure.

    """
    # setup array range
    if not array_range:
        array_range = (np.nanmin(array_list[0]), np.nanmax(array_list[0]))
        for array in array_list[1:]:
            array_range = (
                min(np.nanmin(array), array_range[0]),
                max(np.nanmax(array), array_range[1]))
    # setup bins
    if not bins:
        bins = int(mrb.range_size(array_range) / bin_size + 1)
    # setup histogram reange
    hist_range = tuple([mrb.to_range(val, out_range=array_range)
                        for val in hist_range])
    # calculate histogram
    if scale == 'normed':
        is_normed = True
    else:
        is_normed = False
    # prepare figure
    if use_new_figure:
        plot = plt.figure()
    # prepare style list
    if styles is None:
        styles = [linestyle + color
                  for linestyle in PLOT_LINESTYLES for color in PLOT_COLORS]
    style_cycler = itertools.cycle(styles)
    # prepare histograms
    for idx, array in enumerate(array_list):
        hist, bin_edges = np.histogram(
            array, bins=bins, range=hist_range, normed=is_normed)
        # adjust scale
        if scale == 'log':
            hist[hist != 0.0] = np.log(hist[hist != 0.0])
        elif scale == 'log10':
            hist[hist != 0.0] = np.log10(hist[hist != 0.0])
        # prepare legend
        if legends is not None and idx < len(legends):
            legend = legends[idx]
        else:
            legend = '_nolegend_'
        # plot figure
        plt.plot(
            mrb.mid_val_array(bin_edges), hist, next(style_cycler),
            label=legend)
        plt.legend()
    # setup title and labels
    if title:
        plt.title(title)
    if labels[0]:
        plt.xlabel(labels[0])
    if labels[1]:
        plt.ylabel(labels[1] + ' ({})'.format(scale))
    else:
        plt.ylabel('{}'.format(scale))
    # save figure to file
    if save_path is not None:
        plt.tight_layout()
        plt.savefig(save_path, dpi=D_PLOT_DPI)
    # closing figure
    if close_figure:
        plt.close()
    return hist, bin_edges, plot


# ======================================================================
def histogram2d(
        array1,
        array2,
        bin_size=1,
        hist_range=(0.0, 1.0),
        bins=None,
        array_range=None,
        use_separate_range=False,
        scale='linear',
        interpolation='bicubic',
        title='2D Histogram',
        labels=('Array 1 Values', 'Array 2 Values'),
        cmap=plt.cm.jet,
        bisector=None,
        use_new_figure=True,
        close_figure=False,
        save_path=None):
    """
    Plot 2D histogram of two arrays with MatPlotLib.

    Parameters
    ==========
    array1 : ndarray
        The array 1 for which the 2D histogram is to be plotted.
    array2 : ndarray
        The array 1 for which the 2D histogram is to be plotted.
    bin_size : int or float | int 2-tuple (optional)
        The size of the bins.
    hist_range : float 2-tuple | 2-tuple of float 2-tuple (optional)
        The range of the histogram to display in percentage.
    bins : int | int 2-tuple (optional)
        The number of bins to use. If set, it overrides bin_size parameter.
    array_range : float 2-tuple | 2-tuple of float 2-tuple (optional)
        Theoretical range of values for the array. If unset, uses min and max.
    use_separate_range : bool (optional)
        Select if display ranges in each dimension are determined separately.
    scale : ['linear'|'log'|'log10'|'normed'] string (optional)
        The frequency value scaling.
    interpolation : str (optional)
        Interpolation method (see imshow()).
    title : str (optional)
        The title of the plot.
    labels : str 2-tuple (optional)
        A 2-tuple of strings containing x-labels and y-labels.
    cmap : MatPlotLib ColorMap (optional)
        The colormap to be used for displaying the histogram.
    bisector : str or None (optional)
        If not None, show the first bisector using specified line style.
    use_new_figure : bool (optional)
        Plot the histogram in a new figure.
    close_figure : bool (optional)
        Close the figure after saving (useful for batch processing).
    save_path : str (optional)
        The path to which the plot is to be saved. If unset, no output.

    Returns
    =======
    hist2d : array
        The calculated 2D histogram.

    """
    # setup array range
    if not array_range:
        if use_separate_range:
            array_range = (
                (np.nanmin(array1), np.nanmax(array1)),
                (np.nanmin(array2), np.nanmax(array2)))
        else:
            array_range = (
                min(np.nanmin(array1), np.nanmin(array2)),
                max(np.nanmax(array1), np.nanmax(array2)))
    try:
        array_range[0].__iter__
    except AttributeError:
        array_range = (array_range, array_range)
    # setup bins
    if not bins:
        bins = tuple([int(mrb.range_size(a_range) / bin_size + 1)
                      for a_range in array_range])
    else:
        try:
            bins.__iter__
        except AttributeError:
            bins = (bins, bins)
    # setup histogram range
    try:
        hist_range[0].__iter__
    except AttributeError:
        hist_range = (hist_range, hist_range)
    hist_range = list(hist_range)
    for idx in range(2):
        hist_range[idx] = tuple([
                                    mrb.to_range(val,
                                                 out_range=array_range[idx])
                                    for val in hist_range[idx]])
    hist_range = tuple(hist_range)
    # calculate histogram
    if scale == 'normed':
        is_normed = True
    else:
        is_normed = False
    # prepare histogram
    hist, x_edges, y_edges = np.histogram2d(
        array1.ravel(), array2.ravel(),
        bins=bins, range=hist_range, normed=is_normed)
    hist = hist.transpose()
    # adjust scale
    if scale == 'log':
        hist[hist != 0.0] = np.log(hist[hist != 0.0])
    elif scale == 'log10':
        hist[hist != 0.0] = np.log10(hist[hist != 0.0])
    # prepare figure
    if use_new_figure:
        plot = plt.figure()
    # plot figure
    plt.imshow(
        hist, cmap=cmap, origin='lower', interpolation=interpolation,
        vmin=np.floor(np.min(hist)), vmax=np.ceil(np.max(hist)),
        extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]])
    # plot the color bar
    plt.colorbar(use_gridspec=True)
    # plot first bisector
    if bisector:
        plt.autoscale(False)
        x_val, y_val = [np.linspace(*val_range) for val_range in array_range]
        plt.plot(array_range[0], array_range[1], bisector, label='bisector')
    # setup title and labels
    plt.title(title + ' ({} scale)'.format(scale))
    plt.xlabel(labels[0])
    plt.ylabel(labels[1])
    # save figure to file
    if save_path is not None:
        plt.tight_layout()
        plt.savefig(save_path, dpi=D_PLOT_DPI)
    # closing figure
    if close_figure:
        plt.close()
    return hist, x_edges, y_edges, plot
