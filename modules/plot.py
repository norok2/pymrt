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
from __future__ import unicode_literals
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
import matplotlib as mpl  # Matplotlib (2D/3D plotting library)
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
import mri_tools.modules.base as mrb

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
def quick_1d(array):
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
    # todo: plot 1d projections (use Qt)
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
def quick_2d(array):
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
    # todo: plot 2d projections (use Qt)
    if array.ndim == 2:
        # using Matplotlib
        fig = plt.subplots()
        plt.imshow(array.astype(float), cmap=plt.cm.binary)
        plt.draw()
        plt.show()
    elif 2 < array.ndim <= 4:
        # using Matplotlib
        nrows = 1
        fig = plt.subplots()
        plt.imshow(array.astype(float), cmap=plt.cm.binary)
        plt.draw()
        plt.show()
    else:
        print('W: cannot plot current array with 2d projections.')


# ======================================================================
def sample2d(
        array,
        axis=None,
        index=None,
        title=None,
        array_range=None,
        ticks_limit=None,
        orientation=None,
        cmap=None,
        colorbar_opts=None,
        colorbar_text=None,
        text_list=None,
        use_new_figure=True,
        close_figure=False,
        save_path=None,
        ax=None):
    """
    Plot a 2D sample image of a 3D array.

    Parameters
    ==========
    array : ndarray
        The original 3D array.
    axis : int (optional)
        The slicing axis. If None, use the shortest one.
    index : int (optional)
        The slicing index. If None, mid-value is taken.
    title : str (optional)
        The title of the plot.
    array_range : 2-tuple (optional)
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
    plot : matplotlib.pyplot.Figure
        The figure object containing the plot.

    """
    if array.ndim != 3:
        raise IndexError('3D array required')
    if use_new_figure:
        fig = plt.figure()
    if ax is None:
        ax = plt.gca()
    if axis is None:
        axis = np.argmin(array.shape)
    sample = mrb.slice_array(array, axis, index)
    if title:
        ax.set_title(title)
    if array_range is None:
        array_range = mrb.range_array(array)
    if not cmap:
        if array_range[0] * array_range[1] < 0:
            cmap = plt.cm.seismic
        else:
            cmap = plt.cm.binary
    ax.set_aspect('equal')
    if (orientation == 'portrait' and sample.shape[0] < sample.shape[1]) or \
            (orientation == 'landscape' and sample.shape[0] > sample.shape[1]):
        sample = sample.transpose()
    plot = ax.imshow(sample, cmap=cmap, vmin=array_range[0],
                     vmax=array_range[1])
    if ticks_limit is not None:
        if ticks_limit > 0:
            ax.locator_params(nbins=ticks_limit)
        else:
            ax.set_xticks([])
            ax.set_yticks([])
    if colorbar_opts is not None:
        cbar = ax.figure.colorbar(plot, ax=ax, **colorbar_opts)
        if colorbar_text is not None:
            cbar.set_label(colorbar_text)
    # include additional text
    if text_list is not None:
        for text_kwarg in text_list:
            ax.text(**text_kwarg)
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
        ticks_limits=None,
        scale='linear',
        title='Histogram',
        labels=('Value', 'Value Frequency'),
        style='-k',
        text_list=None,
        use_new_figure=True,
        close_figure=False,
        save_path=None,
        ax=None):
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
    bin_edges : array
        The bin edges of the calculated histogram.
    fig : matplotlib.pyplot.Figure
        The figure object containing the plot.

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
        fig = plt.figure()
    # create histogram
    hist, bin_edges = np.histogram(
        array, bins=bins, range=hist_range, normed=is_normed)
    # adjust scale
    hist = hist.astype(float)
    if scale == 'log':
        hist[hist > 0.0] = np.log(hist[hist > 0.0])
    elif scale == 'log10':
        hist[hist > 0.0] = np.log10(hist[hist > 0.0])
    # plot figure
    if ax is None:
        ax = plt.gca()
    plot = ax.plot(mrb.mid_val_array(bin_edges), hist, style)
    # setup title and labels
    if title:
        ax.set_title(title)
    if labels[0]:
        ax.set_xlabel(labels[0])
    if labels[1]:
        ax.set_ylabel(labels[1] + ' ({})'.format(scale))
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
        arrays,
        bin_size=1,
        hist_range=(0.0, 1.0),
        bins=None,
        array_range=None,
        ticks_limit=None,
        scale='linear',
        title='Histogram',
        labels=('Value', 'Value Frequency'),
        legends=None,
        legend_opts=None,
        styles=None,
        text_list=None,
        use_new_figure=True,
        close_figure=False,
        save_path=None,
        ax=None):
    """

    Args:
        arrays list[ndarray]: The array for which histogram is to be plotted.
        bin_size [int|float]: The size of the bins.
        hist_range:
        bins:
        array_range:
        scale:
        title:
        labels:
        legends:
        styles:
        text_list:
        use_new_figure:
        close_figure:
        save_path:

    Returns:

    """

    """
    Plot 1D histograms of multiple arrays with MatPlotLib.

    Parameters
    ==========
    array : ndarray
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
    fig : matplotlib.pyplot.Figure
        The figure object containing the plot.

    """
    # setup array range
    if not array_range:
        array_range = (np.nanmin(arrays[0]), np.nanmax(arrays[0]))
        for array in arrays[1:]:
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
        fig = plt.figure()
    # prepare style list
    if styles is None:
        styles = [linestyle + color
                  for linestyle in PLOT_LINESTYLES for color in PLOT_COLORS]
    style_cycler = itertools.cycle(styles)

    # prepare histograms
    if ax is None:
        ax = plt.gca()
    ax.set_aspect('auto')
    plots = []
    for i, array in enumerate(arrays):
        hist, bin_edges = np.histogram(
            array, bins=bins, range=hist_range, normed=is_normed)
        # adjust scale
        hist = hist.astype(float)
        if scale == 'log':
            hist[hist > 0.0] = np.log(hist[hist > 0.0])
        elif scale == 'log10':
            hist[hist > 0.0] = np.log10(hist[hist > 0.0])
        # prepare legend
        if legends is not None and i < len(legends):
            legend = legends[i]
        else:
            legend = '_nolegend_'
        # plot figure
        plot = ax.plot(
            mrb.mid_val_array(bin_edges), hist, next(style_cycler),
            label=legend)
        plots.append(plot)
    # create the legend for the first line.
    ax.legend(**(legend_opts if legend_opts is not None else {}))
    # fine-tune ticks
    if ticks_limit is not None:
        if ticks_limit > 0:
            ax.locator_params(nbins=ticks_limit)
        else:
            ax.set_xticks([])
            ax.set_yticks([])
    # setup title and labels
    if title:
        ax.set_title(title.format(bins=bins, scale=scale))
    if labels[0]:
        ax.set_xlabel(labels[0])
    if labels[1]:
        ax.set_ylabel(labels[1] + ' ({})'.format(scale))
    else:
        plt.ylabel('{}'.format(scale))
    # save figure to file
    if save_path is not None:
        plt.tight_layout()
        plt.savefig(save_path, dpi=D_PLOT_DPI)
    # closing figure
    if close_figure:
        plt.close()
    return hist, bin_edges, plots


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
        hist_val_range=None,
        ticks_limit=None,
        interpolation='bicubic',
        title='2D Histogram',
        labels=('Array 1 Values', 'Array 2 Values'),
        text_list=None,
        cmap=plt.cm.hot_r,
        bisector=None,
        stats_opts=None,
        colorbar_opts=None,
        colorbar_text=None,
        use_new_figure=True,
        close_figure=False,
        save_path=None,
        ax=None):
    """

    Args:
        array1 (ndarray):
        array2:
        bin_size:
        hist_range:
        bins:
        array_range:
        use_separate_range:
        scale:
        hist_val_range:
        ticks_limit:
        interpolation:
        title:
        labels:
        text_list:
        cmap:
        bisector:
        show_stats:
        colorbar_opts:
        colorbar_text:
        use_new_figure:
        close_figure:
        save_path:
        ax:

    Returns:

    """

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
    hist_val_range : float 2-tuple (optional)
        The range of histogram values. If None, it is calculated automatically.
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
    hist2d : ndarray
        The calculated 2D histogram.
    x_edges : ndarray
        The bin edges on the x-axis.
    y_edges : ndarray
        The bin edges on the y-axis.
    plot : matplotlib.pyplot.Figure
        The figure object containing the plot.

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
    for i in range(2):
        hist_range[i] = tuple(
            [mrb.to_range(val, out_range=array_range[i])
             for val in hist_range[i]])
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
    hist = hist.astype(float)
    if scale == 'log':
        hist[hist > 0.0] = np.log(hist[hist > 0.0])
    elif scale == 'log10':
        hist[hist > 0.0] = np.log10(hist[hist > 0.0])
    # adjust histogram intensity range
    if hist_val_range is None:
        hist_val_range = (np.floor(np.min(hist)), np.ceil(np.max(hist)))
    # prepare figure
    if use_new_figure:
        fig = plt.figure()
    if ax is None:
        ax = plt.gca()
    # plot figure
    plot = ax.imshow(
        hist, cmap=cmap, origin='lower', interpolation=interpolation,
        vmin=hist_val_range[0], vmax=hist_val_range[1],
        extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]])
    # plot the color bar
    if colorbar_opts is not None:
        cbar = ax.figure.colorbar(plot, ax=ax, **colorbar_opts)
        if colorbar_text is not None:
            cbar.set_label(colorbar_text)
        if ticks_limit is not None:
            if ticks_limit > 0:
                cbar.locator = mpl.ticker.MaxNLocator(nbins=ticks_limit)
            else:
                cbar.set_ticks([])
            cbar.update_ticks()
    # plot first bisector
    if bisector:
        ax.autoscale(False)
        ax.plot(array_range[0], array_range[1], bisector, label='bisector')
    if stats_opts is not None:
        mask = np.ones_like(array1 * array2).astype(bool)
        mask *= (array1 > hist_range[0][0]).astype(bool)
        mask *= (array1 < hist_range[0][1]).astype(bool)
        mask *= (array2 > hist_range[1][0]).astype(bool)
        mask *= (array2 < hist_range[1][1]).astype(bool)
        stats_dict = mrb.calc_stats(
            array1[mask] - array2[mask], **stats_opts)
        stats_text = '$\\mu_D = {}$\n$\\sigma_D = {}$'.format(
            *mrb.format_value_error(stats_dict['avg'], stats_dict['std'], 3))
        ax.text(
            1 / 2, 31 / 32, stats_text,
            horizontalalignment='center', verticalalignment='top',
            transform=ax.transAxes)
    # setup title and labels
    ax.set_title(title.format(bins=bins, scale=scale))
    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])
    # fine-tune ticks
    if ticks_limit is not None:
        if ticks_limit > 0:
            ax.locator_params(nbins=ticks_limit)
        else:
            ax.set_xticks([])
            ax.set_yticks([])
    # include additional text
    if text_list is not None:
        for text_kwarg in text_list:
            ax.text(**text_kwarg)
    # save figure to file
    if save_path is not None:
        plt.tight_layout()
        plt.savefig(save_path, dpi=D_PLOT_DPI)
    # closing figure
    if close_figure:
        plt.close()
    plt.figure()
    return hist, x_edges, y_edges, plot
