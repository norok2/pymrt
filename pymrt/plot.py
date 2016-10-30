#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
PyMRT: useful basic utilities.
"""

# ======================================================================
# :: Future Imports
from __future__ import (
    division, absolute_import, print_function, unicode_literals)

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
import warnings  # Warning control

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
import matplotlib.animation as anim  # Matplotlib's animations
from mpl_toolkits.mplot3d.axes3d import Axes3D
# import scipy.optimize  # SciPy: Optimization Algorithms
# import scipy.integrate  # SciPy: Integrations facilities
# import scipy.constants  # SciPy: Mathematal and Physical Constants
# import scipy.ndimage  # SciPy: ND-image Manipulation
import scipy.stats  # SciPy: Statistical functions

# :: Local Imports
import pymrt.utils as pmu

# from pymrt import INFO
from pymrt import VERB_LVL, D_VERB_LVL, VERB_LVL_NAMES
from pymrt import msg, dbg

# ======================================================================
# :: MatPlotLib-related constants
# standard plot resolutions
D_PLOT_DPI = 72
# colors and linestyles
PLOT_COLORS = mpl.rcParams['axes.prop_cycle']
PLOT_LINESTYLES = ('-', '--', '-.', ':')


# ======================================================================
# def plot_with_adjusting_parameters()
# TODO: make a plot with possibility to adjust params


def explore(array):
    """
    Generate a visualization of an ND-array.

    Args:
        array:

    Returns:
        None
    """
    # todo: implement!
    pass


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
        quick_1d(array)
    elif array.ndim == 2:
        quick_2d(array)
    elif array.ndim == 3:
        quick_3d(array)
    else:
        warnings.warn('cannot quickly plot this array (try `explore`)')
    plt.show()


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
    if array.ndim == 1:
        # using Matplotlib
        plt.figure()
        plt.plot(np.arange(len(array)), array.astype(float))
    elif array.ndim > 1:
        # todo: 1D projection
        pass
    else:
        warnings.warn('cannot plot (1D projection of) current array')


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
    if array.ndim == 2:
        # using Matplotlib
        fig = plt.subplots()
        plt.imshow(array.astype(float), cmap=plt.cm.gray)
    elif array.ndim > 2:
        # todo: 2D projection
        pass
    else:
        warnings.warn('cannot plot (2D projection of) current array')


def quick_3d(array):
    """
    TODO: DOCSTRING.

    Args:
        array:

    Returns:
        None
    """
    warnings.warn('3D-support plots might be slow (consider using `explore`)')
    if array.ndim == 3:
        # using Matplotlib
        from skimage import measure

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1, projection='3d')
        # zz, xx, yy = array.nonzero()
        # ax.scatter(xx, yy, zz, cmap=plt.cm.hot)

        verts, faces = measure.marching_cubes(array, 0.5, (2,) * 3)
        ax.plot_trisurf(
            verts[:, 0], verts[:, 1], faces, verts[:, 2], cmap='Spectral',
            antialiased=False, linewidth=0.0)
    elif array.ndim > 3:
        # todo: 3D projection
        pass
    else:
        warnings.warn('cannot plot (3D projection of) current array')


# ======================================================================
def sample2d(
        arr,
        axis=None,
        index=None,
        title=None,
        array_interval=None,
        ticks_limit=None,
        orientation=None,
        cmap=None,
        cbar_kws=None,
        cbar_txt=None,
        text_color=None,
        text_list=None,
        resolution=None,
        size_info=None,
        use_new_figure=True,
        close_figure=False,
        save_filepath=None,
        ax=None):
    """
    Plot a 2D sample image of a 3D array.

    Parameters
    ==========
    arr : ndarray
        The original 3D array.
    axis : int (optional)
        The slicing axis. If None, use the shortest one.
    index : int (optional)
        The slicing index. If None, mid-value is taken.
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
    ndim = 2
    if use_new_figure:
        fig = plt.figure()
    if ax is None:
        ax = plt.gca()
    if axis is None:
        axis = np.argsort(arr.shape)[:-ndim]
    else:
        axis = pmu.auto_repeat(axis, 1)
    if arr.ndim - len(axis) != 2:
        raise IndexError(
            'Mismatching dimensions ({ndim}) and axis ({naxes}): '
            '{ndim} - {naxes} != 2'.format(ndim=arr.ndim, naxes=len(axis)))
    sample = pmu.ndim_slice(arr, axis, index)
    if title:
        ax.set_title(title)
    if array_interval is None:
        array_interval = pmu.minmax(arr)
    if not cmap:
        if array_interval[0] * array_interval[1] < 0:
            cmap = plt.cm.RdBu_r
        else:
            cmap = plt.cm.gray_r
    if not text_color:
        if array_interval[0] * array_interval[1] < 0:
            text_color = 'k'
        else:
            text_color = 'k'
    ax.set_aspect('equal')
    if (orientation == 'portrait' and sample.shape[0] < sample.shape[1]) or \
            (orientation == 'landscape' and sample.shape[0] > sample.shape[1]):
        sample = sample.transpose()
    plot = ax.imshow(sample, cmap=cmap, vmin=array_interval[0],
                     vmax=array_interval[1], interpolation='none')
    if ticks_limit is not None:
        if ticks_limit > 0:
            ax.locator_params(nbins=ticks_limit)
        else:
            ax.set_xticks([])
            ax.set_yticks([])
    if cbar_kws is not None:
        from mpl_toolkits.axes_grid1 import make_axes_locatable, axes_size
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        cbar = ax.figure.colorbar(plot, cax=cax, **cbar_kws)
        # cbar = ax.figure.colorbar(plot, ax=ax, **cbar_kws)
        if cbar_txt is not None:
            only_extremes = 'ticks' in cbar_kws and len(cbar_kws['ticks']) == 2
            if only_extremes:
                cbar.ax.text(2.0, 0.5, cbar_txt, fontsize='medium', rotation=90)
            else:
                cbar.set_label(cbar_txt)
    # print resolution information and draw a ruler
    if size_info is not None and resolution is not None:
        if size_info >= 0.0:
            # print resolution information
            if resolution[0] == resolution[1] == resolution[2]:
                x = resolution[0]
                res_str = '{} {} iso.'.format(str(x), 'mm')
            else:
                res_str = 'x'.join([str(x) for x in resolution[0:3]]) \
                          + ' ' + 'mm'
            ax.text(
                0.975, 0.975, res_str, rotation=0, color=text_color,
                horizontalalignment='right', verticalalignment='top',
                transform=ax.transAxes)
        if size_info != 0.0:
            res = resolution[1]
            size_info_size = round(abs(size_info) * (sample.shape[1] * res), -1)
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
    # include additional text
    if text_list is not None:
        for text_kwarg in text_list:
            ax.text(**text_kwarg)
    if save_filepath is not None:
        plt.tight_layout()
        plt.savefig(save_filepath, dpi=D_PLOT_DPI)
    if close_figure:
        plt.close()
    return sample, plot


# ======================================================================
def sample2d_anim(
        array,
        axis=None,
        step=1,
        duration=10,
        title=None,
        array_interval=None,
        ticks_limit=None,
        orientation=None,
        cmap=None,
        cbar_kws=None,
        cbar_txt=None,
        text_color=None,
        text_list=None,
        resolution=None,
        size_info=None,
        dpi=300,
        close_figure=False,
        save_filepath=None,
        ax=None):
    """
    Plot a 2D sample image of a 3D array.

    Parameters
    ==========
    array : ndarray
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
    if array.ndim != 3:
        raise IndexError('3D array required')
    fig = plt.figure()
    if ax is None:
        ax = plt.gca()
    if axis is None:
        axis = np.argmin(array.shape)
    sample = pmu.ndim_slice(array, axis, 0)
    if title:
        ax.set_title(title)
    if array_interval is None:
        array_interval = pmu.minmax(array)
    if not cmap:
        if array_interval[0] * array_interval[1] < 0:
            cmap = plt.cm.RdBu
        else:
            cmap = plt.cm.gray
    if not text_color:
        if array_interval[0] * array_interval[1] < 0:
            text_color = 'k'
        else:
            text_color = 'k'
    ax.set_aspect('equal')
    if (orientation == 'portrait' and sample.shape[0] < sample.shape[1]) or \
            (orientation == 'landscape' and sample.shape[0] > sample.shape[1]):
        sample = sample.transpose()
    if ticks_limit is not None:
        if ticks_limit > 0:
            ax.locator_params(nbins=ticks_limit)
        else:
            ax.set_xticks([])
            ax.set_yticks([])

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
            size_info_size = round(abs(size_info) * (sample.shape[1] * res), -1)
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
    # include additional text
    if text_list is not None:
        for text_kwarg in text_list:
            ax.text(**text_kwarg)
    n_frames = array.shape[axis]
    plots = []
    for i in range(0, n_frames, step):
        plot = ax.imshow(
            pmu.ndim_slice(array, axis, i), cmap=cmap,
            vmin=array_interval[0], vmax=array_interval[1], animated=True)
        if len(plots) <= 0:
            if cbar_kws is not None:
                cbar = ax.figure.colorbar(plot, ax=ax, **cbar_kws)
                if cbar_txt is not None:
                    only_extremes = 'ticks' in cbar_kws and \
                                    len(cbar_kws['ticks']) == 2
                    if only_extremes:
                        cbar.ax.text(
                            2.0, 0.5, cbar_txt, fontsize='small', rotation=90)
                    else:
                        cbar.set_label(cbar_txt)
        plots.append([plot])
    mov = anim.ArtistAnimation(fig, plots, blit=False)
    if save_filepath is not None:
        plt.tight_layout()
        mov.save(save_filepath, dpi=dpi, fps=n_frames / step / duration)
    if close_figure:
        plt.close()
    return mov


# ======================================================================
def histogram1d(
        array,
        bin_size=1,
        hist_interval=(0.0, 1.0),
        bins=None,
        array_interval=None,
        ticks_limits=None,
        scale='linear',
        title='Histogram',
        labels=('Value', 'Value Frequency'),
        style='-k',
        text_list=None,
        use_new_figure=True,
        close_figure=False,
        save_filepath=None,
        ax=None):
    """
    Plot 1D histogram of array with MatPlotLib.

    Parameters
    ==========
    array : nd-array
        The array for which histogram is to be plotted.
    bin_size : int or float or float (optional)
        The size of the bins.
    hist_interval : float 2-tuple (optional)
        The range of the histogram to display in percentage.
    bins : int (optional)
        The number of bins to use. If set, it overrides bin_size parameter.
    array_interval : float 2-tuple (optional)
        Theoretical range of values for the array. If unset, uses min and max.
    scale : ['linear'|'log'|'log10'|'density'] string (optional)
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
    save_filepath : str (optional)
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
    # todo: implement optimal bin size

    # setup array range
    if not array_interval:
        array_interval = (np.nanmin(array), np.nanmax(array))
    # setup bins
    if not bins:
        bins = int(np.ptp(array_interval) / bin_size + 1)
    # setup histogram reange
    hist_interval = tuple([pmu.scale(val, array_interval)
                           for val in hist_interval])
    # calculate histogram
    # prepare figure
    if use_new_figure:
        fig = plt.figure()
    # create histogram
    hist, bin_edges = np.histogram(
        array, bins=bins, range=hist_interval, density=(scale == 'density'))
    # adjust scale
    hist = hist.astype(float)
    if scale == 'log':
        hist[hist > 0.0] = np.log(hist[hist > 0.0])
    elif scale == 'log10':
        hist[hist > 0.0] = np.log10(hist[hist > 0.0])
    # plot figure
    if ax is None:
        ax = plt.gca()
    plot = ax.plot(pmu.midval(bin_edges), hist, style)
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
    if save_filepath is not None:
        plt.tight_layout()
        plt.savefig(save_filepath, dpi=D_PLOT_DPI)
    # closing figure
    if close_figure:
        plt.close()
    return hist, bin_edges, plot


# ======================================================================
def histogram1d_list(
        arrays,
        bin_size=1,
        hist_interval=(0.0, 1.0),
        bins=None,
        array_interval=None,
        ticks_limit=None,
        scale='linear',
        title='Histogram',
        labels=('Value', 'Value Frequency'),
        legends=None,
        legend_kws=None,
        styles=None,
        text_list=None,
        use_new_figure=True,
        close_figure=False,
        save_filepath=None,
        ax=None):
    """

    Args:
        arrays list[ndarray]: The array for which histogram is to be plotted.
        bin_size [int|float]: The size of the bins.
        hist_interval:
        bins:
        array_interval:
        scale:
        title:
        labels:
        legends:
        styles:
        text_list:
        use_new_figure:
        close_figure:
        save_filepath:

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
    hist_interval : float 2-tuple (optional)
        The range of the histogram to display in percentage.
    bins : int (optional)
        The number of bins to use. If set, it overrides bin_size parameter.
    array_interval : float 2-tuple (optional)
        Theoretical range of values for the array. If unset, uses min and max.
    scale : ['linear'|'log'|'log10'|'density'] string (optional)
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
    save_filepath : str (optional)
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
    if not array_interval:
        array_interval = (np.nanmin(arrays[0]), np.nanmax(arrays[0]))
        for array in arrays[1:]:
            array_interval = (
                min(np.nanmin(array), array_interval[0]),
                max(np.nanmax(array), array_interval[1]))
    # setup bins
    if not bins:
        bins = int(np.ptp(array_interval) / bin_size + 1)
    # setup histogram reange
    hist_interval = tuple([pmu.scale(val, array_interval)
                           for val in hist_interval])
    # calculate histogram
    # prepare figure
    if use_new_figure:
        fig = plt.figure()
    # prepare style list
    if styles is None:
        styles = []
        for linestyle in PLOT_LINESTYLES:
            for color in PLOT_COLORS:
                style = {'linestyle': linestyle}
                style.update(color)
                styles.append(style)

    style_cycler = itertools.cycle(styles)

    # prepare histograms
    if ax is None:
        ax = plt.gca()
    ax.set_aspect('auto')
    plots = []
    for i, array in enumerate(arrays):
        hist, bin_edges = np.histogram(
            array, bins=bins, range=hist_interval, density=(scale == 'density'))
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
            pmu.midval(bin_edges), hist,
            **next(style_cycler),
            label=legend)
        plots.append(plot)
    # create the legend for the first line.
    ax.legend(**(legend_kws if legend_kws is not None else {}))
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
    if save_filepath is not None:
        plt.tight_layout()
        plt.savefig(save_filepath, dpi=D_PLOT_DPI)
    # closing figure
    if close_figure:
        plt.close()
    return hist, bin_edges, plots


# ======================================================================
def histogram2d(
        arr1,
        arr2,
        bin_size=1,
        bins=None,
        array_interval=None,
        hist_interval=((0.0, 1.0), (0.0, 1.0)),
        use_separate_interval=False,
        subplot_aspect=None,
        scale='linear',
        hist_val_interval=None,
        ticks_limit=None,
        interpolation='bicubic',
        title='2D Histogram',
        labels=('Array 1 Values', 'Array 2 Values'),
        text_list=None,
        cmap=plt.cm.afmhot_r,
        bisector=None,
        stats_kws=None,
        cbar_kws=None,
        cbar_txt=None,
        use_new_figure=True,
        close_figure=False,
        save_filepath=None,
        save_kws=None,
        ax=None):
    """
    Plot 2D histogram of two arrays.

    Args:
        arr1 (np.ndarray):
        arr2 (np.ndarray):
        bin_size (float|tuple[float]): The size of the bins.
            If a single number is given, the same number is used for both axes.
            Otherwise, the first number is used for the x-axis and the second
            number is used for the y-axis.
            It can be overridden by setting the 'bins' parameter.
        bins (int|tuple[int]|None): The number of bins to use.
            If a single number is given, the same number is used for both axes.
            Otherwise, the first number is used for the x-axis and the second
            number is used for the y-axis.
            It overrides the 'bin_size' parameter.
        array_interval (tuple[float|tuple[float]]|None): The values interval.
            If a tuple of float is given, the values outside of the (min, max)
            interval are not considered, and it is assumed identical for both
            axes.
            Otherwise, if a tuple of tuple of float is given, the first tuple of
            float is interpreted as the (min, max) for the x-axis, and the
            second tuple of float is for the y-axis.
        hist_interval (tuple[float|tuple[float]]): The histogram interval.
            If a tuple of float is given, it is interpreted as the (min, max)
            of the interval to use for the histogram as percentage of the array
            interval (specified or calculated), and it is assumed identical for
            both axes.
            Otherwise, if a tuple of tuple of float is given, the first tuple of
            float is interpreted as the (min, max) for the x-axis, and the
            second tuple of float is for the y-axis.
        use_separate_interval (bool): Generate 'array_interval' separately.
            If set, the array_interval is generated as:
            ((min(array1), max(array1)), (min(array2), max(array2)).
            Otherwise, uses information for both arrays to determine a common
            identical interval for both axis.
        subplot_aspect (float): subplot_aspect ratio of the histogram.
            If None, it is calculated to result in squared proportions.
        scale (str): The histogram frequency value transformation method.
            - 'linear': no transformation is performed;
            - 'log': the natual logarithm of the histogram frequency is used;
            - 'log10': the base-10 logarithm of the histogram frequency is used;
            - 'density': calculate the histogram setting the 'density' keyword
              parameter to True.
        hist_val_interval (tuple[float]|None): The interval of histogram values.
            If None, it is calculated automatically as the (min, max) of the
            histogram values rounded to the most comprehensive integer interval.
        ticks_limit (None): TODO
        interpolation (str): Image display interpolation method.
            See matplotlib.imshow for more details.
        title (str): The title of the plot.
        labels (tuple[str]): The x- and y- labels.
        text_list (tuple[dict]): The keyword arguments defining texts.
        cmap (mpl.cm): The colormap for the histogram.
        bisector (str|None): If not None, show the first bisector.
            The line style must be specified in the string format,
            as specified by matplotlib specifications.
        stats_kws (None): TODO
        cbar_kws (None): TODO
        cbar_txt (None): TODO
        use_new_figure (bool): Plot the histogram in a new figure.
        close_figure (bool): Close the figure after saving.
        save_filepath (str): The file path where the plot is saved to.
            If unset, no output.
        ax (matplotlib.axes): The Axes object to use for plotting.
            If None, gets the current Axes object.

    Returns:
        hist2d (np.ndarray): The calculated 2D histogram.
        x_edges (np.ndarray): The bin edges on the x-axis.
        y_edges (np.ndarray): The bin edges on the y-axis.
        plot (matplotlib.pyplot.Figure): The Figure object containing the plot.
    """

    def _ensure_all_axis(obj, n=2):
        try:
            iter(obj[0])
        except TypeError:
            obj = (obj,) * n
        return obj

    # setup array range
    if not array_interval:
        if use_separate_interval:
            array_interval = (
                (np.nanmin(arr1), np.nanmax(arr1)),
                (np.nanmin(arr2), np.nanmax(arr2)))
        else:
            array_interval = (
                min(np.nanmin(arr1), np.nanmin(arr2)),
                max(np.nanmax(arr1), np.nanmax(arr2)))

    array_interval = _ensure_all_axis(array_interval)
    # setup image subplot_aspect ratio
    if not subplot_aspect:
        x_axis_size = np.ptp(array_interval[0])
        y_axis_size = np.ptp(array_interval[1])
        if x_axis_size != y_axis_size:
            subplot_aspect = x_axis_size / y_axis_size
        else:
            subplot_aspect = 1.0
    # setup bins
    if not bins:
        bins = tuple([int(np.ptp(val) / bin_size + 1)
                      for val in array_interval])
    else:
        bins = _ensure_all_axis(bins)
    # setup histogram range
    hist_interval = _ensure_all_axis(hist_interval)
    hist_interval = tuple([[pmu.scale(val, array_interval[i])
                            for val in hist_interval[i]] for i in range(2)])
    # calculate histogram
    # prepare histogram
    hist, x_edges, y_edges = np.histogram2d(
        arr1.ravel(), arr2.ravel(),
        bins=bins, range=hist_interval, normed=(scale == 'density'))
    hist = hist.transpose()
    # adjust scale
    hist = hist.astype(float)
    if scale == 'log':
        hist[hist > 0.0] = np.log(hist[hist > 0.0])
    elif scale == 'log10':
        hist[hist > 0.0] = np.log10(hist[hist > 0.0])
    # adjust histogram intensity range
    if hist_val_interval is None:
        hist_val_interval = (np.floor(np.min(hist)), np.ceil(np.max(hist)))
    # prepare figure
    if use_new_figure:
        fig = plt.figure()
    if ax is None:
        ax = plt.gca()
    # plot figure
    plot = ax.imshow(
        hist, cmap=cmap, origin='lower', interpolation=interpolation,
        subplot_aspect=subplot_aspect,
        vmin=hist_val_interval[0], vmax=hist_val_interval[1],
        extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]])
    # plot the color bar
    if cbar_kws is not None:
        cbar = ax.figure.colorbar(plot, ax=ax, **cbar_kws)
        if cbar_txt is not None:
            only_extremes = 'ticks' in cbar_kws and len(cbar_kws['ticks']) == 2
            if only_extremes:
                cbar.ax.text(2.0, 0.5, cbar_txt, fontsize='small', rotation=90)
            else:
                cbar.set_label(cbar_txt)
        if ticks_limit is not None:
            if ticks_limit > 0:
                cbar.locator = mpl.ticker.MaxNLocator(nbins=ticks_limit)
            else:
                cbar.set_ticks([])
            cbar.update_ticks()
    # plot first bisector
    if bisector:
        ax.autoscale(False)
        ax.plot(array_interval[0], array_interval[1], bisector,
                label='bisector')
    if stats_kws is not None:
        mask = np.ones_like(arr1 * arr2).astype(bool)
        mask *= (arr1 > array_interval[0][0]).astype(bool)
        mask *= (arr1 < array_interval[0][1]).astype(bool)
        mask *= (arr2 > array_interval[1][0]).astype(bool)
        mask *= (arr2 < array_interval[1][1]).astype(bool)
        stats_dict = pmu.calc_stats(
            arr1[mask] - arr2[mask], **stats_kws)
        stats_text = '$\\mu_D = {}$\n$\\sigma_D = {}$'.format(
            *pmu.format_value_error(stats_dict['avg'], stats_dict['std'], 3))
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
    if save_filepath is not None:
        plt.tight_layout()
        plt.savefig(save_filepath, **save_kws)
    # closing figure
    if close_figure:
        plt.close()
    plt.figure()
    return hist, x_edges, y_edges, plot


# ======================================================================
def subplots(
        plots,
        rows=None,
        cols=None,
        num_row=None,
        num_col=None,
        aspect_ratio=None,
        width_height=None,
        swap_filling=False,
        title=None,
        row_labels=None,
        col_labels=None,
        row_label_width=0.030,
        col_label_width=0.030,
        border_top=0.010,
        border_left=0.010,
        pad=1.0,
        w_pad=2.0,
        h_pad=2.0,
        save_filepath=None,
        save_kws=None,
        force=False,
        verbose=D_VERB_LVL):
    """
    
    Args:
        plots (): 
        rows (): 
        cols (): 
        num_row (): 
        num_col (): 
        aspect_ratio ():
        width_height ():
        swap_filling (): 
        title (): 
        row_labels (): 
        col_labels (): 
        row_label_width (): 
        col_label_width (): 
        border_top (): 
        border_left (): 
        pad (): 
        w_pad (): 
        h_pad (): 
        save_filepath ():
        save_kws ():
        force (): 
        verbose (): 

    Returns:

    """
    num_plots = len(plots)

    # determine rows and cols if not specified
    if rows is None and cols is None:
        if num_row is None and num_col is None:
            if isinstance(aspect_ratio, float):
                num_col = np.ceil(np.sqrt(num_plots * aspect_ratio))
            elif 'exact' in aspect_ratio:
                num_col, num_row = pmu.optimal_ratio(num_plots)
                if 'portrait' in aspect_ratio:
                    num_row, num_col = num_col, num_row
            else:
                if aspect_ratio == 'portrait':
                    num_row = np.ceil(np.sqrt(num_plots))
                else:  # plot_aspect == 'landscape'
                    num_row = np.floor(np.sqrt(num_plots))
        if num_row is None and num_col > 0:
            num_row = np.ceil(num_plots / num_col)
        if num_row > 0 and num_col is None:
            num_row = np.ceil(num_plots / num_row)

        if width_height is None:
            width, height = 1, 1
        else:
            width, height = width_height
        rows = (width,) * num_row
        cols = (height,) * num_col
    else:
        num_row = len(rows)
        num_col = len(cols)
    assert (num_row * num_col >= num_plots)

    if row_labels is None:
        row_labels = pmu.auto_repeat(None, num_row)
        row_label_width = 0.0
        border_left = 0.0

    if col_labels is None:
        col_labels = pmu.auto_repeat(None, num_col)
        col_label_width = 0.0
        border_top = 0.0

    # generate plot
    fig, axs = plt.subplots(
        num_row, num_col, figsize=(sum(cols), sum(rows)))

    assert (num_row == len(row_labels))
    assert (num_col == len(col_labels))

    for i, row_label in enumerate(row_labels):
        for j, col_label in enumerate(col_labels):
            if swap_filling:
                n = i * num_row + j
            else:
                n = j * num_col + i
            axs[i, j] = plots[n]

            fig.text(
                pmu.scale((j * 2 + 1) / (num_col * 2),
                          out_interval=(
                              row_label_width + border_left, 1.0)),
                1.0 - border_top,
                col_label, rotation=0,
                fontweight='bold', fontsize='x-large',
                horizontalalignment='center', verticalalignment='top')

        fig.text(
            border_left,
            pmu.scale(1.0 - (i * 2 + 1) / (num_row * 2),
                      out_interval=(
                          border_top, 1 - col_label_width - border_top)),
            row_label, rotation=0,
            fontweight='bold', fontsize='x-large',
            horizontalalignment='left', verticalalignment='center')

    fig.tight_layout(
        rect=[0.0 + row_label_width, 0.0, 1.0, 1.0 - col_label_width],
        pad=pad, w_pad=w_pad, h_pad=h_pad)
    fig.savefig(save_filepath, **save_kws)
    msg('Plot: {}'.format(save_filepath), verbose, D_VERB_LVL['medium'])
    plt.close(fig)
