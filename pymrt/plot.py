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

# :: External Imports Submodules
import matplotlib.pyplot as plt  # Matplotlib's pyplot: MATLAB-like syntax
import matplotlib.gridspec
import matplotlib.animation as anim  # Matplotlib's animations
from mpl_toolkits.mplot3d.axes3d import Axes3D
# import scipy.optimize  # SciPy: Optimization Algorithms
# import scipy.integrate  # SciPy: Integrations facilities
# import scipy.constants  # SciPy: Mathematal and Physical Constants
# import scipy.ndimage  # SciPy: ND-image Manipulation
import scipy.stats  # SciPy: Statistical functions

# :: Local Imports
import pymrt as mrt
import pymrt.utils

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
def quick(array, values_range=None):
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
        quick_1d(array, values_range)
    elif array.ndim == 2:
        quick_2d(array, values_range)
    elif array.ndim == 3:
        quick_3d(array, values_range)
    else:
        warnings.warn('cannot quickly plot this array (try `explore`)')
    plt.show()


# ======================================================================
def quick_1d(array, values_range=None):
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
def quick_2d(arr, values_range=None):
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
    if arr.ndim == 2:
        img = arr
    else:
        warnings.warn('current array was not 2D, performing brute conversion')
        img = arr.reshape(mrt.utils.optimal_ratio(arr.size))

    if not values_range:
        values_range = mrt.utils.minmax(img)

    print(values_range)
    # using Matplotlib
    fig = plt.subplots()
    plt.imshow(
        img.astype(float), cmap=plt.cm.gray,
        vmin=values_range[0], vmax=values_range[1])


def quick_3d(array, values_range=None):
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
        # ax.scatter(xx, yy, zz, cmap=plt.cm.afmhot_r)

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
def simple(
        x_datas, y_datas,
        title=None,
        labels=(None, None),
        styles=None,
        legends=None,
        legend_kws=None,
        more_texts=None,
        ax=None,
        save_filepath=None,
        save_kws=None,
        force=False,
        verbose=D_VERB_LVL):
    # plot figure
    if ax is None:
        fig = plt.figure()
        ax = fig.gca()
    else:
        fig = plt.gcf()
    if isinstance(x_datas, np.ndarray):
        x_datas = mrt.utils.auto_repeat(x_datas, len(y_datas), True, True)
    if legends is None:
        legends = mrt.utils.auto_repeat(None, len(y_datas), check=True)
    for x_data, y_data, legend in zip(x_datas, y_datas, legends):
        plot = ax.plot(x_data, y_data, label=legend)
    # setup title and labels
    if title:
        ax.set_title(title.format_map(locals()))
    if labels[0]:
        ax.set_xlabel(labels[0].format_map(locals()))
    if labels[1]:
        ax.set_ylabel(labels[1].format_map(locals()))
    if any([legend for legend in legends]):
        ax.legend(**(legend_kws if legend_kws is not None else {}))
    # include additional text
    if more_texts is not None:
        for text_kws in more_texts:
            ax.text(**dict(text_kws))
    # save figure to file
    if save_filepath and mrt.utils.check_redo(None, [save_filepath], force):
        fig.tight_layout()
        if save_kws is None:
            save_kws = {}
        fig.savefig(save_filepath, **dict(save_kws))
        msg('Plot: {}'.format(save_filepath, verbose, VERB_LVL['medium']))
        plt.close(fig)
    return (x_datas, y_datas), fig


# ======================================================================
def sample2d(
        arr,
        axis=None,
        index=None,
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
        ax=None,
        save_filepath=None,
        save_kws=None,
        force=False,
        verbose=D_VERB_LVL):
    """
    Plot a 2D sample image of an ND array.

    TODO: fix documentation!!!

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
    data : ndarray
        The sliced (N-1)D-array.
    plot : matplotlib.pyplot.Figure
        The figure object containing the plot.

    """
    data_dim = 2

    if ax is None:
        fig = plt.figure()
        ax = fig.gca()
    else:
        fig = plt.gcf()

    # prepare data
    if axis is None:
        axis = np.argsort(arr.shape)[:-data_dim]
    else:
        axis = mrt.utils.auto_repeat(axis, 1)
    if index is not None:
        index = mrt.utils.auto_repeat(index, 1)
        if len(index) != len(axis):
            raise IndexError(
                'Mismatching number of axis ({num_axis}) and index '
                '({num_index})'.format(
                    num_axis=len(axis), num_index=len(index)))

    if arr.ndim - len(axis) == data_dim:
        data = mrt.utils.ndim_slice(arr, axis, index)
    elif arr.ndim == data_dim:
        data = arr
    else:
        raise IndexError(
            'Mismatching dimensions ({dim}) and axis ({num_axes}): '
            '{dim} - {num_axes} != {data_dim}'.format(
                dim=arr.ndim, num_axes=len(axis), data_dim=data_dim))
    if ((orientation == 'transpose') or
            (orientation == 'landscape' and data.shape[0] > data.shape[1]) or
            (orientation == 'portrait' and data.shape[0] < data.shape[1])):
        data = data.transpose()
    if orientation == 'rot90':
        data = np.rot90(data)
    if flip_ud:
        data = data[::-1, :]
    if flip_lr:
        data = data[:, ::-1]

    # prepare plot
    if title:
        ax.set_title(title)
    if array_interval is None:
        array_interval = mrt.utils.minmax(arr)
    if not cmap:
        if not mrt.utils.is_same_sign(array_interval):
            cmap = mpl.cm.RdBu_r
        else:
            cmap = mpl.cm.gray_r
    if not text_color:
        if not mrt.utils.is_same_sign(array_interval):
            text_color = 'k'
        else:
            text_color = 'k'
    ax.set_aspect('equal')

    # plot data
    plot = ax.imshow(
        data, cmap=cmap, vmin=array_interval[0], vmax=array_interval[1],
        interpolation='none')

    # plot ticks in plotting axes
    if ticks_limit is not None:
        if ticks_limit > 0:
            ax.locator_params(nbins=ticks_limit)
        else:
            ax.set_xticks([])
            ax.set_yticks([])

    # set colorbar
    if cbar_kws is not None:
        from mpl_toolkits.axes_grid1 import make_axes_locatable, axes_size
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        cbar = ax.figure.colorbar(plot, cax=cax, **dict(cbar_kws))
        # cbar = ax.figure.colorbar(plot, ax=ax, **dict(cbar_kws))
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
            size_info_size = round(abs(size_info) * (data.shape[1] * res), -1)
            size_info_str = '{} {}'.format(size_info_size, 'mm')
            size_info_px = size_info_size / res
            ax.text(
                0.025, 0.050, size_info_str, rotation=0, color=text_color,
                horizontalalignment='left', verticalalignment='bottom',
                transform=ax.transAxes)
            ax.plot(
                (data.shape[1] * 0.025,
                 data.shape[1] * 0.025 + size_info_px),
                (data.shape[0] * 0.965, data.shape[0] * 0.965),
                color=text_color, linewidth=2.5)

    # include additional text
    if more_texts is not None:
        for text_kws in more_texts:
            ax.text(**dict(text_kws))

    # save plot
    if save_filepath and mrt.utils.check_redo(None, [save_filepath], force):
        fig.tight_layout()
        if save_kws is None:
            save_kws = {}
        fig.savefig(save_filepath, **dict(save_kws))
        msg('Plot: {}'.format(save_filepath, verbose, VERB_LVL['medium']))
        plt.close(fig)

    return data, fig


# ======================================================================
def sample3d_view2d(
        arr,
        transform=None,
        axis=None,
        index=None,
        view_axes=None,
        view_indexes=None,
        view_flip=(False, False, False),
        orientation='landscape',
        flip_ud=(False, False, False),
        flip_lr=(False, False, False),
        rot90=(False, False, False),
        transpose=(False, False, False),
        mode='standard',
        title=None,
        array_interval=None,
        ticks_limit=None,
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
    warnings.warn('Experimental!')
    data_dim = 3
    view_dim = 2

    if ax is None:
        fig = plt.figure()
        ax = fig.gca()
    else:
        fig = plt.gcf()

    # prepare data
    if axis is None:
        axis = np.argsort(arr.shape)[:-data_dim]
    else:
        axis = mrt.utils.auto_repeat(axis, 1)
    if index is not None:
        index = mrt.utils.auto_repeat(index, 1)
        if len(index) != len(axis):
            raise IndexError(
                'Mismatching number of axis ({num_axis}) and index '
                '({num_index})'.format(
                    num_axis=len(axis), num_index=len(index)))

    if arr.ndim - len(axis) == data_dim:
        data = mrt.utils.ndim_slice(arr, axis, index)
    elif arr.ndim == data_dim:
        data = arr
    else:
        raise IndexError(
            'Mismatching dimensions ({dim}) and axis ({num_axes}): '
            '{dim} - {num_axes} != {data_dim}'.format(
                dim=arr.ndim, num_axes=len(axis), data_dim=data_dim))
    if view_flip:
        data = data[
            [slice(None, None, -1) if f else slice(None) for f in view_flip]]

    # prepare view
    if view_axes is None:
        view_axes = np.argsort(data.shape)
    if view_indexes is None:
        view_indexes = mrt.utils.auto_repeat(None, data_dim)
    if len(view_axes) != data_dim:
        raise IndexError('Incorrect number of view axes.')
    if len(view_indexes) != data_dim:
        raise IndexError('Incorrect number of view indexes.')

    views = []
    for view_axis, view_index in zip(view_axes, view_indexes):
        views.append(mrt.utils.ndim_slice(data, view_axis, view_index))
    if ((orientation == 'transpose') or
            (orientation == 'landscape'
             and views[0].shape[0] > views[0].shape[1]) or
            (orientation == 'portrait'
             and views[0].shape[0] < views[0].shape[1])):
        views[0] = views[0].transpose()
    if orientation == 'rot90':
        views[0] = np.rot90(views[0])

    # perform flipping
    for i, (v, f_ud, f_lr, r90, tr) in \
            enumerate(zip(views, flip_ud, flip_lr, rot90, transpose)):
        if f_ud:
            views[i] = views[i][::-1, :]
        if f_lr:
            views[i] = views[i][:, ::-1]
        if r90:
            views[i] = np.rot90(views[i])
        if tr:
            views[i] = views[i].transpose()
    # print([v.shape for v in views])  # DEBUG

    if mode == ('std', 'standard'):
        data_shape = list(data.shape)
        view_shape = list(views[0].shape)
        other_size = [
            e for e in data_shape
            if not e in view_shape or view_shape.remove(e)][0]
        x_size = views[0].shape[0] + other_size
        y_size = views[0].shape[1] + other_size
        view = np.zeros((x_size, y_size))

        # transpose to match size
        # todo: fix this
        for i, v in enumerate(views):
            if v.shape[0] != views[0].shape[0] and \
                            v.shape[1] != views[0].shape[1]:
                views[i] = v.transpose()

        x0s, y0s = [0, views[0].shape[0], 0], [0, 0, views[0].shape[1]]
        if other_size != views[2].shape[1] and other_size != views[1].shape[0]:
            x0s[1], y0s[1], x0s[2], y0s[2] = x0s[2], y0s[2], x0s[1], y0s[1]

    elif mode in ('hor', 'horizontal'):
        x_size = max([v.shape[0] for v in views])
        y_size = sum([v.shape[1] for v in views])
        view = np.zeros((x_size, y_size))
        x0s = [(x_size - v.shape[0]) // 2 for v in views]
        y0s = [0] + list(np.cumsum([v.shape[1] for v in views])[:-1])

    elif mode in ('ver', 'vertical'):
        x_size = sum([v.shape[0] for v in views])
        y_size = max([v.shape[1] for v in views])
        view = np.zeros((x_size, y_size))
        x0s = [0] + list(np.cumsum([v.shape[0] for v in views])[:-1])
        y0s = [(y_size - v.shape[1]) // 2 for v in views]

    # assemble the image
    print([v.shape for v in views])  # DEBUG
    print(view.shape)  # DEBUG
    print(x0s, y0s)  # DEBUG
    for i, (v, x0, y0) in enumerate(zip(views, x0s, y0s)):
        x1, y1 = x0 + v.shape[0], y0 + v.shape[1]
        # print(x0, y0, x1, y1, v.shape, view[x0:x1, y0:y1].shape)  #DEBUG
        view[x0:x1, y0:y1] = v

    # prepare plot
    if title:
        ax.set_title(title)
    if array_interval is None:
        array_interval = mrt.utils.minmax(arr)
    if not cmap:
        if not mrt.utils.is_same_sign(array_interval):
            cmap = mpl.cm.RdBu_r
        else:
            cmap = mpl.cm.gray_r
    if not text_color:
        if not mrt.utils.is_same_sign(array_interval):
            text_color = 'k'
        else:
            text_color = 'k'
    ax.set_aspect('equal')

    # plot data
    plot = ax.imshow(
        view, cmap=cmap, vmin=array_interval[0], vmax=array_interval[1],
        interpolation='none')

    # plot ticks in plotting axes
    if ticks_limit is not None:
        if ticks_limit > 0:
            ax.locator_params(nbins=ticks_limit)
        else:
            ax.set_xticks([])
            ax.set_yticks([])

    # set colorbar
    if cbar_kws is not None:
        from mpl_toolkits.axes_grid1 import make_axes_locatable, axes_size
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        cbar = ax.figure.colorbar(plot, cax=cax, **dict(cbar_kws))
        # cbar = ax.figure.colorbar(plot, ax=ax, **dict(cbar_kws))
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
            size_info_size = round(abs(size_info) * (view.shape[1] * res), -1)
            size_info_str = '{} {}'.format(size_info_size, 'mm')
            size_info_px = size_info_size / res
            ax.text(
                0.025, 0.050, size_info_str, rotation=0, color=text_color,
                horizontalalignment='left', verticalalignment='bottom',
                transform=ax.transAxes)
            ax.plot(
                (view.shape[1] * 0.025,
                 view.shape[1] * 0.025 + size_info_px),
                (view.shape[0] * 0.965, view.shape[0] * 0.965),
                color=text_color, linewidth=2.5)

    # include additional text
    if more_texts is not None:
        for text_kws in more_texts:
            ax.text(**dict(text_kws))

    # save plot
    if save_filepath and mrt.utils.check_redo(None, [save_filepath], force):
        fig.tight_layout()
        if save_kws is None:
            save_kws = {}
        fig.savefig(save_filepath, **dict(save_kws))
        msg('Plot: {}'.format(save_filepath, verbose, VERB_LVL['medium']))
        plt.close(fig)

    return data, fig


# ======================================================================
def sample2d_multi():
    pass


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
    if ax is None:
        fig = plt.figure()
        ax = fig.gca()
    else:
        fig = plt.gcf()
    if axis is None:
        axis = np.argmin(array.shape)
    sample = mrt.utils.ndim_slice(array, axis, 0)
    if title:
        ax.set_title(title)
    if array_interval is None:
        array_interval = mrt.utils.minmax(array)
    if not cmap:
        if not mrt.utils.is_same_sign(array_interval):
            cmap = mpl.cm.RdBu_r
        else:
            cmap = mpl.cm.gray_r
    if not text_color:
        if not mrt.utils.is_same_sign(array_interval):
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
        for text_kws in text_list:
            ax.text(**dict(text_kws))
    n_frames = array.shape[axis]
    plots = []
    data = []
    for i in range(0, n_frames, step):
        sample = mrt.utils.ndim_slice(array, axis, i)
        plot = ax.imshow(
            sample, cmap=cmap,
            vmin=array_interval[0], vmax=array_interval[1], animated=True)
        # include additional text
        if more_texts is not None:
            for text_kws in more_texts:
                ax.text(**dict(text_kws))
        data.append(sample)
        if len(plots) <= 0:
            if cbar_kws is not None:
                cbar = ax.figure.colorbar(plot, ax=ax, **dict(cbar_kws))
                if cbar_txt is not None:
                    only_extremes = \
                        'ticks' in cbar_kws and len(cbar_kws['ticks']) == 2
                    if only_extremes:
                        cbar.ax.text(
                            2.0, 0.5, cbar_txt, fontsize='small', rotation=90)
                    else:
                        cbar.set_label(cbar_txt)
        plots.append([plot])
    mov = anim.ArtistAnimation(fig, plots, blit=False)
    if save_filepath and mrt.utils.check_redo(None, [save_filepath], force):
        fig.tight_layout()
        save_kwargs = {'fps': n_frames / step / duration}
        if save_kws is None:
            save_kws = {}
        save_kwargs.update(save_kws)
        mov.save(save_filepath, **dict(save_kws))
        msg('Anim: {}'.format(save_filepath, verbose, VERB_LVL['medium']))
        plt.close(fig)
    return data, fig


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
        style=(('linestyle', 'solid'), ('color', 'black')),
        more_texts=None,
        ax=None,
        save_filepath=None,
        save_kws=None,
        force=False,
        verbose=D_VERB_LVL):
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
        The value count scaling.
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
    hist_interval = tuple([mrt.utils.scale(val, array_interval)
                           for val in hist_interval])

    # create histogram
    hist, bin_edges = np.histogram(
        array, bins=bins, range=hist_interval, density=(scale == 'density'))
    # adjust scale
    hist = hist.astype(float)
    if scale in ('linear', 'density'):
        pass
    elif scale in ('log', 'log10', 'log2'):
        hist[hist > 0.0] = getattr(np, scale)(hist[hist > 0.0])
    elif scale:
        hist[hist > 0.0] = scale(hist[hist > 0.0])
        scale = 'custom'
    # plot figure
    if ax is None:
        fig = plt.figure()
        ax = fig.gca()
    else:
        fig = plt.gcf()
    plot = ax.plot(mrt.utils.midval(bin_edges), hist, **dict(style))
    # setup title and labels
    if title:
        ax.set_title(title.format_map(locals()))
    if labels[0]:
        ax.set_xlabel(labels[0].format_map(locals()))
    if labels[1]:
        ax.set_ylabel(labels[1].format_map(locals()))
    # include additional text
    if more_texts is not None:
        for text_kws in more_texts:
            ax.text(**dict(text_kws))
    # save figure to file
    if save_filepath and mrt.utils.check_redo(None, [save_filepath], force):
        fig.tight_layout()
        if save_kws is None:
            save_kws = {}
        fig.savefig(save_filepath, **dict(save_kws))
        msg('Plot: {}'.format(save_filepath, verbose, VERB_LVL['medium']))
        plt.close(fig)
    return (hist, bin_edges), fig


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
        labels=('Value', 'Value Count ({scale})'),
        legends=None,
        legend_kws=None,
        styles=None,
        more_texts=None,
        ax=None,
        save_filepath=None,
        save_kws=None,
        force=False,
        verbose=D_VERB_LVL):
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
    hist_interval = tuple([mrt.utils.scale(val, array_interval)
                           for val in hist_interval])

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
        fig = plt.figure()
        ax = fig.gca()
    else:
        fig = plt.gcf()
    ax.set_aspect('auto')
    data = []
    for i, array in enumerate(arrays):
        hist, bin_edges = np.histogram(
            array, bins=bins, range=hist_interval, density=(scale == 'density'))
        # adjust scale
        hist = hist.astype(float)
        if scale in ('linear', 'density'):
            pass
        elif scale in ('log', 'log10', 'log2'):
            hist[hist > 0.0] = getattr(np, scale)(hist[hist > 0.0])
        elif scale:
            hist[hist > 0.0] = scale(hist[hist > 0.0])
            scale = 'custom'
        # prepare legend
        if legends is not None and i < len(legends):
            legend = legends[i]
        else:
            legend = '_nolegend_'
        # plot figure
        plot = ax.plot(
            mrt.utils.midval(bin_edges), hist,
            **next(style_cycler),
            label=legend)
        data.append((hist, bin_edges))
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
        ax.set_title(title.format_map(locals()))
    if labels[0]:
        ax.set_xlabel(labels[0].format_map(locals()))
    if labels[1]:
        ax.set_ylabel(labels[1].format_map(locals()))
    # include additional text
    if more_texts is not None:
        for text_kws in more_texts:
            ax.text(**dict(text_kws))
    # save figure to file
    if save_filepath and mrt.utils.check_redo(None, [save_filepath], force):
        fig.tight_layout()
        if save_kws is None:
            save_kws = {}
        fig.savefig(save_filepath, **dict(save_kws))
        msg('Plot: {}'.format(save_filepath, verbose, VERB_LVL['medium']))
        plt.close(fig)
    return data, fig


# ======================================================================
def histogram2d(
        arr1,
        arr2,
        bin_size=1,
        bins=None,
        array_interval=None,
        hist_interval=((0.0, 1.0), (0.0, 1.0)),
        use_separate_interval=False,
        aspect=None,
        scale='linear',
        hist_val_interval=None,
        ticks_limit=None,
        interpolation='bicubic',
        title='2D Histogram ({bins} bins, {scale})',
        labels=('Array 1 Values', 'Array 2 Values'),
        cmap=mpl.cm.afmhot_r,
        bisector=None,
        stats_kws=None,
        cbar_kws=None,
        cbar_txt=None,
        more_texts=None,
        ax=None,
        save_filepath=None,
        save_kws=None,
        force=False,
        verbose=D_VERB_LVL):
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
        aspect (float): aspect ratio of the histogram.
            If None, it is calculated to result in squared proportions.
        scale (str|callable): The frequency value transformation method.

            - 'linear': no transformation is performed.
            - 'log': uses the natural logarithm of the histogram frequency.
            - 'log2': uses the base-2 logarithm of the histogram frequency.
            - 'log10': uses the base-10 logarithm of the histogram frequency.
            - 'density': uses the normalized histogram frequency,
              obtained by setting `np.histogram()` parameter `density=True`.
            - callable (func(float)->float): apply the function to the values.
              The corresponding scale name is then set to `custom`.
        hist_val_interval (tuple[float]|None): The interval of histogram values.
            If None, it is calculated automatically as the (min, max) of the
            histogram values rounded to the most comprehensive integer interval.
        ticks_limit (None): TODO
        interpolation (str): Image display interpolation method.
            See `matplotlib.imshow()` for more details.
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
        fig (matplotlib.pyplot.Figure): The Figure object containing the plot.
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
    # setup image aspect ratio
    if not aspect:
        x_axis_size = np.ptp(array_interval[0])
        y_axis_size = np.ptp(array_interval[1])
        if x_axis_size != y_axis_size:
            aspect = x_axis_size / y_axis_size
        else:
            aspect = 1.0
    # setup bins
    if not bins:
        bins = tuple([int(np.ptp(val) / bin_size + 1)
                      for val in array_interval])
    else:
        bins = _ensure_all_axis(bins)
    # setup histogram range
    hist_interval = _ensure_all_axis(hist_interval)
    hist_interval = tuple([[mrt.utils.scale(val, array_interval[i])
                            for val in hist_interval[i]] for i in range(2)])
    # calculate histogram
    # prepare histogram
    hist, x_edges, y_edges = np.histogram2d(
        arr1.ravel(), arr2.ravel(),
        bins=bins, range=hist_interval, normed=(scale == 'density'))
    hist = hist.transpose()
    # adjust scale
    hist = hist.astype(float)
    if scale in ('linear', 'density'):
        pass
    elif scale in ('log', 'log10', 'log2'):
        hist[hist > 0.0] = getattr(np, scale)(hist[hist > 0.0])
    elif scale:
        hist[hist > 0.0] = scale(hist[hist > 0.0])
        scale = 'custom'
    # adjust histogram intensity range
    if hist_val_interval is None:
        hist_val_interval = (np.floor(np.min(hist)), np.ceil(np.max(hist)))
    # create a new figure
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
    else:
        fig = plt.gcf()
    # plot figure
    plot = ax.imshow(
        hist, cmap=cmap, origin='lower', interpolation=interpolation,
        aspect=aspect,
        vmin=hist_val_interval[0], vmax=hist_val_interval[1],
        extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]])
    # plot the color bar
    if cbar_kws is not None:
        cbar = ax.figure.colorbar(plot, ax=ax, **dict(cbar_kws))
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
        stats_dict = mrt.utils.calc_stats(
            arr1[mask] - arr2[mask], **stats_kws)
        stats_text = '$\\mu_D = {}$\n$\\sigma_D = {}$'.format(
            *mrt.utils.format_value_error(stats_dict['avg'], stats_dict['std'], 3))
        ax.text(
            1 / 2, 31 / 32, stats_text,
            horizontalalignment='center', verticalalignment='top',
            transform=ax.transAxes)
    # setup title and labels
    if title:
        ax.set_title(title.format_map(locals()))
    if labels[0]:
        ax.set_xlabel(labels[0].format_map(locals()))
    if labels[1]:
        ax.set_ylabel(labels[1].format_map(locals()))
    # fine-tune ticks
    if ticks_limit is not None:
        if ticks_limit > 0:
            ax.locator_params(nbins=ticks_limit)
        else:
            ax.set_xticks([])
            ax.set_yticks([])
    # include additional text
    if more_texts is not None:
        for text_kws in more_texts:
            ax.text(**dict(text_kws))
    # save figure to file
    if save_filepath and mrt.utils.check_redo(None, [save_filepath], force):
        fig.tight_layout()
        if save_kws is None:
            save_kws = {}
        fig.savefig(save_filepath, **dict(save_kws))
        msg('Plot: {}'.format(save_filepath, verbose, VERB_LVL['medium']))
        plt.close(fig)
    return (hist, x_edges, y_edges), fig


# ======================================================================
def bar_chart(
        data,
        err,
        series,
        groups,
        x_label=None,
        y_label=None,
        title=None,
        y_limits=None,
        legend_kws=None,
        bar_width=None,
        ax=None):
    """
    Plot a bar chart.

    WIP

    Args:
        data:
        err:
        series:
        groups:
        x_label:
        y_label:
        title:
        y_limits:
        legend_kws:
        bar_width:
        ax:

    Returns:

    """
    #todo: polish code and documentation
    #todo: add support for orientation
    #todo: add support for stacked bars
    #todo: add support for variable bar_width

    # create a new figure
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
    else:
        fig = plt.gcf()
    if y_limits:
        ax.set_ylim(y_limits)
    num_series = len(series)
    num_groups = len(groups)
    indices = np.arange(num_groups)
    if not bar_width:
        bar_width = 1 / (num_series + 0.5)

    color_cycler = itertools.cycle(mrt.plot.PLOT_COLORS)
    bcs = []
    for j, serie in enumerate(series):
        bar_data = data[serie]
        bar_err = err[serie]
        bc = ax.bar(
            indices + (j * bar_width),
            bar_data,
            bar_width,
            yerr=bar_err,
            color=next(color_cycler)['color'])
        bcs.append(bc)
    ax.set_xticks(indices + bar_width)
    if groups:
        ax.set_xticklabels(groups)
    if x_label:
        ax.set_xlabel(x_label)
    if y_label:
        ax.set_ylabel(y_label)
    if title:
        ax.set_title(title)

    if legend_kws is not None:
        ax.legend(tuple(bc[0] for bc in bcs), series, **dict(legend_kws))
    # barplot_autolabel(ax, bc)
    return data, fig


# ======================================================================
def subplots(
        plots,
        rows=None,
        cols=None,
        num_row=None,
        num_col=None,
        aspect_ratio=None,
        width_height=None,
        figsize_factors=3,
        pads=2.0,
        swap_filling=False,
        title=None,
        subplot_title_fmt='({letter}) {t}',
        row_labels=None,
        col_labels=None,
        label_pads=0.3,
        more_texts=None,
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
        tight_layout_kws ():
        save_filepath ():
        savefig_args ():
        savefig_kwargs ():
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
            elif isinstance(aspect_ratio, str):
                if 'exact' in aspect_ratio:
                    num_col, num_row = mrt.utils.optimal_ratio(num_plots)
                    if 'portrait' in aspect_ratio:
                        num_row, num_col = num_col, num_row
                if aspect_ratio == 'portrait':
                    num_row = np.ceil(np.sqrt(num_plots))
            else:  # plot_aspect == 'landscape'
                num_row = int(np.floor(np.sqrt(num_plots)))

        if num_row is None and num_col > 0:
            num_row = int(np.ceil(num_plots / num_col))
        if num_row > 0 and num_col is None:
            num_col = int(np.ceil(num_plots / num_row))

        if width_height is None:
            width, height = 1, 1
        else:
            width, height = width_height
        rows = (height,) * num_row
        cols = (width,) * num_col
    else:
        num_row = len(rows)
        num_col = len(cols)
    assert (num_row * num_col >= num_plots)

    pads = list(mrt.utils.auto_repeat(pads, 2, False, True))
    label_pads = list(mrt.utils.auto_repeat(label_pads, 2, False, True))
    figsize_factors = list(mrt.utils.auto_repeat(figsize_factors, 2, False, True))

    # fix row/col labels
    if row_labels is None:
        row_labels = mrt.utils.auto_repeat(None, num_row)
        label_pads[0] = 0.0
    if col_labels is None:
        col_labels = mrt.utils.auto_repeat(None, num_col)
        label_pads[1] = 0.0
    assert (num_row == len(row_labels))
    assert (num_col == len(col_labels))

    # generate plot
    figsizes = [
        factor * sum(items)
        for factor, items, pad in zip(figsize_factors, (rows, cols), pads)]

    # fig, axs = plt.subplots(
    #     nrows=num_row, ncols=num_col,
    #     gridspec_kw={'width_ratios': cols, 'height_ratios': rows},
    #     figsize=figsizes[::-1])

    fig = plt.figure(figsize=figsizes[::-1])
    label_pads = [
        label_pad / figsize
        for label_pad, figsize in zip(label_pads, figsizes)]

    gs = mpl.gridspec.GridSpec(
        num_row, num_col, width_ratios=cols, height_ratios=rows)

    axs = [plt.subplot(gs[n]) for n in range(num_row * num_col)]
    axs = np.array(axs).reshape((num_row, num_col))

    for i, row_label in enumerate(row_labels):
        for j, col_label in enumerate(col_labels):
            if not swap_filling:
                n_plot = i * num_col + j
            else:
                n_plot = j * num_row + i

            if n_plot < num_plots:
                plot_func, plot_args, plot_kwargs = plots[n_plot]
                plot_kwargs['ax'] = axs[i, j]
                if subplot_title_fmt:
                    t = plot_kwargs['title'] if 'title' in plot_kwargs else ''
                    roman = numeral.int2roman(n_plot + 1, only_ascii=True)
                    roman_uppercase = roman.upper()
                    roman_lowercase = roman.lower()
                    letter = numeral.int2letter(n_plot)
                    letter_uppercase = letter.upper()
                    letter_lowercase = letter.lower()
                    plot_kwargs['title'] = \
                        subplot_title_fmt.format_map(locals())
                plot_func(*plot_args, **plot_kwargs)
            else:
                axs[i, j].clear()
                axs[i, j].set_axis_off()

            if col_label:
                fig.text(
                    mrt.utils.scale(
                        (j * 2 + 1) / (num_col * 2),
                        out_interval=(label_pads[0], 1.0 - label_pads[0] / 5)),
                    1.0 - label_pads[1] / 2,
                    col_label, rotation=0,
                    fontweight='bold', fontsize='large',
                    horizontalalignment='center', verticalalignment='top')

        if row_label:
            fig.text(
                label_pads[0] / 2,
                mrt.utils.scale(
                    1.0 - (i * 2 + 1) / (num_row * 2),
                    out_interval=(label_pads[1] / 5, 1.0 - label_pads[1])),
                row_label, rotation=90,
                fontweight='bold', fontsize='large',
                horizontalalignment='left', verticalalignment='center')

    # include additional text
    if more_texts is not None:
        for text_kws in more_texts:
            fig.text(**dict(text_kws))

    # save figure to file
    if save_filepath and mrt.utils.check_redo(None, [save_filepath], force):
        fig.tight_layout(
            rect=[0.0 + label_pads[0], 0.0, 1.0, 1.0 - label_pads[1]],
            pad=1.0, h_pad=pads[0], w_pad=pads[1])
        if save_kws is None:
            save_kws = {}
        fig.savefig(save_filepath, **dict(save_kws))
        msg('Plot: {}'.format(save_filepath, verbose, VERB_LVL['medium']))
        plt.close(fig)
    return fig

# # ======================================================================
# def subplots(
#         plots,
#         rows=None,
#         cols=None,
#         num_row=None,
#         num_col=None,
#         aspect_ratio=None,
#         width_height=None,
#         figsize_factors=3,
#         pads=2.0,
#         swap_filling=False,
#         title=None,
#         subplot_title_fmt='({letter}) {t}',
#         row_labels=None,
#         col_labels=None,
#         label_pos=(0, 0),
#         more_texts=None,
#         save_filepath=None,
#         save_kws=None,
#         force=False,
#         verbose=D_VERB_LVL):
#     """
#
#     Args:
#         plots ():
#         rows ():
#         cols ():
#         num_row ():
#         num_col ():
#         aspect_ratio ():
#         width_height ():
#         swap_filling ():
#         title ():
#         row_labels ():
#         col_labels ():
#         row_label_width ():
#         col_label_width ():
#         border_top ():
#         border_left ():
#         tight_layout_kws ():
#         save_filepath ():
#         savefig_args ():
#         savefig_kwargs ():
#         force ():
#         verbose ():
#
#     Returns:
#
#     """
#     num_plots = len(plots)
#
#     # determine rows and cols if not specified
#     if rows is None and cols is None:
#         if num_row is None and num_col is None:
#             if isinstance(aspect_ratio, float):
#                 num_col = np.ceil(np.sqrt(num_plots * aspect_ratio))
#             elif isinstance(aspect_ratio, str):
#                 if 'exact' in aspect_ratio:
#                     num_col, num_row = mrt.utils.optimal_ratio(num_plots)
#                     if 'portrait' in aspect_ratio:
#                         num_row, num_col = num_col, num_row
#                 if aspect_ratio == 'portrait':
#                     num_row = np.ceil(np.sqrt(num_plots))
#             else:  # plot_aspect == 'landscape'
#                 num_row = int(np.floor(np.sqrt(num_plots)))
#
#         if num_row is None and num_col > 0:
#             num_row = int(np.ceil(num_plots / num_col))
#         if num_row > 0 and num_col is None:
#             num_col = int(np.ceil(num_plots / num_row))
#
#         if width_height is None:
#             width, height = 1, 1
#         else:
#             width, height = width_height
#         rows = [height,] * num_row
#         cols = [width,] * num_col
#     else:
#         rows = [pad_labels[0]] + list(rows)
#         cols = [pad_labels[1]] + list(cols)
#         num_row = len(rows)
#         num_col = len(cols)
#     assert (num_row * num_col >= num_plots)
#
#     pads = mrt.utils.auto_repeat(pads, 2, False, True)
#     figsize_factors = mrt.utils.auto_repeat(figsize_factors, 2, False, True)
#
#     # fix row/col labels
#     row_labels = mrt.utils.auto_repeat(None, num_row) \
#         if row_labels is None else [None] + list(row_labels)
#     col_labels = mrt.utils.auto_repeat(None, num_col) \
#         if col_labels is None else [None] + list(row_labels)
#     assert (num_row + 1 == len(row_labels))
#     assert (num_col + 1 == len(col_labels))
#
#     # generate plot
#     figsizes = [
#         factor * sum(items)
#         for factor, items, pad in zip(figsize_factors, (rows, cols), pads)]
#     fig = plt.figure(figsize=figsizes[::-1])
#
#     gs = mpl.gridspec.GridSpec(
#         num_row, num_col + 1, width_ratios=cols, height_ratios=rows)
#
#     axs = [plt.subplot(gs[n]) for n in range((num_row) * (num_col))]
#     axs = np.array(axs).reshape((num_row, num_col))
#
#     def is_label(row, col, shape, pos):
#         if pos in ('nw', 'tl', 'northwest', 'topleft'):
#             return row == 0 or col == 0
#         elif pos in ('ne', 'tr', 'northeast', 'topright'):
#             return row == 0 or col == shape[1]
#         elif pos in ('se', 'br', 'southwest', 'bottomleft'):
#             return row == shape[0] or col == 0
#         elif pos in ('se', 'br', 'northeast', 'bottomright'):
#             return row == shape[0] or col == shape[1]
#
#
#     for i, row_label in enumerate(row_labels):
#         for j, col_label in enumerate(col_labels):
#             if not swap_filling:
#                 n_plot = i * num_col + j
#             else:
#                 n_plot = j * num_row + i
#
#             plot_func, plot_args, plot_kwargs = plots[n_plot]
#             plot_kwargs['ax'] = axs[i, j]
#             if subplot_title_fmt:
#                 t = plot_kwargs['title'] if 'title' in plot_kwargs else ''
#                 number = n_plot + 1
#                 if roman_numbers:  # todo: switch to numerals?
#                     roman_lowercase = roman_numbers.toRoman(number)
#                     roman_uppercase = roman_numbers.toRoman(number)
#                     roman = roman_uppercase
#                 letter_lowercase = string.ascii_lowercase[n_plot]
#                 letter_uppercase = string.ascii_uppercase[n_plot]
#                 letter = letter_lowercase
#                 plot_kwargs['title'] = subplot_title_fmt.format_map(locals())
#             plot_func(*tuple(plot_args), **(plot_kwargs))
#
#             if col_label and is_label(i, j, axs.shape, label_pos):
#                 fig.text(
#                     mrt.utils.scale(
#                         (j * 2 + 1) / (num_col * 2),
#                         out_interval=(label_pads[0], 1.0)),
#                     1.0 - label_pads[1] / 2,
#                     col_label, rotation=0,
#                     fontweight='bold', fontsize='large',
#                     horizontalalignment='center', verticalalignment='top')
#
#         if row_label:
#             fig.text(
#                 label_pads[0] / 2,
#                 mrt.utils.scale(
#                     1.0 - (i * 2 + 1) / (num_row * 2),
#                     out_interval=(0, 1.0 - label_pads[1])),
#                 row_label, rotation=90,
#                 fontweight='bold', fontsize='large',
#                 horizontalalignment='left', verticalalignment='center')
#
#     # include additional text
#     if more_texts is not None:
#         for text_kws in more_texts:
#             fig.text(**dict(text_kws))
#
#     # save figure to file
#     if save_filepath and mrt.utils.check_redo(None, [save_filepath], force):
#         fig.tight_layout(
#             rect=[0.0 + label_pads[0], 0.0, 1.0, 1.0 - label_pads[1]],
#             pad=1.0, h_pad=pads[0], w_pad=pads[1])
#         if save_kws is None:
#             save_kws = {}
#         fig.savefig(save_filepath, **dict(save_kws))
#         msg('Plot: {}'.format(save_filepath, verbose, VERB_LVL['medium']))
#         plt.close(fig)
#     return fig
