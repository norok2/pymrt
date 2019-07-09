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
from mpl_toolkits.axes_grid1 import make_axes_locatable

# :: Local Imports
import pymrt as mrt
import pymrt.utils

from pymrt import INFO, PATH
from pymrt import VERB_LVL, D_VERB_LVL, VERB_LVL_NAMES
from pymrt import elapsed, report, run_doctests
from pymrt import msg, dbg

# ======================================================================
# :: MatPlotLib-related constants
# standard plot resolutions
D_PLOT_DPI = mpl.rcParams['figure.dpi']
# colors and linestyles
PLOT_COLORS = tuple(x['color'] for x in mpl.rcParams['axes.prop_cycle'])
PLOT_LINESTYLES = ('-', '--', '-.', ':')


# ======================================================================
# def plot_with_adjusting_parameters()
# TODO: make a plot with possibility to adjust params


# ======================================================================
def _ensure_fig_ax(ax):
    """
    Ensure valid Matplotlib axes and figure, even if the input is None.

    Args:
        ax (mpl.axes|None): A Matplotlib axes.
            If None, it ensures a valid Matplotlib axes.
            Otherwise, the ax input is unchanged.

    Returns:
        result (tuple): The tuple
            contains:
            - fig (mpl.Figure): A valid Matplotlib figure.
            - ax (mpl.axes): A valid Matplotlib axes.
    """
    if ax is None:
        fig = plt.figure()
        ax = fig.gca()
    else:
        fig = plt.gcf()
    return fig, ax


# ======================================================================
def _color_series(
        name,
        num=8,
        asym=0.3,
        min_=None,
        max_=None):
    """
    Generate a color series from a Matplotlib Colormap.

    Args:
        name (str): Name of the colormap.
        num (int): Number of colors in the series.
        asym (float): Asymmetry factor.
            Determines which portion of the colormap is used.
            Use 0 for no asymmetry, 1 for right asymmetry and -1 for left
            asymmetry.
            Must be in the [-1, 1] range.
            This is only used if `min_` and/or `max_` are None.
        min_ (float): Minimum of the colormap range.
            If None, this is automatically calculated from `asym` parameter.
            Must be in the [0, 1] range and must be larger than `max_`.
        max_ (float): Maximum of the colormap range.
            If None, this is automatically calculated from `asym` parameter.
            Must be in the [0, 1] range and must be smaller than `min_`.

    Returns:
        result (list): The generated colors.
    """
    if max_ is None:
        if asym >= 0:
            max_ = (num * (1 + asym)) / (num * (np.abs(asym) + 1.0) + 1.0)
        else:
            max_ = num / (num * (np.abs(asym) + 1.0) + 1.0)
    if min_ is None:
        if asym >= 0:
            min_ = (1.0 + asym * num) / (num * (np.abs(asym) + 1.0) + 1.0)
        else:
            min_ = 1.0 / (num * (np.abs(asym) + 1.0) + 1.0)
    return [mpl.cm.get_cmap(name)(x) for x in np.linspace(min_, max_, num)]


# ======================================================================
def _transparent_cmap_from_color(
        color,
        threshold=0.0):
    """
    Generate transparent colormap from a single color.

    Args:
        color (str|tuple): A valid color.
        threshold (int|float): Threshold for transparency.
            Values below the threshold are set to be transparent.

    Returns:
        result (mpl.Colormap): A Matplotlib colormap.
    """
    return mpl.colors.LinearSegmentedColormap.from_list(
        'my_cmap_{}'.format(color),
        [(0.00, (0, 0, 0, 0)),
         (threshold, mpl.colors.to_rgba(color, alpha=0)),
         (1.00, mpl.colors.to_rgba(color, alpha=1))])


# ======================================================================
def _ax_sizes_pt(ax):
    """
    Compute the size of the Matplotlib axes in pt.

    Args:
        ax (mpl.axes): A Matplotlib axes.

    Returns:
        sizes (tuple[float]): The size of the Matplotlib axes in pt.
    """
    bbox = ax.get_window_extent().transformed(
        ax.figure.dpi_scale_trans.inverted())
    sizes = tuple(size * ax.figure.dpi for size in (bbox.width, bbox.height))
    return sizes


# ======================================================================
def _reorient(
        arr,
        transposes=None,
        rot90s=None,
        flips=None):
    """
    Transform the input data according to the preferred orientation.

    The transformations are performed in order of appearance.

    Args:
        arr (np.ndarray): The input array.
        transposes (Iterable[int, int]): List of transposition axes.
        rot90s (Iterable[int, (int, int)]|None): List of rotation parameters.
            The first element of the iterable (int) indicates the number of
            90 degree rotations, while the second element of the iterable
            (int, int) indicates the axes corresponding to the plane of
            rotation (the direction is from the first to the second axis).
        flips (Iterable[int]|None): List of flipping axes.

    Returns:
        arr (np.ndarray): The transformed array.
    """
    if transposes:
        for axes in transposes:
            arr = np.transpose(arr, axes)
    if rot90s:
        for k, axes in rot90s:
            arr = np.rot90(arr, k, axes)
    if flips:
        for axis in flips:
            arr = np.flip(arr, axis)
    return arr


# ======================================================================
def _reorient_2d(
        arr,
        orientation=None,
        flip_ud=False,
        flip_lr=False):
    """
    Transform the input data according to the preferred orientation.

    This is specialized for 2D inputs.

    Args:
        arr (np.ndarray): The input array.
        orientation (str): The orientation.
            Accepted values are:
             - 'transpose': transpose the input
             - 'landscape': first axis gets smaller then the second
             - 'portrait': first axis gets larger then the second
             - 'rot90': rotate the input by 90 degrees.
        flip_ud (bool): Flip in the first dim (up/down or vertically).
        flip_lr (bool): Flip in the second dim (left/right or horizontally).

    Returns:
        arr (np.ndarray): The transformed array.
    """
    if ((orientation == 'transpose') or
            (orientation == 'landscape' and arr.shape[0] > arr.shape[1]) or
            (orientation == 'portrait' and arr.shape[0] < arr.shape[1])):
        arr = np.transpose(arr)
    if orientation == 'rot90':
        arr = np.rot90(arr)
    if flip_ud:
        arr = np.flipud(arr)
    if flip_lr:
        arr = np.fliplr(arr)
    return arr


# ======================================================================
def _manage_resolution_info(
        size_info,
        resolution,
        shape,
        text_color,
        ax):
    """
    Create a ruler and indications for resolution info.

    Assumes that the resolution is in `px / mm`.

    Note that this modifies the `ax` object.

    Args:
        size_info:
        resolution (Iterable[int|float]):
        shape (Iterable[int]):
        text_color:
        ax:

    Returns:
        None.
    """
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
            size_info_size = round(abs(size_info) * (shape[1] * res), -1)
            size_info_str = '{} {}'.format(size_info_size, 'mm')
            size_info_px = size_info_size / res
            ax.text(
                0.025, 0.050, size_info_str, rotation=0, color=text_color,
                horizontalalignment='left', verticalalignment='bottom',
                transform=ax.transAxes)
            ax.plot(
                (shape[1] * 0.025,
                 shape[1] * 0.025 + size_info_px),
                (shape[0] * 0.965, shape[0] * 0.965),
                color=text_color, linewidth=2.5)


# ======================================================================
def _manage_colorbar(
        cbar_kws,
        cbar_txt,
        ax,
        pax):
    """
    Manage the input for displaying a Matplotlib colorbar.

    Note that this modifies the `ax` (and `pax`) objects.

    Args:
        cbar_kws (dict|tuple): The keyword arguments to pass to `colorbar()`
            See `mpl.figure.colorbar()` for more.
        cbar_txt (str): The text to use as label for the colorbar.
        ax (mpl.axes): A Matplotlib axes.
        pax (mpl.axes): The plotted Matplotlib AxesImage.

    Returns:
        None.
    """
    if cbar_kws is not None or cbar_txt is not None:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        cbar = ax.figure.colorbar(pax, cax=cax, **dict(cbar_kws))
        if cbar_txt is not None:
            only_extremes = 'ticks' in cbar_kws and len(cbar_kws['ticks']) == 2
            if only_extremes:
                cbar.ax.text(
                    2.0, 0.5, cbar_txt, fontsize='medium', rotation=90,
                    verticalalignment='center', horizontalalignment='left',
                    transform=cbar.ax.transAxes)
                # ax_sizes_pt = _ax_sizes_pt(ax)
                # cbar.set_label(cbar_txt, labelpad=-ax_sizes_pt[0] * 0.075)
                # cbar.ax.yaxis.set_label_position('left')
            else:
                cbar.set_label(cbar_txt)


# ======================================================================
def _manage_ticks_limit(
        ticks_limit,
        ax):
    """
    Manage the input for displaying the Matplotlib ticks in the axis.

    Note that this modifies the `ax` object.

    Args:
        ticks_limit (int): The parameter determining the ticks behavior.
            This affects both x- and y- axis.
            If `ticks_limit > 0`, this is the number of bins, and the number
            of ticks will be `ticks_limit + 1`.
            If `ticks_limit == 0` no ticks are being displayed.
            If `ticks_limit < 0` the axis itself is not displayed.
        ax (mpl.axes): A Matplotlib axes.

    Returns:
        None.
    """
    if ticks_limit is not None:
        if ticks_limit > 0:
            ax.locator_params(nbins=ticks_limit)
        elif ticks_limit == 0:
            ax.set_xticks([])
            ax.set_yticks([])
        else:
            ax.set_axis_off()


# ======================================================================
def _more_texts(more_texts, ax):
    """
    Manage the input for adding more texts.

    Note that this modifies the `ax` object.

    Args:
        more_texts (Iterable): List of keyword arguments for `mpl.axes.text()`.
            See `pymrt.plot._more_texts()` for more info.
        ax (mpl.axes): A Matplotlib axes.

    Returns:
        None.
    """
    if more_texts is not None:
        for text_kws in more_texts:
            text_kws = dict(text_kws) if text_kws is not None else {}
            ax.text(**text_kws)


# ======================================================================
def _more_elements(more_elements, ax):
    """
    Manage the input for adding more elements.

    Note that this modifies the `ax` object.

    Args:
        more_elements (Iterable[Iterable]): List of extra methods/parameters.
            Each item consists of:
            - element_func (str): Name of the method of `mpl.axes`.
              `ax.element_func` must be callable.
            - element_args (Iterable): Positional arguments.
            - element_kws (dict|tuple): Keyword arguments.
            See `pymrt.plot._more_elements()` for more info.
        ax (mpl.axes): A Matplotlib axes.

    Returns:
        None.
    """
    if more_elements is not None:
        for element_func, element_kargs, element_kws in more_elements:
            getattr(ax, element_func)(
                *tuple(element_kargs), **dict(element_kws))


# ======================================================================
def _save_plot(
        save_filepath,
        save_kws,
        tight_layout_kws,
        fig,
        force,
        verbose):
    """
    Manage the input for saving the plot to file.

    Args:
        save_filepath (str|None): The file path where the plot is saved to.
            If None, no output is saved.
            If filepath exists a new plot is saved only if `force == True`.
        save_kws (dict|tuple): Keyword arguments for `mpl.figure.savefig()`.
            Frequently used parameters are: `dpi` and `format`.
        tight_layout_kws (dict|tuple): Keyword arguments for `tight_layout()`.
            Uses the method of `mpl.figure`.
        fig (mpl.Figure): A Matplotlib figure.
        force (bool): Force new processing.
        verbose (int): Set level of verbosity.

    Returns:
        None.
    """
    if save_filepath and fc.base.check_redo(None, [save_filepath], force):
        tight_layout_kws = dict(tight_layout_kws) if tight_layout_kws else {}
        fig.tight_layout(**tight_layout_kws)
        save_kws = dict(save_kws) if save_kws is not None else {}
        fig.savefig(save_filepath, **save_kws)
        msg('Plot: {}'.format(save_filepath, verbose, VERB_LVL['medium']))
        plt.close(fig)


# ======================================================================
def simple(
        x_datas,
        y_datas,
        labels=(None, None),
        limits=(None, None),
        styles=None,
        legends=None,
        legend_kws=None,
        title=None,
        more_texts=None,
        more_elements=None,
        ax=None,
        save_filepath=None,
        save_kws=None,
        force=False,
        verbose=D_VERB_LVL):
    """

    Args:
        x_datas:
        y_datas:
        labels:
        limits:
        styles:
        legends:
        legend_kws (dict|tuple): Keyword arguments for `mpl.axes.legend()`.
        title (str|None): The plot title.
            If None, no title is set.
        more_texts (Iterable|None): List of arguments for `mpl.axes.text()`.
            See `pymrt.plot._more_texts()` for more info.
        more_elements (Iterable[Iterable]|None): List of extra elements.
            See `pymrt.plot._more_elements()` for more info.
        ax (mpl.axes): A Matplotlib axes.
        save_filepath (str|None): The file path where the plot is saved to.
            See `pymrt.plot._save_plot()` for more info.
        save_kws (dict|tuple): Keyword arguments for `mpl.figure.savefig()`.
            See `pymrt.plot._save_plot()` for more info.
        force (bool): Force new processing.
        verbose (int): Set level of verbosity.

    Returns:
        result (tuple): The tuple
            contains:
            - data: The plotted data.
            - fig: The plotted figure.
    """
    fig, ax = _ensure_fig_ax(ax)
    if isinstance(x_datas, np.ndarray):
        x_datas = fc.base.auto_repeat(x_datas, len(y_datas), True, True)
    if legends is None:
        legends = fc.base.auto_repeat(None, len(y_datas), check=True)
    if styles is None:
        styles = fc.base.auto_repeat(None, len(y_datas), check=True)
    for x_data, y_data, legend, style \
            in zip(x_datas, y_datas, legends, styles):
        if style is None:
            style = {}
        pax = ax.plot(x_data, y_data, label=legend, **dict(style))
    # setup title and labels
    if title:
        ax.set_title(title.format(**locals()))
    if labels[0]:
        ax.set_xlabel(labels[0].format(**locals()))
    if labels[1]:
        ax.set_ylabel(labels[1].format(**locals()))
    if limits[0]:
        ax.set_xlim(limits[0])
    if limits[1]:
        ax.set_ylim(limits[1])
    if any([legend for legend in legends]):
        ax.legend(**(legend_kws if legend_kws is not None else {}))
    _more_texts(more_texts, ax)
    _more_elements(more_elements, ax)
    _save_plot(save_filepath, save_kws, None, fig, force, verbose)
    return (x_datas, y_datas), fig


# ======================================================================
def empty(
        title='',
        more_texts=None,
        more_elements=None,
        ax=None,
        save_filepath=None,
        save_kws=None,
        force=False,
        verbose=D_VERB_LVL):
    """

    Args:
        title (str|None): The plot title.
            If None, no title is set.
        more_texts (Iterable|None): List of arguments for `mpl.axes.text()`.
            See `pymrt.plot._more_texts()` for more info.
        more_elements (Iterable[Iterable]|None): List of extra elements.
            See `pymrt.plot._more_elements()` for more info.
        ax (mpl.axes): A Matplotlib axes.
        save_filepath (str|None): The file path where the plot is saved to.
            See `pymrt.plot._save_plot()` for more info.
        save_kws (dict|tuple): Keyword arguments for `mpl.figure.savefig()`.
            See `pymrt.plot._save_plot()` for more info.
        force (bool): Force new processing.
        verbose (int): Set level of verbosity.

    Returns:

    """
    fig, ax = _ensure_fig_ax(ax)

    if title:
        ax.set_title(title)

    ax.axis('off')
    ax.set_aspect(1)

    _more_texts(more_texts, ax)
    _more_elements(more_elements, ax)
    _save_plot(save_filepath, save_kws, None, fig, force, verbose)
    return None, fig


# ======================================================================
def multi(
        x_arrs,
        y_arrs,
        dy_arrs=None,
        y_lbls=None,
        dy_lbls=None,
        x_label=None,
        y_label=None,
        x_limits=None,
        y_limits=None,
        twin_limits=None,
        twin_indexes=None,
        shared_axis='y',
        groups=None,
        colors=PLOT_COLORS,
        legend_kws=None,
        method='errorbars',  # 'errorarea', # 'dotted+solid',
        title=None,
        more_texts=None,
        more_elements=None,
        ax=None,
        save_filepath=None,
        save_kws=None,
        force=False,
        verbose=D_VERB_LVL):
    """
    Plot multiple curves including optional errorbars and twin axes.

    Args:
        x_arrs ():
        y_arrs ():
        dy_arrs ():
        y_lbls ():
        dy_lbls ():
        x_label ():
        y_label ():
        x_limits ():
        y_limits ():
        twin_limits ():
        twin_indexes ():
        shared_axis ():
        groups ():
        colors ():
        legend_kws ():
        method ():
        title (str|None): The plot title.
            If None, no title is set.
        more_texts (Iterable|None): List of arguments for `mpl.axes.text()`.
            See `pymrt.plot._more_texts()` for more info.
        more_elements (Iterable[Iterable]|None): List of extra elements.
            See `pymrt.plot._more_elements()` for more info.
        ax (mpl.axes): A Matplotlib axes.
        save_filepath (str|None): The file path where the plot is saved to.
            See `pymrt.plot._save_plot()` for more info.
        save_kws (dict|tuple): Keyword arguments for `mpl.figure.savefig()`.
            See `pymrt.plot._save_plot()` for more info.
        force (bool): Force new processing.
        verbose (int): Set level of verbosity.

    Returns:
        TODO:
    """
    method = method.lower()
    shared_axis = shared_axis.lower()

    fig, ax = _ensure_fig_ax(ax)

    handles = []

    # : prepare for plotting
    num = len(y_arrs)
    if isinstance(x_arrs, np.ndarray):
        x_arrs = fc.base.auto_repeat(x_arrs, num, force=True, check=True)
    elif x_arrs is None:
        x_arrs = tuple(np.arange(len(y_arr)) for y_arr in y_arrs)

    if dy_arrs is None:
        dy_arrs = tuple(np.zeros_like(y_arr) for y_arr in y_arrs)
    y_lbls = ('_nolegend_',) * num if y_lbls is None else y_lbls
    dy_lbls = ('_nolegend_',) * num if dy_lbls is None else dy_lbls
    twin_indexes = () if twin_indexes is None else twin_indexes
    plotters = (x_arrs, y_arrs, dy_arrs, y_lbls, dy_lbls)

    # : set twin axes
    if len(twin_indexes) > 0:
        if shared_axis == 'y':
            twin_ax = ax.twiny()
        else:  # if shared_axis == 'x':
            twin_ax = ax.twinx()
    else:
        twin_ax = None

    if isinstance(x_label, str):
        ax.set_xlabel(x_label)
    elif x_label:
        ax.set_xlabel(x_label[0])
        if shared_axis == 'y':
            twin_ax.set_xlabel(x_label[1])

    if isinstance(y_label, str):
        ax.set_ylabel(y_label)
    elif y_label:
        ax.set_ylabel(y_label[0])
        if shared_axis == 'x':
            twin_ax.set_ylabel(y_label[1])

    if groups:
        if sum(groups) < num:
            groups = tuple(groups) + (num - sum(groups),)

        num = max(groups)
        tmp_colors = []
        for group, color in zip(groups, itertools.cycle(colors)):
            if callable(color):
                tmp_color = [color(i) for i in range(group)]
            elif isinstance(color, str):
                tmp_color = _color_series(color, num)
            else:
                tmp_color = color
            tmp_colors.extend(tmp_color[:group])
        colors = tmp_colors

    colors = itertools.cycle(colors)

    for i, (x_arr, y_arr, dy_arr, y_lbl, dy_lbl) in enumerate(zip(*plotters)):
        _ax = twin_ax if i in twin_indexes else ax

        color = next(colors)

        if '+' in method:
            y_ls, dy_ls = method.split('+')
            handles.extend(
                _ax.plot(
                    x_arr, y_arr, linestyle=y_ls, color=color, label=y_lbl))
            handles.extend(
                _ax.plot(
                    x_arr, dy_arr, linestyle=dy_ls, color=color, label=dy_lbl))

        elif method == 'errorbars':
            _ax.errorbar(
                x_arr, y_arr, dy_arr, color=color, label=y_lbl)
            handles.extend(
                _ax.plot([], [], color=color, label=y_lbl))

        elif method == 'errorarea':
            handles.extend(
                _ax.plot(
                    x_arr, y_arr, color=color, label=y_lbl))
            _ax.fill_between(
                x_arr, y_arr - dy_arr, y_arr + dy_arr,
                color=color, alpha=0.25, label=dy_lbl)
            handles.extend(
                _ax.plot([], [], color=color, alpha=0.25, label=dy_lbl))

    if legend_kws is not None:
        ax.legend(handles=handles, **dict(legend_kws))

    # set limits
    if x_limits is not None:
        ax.set_xlim(x_limits)
    if y_limits is not None:
        ax.set_ylim(y_limits)
    if twin_limits is not None:
        if shared_axis == 'x':
            twin_ax.set_ylim(twin_limits)
        else:
            twin_ax.set_xlim(twin_limits)

    # setup title and labels
    if title:
        # y=1.08 if twin_ax is not None and shared_axis='y' else None
        if twin_ax is None:
            ax.set_title(title.format(**locals()))
        else:
            twin_ax.set_title(title.format(**locals()))

    _more_texts(more_texts, ax)
    _more_elements(more_elements, ax)
    _save_plot(save_filepath, save_kws, None, fig, force, verbose)
    return handles, fig


# ======================================================================
def legend(
        y_labels=None,
        dy_labels=None,
        groups=None,
        colors=PLOT_COLORS,
        legend_kws=None,
        method='errorbars',  # 'errorarea', # 'dotted+solid',
        more_texts=None,
        more_elements=None,
        ax=None,
        save_filepath=None,
        save_kws=None,
        force=False,
        verbose=D_VERB_LVL):
    """
    Plot multiple curves including optional errorbars and twin axes.

    Args:
        y_labels:
        dy_labels:
        groups:
        colors:
        legend_kws:
        method:
        title (str|None): The plot title.
            If None, no title is set.
        more_texts (Iterable|None): List of arguments for `mpl.axes.text()`.
            See `pymrt.plot._more_texts()` for more info.
        more_elements (Iterable[Iterable]|None): List of extra elements.
            See `pymrt.plot._more_elements()` for more info.
        ax (mpl.axes): A Matplotlib axes.
        save_filepath (str|None): The file path where the plot is saved to.
            See `pymrt.plot._save_plot()` for more info.
        save_kws (dict|tuple): Keyword arguments for `mpl.figure.savefig()`.
            See `pymrt.plot._save_plot()` for more info.
        force (bool): Force new processing.
        verbose (int): Set level of verbosity.

    Returns:

    """
    method = method.lower()

    fig, ax = _ensure_fig_ax(ax)

    handles = []

    # : prepare for plotting
    num = len(y_labels)
    x_arrs = y_arrs = dy_arrs = [() for _ in range(num)]
    dy_labels = ('_nolegend_',) * num if dy_labels is None else dy_labels
    plotters = (x_arrs, y_arrs, dy_arrs, y_labels, dy_labels)

    if groups:
        if sum(groups) < num:
            groups = tuple(groups) + (num - sum(groups),)

        num = max(groups)
        tmp_colors = []
        for group, color in zip(groups, itertools.cycle(colors)):
            if callable(color):
                tmp_color = [color(i) for i in range(group)]
            elif isinstance(color, str):
                tmp_color = _color_series(color, num)
            else:
                tmp_color = color
            tmp_colors.extend(tmp_color[:group])
        colors = tmp_colors

    colors = itertools.cycle(colors)

    for i, (x_arr, y_arr, dy_arr, y_lbl, dy_lbl) in enumerate(zip(*plotters)):
        color = next(colors)

        if '+' in method:
            y_ls, dy_ls = method.split('+')
            handles.extend(
                ax.plot(
                    x_arr, y_arr, linestyle=y_ls, color=color,
                    label=y_lbl))
            handles.extend(
                ax.plot(
                    x_arr, dy_arr, linestyle=dy_ls, color=color,
                    label=dy_lbl))

        elif method == 'errorbars':
            ax.errorbar(
                x_arr, y_arr, dy_arr, color=color, label=y_lbl)
            handles.extend(
                ax.plot([], [], color=color, label=y_lbl))

        elif method == 'errorarea':
            handles.extend(
                ax.plot(
                    x_arr, y_arr, color=color, label=y_lbl))
            ax.fill_between(
                x_arr, y_arr - dy_arr, y_arr + dy_arr,
                color=color, alpha=0.25, label=dy_lbl)
            handles.extend(
                ax.plot([], [], color=color, alpha=0.25, label=dy_lbl))

    if legend_kws is not None:
        ax.legend(handles=handles, **dict(legend_kws))

    ax.axis('off')
    ax.set_aspect(1)

    _more_texts(more_texts, ax)
    _more_elements(more_elements, ax)
    _save_plot(save_filepath, save_kws, None, fig, force, verbose)
    return handles


# ======================================================================
def sample2d(
        arr,
        axis=None,
        index=None,
        array_interval=None,
        ticks_limit=None,
        interpolation='nearest',
        orientation=None,
        flip_ud=False,
        flip_lr=False,
        text_color=None,
        resolution=None,
        size_info=None,
        cmap=None,
        cbar_kws=None,
        cbar_txt=None,
        title=None,
        more_texts=None,
        more_elements=None,
        ax=None,
        save_filepath=None,
        save_kws=None,
        force=False,
        verbose=D_VERB_LVL):
    """
    Plot a 2D sample image of an N-dim array.

    Args:
        title (str|None): The plot title.
            If None, no title is set.
        more_texts (Iterable|None): List of arguments for `mpl.axes.text()`.
            See `pymrt.plot._more_texts()` for more info.
        more_elements (Iterable[Iterable]|None): List of extra elements.
            See `pymrt.plot._more_elements()` for more info.
        ax (mpl.axes): A Matplotlib axes.
        save_filepath (str|None): The file path where the plot is saved to.
            See `pymrt.plot._save_plot()` for more info.
        save_kws (dict|tuple): Keyword arguments for `mpl.figure.savefig()`.
            See `pymrt.plot._save_plot()` for more info.
        force (bool): Force new processing.
        verbose (int): Set level of verbosity.

    Returns:
        todo:
    """
    # todo: transpose/swapaxes/moveaxes/rollaxes
    data_dim = 2

    fig, ax = _ensure_fig_ax(ax)

    # prepare data
    if axis is None:
        axis = np.argsort(arr.shape)[:-data_dim]
    else:
        axis = fc.base.auto_repeat(axis, 1)
    if index is not None:
        index = fc.base.auto_repeat(index, 1)
        if len(index) != len(axis):
            raise IndexError(
                'Mismatching number of axis ({num_axis}) and index '
                '({num_index})'.format(
                    num_axis=len(axis), num_index=len(index)))

    if arr.ndim - len(axis) == data_dim:
        data = fc.extra.ndim_slice(arr, axis, index)
    elif arr.ndim == data_dim:
        data = arr
    else:
        raise IndexError(
            'Mismatching dimensions ({dim}) and axis ({num_axes}): '
            '{dim} - {num_axes} != {data_dim}'.format(
                dim=arr.ndim, num_axes=len(axis), data_dim=data_dim))
    data = _reorient_2d(data, orientation, flip_ud, flip_lr)

    # prepare plot
    if title:
        ax.set_title(title)
    if array_interval is None:
        array_interval = fc.extra.minmax(arr)
    same_sign = fc.base.is_same_sign(array_interval)
    if not cmap:
        cmap = mpl.cm.get_cmap('RdBu_r' if not same_sign else 'gray_r')
    if not text_color:
        text_color = 'k'
    ax.set_aspect('equal')

    # plot data
    pax = ax.imshow(
        data, cmap=cmap, vmin=array_interval[0], vmax=array_interval[1],
        interpolation=interpolation)

    _manage_ticks_limit(ticks_limit, ax)
    _manage_colorbar(cbar_kws, cbar_txt, ax, pax)

    # print resolution information and draw a ruler
    _manage_resolution_info(size_info, resolution, data.shape, text_color, ax)

    _more_texts(more_texts, ax)
    _more_elements(more_elements, ax)
    _save_plot(save_filepath, save_kws, None, fig, force, verbose)
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
        interpolation='nearest',
        ticks_limit=None,
        frame=True,
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

    fig, ax = _ensure_fig_ax(ax)

    # prepare data
    if axis is None:
        axis = np.argsort(arr.shape)[:-data_dim]
    else:
        axis = fc.base.auto_repeat(axis, 1)
    if index is not None:
        index = fc.base.auto_repeat(index, 1)
        if len(index) != len(axis):
            raise IndexError(
                'Mismatching number of axis ({num_axis}) and index '
                '({num_index})'.format(
                    num_axis=len(axis), num_index=len(index)))

    if arr.ndim - len(axis) == data_dim:
        data = fc.extra.ndim_slice(arr, axis, index)
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
        view_indexes = fc.base.auto_repeat(None, data_dim)
    if len(view_axes) != data_dim:
        raise IndexError('Incorrect number of view axes.')
    if len(view_indexes) != data_dim:
        raise IndexError('Incorrect number of view indexes.')

    views = []
    for view_axis, view_index in zip(view_axes, view_indexes):
        views.append(fc.extra.ndim_slice(data, view_axis, view_index))

    views[0] = _reorient_2d(views[0], orientation, False, False)

    # perform flipping
    for i, (v, f_ud, f_lr, r90, tr) in \
            enumerate(zip(views, flip_ud, flip_lr, rot90, transpose)):
        if f_ud:
            views[i] = np.flipud(views[i])
        if f_lr:
            views[i] = np.fliplr(views[i])
        if r90:
            views[i] = np.rot90(views[i])
        if tr:
            views[i] = np.transpose(views[i])
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
                views[i] = np.transpose(v)

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
        array_interval = fc.extra.minmax(arr)
    same_sign = fc.base.is_same_sign(array_interval)
    if not cmap:
        cmap = mpl.cm.get_cmap('RdBu_r' if not same_sign else 'gray_r')
    if not text_color:
        text_color = 'k'
    ax.set_aspect('equal')

    # plot data
    pax = ax.imshow(
        view, cmap=cmap, vmin=array_interval[0], vmax=array_interval[1],
        interpolation=interpolation)

    _manage_ticks_limit(ticks_limit, ax)
    _manage_colorbar(cbar_kws, cbar_txt, ax, pax)

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

    if not frame:
        ax.axis('off')

    _more_texts(more_texts, ax)
    _save_plot(save_filepath, save_kws, None, fig, force, verbose)
    return data, fig


# ======================================================================
def sample2d_multi(
        arrs,
        alphas=1.0,
        axis=None,
        index=None,
        title=None,
        array_intervals=None,
        ticks_limit=None,
        interpolation='nearest',
        orientation=None,
        flip_ud=False,
        flip_lr=False,
        cmaps=None,
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

    EXPERIMENTAL!


    Args:
        arrs ():
        alphas ():
        axis ():
        index ():
        title ():
        array_interval ():
        ticks_limit ():
        interpolation ():
        orientation ():
        flip_ud ():
        flip_lr ():
        cmaps ():
        cbar_kws ():
        cbar_txt ():
        text_color ():
        resolution ():
        size_info ():
        more_texts ():
        ax ():
        save_filepath ():
        save_kws ():
        force ():
        verbose ():

    Returns:

    """
    # todo: transpose/swapaxes/moveaxes/rollaxes
    data_dim = 2

    fig, ax = _ensure_fig_ax(ax)

    assert all([arr.shape == arrs[0].shape for arr in arrs])
    num_arrs = len(arrs)
    alphas = fc.base.auto_repeat(alphas, num_arrs, False, True)
    cmaps = fc.base.auto_repeat(cmaps, num_arrs, False, True)
    array_intervals = fc.base.auto_repeat(
        array_intervals, num_arrs, False, True)

    # prepare data
    if axis is None:
        axis = np.argsort(arrs[0].shape)[:-data_dim]
    else:
        axis = fc.base.auto_repeat(axis, 1)
    if index is not None:
        index = fc.base.auto_repeat(index, 1)
        if len(index) != len(axis):
            raise IndexError(
                'Mismatching number of axis ({num_axis}) and index '
                '({num_index})'.format(
                    num_axis=len(axis), num_index=len(index)))

    # plot title
    if title:
        ax.set_title(title)
    _manage_ticks_limit(ticks_limit, ax)
    # aspect ratio
    ax.set_aspect('equal')

    for arr, alpha, cmap, array_interval in \
            zip(arrs, alphas, cmaps, array_intervals):
        if arr.ndim - len(axis) == data_dim:
            data = fc.extra.ndim_slice(arr, axis, index)
        elif arr.ndim == data_dim:
            data = arr
        else:
            raise IndexError(
                'Mismatching dimensions ({dim}) and axis ({num_axes}): '
                '{dim} - {num_axes} != {data_dim}'.format(
                    dim=arr.ndim, num_axes=len(axis), data_dim=data_dim))
        data = _reorient_2d(data, orientation, flip_ud, flip_lr)

        if array_interval is None:
            array_interval = fc.extra.minmax(arr)
        same_sign = fc.base.is_same_sign(array_interval)
        if not cmap:
            cmap = mpl.cm.get_cmap('RdBu_r' if not same_sign else 'gray_r')
        if not text_color:
            text_color = 'k'

        # plot data
        pax = ax.imshow(
            data, cmap=cmap, vmin=array_interval[0], vmax=array_interval[1],
            interpolation=interpolation)

    _manage_colorbar(cbar_kws, cbar_txt, ax, pax)

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

    _more_texts(more_texts, ax)
    _save_plot(save_filepath, save_kws, None, fig, force, verbose)
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
    hist_interval = tuple([fc.extra.scale(val, array_interval)
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
    fig, ax = _ensure_fig_ax(ax)
    pax = ax.plot(fc.base.midval(bin_edges), hist, **dict(style))
    # setup title and labels
    if title:
        ax.set_title(title.format(**locals()))
    if labels[0]:
        ax.set_xlabel(labels[0].format(**locals()))
    if labels[1]:
        ax.set_ylabel(labels[1].format(**locals()))
    _more_texts(more_texts, ax)
    _save_plot(save_filepath, save_kws, None, fig, force, verbose)
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
    hist_interval = tuple([fc.extra.scale(val, array_interval)
                           for val in hist_interval])

    # prepare style list
    if styles is None:
        styles = []
        for linestyle in PLOT_LINESTYLES:
            for color in PLOT_COLORS:
                style = dict(linestyle=linestyle, color=color)
                styles.append(style)

    style_cycler = itertools.cycle(styles)

    # prepare histograms
    fig, ax = _ensure_fig_ax(ax)
    ax.set_aspect('auto')
    data = []
    for i, array in enumerate(arrays):
        hist, bin_edges = np.histogram(
            array, bins=bins, range=hist_interval,
            density=(scale == 'density'))
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
        pax = ax.plot(
            fc.base.midval(bin_edges), hist,
            label=legend,
            **next(style_cycler))
        data.append((hist, bin_edges))
    # create the legend for the first line.
    ax.legend(**(legend_kws if legend_kws is not None else {}))
    # fine-tune ticks
    _manage_ticks_limit(ticks_limit, ax)
    # setup title and labels
    if title:
        ax.set_title(title.format(**locals()))
    if labels[0]:
        ax.set_xlabel(labels[0].format(**locals()))
    if labels[1]:
        ax.set_ylabel(labels[1].format(**locals()))
    _more_texts(more_texts, ax)
    _save_plot(save_filepath, save_kws, None, fig, force, verbose)
    return data, fig


# ======================================================================
def histogram2d(
        arr1,
        arr2,
        bin_size=1,
        bins=None,
        array_interval=None,
        hist_interval=((0.0, 1.0), (0.0, 1.0)),
        use_sn_split_interval=False,
        aspect=None,
        scale='linear',
        hist_val_interval=None,
        ticks_limit=None,
        interpolation='bicubic',
        title='2D Histogram ({bins} bins, {scale})',
        labels=('Array 1 Values', 'Array 2 Values'),
        cmap='afmhot_r',
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
            Otherwise, if a tuple of tuple of float is given, the first
            tuple of
            float is interpreted as the (min, max) for the x-axis, and the
            second tuple of float is for the y-axis.
        hist_interval (tuple[float|tuple[float]]): The histogram interval.
            If a tuple of float is given, it is interpreted as the (min, max)
            of the interval to use for the histogram as percentage of the array
            interval (specified or calculated), and it is assumed identical for
            both axes.
            Otherwise, if a tuple of tuple of float is given, the first
            tuple of
            float is interpreted as the (min, max) for the x-axis, and the
            second tuple of float is for the y-axis.
        use_sn_split_interval (bool): Generate 'array_interval' separately.
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
        hist_val_interval (tuple[float]|None): The interval of histogram
        values.
            If None, it is calculated automatically as the (min, max) of the
            histogram values rounded to the most comprehensive integer
            interval.
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
        save_filepath (str): The file path where the plot is saved to.
            If unset, no output is saved.
        ax (matplotlib.axes): The Matplotlib Axes object used for plotting.

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
        if use_sn_split_interval:
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
    hist_interval = tuple([[fc.extra.scale(val, array_interval[i])
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
    fig, ax = _ensure_fig_ax(ax)
    # plot figure
    pax = ax.imshow(
        hist, cmap=cmap, origin='lower', interpolation=interpolation,
        aspect=aspect,
        vmin=hist_val_interval[0], vmax=hist_val_interval[1],
        extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]])
    _manage_ticks_limit(ticks_limit, ax)
    _manage_colorbar(cbar_kws, cbar_txt, ax, pax)
    # # plot the color bar
    # if cbar_kws is not None:
    #     cbar = ax.figure.colorbar(pax, ax=ax, **dict(cbar_kws))
    #     if cbar_txt is not None:
    #         only_extremes = 'ticks' in cbar_kws and len(cbar_kws['ticks'])
    #  == 2
    #         if only_extremes:
    #             cbar.ax.text(2.0, 0.5, cbar_txt, fontsize='small',
    # rotation=90)
    #         else:
    #             cbar.set_label(cbar_txt)
    #     if ticks_limit is not None:
    #         if ticks_limit > 0:
    #             cbar.locator = mpl.ticker.MaxNLocator(nbins=ticks_limit)
    #         else:
    #             cbar.set_ticks([])
    #         cbar.update_ticks()
    # plot first bisector
    if bisector:
        ax.autoscale(False)
        ax.plot(
            array_interval[0], array_interval[1], bisector, label='bisector')
    if stats_kws is not None:
        mask = np.ones_like(arr1 * arr2).astype(bool)
        mask *= (arr1 > array_interval[0][0]).astype(bool)
        mask *= (arr1 < array_interval[0][1]).astype(bool)
        mask *= (arr2 > array_interval[1][0]).astype(bool)
        mask *= (arr2 < array_interval[1][1]).astype(bool)
        stats_dict = fc.extra.calc_stats(
            arr1[mask] - arr2[mask], **dict(stats_kws))
        stats_text = '$\\mu_D = {}$\n$\\sigma_D = {}$'.format(
            *fc.base.format_value_error(
                stats_dict['avg'], stats_dict['std'], 3))
        ax.text(
            1 / 2, 31 / 32, stats_text,
            horizontalalignment='center', verticalalignment='top',
            transform=ax.transAxes)
    # setup title and labels
    if title:
        ax.set_title(title.format(**locals()))
    if labels[0]:
        ax.set_xlabel(labels[0].format(**locals()))
    if labels[1]:
        ax.set_ylabel(labels[1].format(**locals()))

    _manage_ticks_limit(ticks_limit, ax)
    _more_texts(more_texts, ax)
    _save_plot(save_filepath, save_kws, None, fig, force, verbose)
    return (hist, x_edges, y_edges), fig


# ======================================================================
def bar_chart(
        data,
        err,
        series,
        groups,
        x_label=None,
        y_label=None,
        colors=PLOT_COLORS,
        limits=None,
        legend_kws=None,
        orientation='h',
        bar_width=None,
        title=None,
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
        limits:
        legend_kws:
        bar_width:
        ax:

    Returns:

    """
    # todo: polish code and documentation
    # todo: add support for orientation
    # todo: add support for stacked bars
    # todo: add support for variable bar_width

    fig, ax = _ensure_fig_ax(ax)

    num_series = len(series)
    if groups is not None:
        num_groups = len(groups)
    else:
        num_groups = 1
    indices = np.arange(num_groups)
    if not bar_width:
        bar_width = 1 / (num_series + 0.5)

    orientation = orientation.lower()
    if orientation in ('h', 'horizontal'):
        is_hor = True
    elif orientation in ('v', 'vertical'):
        is_hor = False
    else:
        text = 'Unknown orientation `{}`. ' \
               'Fallback to `horizontal`.'.format(orientation)
        warnings.warn(text)
        is_hor = True

    if not is_hor:
        ax.invert_yaxis()

    if limits is not None:
        set_lim = ax.set_ylim if is_hor else ax.set_xlim
        set_lim(limits)

    colors = itertools.cycle(colors if is_hor else colors)
    bcs = []
    for j, serie in enumerate(series):
        bar_data = data[serie]
        bar_err = err[serie] if err is not None else None
        bar_plot = ax.bar if is_hor else ax.barh
        bc = bar_plot(
            indices + (j * bar_width),
            bar_data,
            bar_width,
            xerr=None if is_hor else bar_err,
            yerr=bar_err if is_hor else None,
            color=next(colors))
        bcs.append(bc)
    the_axis = ax.xaxis if is_hor else ax.yaxis
    if groups is not None:
        the_axis.set_ticks(
            np.arange(num_groups + 1) - bar_width * 3 / 4, minor=False)
        the_axis.set_ticks(
            np.arange(num_groups) + (len(series) - 1) * bar_width / 2,
            minor=True)
        the_axis.set_ticklabels([], minor=False)
        the_axis.set_ticklabels(groups, minor=True)
    else:
        the_axis.set_ticks(
            np.arange(num_series + 1) * bar_width - bar_width / 2,
            minor=False)
        the_axis.set_ticks(
            np.arange(num_series) * bar_width, minor=True)
        the_axis.set_ticklabels([], minor=False)
        the_axis.set_ticklabels(series, minor=True)
    if x_label:
        ax.set_xlabel(x_label)
    if y_label:
        ax.set_ylabel(y_label)
    if title:
        ax.set_title(title)

    if legend_kws is not None:
        ax.legend(tuple(bc[0] for bc in bcs), series, **dict(legend_kws))
    return data, fig


# ======================================================================
def heatmap(
        data,
        x_ticks=None,
        y_ticks=None,
        x_label=None,
        y_label=None,
        title=None,
        tick_top=False,
        y_axis_invert=False,
        ax=None,
        **_kws):
    fig, ax = _ensure_fig_ax(ax)

    ax = sns.heatmap(data, ax=ax, **_kws)

    if x_ticks is not None:
        ax.set_xticks(x_ticks)
    if y_ticks is not None:
        ax.set_yticks(y_ticks)

    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    if tick_top:
        ax.xaxis.tick_top()
    else:
        ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    if y_axis_invert:
        ax.invert_yaxis()

    if x_label:
        ax.set_xlabel(x_label)
    if y_label:
        ax.set_ylabel(y_label)
    if title:
        ax.set_title(title, y=1.08 if tick_top else 1.0)

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
        size_factors=3,
        pads=0.03,
        swap_filling=False,
        title=None,
        subplot_title_fmt='({letter}) {t}',
        row_labels=None,
        col_labels=None,
        label_pads=0.03,
        borders=0.0,
        subplot_kws=None,
        legend_kws=None,
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
        savefig_kws ():
        force (): 
        verbose (): 

    Returns:

    """
    num_plots = len(plots)
    if num_plots == 0:
        warnings.warn('Nothing to plot.')
        return

    # determine rows and cols if not specified
    if rows is None and cols is None:
        if num_row is None and num_col is None:
            if isinstance(aspect_ratio, float):
                num_col = np.ceil((num_plots * aspect_ratio) ** 0.5)
            elif isinstance(aspect_ratio, str):
                if 'exact' in aspect_ratio:
                    num_col, num_row = fc.base.optimal_shape(num_plots)
                    if 'portrait' in aspect_ratio:
                        num_row, num_col = num_col, num_row
                if aspect_ratio == 'portrait':
                    num_row = np.ceil((num_plots) ** 0.5)
            else:  # plot_aspect == 'landscape'
                num_row = int(np.floor((num_plots) ** 0.5))

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

    pads = list(fc.base.auto_repeat(pads, 2, False, True))
    label_pads = list(fc.base.auto_repeat(label_pads, 2, False, True))
    borders = list(fc.base.auto_repeat(borders, 4, False, True))
    size_factors = list(
        fc.base.auto_repeat(size_factors, 2, False, True))

    # fix row/col labels
    if row_labels is None:
        row_labels = fc.base.auto_repeat(None, num_row)
        label_pads[0] = 0.0
    if col_labels is None:
        col_labels = fc.base.auto_repeat(None, num_col)
        label_pads[1] = 0.0
    assert (num_row == len(row_labels))
    assert (num_col == len(col_labels))

    # generate plot
    label_pads = [k * x for k, x in zip(size_factors, label_pads)]
    pads = [k * x for k, x in zip(size_factors, pads)]

    fig_sizes = [
        (k * sum(items) + pad * (len(items) - 1) + label_pad)
        for k, items, pad, label_pad
        in zip(size_factors, (rows, cols), pads, label_pads)]

    label_pads = [x / k for k, x in zip(size_factors, label_pads)]
    pads = [k * x for k, x in zip(fig_sizes, pads)]

    # fig, axs = plt.subplots(
    #     nrows=num_row, ncols=num_col,
    #     gridspec_kw={'width_ratios': cols, 'height_ratios': rows},
    #     figsize=fig_sizes[::-1])

    legend_pad = 0
    if legend_kws is not None:
        legend_pad = legend_kws.pop('pad') if 'pad' in legend_kws else 0
        legend_row = legend_kws.pop('row') if 'row' in legend_kws else None
        legend_col = legend_kws.pop('col') if 'col' in legend_kws else None
        if legend_row is not None and legend_col is not None:
            legend_pad = 0

    fig = plt.figure(figsize=fig_sizes[::-1])
    gs = mpl.gridspec.GridSpec(
        num_row, num_col, width_ratios=cols, height_ratios=rows)

    if subplot_kws is None:
        subplot_kws = ({},) * (num_row * num_col)
    else:
        subplot_kws = tuple(subplot_kws)
        subplot_kws += ({},) * (num_row * num_col - len(subplot_kws))
        if swap_filling:
            subplot_kws = tuple(
                np.ravel(np.transpose(
                    np.array(subplot_kws).reshape((num_col, num_row)))))

    axs = [
        fig.add_subplot(gs[n], **subplot_kws[n])
        for n in range(num_row * num_col)]
    axs = np.array(axs).reshape((num_row, num_col))

    skip_offset = 0
    for i, row_label in enumerate(row_labels):
        if row_label:
            fig.text(
                label_pads[0] / 2,
                fc.extra.scale(
                    1.0 - (i * 2 + 1) / (num_row * 2),
                    out_interval=(0, 1.0 - label_pads[1])),
                row_label, rotation=90,
                fontweight='bold', fontsize='large',
                horizontalalignment='left', verticalalignment='center')

    for j, col_label in enumerate(col_labels):
        if col_label:
            fig.text(
                fc.extra.scale(
                    (j * 2 + 1) / (num_col * 2),
                    out_interval=(
                        label_pads[0], 1.0 - legend_pad)),
                1.0 - label_pads[1] / 2,
                col_label, rotation=0,
                fontweight='bold', fontsize='large',
                horizontalalignment='center', verticalalignment='top')

    for n_plot in range(num_row * num_col):
        i, j = np.unravel_index(
            n_plot, (num_row, num_col), 'F' if swap_filling else 'C')

        if n_plot < num_plots:
            plot_func, plot_args, plot_kws = plots[n_plot]
            plot_kws['ax'] = axs[i, j]
            if subplot_title_fmt:
                t = plot_kws['title'] if 'title' in plot_kws else ''
                if t is not None:
                    roman = numeral.int2roman(n_plot + 1, only_ascii=True)
                    roman_uppercase = roman.upper()
                    roman_lowercase = roman.lower()
                    letter = numeral.int2letter(n_plot - skip_offset)
                    letter_uppercase = letter.upper()
                    letter_lowercase = letter.lower()
                    plot_kws['title'] = \
                        subplot_title_fmt.format(**locals())
                else:
                    skip_offset += 1
            plot_func(*plot_args, **plot_kws)
        else:
            axs[i, j].clear()
            axs[i, j].set_axis_off()

    if legend_kws is not None:
        if 'handles' in legend_kws and 'labels' not in legend_kws:
            legend_kws['labels'] = tuple(
                h.get_label() for h in legend_kws['handles'])
        if legend_row is not None and legend_col is not None:
            axs[legend_row, legend_col].legend(**dict(legend_kws))
            legend_pad = 0
        else:
            fig.legend(**dict(legend_kws))

    _more_texts(more_texts, fig)
    tight_layout_kws = dict(
        rect=[
            0.0 + label_pads[0] + borders[0],
            0.0 + borders[1],
            1.0 - legend_pad - borders[2],
            1.0 - label_pads[1] - borders[3]],
        pad=1.0, h_pad=pads[0], w_pad=pads[1])
    _save_plot(save_filepath, save_kws, tight_layout_kws, fig, force, verbose)
    return fig


# ======================================================================
elapsed(__file__[len(PATH['base']) + 1:])

# ======================================================================
if __name__ == '__main__':
    run_doctests(__doc__)
