#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PyMRT: Generation of interactive plots.

Generate interactive plots from a minimal set of specification.
In particular, two objects need to be defined:
- a plotting function, which must accept:
    - a `matplotlib.Axes` where the plot is shown (this is used internally)
    - the dictionary of parameters for which interactivity is desired
- an ordered dictionary with interactivity information, where the key 
  correspond to the internal name of the parameter (useful for **kwargs magic),
  and the value is a dictionary with the following required fields:
    - 'label': the printing name of the variable (will show in GUI)
    - 'default': the default value
    - 'start': the minimum value of the parameter
    - 'stop': the maximum value of the parameter
    - 'step': the step size for the variation

Examples:
    >>> import numpy as np
    >>> interactives = collections.OrderedDict([
    ...     ('a', dict(
    ...         label='a (arb.units)',
    ...         default=10, start=-100, stop=100, step=0.01)), ])
    >>> def plot_func(
    ...         ax,
    ...         params=None,
    ...         title='Test'):
    ...     x = np.linspace(-100, 100, 128)
    ...     try:
    ...         y = np.sin(x / params['a'])
    ...         ax.plot(x, y, label=r'$\sin(x / a)$')
    ...     except:
    ...         ax.set_title('\\n'.join(('WARNING! Some plot failed!', title)))
    ...     else:
    ...         ax.set_title(title)
    ...     finally:
    ...         ax.set_xlabel(r'x (arb.units)')
    ...         ax.set_ylabel(r'y (arb.units)')
    ...         ax.legend()
    ...     return ax
    >>> mpl_plot(plot_func, interactives, title='Test')
    <tkinter.Tk object .>
"""

# ======================================================================
# :: Future Imports
from __future__ import (
    division, absolute_import, print_function, unicode_literals)

# ======================================================================
# :: Python Standard Library Imports
import collections  # Container datatypes
import datetime  # Basic date and time types
import doctest  # Test interactive Python examples

# :: External Imports
import matplotlib as mpl  # Matplotlib (2D/3D plotting library)
import pytk
import pytk.widgets

import matplotlib.backends.backend_tkagg as tkagg

# :: Local Imports
import pymrt

from pymrt import INFO, DIRS, MY_GREETINGS
from pymrt import VERB_LVL, D_VERB_LVL
from pymrt import msg, dbg
from pymrt import elapsed, print_elapsed

# ======================================================================
_MIN_WIDTH = 320
_MIN_HEIGHT = 200
_WIDTH = 960
_HEIGHT = 600


# ======================================================================
class PytkAbout(pytk.Window):
    def __init__(self, parent, about=__doc__):
        self.win = super(PytkAbout, self).__init__(parent)
        self.transient(parent)
        self.parent = parent
        self.title('About {}'.format(INFO['name']))
        self.resizable(False, False)
        self.frm = pytk.widgets.Frame(self)
        self.frm.pack(fill='both', expand=True)
        self.frmMain = pytk.widgets.Frame(self.frm)
        self.frmMain.pack(fill='both', padx=1, pady=1, expand=True)

        about_txt = '\n'.join((
            MY_GREETINGS[1:],
            pymrt.__doc__,
            about,
            '{} - ver. {}\n{} {}\n{}'.format(
                INFO['name'], INFO['version'],
                INFO['copyright'], INFO['author'], INFO['notice'])
        ))
        msg(about_txt)
        self.lblInfo = pytk.widgets.Label(
            self.frmMain, text=about_txt, anchor='center',
            background='#333', foreground='#ccc', font='TkFixedFont')
        self.lblInfo.pack(padx=8, pady=8, ipadx=8, ipady=8)

        self.btnClose = pytk.widgets.Button(
            self.frmMain, text='Close', command=self.destroy)
        self.btnClose.pack(side='bottom', padx=8, pady=8)
        self.bind('<Return>', self.destroy)
        self.bind('<Escape>', self.destroy)

        pytk.utils.center(self, self.parent)

        self.grab_set()
        self.wait_window(self)


# ======================================================================
class PytkMain(pytk.widgets.Frame):
    def __init__(
            self, parent, func, interactives,
            title=__doc__.strip().split('\n')[0],
            about=__doc__,
            width=_WIDTH, height=_HEIGHT,
            min_width=_MIN_WIDTH, min_height=_MIN_HEIGHT,
            *args, **kwargs):
        self.func = func
        self.interactives = interactives
        self.about = about

        # :: initialization of the UI
        self.win = super(PytkMain, self).__init__(
            parent, width=width, height=height)
        self.parent = parent
        self.parent.title(title)
        self.parent.protocol('WM_DELETE_WINDOW', self.actionExit)
        self.parent.minsize(min_width, min_height)

        self.style = pytk.Style()
        # print(self.style.theme_names())
        self.style.theme_use('clam')
        self.pack(fill='both', expand=True)
        pytk.utils.center(self.parent)

        self._make_menu()

        # :: define UI items
        # : main
        self.frmMain = pytk.widgets.Frame(self)
        self.frmMain.pack(fill='both', padx=4, pady=4, expand=True)
        self.frmSpacers = []

        # : left frame
        self.frmLeft = pytk.widgets.Frame(self.frmMain)
        self.frmLeft.pack(
            side='left', fill='both', padx=4, pady=4, expand=True)

        self.fig = mpl.figure.Figure(figsize=(0.1, 0.1))
        self.ax = self.fig.add_subplot(111)
        self.canvas = tkagg.FigureCanvasTkAgg(
            self.fig, self.frmLeft)
        self.nav_toolbar = tkagg.NavigationToolbar2TkAgg(
            self.canvas, self.frmLeft)
        self.canvas.get_tk_widget().pack(fill='both', expand=True)
        self.actionPlotUpdate()

        # : right frame
        self.frmRight = pytk.widgets.ScrollingFrame(
            self.frmMain,
            label_kws=dict(text='Parameters'),
            label_pack_kws=dict(
                side='top', padx=0, pady=0, expand=False, anchor='n'))
        self.frmRight.pack(
            side='right', fill='both', padx=4, pady=4, expand=False)

        self.frmParams = self.frmRight.scrolling
        spacer = pytk.widgets.Frame(self.frmParams)
        spacer.pack(side='top', padx=4, pady=4)
        self.frmSpacers.append(spacer)
        self.wdgInteractives = collections.OrderedDict()
        for name, info in self.interactives.items():
            if isinstance(info['default'], bool):
                var = pytk.tk.BooleanVar()
                chk = pytk.widgets.Checkbox(
                    self.frmParams, text=info['label'], variable=var)
                chk.pack(fill='x', padx=1, pady=1)
                self.wdgInteractives[name] = {'var': var, 'chk': chk}
            elif isinstance(info['default'], (int, float)):
                var = pytk.tk.StringVar()
                var.set(info['default'])
                frm = pytk.widgets.Frame(self.frmParams)
                frm.pack(fill='x', padx=1, pady=1, expand=True)
                lbl = pytk.widgets.Label(frm, text=info['label'])
                lbl.pack(side='left', fill='x', padx=1, pady=1, expand=False)
                rng = pytk.widgets.Range(
                    frm,
                    start=info['start'], stop=info['stop'], step=info['step'],
                    orient='horizontal', variable=var)
                rng.pack(
                    side='right', fill='x', anchor='w', padx=1, pady=1,
                    expand=False)
                spb = pytk.widgets.Spinbox(
                    frm,
                    start=info['start'], stop=info['stop'], step=info['step'],
                    textvariable=var)
                spb.pack(
                    side='right', fill='none', anchor='w', padx=1, pady=1,
                    expand=False)
                self.wdgInteractives[name] = {
                    'var': var, 'frm': frm, 'lbl': lbl, 'spb': spb, 'rng': rng}
            elif isinstance(info['default'], (str, bytes)):
                pass
        self._bind_interactions()

    def _make_menu(self):
        self.mnuMain = pytk.widgets.Menu(self.parent, tearoff=False)
        self.parent.config(menu=self.mnuMain)
        self.mnuPlot = pytk.widgets.Menu(self.mnuMain, tearoff=False)
        self.mnuMain.add_cascade(label='Plot', menu=self.mnuPlot)
        self.mnuPlot.add_command(label='Reset', command=self.actionReset)
        self.mnuPlot.add_separator()
        self.mnuPlot.add_command(label='Exit', command=self.actionExit)
        self.mnuHelp = pytk.widgets.Menu(self.mnuMain, tearoff=False)
        self.mnuMain.add_cascade(label='Help', menu=self.mnuHelp)
        self.mnuHelp.add_command(label='About', command=self.actionAbout)

    def _bind_interactions(self):
        for k, v in self.wdgInteractives.items():
            v['trace'] = v['var'].trace('w', self.actionPlotUpdate)

    def _unbind_interactions(self):
        for k, v in self.wdgInteractives.items():
            v['var'].trace_vdelete('w', v['trace'])

    def actionPlotUpdate(self, *args):
        """Update the plot."""
        self.ax.clear()
        if hasattr(self, 'wdgInteractives'):
            params = {}
            for k, v in self.wdgInteractives.items():
                val = v['var'].get()
                try:
                    val = float(val)
                except ValueError:
                    val = self.interactives[k]['default']
                params[k] = val
        else:
            params = {k: v['default'] for k, v in self.interactives.items()}
        self.func(ax=self.ax, params=params)
        self.canvas.draw()

    def actionExit(self, event=None):
        """Action on Exit."""
        if pytk.messagebox.askokcancel(
                'Quit', 'Are you sure you want to quit?'):
            self.parent.destroy()

    def actionAbout(self, event=None):
        """Action on About."""
        self.winAbout = PytkAbout(self.parent, self.about)

    def actionReset(self, event=None):
        """Action on Reset."""
        self._unbind_interactions()
        for name, info in self.interactives.items():
            self.wdgInteractives[name]['var'].set(info['default'])
        self._bind_interactions()
        self.actionPlotUpdate()


# ======================================================================
def mpl_plot(func, interactives, gui_main=PytkMain, *args, **kwargs):
    root = pytk.tk.Tk()
    app = gui_main(root, func, interactives, *args, **kwargs)
    pytk.utils.set_icon(root, 'icon', DIRS['data'])
    root.mainloop()
    return root


# ======================================================================
if __name__ == '__main__':
    msg(__doc__.strip())
    begin_time = datetime.datetime.now()
    doctest.testmod()
    end_time = datetime.datetime.now()
    print('ExecTime: {}'.format(end_time - begin_time))
