#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PyMRT - Python Magnetic Resonance Tools: data analysis for quantitative MRI.
"""

# Copyright (c) Riccardo Metere <rick@metere.it>

# ======================================================================
# :: Future Imports
from __future__ import (
    division, absolute_import, print_function, unicode_literals, )

# ======================================================================
# :: Python Standard Library Imports
import datetime  # Basic date and time types
# import inspect  # Inspect live objects
import os  # Miscellaneous operating system interfaces
import appdirs  # Determine appropriate platform-specific dirs
import pkg_resources  # Manage package resource (from setuptools module)

# ======================================================================
# :: Version
from ._version import __version__

# ======================================================================
# :: Project Details
INFO = {
    'name': 'PyMRT',
    'author': 'PyMRT developers',
    'contrib': (
        'Riccardo Metere <rick@metere.it>',
    ),
    'copyright': 'Copyright (C) 2015-2017',
    'license': 'GNU General Public License version 3 or later (GPLv3+)',
    'notice':
        """
This program is free software and it comes with ABSOLUTELY NO WARRANTY.
It is covered by the GNU General Public License version 3 (GPLv3+).
You are welcome to redistribute it under its terms and conditions.
        """,
    'version': __version__
}

# ======================================================================
# :: Supported Verbosity Levels
VERB_LVL_NAMES = (
    'none', 'lowest', 'lower', 'low', 'medium', 'high', 'higher', 'highest',
    'warning', 'debug')
VERB_LVL = {k: v for v, k in enumerate(VERB_LVL_NAMES)}
D_VERB_LVL = VERB_LVL['lowest']

# ======================================================================
# :: quick and dirty timing facility
_EVENTS = []

# ======================================================================
# Greetings
MY_GREETINGS = r"""
 ____        __  __ ____ _____
|  _ \ _   _|  \/  |  _ \_   _|
| |_) | | | | |\/| | |_) || |
|  __/| |_| | |  | |  _ < | |
|_|    \__, |_|  |_|_| \_\|_|
       |___/
"""
# generated with: figlet 'PyMRT' -f standard

# :: Causes the greetings to be printed any time the library is loaded.
print(MY_GREETINGS)


# ======================================================================
def msg(
        text,
        verb_lvl=D_VERB_LVL,
        verb_threshold=D_VERB_LVL,
        fmt=None,
        *args,
        **kwargs):
    """
    Display a feedback message to the standard output.

    Args:
        text (str|Any): Message to display or object with `__repr__`.
        verb_lvl (int): Current level of verbosity.
        verb_threshold (int): Threshold level of verbosity.
        fmt (str): Format of the message (if `blessed` supported).
            If None, a standard formatting is used.
        *args (*tuple): Positional arguments to be passed to `print`.
        **kwargs (**dict): Keyword arguments to be passed to `print`.

    Returns:
        None.

    Examples:
        >>> s = 'Hello World!'
        >>> msg(s)
        Hello World!
        >>> msg(s, VERB_LVL['medium'], VERB_LVL['low'])
        Hello World!
        >>> msg(s, VERB_LVL['low'], VERB_LVL['medium'])  # no output
        >>> msg(s, fmt='{t.green}')  # if ANSI Terminal, green text
        Hello World!
        >>> msg('   :  a b c', fmt='{t.red}{}')  # if ANSI Terminal, red text
           :  a b c
        >>> msg(' : a b c', fmt='cyan')  # if ANSI Terminal, cyan text
         : a b c
    """
    if verb_lvl >= verb_threshold and text is not None:
        # if blessed/blessings is not present, no coloring
        try:
            import blessed
        except ImportError:
            try:
                import blessings as blessed
            except ImportError:
                blessed = None

        try:
            t = blessed.Terminal()
        except (ValueError, AttributeError):
            t = None

        if blessed and t:
            text = str(text)
            if not fmt:
                if VERB_LVL['low'] < verb_threshold <= VERB_LVL['medium']:
                    e = t.cyan
                elif VERB_LVL['medium'] < verb_threshold < VERB_LVL['debug']:
                    e = t.magenta
                elif verb_threshold >= VERB_LVL['debug']:
                    e = t.blue
                elif text.startswith('I:'):
                    e = t.green
                elif text.startswith('W:'):
                    e = t.yellow
                elif text.startswith('E:'):
                    e = t.red
                else:
                    e = t.white
                # first non-whitespace word
                txt1 = text.split(None, 1)[0] if len(text) > 0 else ''
                # initial whitespaces
                n = text.find(txt1)
                txt0 = text[:n]
                # rest
                txt2 = text[n + len(txt1):]
                txt_kws = dict(
                    e1=e + (t.bold if e == t.white else ''),
                    e2=e + (t.bold if e != t.white else ''),
                    t0=txt0, t1=txt1, t2=txt2, n=t.normal)
                text = '{t0}{e1}{t1}{n}{e2}{t2}{n}'.format_map(txt_kws)
            else:
                if 't.' not in fmt:
                    fmt = '{{t.{}}}'.format(fmt)
                if '{}' not in fmt:
                    fmt += '{}'
                text = fmt.format(text, t=t) + t.normal
        print(text, *args, **kwargs)


# ======================================================================
def dbg(obj, fmt=None):
    """
    Print content of a variable for debug purposes.

    Args:
        obj: The name to be inspected.
        fmt (str): Format of the message (if `blessed` supported).
            If None, a standard formatting is used.

    Returns:
        None.

    Examples:
        >>> my_dict = {'a': 1, 'b': 1}
        >>> dbg(my_dict)
        dbg(my_dict): (('a', 1), ('b', 1))
        >>> dbg(my_dict['a'])
        dbg(my_dict['a']): 1
    """
    import inspect


    outer_frame = inspect.getouterframes(inspect.currentframe())[1]
    name_str = outer_frame[4][0][:-1]
    msg(name_str, fmt=fmt, end=': ')
    if isinstance(obj, dict):
        obj = tuple(sorted(obj.items()))
    text = repr(obj)
    msg(text, fmt='')


# ======================================================================
def elapsed(
        name=None,
        time_point=None,
        events=_EVENTS):
    """
    Append a named event point to the events list.

    Args:
        name (str): The name of the event point.
        time_point (float): The time in seconds since the epoch.
        events (list[(str,datetime.datetime)]): A list of named time points.
            Each event is a 2-tuple: (label, datetime.datetime).

    Returns:
        None.
    """
    if name is None:
        # outer_frame = inspect.getouterframes(inspect.currentframe())[1]
        filename = __file__
        name = os.path.basename(filename)

    if not time_point:
        time_point = datetime.datetime.now()
    events.append((name, time_point))


# ======================================================================
def report(
        events=_EVENTS,
        title='Elapsed Time(s)',
        labels=('Label', 'Duration / s', 'Cum. Duration / s'),
        max_col_widths=(36, 20, 20),
        title_sep='=',
        label_sep='-',
        only_last=False):
    """
    Print quick-and-dirty elapsed times between named event points.

    Args:
        events (list[(str,datetime.datetime)]): A list of named time points.
            Each event is a 2-tuple: (label, time).
        title (str): heading of the elapsed time table.
        labels (Iterable[str]): Labels for the report.
            Three elements are expected.
        max_col_widths (Iterable[int]): Maximum width of columns in the report.
            Three elements are expected.
        title_sep (str): The separator used to underline the title.
        label_sep (str): The separator used to underline the labels.
        only_last (bool): print only the last event (useful inside a loop).

    Returns:
        None.
    """
    text = '\n'
    if events:
        if not only_last and len(events) > 2:
            fmt = '{{!s:{}s}}  {{!s:>{}s}}  {{!s:>{}s}}\n'.format(
                *max_col_widths)
            title_sep = ((title_sep * len(title))[:len(title)] + '\n') \
                if title_sep else ''
            text += title + (
                '\n' + title_sep + '\n' if len(events) > 2 else ': ')

            if labels and len(events) > 2:
                text += (fmt.format(*labels))

            if label_sep:
                text += (fmt.format(
                    *[(label_sep * max_col_width)[:max_col_width]
                      for max_col_width in max_col_widths]))

            first_elapsed = events[0][1]
            for i in range(len(events) - 1):
                _id = i + 1
                name = events[_id][0]
                curr_elapsed = events[_id][1]
                prev_elapsed = events[_id - 1][1]
                diff_first = curr_elapsed - first_elapsed
                diff_last = curr_elapsed - prev_elapsed
                if diff_first == diff_last:
                    diff_first = '-'
                text += (fmt.format(
                    name[:max_col_widths[0]], diff_last, diff_first))
        elif len(events) > 1:
            fmt = '{{!s:{}s}}  {{!s:>{}s}}'.format(*max_col_widths)
            _id = -1
            name = events[_id][0]
            curr_elapsed = events[_id][1]
            prev_elapsed = events[_id - 1][1]
            text += (fmt.format(name, curr_elapsed - prev_elapsed))
        else:
            events = None

    if not events:
        text += 'No ' + title.lower() + ' to report!'
    return text


# ======================================================================
def _app_dirs(
        name=INFO['name'],
        author=INFO['author'],
        version=INFO['version']):
    """
    Generate application directories.

    Args:
        name (str): Application name.
        author (str): Application author.
        version (str): Application version.

    Returns:
        dirs (dict): The requested directory.
            - 'config': directory for configuration files.
            - 'cache': directory for caching files.
            - 'data': directory for data files.
            - 'log': directory for log files.

    Examples:
        >>> sorted(_app_dirs().keys())
        ['base', 'cache', 'config', 'data', 'log']
    """
    dirpaths = dict((
        ('base', os.path.dirname(__file__)),  # todo: fix for pyinstaller
        ('resources', pkg_resources.resource_filename('pymrt', 'resources')),
        ('config', appdirs.user_config_dir(name, author, version)),
        ('cache', appdirs.user_cache_dir(name, author, version)),
        ('data', appdirs.user_data_dir(name, author, version)),
        ('log', appdirs.user_data_dir(name, author, version)),
    ))
    for name, dirpath in dirpaths.items():
        if not os.path.isdir(dirpath):
            os.makedirs(dirpath)
    return dirpaths


# ======================================================================
DIRS = _app_dirs()

# ======================================================================
elapsed()

# ======================================================================
if __name__ == '__main__':
    import doctest  # Test interactive Python examples

    msg(__doc__.strip())
    doctest.testmod()
    msg(report())
