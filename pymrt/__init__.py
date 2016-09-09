#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
PyMRT: data analysis for quantitative MRI
"""

# Copyright (c) Riccardo Metere <rick@metere.it>

# ======================================================================
# :: Future Imports
from __future__ import division
from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals

# ======================================================================
# :: Python Standard Library Imports


# ======================================================================
# :: Version
__version__ = '0.1.dev106+nbc2e2c7'

# ======================================================================
# :: Project Details
INFO = {
    'authors': (
        'Riccardo Metere <metere@cbs.mpg.de>',
    ),
    'copyright': 'Copyright (C) 2015',
    'license': 'GNU General Public License version 3 or later (GPLv3+)',
    'notice':
        """
This program is free software and it comes with ABSOLUTELY NO WARRANTY.
It is covered by the GNU General Public License version 3 (GPLv3).
You are welcome to redistribute it under its terms and conditions.
        """,
    'version': __version__
}

# ======================================================================
# :: supported verbosity levels
VERB_LVL_NAMES = (
    'none', 'lowest', 'lower', 'low', 'medium', 'high', 'higher', 'highest',
    'warning', 'debug')
VERB_LVL = {k: v for k, v in zip(VERB_LVL_NAMES, range(len(VERB_LVL_NAMES)))}
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
        text (str): Message to display.
        verb_lvl (int): Current level of verbosity.
        verb_threshold (int): Threshold level of verbosity.
        fmt (str): Format of the message (if `blessed` supported).
            If None, a standard formatting is used.
        *args (tuple): Positional arguments to be passed to `print`.
        **kwargs (dict): Keyword arguments to be passed to `print`.

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
    if verb_lvl >= verb_threshold:
        # if blessed is not present, no coloring
        try:
            import blessed
        except ImportError:
            blessed = None

        if blessed:
            t = blessed.Terminal()
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
                txt1 = text.split(None, 1)[0]
                # initial whitespaces
                n = text.find(txt1)
                txt0 = text[:n]
                # rest
                txt2 = text[n + len(txt1):]
                txt_kwargs = {
                    'e1': e + (t.bold if e == t.white else ''),
                    'e2': e + (t.bold if e != t.white else ''),
                    't0': txt0, 't1': txt1, 't2': txt2, 'n': t.normal}
                text = '{t0}{e1}{t1}{n}{e2}{t2}{n}'.format(**txt_kwargs)
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
        events (list[(str,time)]): A list of named event time points.
            Each event is a 2-tuple: (label, time).

    Returns:
        None.
    """
    import datetime
    import inspect
    import os

    if name is None:
        outer_frame = inspect.getouterframes(inspect.currentframe())[1]
        filename = __file__
        name = os.path.basename(filename)

    if not time_point:
        time_point = datetime.datetime.now()
    events.append((name, time_point))


# ======================================================================
def print_elapsed(
        events=_EVENTS,
        label='\nElapsed Time(s): ',
        only_last=False):
    """
    Print quick-and-dirty elapsed times between named event points.

    Args:
        events (list[str,time]): A list of named event time points.
            Each event is a 2-tuple: (label, time).
        label (str): heading of the elapsed time table.
        only_last (bool): print only the last event (useful inside a loop).

    Returns:
        None.
    """
    import datetime

    if not only_last:
        print(label, end='\n' if len(events) > 2 else '')
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
            print('{!s:24s} {!s:>24s}, {!s:>24s}'.format(
                name, diff_last, diff_first))
    else:
        _id = -1
        name = events[_id][0]
        curr_elapsed = events[_id][1]
        prev_elapsed = events[_id - 1][1]
        diff_last = datetime.timedelta(0, curr_elapsed - prev_elapsed)
        print('{!s}: {!s:>24s}'.format(name, diff_last))


# ======================================================================
if __name__ == '__main__':
    import doctest

    msg(__doc__.strip())
    doctest.testmod()

else:
    elapsed()
