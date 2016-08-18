#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
pymrt: data analysis for quantitative MRI
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
# :: supported verbosity levels (level 4 skipped on purpose)
VERB_LVL = {'none': 0, 'low': 1, 'medium': 2, 'high': 3, 'debug': 5}
D_VERB_LVL = VERB_LVL['low']

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
        text (str|unicode): Message to display.
        verb_lvl (int): Current level of verbosity.
        verb_threshold (int): Threshold level of verbosity.
        fmt (str|unicode): Format of the message (if `blessings` supported).
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
        >>> msg(s, fmt='{t.green}')  # if in ANSI Terminal, text is green
        Hello World!
        >>> msg(s, fmt='{t.red}{}')  # if in ANSI Terminal, text is red
        Hello World!
    """
    if verb_lvl >= verb_threshold:
        try:
            import blessings
            term = blessings.Terminal()
            if not fmt:
                if verb_threshold == VERB_LVL['medium']:
                    extra = term.cyan
                elif verb_threshold == VERB_LVL['high']:
                    extra = term.yellow
                elif verb_threshold == VERB_LVL['debug']:
                    extra = term.magenta
                else:
                    extra = term.white
                text = '{e}{t.bold}{first}{t.normal}{e}{rest}{t.normal}'.format(
                    t=term, e=extra,
                    first=text[:text.find(' ')],
                    rest=text[text.find(' '):])
            else:
                if '{}' not in fmt:
                    fmt += '{}'
                text = fmt.format(text, t=term) + term.normal
        except ImportError:
            pass
        finally:
            print(text, *args, **kwargs)


# ======================================================================
def dbg(name):
    """
    Print content of a variable for debug purposes.

    Args:
        name: The name to be inspected.

    Returns:
        None.

    Examples:
        >>> my_dict = {'a': 1, 'b': 1}
        >>> dbg(my_dict)
        dbg(my_dict): {
            "a": 1,
            "b": 1
        }
        <BLANKLINE>
        >>> dbg(my_dict['a'])
        dbg(my_dict['a']): 1
        <BLANKLINE>
    """
    import json
    import inspect

    outer_frame = inspect.getouterframes(inspect.currentframe())[1]
    name_str = outer_frame[4][0][:-1]
    print(name_str, end=': ')
    print(json.dumps(name, sort_keys=True, indent=4))
    print()


# ======================================================================
def elapsed(
        name,
        time_point=None,
        events=_EVENTS):
    """
    Append a named event point to the events list.

    Args:
        name (str|unicode): The name of the event point.
        time_point (float): The time in seconds since the epoch.
        events (list[(str|unicode,time)]): A list of named event time points.
            Each event is a 2-tuple: (label, time).

    Returns:
        None.
    """
    import datetime

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
        events (list[str|unicode,time]): A list of named event time points.
            Each event is a 2-tuple: (label, time).
        label (str|unicode): heading of the elapsed time table.
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
elapsed('pymrt')
