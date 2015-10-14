# ======================================================================
# :: Future Imports
from __future__ import division
from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals


import inspect
import json

# ======================================================================
def dbg(name):
    """Print content of a variable for debug purposes."""
    outer_frame = inspect.getouterframes(inspect.currentframe())[1]
    name_str = outer_frame[4][0][:-1]
    try:
        print(name_str, end=': ')
        print(json.dumps(name, sort_keys=True, indent=4))
        print()
    except NameError:
        print('`{}` undefined'.format(name))
