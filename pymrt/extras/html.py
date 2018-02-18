#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PyMRT: read/write files with BART's CFL structure.

The module is NumPy-aware.

See: https://github.com/mrirecon/bart
"""

# ======================================================================
# :: Future Imports
from __future__ import (
    division, absolute_import, print_function, unicode_literals, )

# ======================================================================
# :: Python Standard Library Imports
import os  # Miscellaneous operating system interfaces

# :: External Imports
import numpy as np  # NumPy (multidimensional numerical arrays library)
import PIL as pil  # Python Imaging Library

import PIL.Image, PIL.ImageChops

# :: Local Imports
import pymrt as mrt
import pymrt.utils

from pymrt import VERB_LVL, D_VERB_LVL, VERB_LVL_NAMES
from pymrt import elapsed, report
from pymrt import msg, dbg

# ======================================================================
# :: additional globals

templates = {
    'table': """<!DOCTYPE html>
<html>
    <head>
        <style>
        body {{
            font-family: sans-serif;}}
        table, thead, tbody, th, tr, td {{
            padding: 0.4ex 0.8em;
            margin: 0.2em;
            border: 0;
            text-align: center;
            border-collapse: collapse;}}
        table {{
            border-bottom: 2px solid #000;
            border-top: 2px solid #000;}}
        thead tr:last-of-type {{
            border-bottom: 1px solid #000;}}
        tbody tr:nth-child(odd) {{
            background-color: #e0ffe2;}}
        tbody tr:nth-child(even) {{
            background-color: #fafffc;}}
        </style>
    </head>
    <body>

{table}

    </body>
</html>
""", }


# ======================================================================
def _trim(filepath):
    """Trim borders from image contained in filepath."""
    # : equivalent to:
    # os.system(
    #     'mogrify "{filepath}" -trim "{filepath}"'.format_map(locals()))
    im = pil.Image.open(filepath)
    bg = pil.Image.new(im.mode, im.size, im.getpixel((0,0)))
    diff = pil.ImageChops.difference(im, bg)
    diff = pil.ImageChops.add(diff, diff, 2.0, -100)
    bbox = diff.getbbox()
    if bbox:
        im.crop(bbox).save(filepath)


# ======================================================================
def to_image(
        html_code,
        save_filepath,
        dpi=96,
        trim=True,
        force=False,
        verbose=D_VERB_LVL):
    """
    Render a HTML page to an image.

    Args:
        html_code (str): The HTML code to render.
        save_filepath (str):
        trim (bool): Remove borders from the image.
        force (bool): Force calculation of output.
        verbose (int): Set level of verbosity.

    Returns:
        None.
    """
    D_DPI = 96
    D_WIDTH = 1024
    zoom = dpi / D_DPI
    width = int(zoom * D_WIDTH)
    html_filepath = mrt.utils.change_ext(save_filepath, 'html')
    with open(html_filepath, 'wb+') as file_obj:
        file_obj.write(html_code.encode('ascii', 'xmlcharrefreplace'))

    os.system(
        'wkhtmltoimage --encoding UTF-8 --zoom {zoom} --width {width} '
        ' "{html_filepath}" "{save_filepath}"'.format_map(locals()))
    if trim:
        _trim(save_filepath)

