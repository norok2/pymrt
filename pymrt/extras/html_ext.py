#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PyMRT: HTML extended support.

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
import shutil  # High-level file operations

# :: External Imports
# import numpy as np  # NumPy (multidimensional numerical arrays library)
import flyingcircus as fc  # Everything you always wanted to have in Python*
import PIL as pil  # Python Imaging Library

import PIL.Image, PIL.ImageChops

# :: Local Imports
# import pymrt as mrt  # Python Magnetic Resonance Tools: the multi-tool of MRI
from pymrt import INFO, PATH
from pymrt import VERB_LVL, D_VERB_LVL, VERB_LVL_NAMES
from pymrt import elapsed, report, run_doctests
from pymrt import msg, dbg, fmt, fmtm

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
    # os.system(fmtm('mogrify "{filepath}" -trim "{filepath}"'))
    im = pil.Image.open(filepath)
    bg = pil.Image.new(im.mode, im.size, im.getpixel((0, 0)))
    diff = pil.ImageChops.difference(im, bg)
    diff = pil.ImageChops.add(diff, diff, 2.0, -100)
    bbox = diff.getbbox()
    if bbox:
        im.crop(bbox).save(filepath)


# ======================================================================
def to_image(
        html_code,
        save_filepath,
        method=None,
        html_filepath=True,
        img_type='png',
        dpi=96,
        trim=True,
        force=False,
        verbose=D_VERB_LVL):
    """
    Render a HTML page to an image.

    Args:
        html_code (str): The HTML code to render.
        save_filepath (str): The image file.
        method (str|None): The HTML-to-image method
        html_filepath (str|bool|None): The intermediate HTML file.
        img_type (str): The image type / extension of the output.
        dpi (int|float): The image density of the output.
        trim (bool): Remove borders from the image.
        force (bool): Force calculation of output.
        verbose (int): Set level of verbosity.

    Returns:
        None.
    """
    available_methods = []
    if shutil.which('wkhtmltoimage'):
        available_methods.append('webkit-png')
    if shutil.which('wkhtmltopdf'):
        available_methods.append('webkit-pdf')
    try:
        import weasyprint
    except ImportError:
        weasyprint = None
    else:
        available_methods.append('weasyprint')

    if not method:
        method = available_methods[0]

    # export intermediate HTML output
    if html_filepath is True or method.startswith('webkit-'):
        html_filepath = fc.base.change_ext(save_filepath, 'htm')
    if html_filepath:
        with open(html_filepath, 'wb+') as file_obj:
            file_obj.write(html_code.encode('ascii', 'xmlcharrefreplace'))

    if method.startswith('webkit-'):
        d_webkit_dpi = 96
        d_webkit_width = 1024
        zoom = dpi / d_webkit_dpi
        width = int(zoom * d_webkit_width)
        if img_type.lower() == 'png':
            webkit_cmd = 'wkhtmltoimage'
        elif img_type.lower() == 'pdf':
            webkit_cmd = 'wkhtmltopdf'
        os.system(fmtm(
            '{webkit_cmd} --encoding UTF-8 --zoom {zoom} --width {width} '
            ' "{html_filepath}" "{save_filepath}"'))
    elif method == 'weasyprint':
        html_obj = weasyprint.HTML(string=html_code)
        if img_type.lower() == 'png':
            html_obj.write_png()
        elif img_type.lower() == 'pdf':
            html_obj.write_pdf()

    if trim:
        _trim(save_filepath)


# ======================================================================
elapsed(__file__[len(PATH['base']) + 1:])

# ======================================================================
if __name__ == '__main__':
    run_doctests(__doc__)
