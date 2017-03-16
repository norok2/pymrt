#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
pymrt.naming: naming for datasets with limited support to metadata information.
"""

# ======================================================================
# :: Future Imports
from __future__ import(
    division, absolute_import, print_function, unicode_literals)

# ======================================================================
# :: Python Standard Library Imports
import os  # Miscellaneous operating system interfaces
import re  # Regular expression operations
import doctest  # Test interactive Python examples

# :: External Imports

# :: External Imports Submodules

# :: Local Imports
import pymrt.utils as pmu

# from dcmpi.lib.common import D_NUM_DIGITS
D_NUM_DIGITS = 3  # synced with: dcmpi.common.D_NUM_DIGITS

# ======================================================================
# :: parsing constants
# old-style separators
D_SEP = '_'
PARAM_BASE_SEP = '_'
PARAM_SEP = ','
PARAM_KEY_VAL_SEP = '='
INFO_SEP = '__'
TOKEN_SEP = '_'
KEY_VAL_SEP = '='

# new-style separators
SEP = {
    'token': '__',
    'info': '_',
    'key_val': '=',
    'name_units': '!',
}

# suffix of new reconstructed image from Siemens
NEW_RECO_ID = 'rr'
SERIES_NUM_ID = 's'


# ======================================================================
def str_to_key_val(
        text,
        kv_sep=KEY_VAL_SEP,
        case_sensitive=False):
    """
    Extract numerical value from string information.
    This expects a string containing a single parameter in the form `key=val`.

    Parameters:
        text (str): The input string.
        kv_sep (str): The string separating the key and the value.
        case_sensitive : bool (optional)
            Parsing of the string is case-sensitive.

    Returns:
        param_val (int|float|str|None): The value of the parameter.

    Examples:
        >>> str_to_key_val('key=1000')
        ('key', 1000)
        >>> str_to_key_val('key1000', '')
        ('key', 1000)

    See Also:
        set_param_val, parse_series_name
    """
    if text:
        if not case_sensitive:
            text = text.lower()
        if kv_sep and kv_sep in text:
            key, val = text.split(kv_sep)
        elif kv_sep == '':
            key = re.findall(r'^[a-zA-Z\-]*', text)[0]
            val = text[len(key):]
        else:
            key = text
            val = None
    else:
        key, val = None, None
    if val:
        val = pmu.auto_convert(val)
    return key, val


# ======================================================================
def key_val_to_str(
        key,
        val=None,
        kv_sep=KEY_VAL_SEP,
        case='lower'):
    """
    Extract numerical value from string information.
    This expects an appropriate string, as retrieved by parse_filename().

    Args:
        val (int|float|None): The value of the parameter.
        key (str): The string containing the label of the parameter.
        kv_sep (str): String separating key from value in parameters.
        case (str): Set the case of the parameter label.
            - 'lower': force the key to be in lower case
            - 'upper': force the key to be in upper case
            - any other value has no effect and it is silently ignored

    Returns:
        text (str): The string containing the information.

    Examples:
        >>> key_val_to_str('key', 1000)
        'key=1000'
        >>> key_val_to_str('key', 1000, '')
        'key1000'

    See Also:
        str_to_key_val, to_series_name
    """
    if case == 'lower':
        key = key.lower()
    elif case == 'upper':
        key = key.upper()
    if val is not None:
        text = kv_sep.join((key, str(val)))
    else:
        text = key
    return text


# ======================================================================
def str_to_info(
        text,
        sep=TOKEN_SEP,
        kv_sep=KEY_VAL_SEP):
    """
    Extract information from Siemens DICOM names.
    Expected format is: [s<###>_]<series_name>[_<#>][_<type>]

    Args:
        text (str):
        sep (str):
        kv_sep (str):

    Returns:
        info (dict):

    Examples:
        >>> d = str_to_info('S001_TEST_NAME_TE=10.6_2_T1')
        >>> for k in sorted(d.keys()): print(k, ':', d[k])  # display dict
        _# : 1
        2 : None
        name : None
        t1 : None
        te : 10.6
        test : None
    """
    tokens = text.split(sep)
    info = {
        '#': pmu.auto_convert(tokens[0][len(SERIES_NUM_ID):])
        if SERIES_NUM_ID.lower() in tokens[0].lower() else None}
    for token in tokens[1:]:
        key, val = str_to_key_val(token, kv_sep)
        info[key] = val
    return info


# ======================================================================
def info_to_str(
        info,
        sep=TOKEN_SEP,
        kv_sep=KEY_VAL_SEP):
    """

    Args:
        info (dict):
        sep (str):
        kv_sep (str):

    Returns:
        text (str):

    Examples:
        >>> info = {'_#': 10, '_id': 'me-mp2rage', 't1': 2000}
    """
    tokens = []
    for key, val in info.items():
        tokens.append(key_val_to_str(key, val, kv_sep))
    return sep.join(tokens)


# ======================================================================
def filepath_to_info(
        filepath,
        file_ext=pmu.EXT['niz'],
        sep=TOKEN_SEP,
        kv_sep=KEY_VAL_SEP):
    """

    Args:
        filepath ():
        file_ext ():
        sep ():
        kv_sep ():

    Returns:
        info (dict): Information extracted from the
    """
    filename = pmu.change_ext(os.path.basename(filepath), '', file_ext)
    return str_to_info(filename)


# ======================================================================
def info_to_filepath(
        info,
        dirpath='.',
        file_ext=pmu.EXT['niz'],
        sep=TOKEN_SEP,
        kv_sep=TOKEN_SEP):
    filename = pmu.change_ext(info_to_str(info, sep, kv_sep), file_ext, '')
    return os.path.join(
        dirpath, + pmu.add_extsep(file_ext))


# ======================================================================
def filepath_set_info(
        in_filepath,
        key,
        val=None,
        force=True):
    """

    Args:
        in_filepath (str):
        key (str):
        val (int|float|str|None):
        force (bool): Force overwrite of previously existing key:val pair.

    Returns:
        out_filepath(str):
    """
    info = filepath_to_info(in_filepath)
    if key not in info or force:
        info[key] = val
    return info_to_filepath(info)


# ======================================================================
def get_param_val(
        param_str,
        param_key='',
        case_sensitive=False):
    """
    Extract numerical value from string information.
    This expects a string containing a single parameter.

    Parameters
    ==========
    name : str
        The string containing the information.
    param_key : str (optional)
        The string containing the label of the parameter.
    case_sensitive : bool (optional)
        Parsing of the string is case-sensitive.

    Returns
    =======
    param_val : int or float
        The value of the parameter.

    See Also
    ========
    set_param_val, parse_series_name

    """
    if param_str:
        if not case_sensitive:
            param_str = param_str.lower()
            param_key = param_key.lower()
        if param_str.startswith(param_key):
            param_val = param_str[len(param_key):]
        elif param_str.endswith(param_key):
            param_val = param_str[:-len(param_key)]
        else:
            param_val = None
    else:
        param_val = None
    return param_val


# ======================================================================
def set_param_val(
        param_val,
        param_key,
        kv_sep=PARAM_KEY_VAL_SEP,
        case='lower'):
    """
    Extract numerical value from string information.
    This expects an appropriate string, as retrieved by parse_filename().

    Args:
        param_val (int|float|None): The value of the parameter.
        param_key (str): The string containing the label of the parameter.
        kv_sep (str): String separating key from value in parameters.
        case ('lower'|'upper'|None): Set the case of the parameter label.

    Returns:
        str: The string containing the information.

    .. _refs:
        get_param_val, to_series_name
    """
    if case == 'lower':
        param_key = param_key.lower()
    elif case == 'upper':
        param_key = param_key.upper()
    if param_val is not None:
        param_str = kv_sep.join((param_key, str(param_val)))
    else:
        param_str = param_key
    return param_str


# ======================================================================
def parse_filename(
        filepath,
        i_sep=INFO_SEP,
        p_sep=PARAM_SEP,
        kv_sep=PARAM_KEY_VAL_SEP,
        b_sep=PARAM_BASE_SEP):
    """
    Extract specific information from SIEMENS data file name/path.
    Expected format is: [s<###>__]<series_name>[__<#>][__<type>].nii.gz

    Parameters
    ==========
    filepath : str
        Full path of the image filename.

    Returns
    =======
    info : dictionary
        Dictionary containing:
            | 'num' : int : identification number of the series.
            | 'name' : str : series name.
            | 'seq' : int or None : sequential number of the series.
            | 'type' : str : image type

    See Also
    ========
    to_filename

    """
    root = pmu.basename(filepath)
    if i_sep != p_sep and i_sep != kv_sep and i_sep != b_sep:
        tokens = root.split(i_sep)
        info = {}
        # initialize end of name indexes
        idx_begin_name = 0
        idx_end_name = len(tokens)
        # check if contains scan ID
        info['num'] = pmu.auto_convert(get_param_val(tokens[0], SERIES_NUM_ID))
        idx_begin_name += (1 if info['num'] is not None else 0)
        # check if contains Sequential Number
        info['seq'] = None
        if len(tokens) > 1:
            for token in tokens[-1:-3:-1]:
                if pmu.is_number(token):
                    info['seq'] = pmu.auto_convert(token)
                    break
        idx_end_name -= (1 if info['seq'] is not None else 0)
        # check if contains Image type
        info['type'] = tokens[-1] if idx_end_name - idx_begin_name > 1 else None
        idx_end_name -= (1 if info['type'] is not None else 0)
        # determine series name
        info['name'] = i_sep.join(tokens[idx_begin_name:idx_end_name])
    else:
        raise TypeError('Cannot parse this file name.')
    return info


# ======================================================================
def to_filename(
        info,
        dirpath=None,
        ext=pmu.EXT['niz']):
    """
    Reconstruct file name/path with SIEMENS-like structure.
    Produced format is: [s<num>__]<series_name>[__<seq>][__<type>].nii.gz

    Parameters
    ==========
    info : dictionary
        Dictionary containing:
            | 'num' : int or None: Identification number of the scan.
            | 'name' : str : Series name.
            | 'seq' : int or None : Sequential number of the volume.
            | 'type' : str or None: Image type
    dirpath : str (optional)
        The base directory path for the filename.
    ext : str (optional)
        Extension to append to the newly generated filename or filepath.

    Returns
    =======
    filepath : str
        Full path of the image filename.

    See Also
    ========
    parse_filename

    """
    tokens = []
    if 'num' in info and info['num'] is not None:
        tokens.append('{}{:0{size}d}'.format(
            SERIES_NUM_ID, info['num'], size=D_NUM_DIGITS))
    if 'name' in info:
        tokens.append(info['name'])
    if 'seq' in info and info['seq'] is not None:
        tokens.append('{:d}'.format(info['seq']))
    if 'type' in info and info['type'] is not None:
        tokens.append(info['type'])
    filepath = INFO_SEP.join(tokens)
    filepath += (pmu.add_extsep(ext) if ext else '')
    filepath = os.path.join(dirpath, filepath) if dirpath else filepath
    return filepath


# ======================================================================
def parse_series_name(
        name,
        p_sep=PARAM_SEP,
        kv_sep=PARAM_KEY_VAL_SEP,
        b_sep=PARAM_BASE_SEP):
    """
    Extract specific information from series name.

    Parameters
    ==========
    name : str
        Full name of the image series.
    p_sep : str (optional)
        String separating parameters.
    kv_sep : str (optional)
        String separating key from value in parameters.
    b_sep : str (optional)
        String separating the parameters from the base name.

    Returns
    =======
    base : str
        Base name of the series, i.e. without parsed parameters.
    params : (string, float or int) dictionary
        List of parameters in the (label, value) format.

    See Also
    ========
    to_series_name

    """
    if p_sep != b_sep and b_sep in name:
        base, tokens = name.split(b_sep)
        tokens = tokens.split(p_sep)
    elif p_sep in name:
        tmp = name.split(p_sep)
        base = tmp[0]
        tokens = tmp[1:]
    else:
        base = name
        tokens = ()
    params = {}
    for token in tokens:
        if kv_sep and kv_sep in token:
            param_id, param_val = token.split(kv_sep)
        else:
            param_id = re.findall('^[a-zA-Z\-]*', token)[0]
            param_val = get_param_val(token, param_id)
        params[param_id] = pmu.auto_convert(param_val) if param_val else None
    return base, params


# ======================================================================
def to_series_name(
        base,
        params,
        p_sep=PARAM_SEP,
        kv_sep=PARAM_KEY_VAL_SEP,
        b_sep=PARAM_BASE_SEP,
        value_case='lower',
        tag_case='lower'):
    """
    Reconstruct series name from specific information.

    Parameters
    ==========
    base : str
        Base name of the series, i.e. without parsed parameters.
    params : (string, float or int) dictionary
        List of parameters in the (label, value) format.
    p_sep : str (optional)
        String separating parameters.
    kv_sep : str (optional)
        String separating key from value in parameters.
    b_sep : str (optional)
        String separating the parameters from the base name.
    value_case : 'lower', 'upper' or None (optional)
        TODO
    tag_case : 'lower', 'upper' or None (optional)
        TODO

    Returns
    =======
    name : str
        Full name of the image series.

    See Also
    ========
    parse_series_name

    """
    values = []
    tags = []
    for key, val in params.items():
        if val is not None:
            values.append(set_param_val(val, key, kv_sep, value_case))
        else:
            tags.append(set_param_val(val, key, kv_sep, tag_case))
    params = p_sep.join(sorted(values) + sorted(tags))
    name = b_sep.join((base, params))
    return name


# ======================================================================
def change_img_type(
        filepath,
        img_type):
    """
    Change the image type of an image filename in a filepath.

    Parameters
    ==========
    filepath : str
        The filepath of the base image.
    img_type : str
        The new image type identifier.

    Returns
    =======
    filepath : str
        The filepath of the image with the new type.

    """
    dirpath = os.path.dirname(filepath)
    info = parse_filename(os.path.basename(filepath))
    info['type'] = img_type
    filepath = to_filename(info, dirpath)
    return filepath


# ======================================================================
def change_param_val(
        filepath,
        param_key,
        param_val):
    """
    Change the parameter value of an image filename in a filepath.

    Parameters
    ==========
    filepath : str
        The image filepath.
    param_key : str
        The identifier of the parameter to change.
    new_param_val : str
        The new value of the parameter to change.

    Returns
    =======
    new_name : str
        The filepath of the image with new type.

    """
    dirpath = os.path.dirname(filepath)
    info = parse_filename(filepath)
    base, params = parse_series_name(info['name'])
    params[param_key] = param_val
    info['name'] = to_series_name(base, params)
    filepath = to_filename(info, dirpath)
    return filepath


# ======================================================================
def extract_param_val(
        filepath,
        param_key):
    """
    Extract the parameter value from an image file name/path.

    Args:
        filepath (str): The image filepath.
        param_key (str): The identifier of the parameter to extract.

    Returns:
        The value of the extracted parameter.

    """
    # todo: add support for lists
    info = parse_filename(filepath)
    base, params = parse_series_name(info['name'])
    param_val = params[param_key] if param_key in params else False
    return param_val


# ======================================================================
def combine_filename(
        prefix,
        filenames):
    """
    Create a new filename, based on a combination of filenames.

    Args:
        prefix:
        filenames:

    Returns:

    """
    # todo: fix doc
    filename = prefix
    for name in filenames:
        filename += 2 * INFO_SEP + \
                    pmu.change_ext(os.path.basename(name), '', pmu.EXT['niz'])
    return filename


# ======================================================================
def filename2label(
        filepath,
        ext=None,
        excludes=None,

        max_length=None):
    """
    Create a sensible but shorter label from filename.

    Parameters
    ==========
    filepath : str
        Path fo the file from which a label is to be extracted.
    excludes : list of string (optional)
        List of string to exclude from filepath.
    max_length : int (optional)
        Maximum length of the label.

    Returns
    =======
    label : str
        The extracted label.

    """
    filename = pmu.change_ext(os.path.basename(filepath), '', ext)
    tokens = filename.split(INFO_SEP)
    # remove unwanted information
    if excludes is None:
        excludes = []
    tokens = [token for token in tokens if token not in excludes]
    label = INFO_SEP.join(tokens)
    if max_length:
        label = label[:max_length]
    return label


# ======================================================================
if __name__ == '__main__':
    msg(__doc__.strip())
    doctest.testmod()
