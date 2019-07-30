#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PyMRT: parse Bruker raw data.

EXPERIMENTAL!
"""

# ======================================================================
# :: Future Imports
from __future__ import (
    division, absolute_import, print_function, unicode_literals, )

# ======================================================================
# :: Python Standard Library Imports
import os  # Miscellaneous operating system interfaces
import itertools  # Functions creating iterators for efficient looping
import struct  # Interpret strings as packed binary data
import warnings  # Warning control

# :: External Imports
import numpy as np  # NumPy (multidimensional numerical arrays library)
import flyingcircus as fc  # Everything you always wanted to have in Python.*

# :: External Imports Submodules

# :: Local Imports
import pymrt as mrt
import pymrt.util
import pymrt.naming
import pymrt.input_output
# import raster_geometry  # Create/manipulate N-dim raster geometric shapes.
from pymrt.extras import jcampdx
from pymrt.recipes import coils

from pymrt import INFO, PATH
from pymrt import VERB_LVL, D_VERB_LVL, VERB_LVL_NAMES
from pymrt import elapsed, report, run_doctests
from pymrt import msg, dbg, fmt, fmtm


# ======================================================================
def _get_shape(base_shape, *extras):
    extras = tuple(fc.base.auto_repeat(extras, 1))
    return tuple(base_shape) + (extras if np.prod(extras) > 1 else ())


# ======================================================================
def _get_load_bin_info_fid(acqp, method):
    _mode = {'little': '<', 'big': '>'}
    _dtype = {
        'GO_32BIT_SGN_INT': 'int',
        'GO_32_BIT_SGN_INT': 'int',
        'GO_16BIT_SGN_INT': 'short',
        'GO_16_BIT_SGN_INT': 'short',
        'GO_32BIT_FLOAT': 'float',
        'GO_32_BIT_FLOAT': 'float'}

    # for ParaVision 6.0.1, who knows for the rest
    info = dict(
        dtype=_dtype[acqp['GO_raw_data_format']],
        mode=_mode[acqp['BYTORDA']],
        user_filter=acqp['ACQ_user_filter'] == 'Yes',
        cx_interleaved=True,
    )
    return info


# ======================================================================
def _get_load_bin_info_reco(reco, method):
    _dtype = {
        '_8BIT_UNSGN_INT': 'uchar',
        '_16BIT_SGN_INT': 'short',
        '_32BIT_SGN_INT': 'int',
        '_32BIT_FLOAT': 'float'}
    _mode = {'littleEndian': '<', 'bigEndian': '>'}

    # for ParaVision 6.0.1, who knows for the rest
    info = dict(
        dtype=_dtype[reco['RECO_wordtype']],
        mode=_mode[reco['RECO_byte_order']],
        user_filter=False,
        cx_interleaved=False,
    )
    return info


# ======================================================================
def _load_bin(
        filepath,
        dtype='int',
        mode='<',
        user_filter=False,
        cx_interleaved=True,
        dry=False):
    byte_size = struct.calcsize(fc.base.DTYPE_STR[dtype])
    with fc.base.zopen(filepath, 'rb') as file_obj:
        file_size = file_obj.seek(0, 2)
        file_obj.seek(0)
        if not dry:
            if user_filter:
                raise NotImplementedError
            arr = np.array(fc.base.read_stream(
                file_obj, dtype, mode, file_size // byte_size))
            if cx_interleaved:
                arr = arr[0::2] + 1j * arr[1::2]
        else:
            arr = np.zeros(file_size)
    return arr


# ======================================================================
def _to_patterns(name, exts):
    return [fc.base.change_ext('*/' + name, ext) for ext in exts]


# ======================================================================
def _get_single(dirpath, name, exts):
    filepaths = fc.base.flistdir(_to_patterns(name, exts), dirpath)
    if len(filepaths) == 1:
        filepath = filepaths[0]
    elif len(filepaths) > 1:
        raise FileExistsError(
            'Multiple `{name}` files found.'.format(name=name))
    else:
        raise FileNotFoundError(
            'No `{name}` file found.'.format(name=name))
    return filepath


# ======================================================================
def _get_scan_num_sample_id(comments):
    lines = comments.split('$$ ')
    lines = [line for line in lines if line.strip().endswith('/acqp')]
    source_filepath = fc.base.multi_split_path(lines[0])
    scan_num = source_filepath[-3]
    sample_id = source_filepath[-4]
    scan_num = '{s}{num:03d}'.format(
        s=mrt.naming.SERIES_NUM_ID, num=int(scan_num))
    return scan_num, sample_id


# ======================================================================
def _get_reco_num(comments):
    lines = comments.split('$$ ')
    lines = [line for line in lines if line.strip().endswith('/reco')]
    source_filepath = fc.base.multi_split_path(lines[0])
    reco_num = source_filepath[-3]
    reco_num = '{s}{num:02d}'.format(
        s=mrt.naming.NEW_RECO_ID[0], num=int(reco_num))
    return reco_num


# ======================================================================
def _reco_from_fid(
        arr,
        acqp,
        method,
        coil_axis=-1,
        images_axis=-2,
        rep_axis=-3,
        avg_axis=-4,
        verbose=D_VERB_LVL):
    is_cartesian = True
    if is_cartesian:
        load_info = _get_load_bin_info_fid(acqp, method)
        dtype_size = struct.calcsize(
            load_info['mode'] + fc.base.DTYPE_STR[load_info['dtype']])
        block_size = acqp['GO_block_size']
        if block_size == 'continuous':
            block_size = 1
        elif block_size == 'Standard_KBlock_Format':
            block_size = (1024 // dtype_size // 2)
        else:
            block_size = int(block_size)

        # number of images per experiment (e.g. multi-echo)
        num_images = acqp['NI']
        msg('num_images={}'.format(num_images), verbose, VERB_LVL['debug'])

        # inner cycle repetition (before phase-encoding) to be combined
        num_accum = acqp['NAE']
        msg('num_accum={}'.format(num_accum), verbose, VERB_LVL['debug'])

        # outer cycle repetition (after phase-encoding) to be combined
        num_avg = acqp['NA']
        msg('num_avg={}'.format(num_avg), verbose, VERB_LVL['debug'])

        # image repetitions that are NOT to be averaged
        num_rep = acqp['NR']
        msg('num_rep={}'.format(num_rep), verbose, VERB_LVL['debug'])

        # number of dummy scans
        # num_ds = acqp['DS']
        # msg('num_ds={}'.format(num_ds), verbose, VERB_LVL['debug'])

        # phase encoding factor
        pe_factor = acqp['ACQ_phase_factor']
        msg('pe_factor={}'.format(pe_factor), verbose, VERB_LVL['debug'])

        acq_shape = acqp['ACQ_size']
        msg('acq_shape={}'.format(acq_shape), verbose, VERB_LVL['debug'])

        base_shape = method['PVM_Matrix']
        msg('base_shape={}'.format(base_shape), verbose, VERB_LVL['debug'])

        ref_gains = acqp['ACQ_CalibratedRG'].ravel()
        msg('ref_gains={}'.format(ref_gains), verbose, VERB_LVL['debug'])

        # number of coils
        num_coils = len([ref_gain for ref_gain in ref_gains if ref_gain > 0])
        # num_coils = acq_shape[0] / base_shape[0]
        msg('num_coils={}'.format(num_coils), verbose, VERB_LVL['debug'])

        try:
            # fp = '/home/raid1/metere/hd3/sandbox/hmri/_/test_{s}.nii.gz'

            msg('fid_size={}'.format(arr.size), verbose, VERB_LVL['debug'])
            fid_shape = (
                fc.base.align(base_shape[0], block_size // num_coils),
                num_coils,
                num_images,
                fc.base.align(base_shape[1], pe_factor, 'lower'),
                acq_shape[2] if len(acq_shape) == 3 else 1,
                num_avg,
                num_rep,
                -1)
            msg('fid_shape={}'.format(fid_shape), verbose, VERB_LVL['debug'])

            arr = arr.reshape(fid_shape, order='F')
            arr = np.moveaxis(arr, (1, 2), (coil_axis, images_axis))
            # remove singleton dimensions
            arr = np.squeeze(arr)

            # remove additional zeros from redout block alignment
            arr = np.delete(arr, slice(base_shape[0], None), 0)
            # remove additional zeros from over-slices
            if len(acq_shape) == 3:
                arr = np.delete(arr, slice(base_shape[2], None), 2)

            # sort and reshape phase encoding steps
            if pe_factor > 1:
                msg('arr_shape={}'.format(arr.shape), verbose,
                    VERB_LVL['debug'])
                pe_size = arr.shape[1]
                pe_step = pe_size // pe_factor
                i = np.argsort(list(itertools.chain(
                    *[range(j, pe_size, pe_step) for j in range(pe_step)])))
                tmp_arr = arr[:, i, ...]
                arr = np.zeros(
                    tuple(base_shape) + tuple(arr.shape[len(base_shape):]),
                    dtype=complex)
                arr[:, :pe_size, ...] = tmp_arr

            # todo: fix phases?

            msg('arr_shape={}'.format(arr.shape), verbose, VERB_LVL['debug'])

            # perform spatial FFT
            # warning: incorrect fft shifts result in checkerboard artifact
            ft_axes = tuple(range(len(base_shape)))
            arr = np.fft.ifftshift(
                np.fft.ifftn(
                    np.fft.fftshift(arr, axes=ft_axes),
                    axes=ft_axes),
                axes=ft_axes)

            # combine coils
            if num_coils > 1:
                if num_images == 1:
                    coils_combine_kws = (
                        ('method', 'block_adaptive_iter'),
                        ('compression_kws', dict((('k_svd', 'quad_weight'),))),
                        ('split_axis', None),)
                else:
                    coils_combine_kws = (
                        ('method', 'multi_svd'),
                        ('compression', None),
                        ('split_axis', None),)
                    # coils_combine_kws = (
                    #     ('method', 'adaptive_iter'),
                    #     ('method_kws', dict((('block', 8),))),
                    #     ('compression_kws', dict((('k_svd', 'quad_weight'),
                    # ))),
                    #     ('split_axis', images_axis),)
                combined_arr = coils.combine(
                    arr, coil_axis=coil_axis,
                    verbose=verbose, **dict(coils_combine_kws))

                qq_arr = coils.quality(arr, combined_arr)
                # mrt.input_output.save(fp.format(s='Q'), qq_arr)
                arr = combined_arr
            if num_avg > 1:
                arr = np.sum(arr, axis=avg_axis)
            if num_rep > 1:
                arr = np.sum(arr, axis=rep_axis)

                # mrt.input_output.save(fp.format(s='M'), np.abs(arr))
                # print('MAG')
                # mrt.input_output.save(fp.format(s='P'), np.angle(arr))
                # print('PHS')

        # except ValueError:
        except NotImplementedError as e:
            msg('Failed at: {}'.format(e))
            fid_shape = fc.base.factorize_k(arr.size, 3)
            warning = ('Could not determine correct shape for FID. '
                       'Using `{}`'.format(fid_shape))
            warnings.warn(warning)
            arr = arr.reshape(fid_shape)
    else:
        raise NotImplementedError
    return arr


def _reco_from_bin(
        arr,
        reco,
        method,
        verbose=D_VERB_LVL):
    warning = 'This is EXPERIMENTAL!'
    warnings.warn(warning)

    is_complex = reco['RECO_image_type'] == 'COMPLEX_IMAGE'
    if is_complex:
        arr = arr[arr.size // 2:] + 1j * arr[:arr.size // 2]

    shape = reco['RECO_size']
    # reco['RecoObjectsPerRepetition'], reco['RecoNumRepetitions']),
    shape = (-1,) + tuple(np.roll(shape, 1))
    # print(shape)
    arr = np.swapaxes(arr.reshape(shape), 0, -1)
    return arr


# ======================================================================
def batch_extract(
        dirpath,
        out_filename='niz/{scan_num}__{acq_method}_{scan_name}_{reco_flag}',
        out_dirpath=None,
        custom_reco=None,
        custom_reco_kws=None,
        fid_name='fid',
        dseq_name='2dseq',
        acqp_name='acqp',
        method_name='method',
        reco_name='reco',
        allowed_ext=('', 'gz'),
        force=False,
        verbose=D_VERB_LVL):
    """
    Extract images from experiment folder.

    EXPERIMENTAL!

    Args:
        dirpath (str):
        out_filename (str|None):
        out_dirpath (str|None):
        custom_reco (str|None):
            Determines how results will be saved.
            Accepted values are:
             - 'mag_phs': saves magnitude and phase.
             - 're_im': saves real and imaginary parts.
             - 'cx': saves the complex data.
        custom_reco_kws (dict|None):
        fid_name ():
        dseq_name ():
        acqp_name ():
        method_name ():
        reco_name ():
        allowed_ext ():
        force ():
        verbose ():

    Returns:

    """
    text = '\n'.join((
        'EXPERIMENTAL!', 'Use at your own risk!',
        'Known issues:',
        ' - orientation not adjusted to method (i.e. 0->RO, 1->PE, 2->SL)',
        ' - FOV is centered out',
        ' - voxel size is not set',
        ''))
    warnings.warn(text)

    if allowed_ext is None:
        allowed_ext = ''
    elif isinstance(allowed_ext, str):
        allowed_ext = (allowed_ext,)
    fid_filepaths = sorted(
        fc.base.flistdir(_to_patterns(fid_name, allowed_ext), dirpath))

    for fid_filepath in sorted(fid_filepaths):
        msg('FID: {}'.format(fid_filepath),
            verbose, D_VERB_LVL)
        fid_dirpath = os.path.dirname(fid_filepath)
        if out_dirpath is None:
            out_dirpath = dirpath
        out_filepath = os.path.join(out_dirpath, out_filename)

        acqp_filepath = _get_single(
            fid_dirpath, acqp_name, allowed_ext)
        method_filepath = _get_single(
            fid_dirpath, method_name, allowed_ext)

        dseq_filepaths = sorted(
            fc.base.flistdir(
                _to_patterns(dseq_name, allowed_ext), fid_dirpath))
        reco_filepaths = sorted(
            fc.base.flistdir(
                _to_patterns(reco_name, allowed_ext), fid_dirpath))

        acqp_s, acqp, acqp_c = jcampdx.read(acqp_filepath)
        method_s, method, method_c = jcampdx.read(method_filepath)
        scan_num, sample_id = _get_scan_num_sample_id(acqp_c)
        scan_name = fc.base.safe_filename(acqp['ACQ_scan_name'])
        acq_method = fc.base.safe_filename(acqp['ACQ_method'])
        reco_flag = mrt.naming.NEW_RECO_ID

        if custom_reco:
            load_info = _get_load_bin_info_fid(acqp, method)

            if custom_reco == 'cx':
                reco_flag = mrt.naming.ITYPES['cx']
                cx_filepath = fc.base.change_ext(
                    fmtm(out_filepath), mrt.util.EXT['niz'])
                if not os.path.isdir(os.path.dirname(cx_filepath)):
                    os.makedirs(os.path.dirname(cx_filepath))

                if fc.base.check_redo(
                        [fid_filepath, acqp_filepath, method_filepath],
                        [cx_filepath], force):
                    arr = _load_bin(fid_filepath, **load_info)
                    arr = _reco_from_fid(arr, acqp, method, verbose=verbose)
                    mrt.input_output.save(cx_filepath, arr)
                    msg('CX:  {}'.format(os.path.basename(cx_filepath)),
                        verbose, D_VERB_LVL)

            elif custom_reco == 'mag_phs':
                reco_flag = mrt.naming.ITYPES['mag']
                mag_filepath = fc.base.change_ext(
                    fmtm(out_filepath), mrt.util.EXT['niz'])
                if not os.path.isdir(os.path.dirname(mag_filepath)):
                    os.makedirs(os.path.dirname(mag_filepath))

                reco_flag = mrt.naming.ITYPES['phs']
                phs_filepath = fc.base.change_ext(
                    fmtm(out_filepath), mrt.util.EXT['niz'])
                if not os.path.isdir(os.path.dirname(phs_filepath)):
                    os.makedirs(os.path.dirname(phs_filepath))

                if fc.base.check_redo(
                        [fid_filepath, acqp_filepath, method_filepath],
                        [mag_filepath, phs_filepath], force):
                    reco_flag = mrt.naming.ITYPES['mag']

                    arr = _load_bin(fid_filepath, **load_info)
                    arr = _reco_from_fid(arr, acqp, method, verbose=verbose)
                    mrt.input_output.save(mag_filepath, np.abs(arr))
                    msg('MAG: {}'.format(os.path.basename(mag_filepath)),
                        verbose, D_VERB_LVL)
                    mrt.input_output.save(phs_filepath, np.angle(arr))
                    msg('PHS: {}'.format(os.path.basename(phs_filepath)),
                        verbose, D_VERB_LVL)

            elif custom_reco == 're_im':
                reco_flag = mrt.naming.ITYPES['re']
                re_filepath = fc.base.change_ext(
                    fmtm(out_filepath), mrt.util.EXT['niz'])
                if not os.path.isdir(os.path.dirname(re_filepath)):
                    os.makedirs(os.path.dirname(re_filepath))

                reco_flag = mrt.naming.ITYPES['im']
                im_filepath = fc.base.change_ext(
                    fmtm(out_filepath), mrt.util.EXT['niz'])
                if not os.path.isdir(os.path.dirname(im_filepath)):
                    os.makedirs(os.path.dirname(im_filepath))

                if fc.base.check_redo(
                        [fid_filepath, acqp_filepath, method_filepath],
                        [re_filepath, im_filepath], force):
                    arr = _load_bin(fid_filepath, **load_info)
                    arr = _reco_from_fid(arr, acqp, method, verbose=verbose)
                    mrt.input_output.save(re_filepath, np.abs(arr))
                    msg('RE: {}'.format(os.path.basename(re_filepath)),
                        verbose, D_VERB_LVL)
                    mrt.input_output.save(im_filepath, np.angle(arr))
                    msg('IM: {}'.format(os.path.basename(im_filepath)),
                        verbose, D_VERB_LVL)

        else:
            text = 'Voxel data and shapes may be incorrect.'
            warnings.warn(text)
            for dseq_filepath, reco_filepath \
                    in zip(dseq_filepaths, reco_filepaths):
                reco_s, reco, reco_c = jcampdx.read(reco_filepath)
                reco_flag = _get_reco_num(reco_c)

                cx_filepath = fc.base.change_ext(
                    fmtm(out_filepath), mrt.util.EXT['niz'])
                if not os.path.isdir(os.path.dirname(cx_filepath)):
                    os.makedirs(os.path.dirname(cx_filepath))

                load_info = _get_load_bin_info_reco(reco, method)

                if fc.base.check_redo(
                        [dseq_filepath, reco_filepath], [cx_filepath], force):
                    arr = _load_bin(dseq_filepath, **load_info)
                    arr = _reco_from_bin(arr, reco, method, verbose=verbose)
                    mrt.input_output.save(cx_filepath, arr)
                    msg('NIZ: {}'.format(os.path.basename(cx_filepath)),
                        verbose, D_VERB_LVL)


# ======================================================================
elapsed(__file__[len(PATH['base']) + 1:])

# ======================================================================
if __name__ == '__main__':
    run_doctests(__doc__)
