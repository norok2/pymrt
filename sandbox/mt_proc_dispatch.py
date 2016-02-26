#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import math

import numpy as np

import mri_tools.input_output as mrio

# ======================================================================
CMD = {
    'dispatch': 'fsl_sub',
    'script': 'python -u mt_fit_quick.py',
}


# ======================================================================
def mt_proc_dispatch(
        num_chunks=20,
        target_filename='target.nii.gz',
        chunk_filename='chunk_[ID].nii.gz',
        dirpath='/media/rick/Q_METERE_IT/FMRIB/mt_preview/',
        dry=False):
    """

    Args:
        num_chunks:
        target_filename:
        chunk_filename:
        dirpath:

    Returns:

    """
    filenames = {'target': target_filename, 'chunk': chunk_filename}
    filepaths = {}
    for item, filename in filenames.items():
        filepaths[item] = os.path.realpath(filename)
        if not os.path.exists(filepaths[item]):
            filepaths[item] = os.path.join(dirpath, filename)
    if num_chunks <= 1:
        cmd = '{} -q verylong.q {} --target {}'.format(
            CMD['dispatch'], CMD['script'], filepaths['target'])
        subtargets = filepaths['target'],
        if dry:
            print(cmd)
        else:
            os.system(cmd)
    else:
        img, aff, hdr = mrio.load(filepaths['target'], True)
        img = img.astype(int)
        msk = img[...].astype(bool)
        msk_size = np.sum(img[msk])
        idx = np.zeros_like(img)
        idx[msk] = np.arange(msk_size) + 1
        chunk_idxs = np.array_split(idx[msk], num_chunks)


        subtargets = []
        for i, chunk_idx in enumerate(chunk_idxs):
            chunk = np.zeros_like(img)
            chunk_mask = (idx > np.min(chunk_idx)) * (idx < np.max(chunk_idx))
            chunk[chunk_mask] = 1
            chunk_filepath = filepaths['chunk'].replace(
                '[ID]', str(i).zfill(math.ceil(math.log10(num_chunks))))
            mrio.save(chunk_filepath, chunk.astype(int))
            cmd = '{} -q short.q {} --target {}'.format(
                CMD['dispatch'], CMD['script'], chunk_filepath)
            if dry:
                print(cmd)
            else:
                os.system(cmd)
    return subtargets


# ======================================================================
if __name__ == '__main__':
    mt_proc_dispatch()
