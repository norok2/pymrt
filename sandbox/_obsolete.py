#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
PyMRT: code that is now deprecated but can still be useful for legacy scripts.
"""

# ======================================================================
# :: Future Imports
from __future__ import(
    division, absolute_import, print_function, unicode_literals)

# ======================================================================
# :: Python Standard Library Imports
# import os  # Miscellaneous operating system interfaces
import sys  # System-specific parameters and functions
import doctest  # Test interactive Python examples

# :: External Imports
import numpy as np  # NumPy (multidimensional numerical arrays library)
import scipy as sp  # SciPy (signal and image processing library)

# :: External Imports Submodules
import scipy.optimize  # SciPy: Optimization Algorithms
import scipy.signal  # SciPy: Signal Processing

import pymrt.base as pmb


# ======================================================================
def tty_colorify(
        text,
        color=None):
    """
    Add color TTY-compatible color code to a string, for pretty-printing.

    Args:
        text (str): The text to color.
        color (str|int|None): Identifier for the color coding.
            Lowercase letters modify the forground color.
            Uppercase letters modify the background color.
            Available colors:
             - r/R: red
             - g/G: green
             - b/B: blue
             - c/C: cyan
             - m/M: magenta
             - y/Y: yellow (brown)
             - k/K: black (gray)
             - w/W: white (gray)

    Returns:
        text (str): The colored text.

    See also:
        tty_colors
    """
    tty_colors = {
        'r': 31, 'g': 32, 'b': 34, 'c': 36, 'm': 35, 'y': 33, 'w': 37, 'k': 30,
        'R': 41, 'G': 42, 'B': 44, 'C': 46, 'M': 45, 'Y': 43, 'W': 47, 'K': 40,
    }

    if color in tty_colors:
        tty_color = tty_colors[color]
    elif color in tty_colors.values():
        tty_color = color
    else:
        tty_color = None
    if tty_color and sys.stdout.isatty():
        return '\x1b[1;{color}m{}\x1b[1;m'.format(text, color=tty_color)
    else:
        return text


# ======================================================================
def interval_size(interval):
    """
    Calculate the (signed) size of an interval given as a 2-tuple (A,B)

    Deprecated: use numpy.ptp instead

    Args:
        interval (float,float): Interval for computation

    Returns:
        val (float): The converted value

    Examples:
        >>> interval_size((0, 1))
        1
    """
    return interval[1] - interval[0]


# :: ndstack obsoleted by: numpy.stack
# ======================================================================
def ndstack(arrays, axis=-1):
    """
    Stack a list of arrays of the same size along a specific axis

    Args:
        arrays (list[ndarray]): A list of (N-1)-dim arrays of the same size
        axis (int): Direction for the concatenation of the arrays

    Returns:
        array (ndarray): The concatenated N-dim array
    """
    array = arrays[0]
    n_dim = array.ndim + 1
    if axis < 0:
        axis += n_dim
    if axis < 0:
        axis = 0
    if axis > n_dim:
        axis = n_dim
    # calculate new shape
    shape = array.shape[:axis] + tuple([len(arrays)]) + array.shape[axis:]
    # stack arrays together
    array = np.zeros(shape, dtype=array.dtype)
    for i, src in enumerate(arrays):
        index = [slice(None)] * n_dim
        index[axis] = i
        array[tuple(index)] = src
    return array


# :: ndsplit obsoleted by: numpy.split
# ======================================================================
def ndsplit(array, axis=-1):
    """
    Split an array along a specific axis into a list of arrays

    Args:
        array (ndarray): The N-dim array to split
        axis (int): Direction for the splitting of the array

    Returns:
        arrays (list[ndarray]): A list of (N-1)-dim arrays of the same size
    """
    # split array apart
    arrays = []
    for i in range(array.shape[axis]):
        # determine index for slicing
        index = [slice(None)] * array.ndim
        index[axis] = i
        arrays.append(array[index])
    return arrays


# ======================================================================
def slice_array(
        arr,
        axis=0,
        index=None):
    """
    Slice a (N-1)-dim sub-array from an N-dim array

    Args:
        arr (np.ndarray): The input N-dim array
        axis (int): The slicing axis
        index (int): The slicing index.
            If None, mid-value is taken.

    Returns:
        sliced (np.ndarray): The sliced (N-1)-dim sub-array

    Raises:
        ValueError: if index is out of bounds

    Examples:
        >>> arr = np.arange(2 * 3 * 4).reshape((2, 3, 4))
        >>> slice_array(arr, 2, 1)
        array([[ 1,  5,  9],
               [13, 17, 21]])
        >>> slice_array(arr, 1, 2)
        array([[ 8,  9, 10, 11],
               [20, 21, 22, 23]])
        >>> slice_array(arr, 0, 0)
        array([[ 0,  1,  2,  3],
               [ 4,  5,  6,  7],
               [ 8,  9, 10, 11]])
        >>> slice_array(arr, 0, 1)
        array([[12, 13, 14, 15],
               [16, 17, 18, 19],
               [20, 21, 22, 23]])
    """
    # initialize slice index
    slab = [slice(None)] * arr.ndim
    # ensure index is meaningful
    if index is None:
        index = np.int(arr.shape[axis] / 2.0)
    # check index
    if (index >= arr.shape[axis]) or (index < 0):
        raise ValueError('Invalid array index in the specified direction')
    # determine slice index
    slab[axis] = index
    # slice the array
    return arr[slab]


# ======================================================================
def sequence(
        start,
        stop,
        step=None,
        precision=None):
    """
    Generate a sequence that steps linearly from start to stop.

    Args:
        start (int|float): The starting value.
        stop (int|float): The final value.
            This value is present in the resulting sequence only if the step is
            a multiple of the interval size.
        step (int|float): The step value.
            If None, it is automatically set to unity (with appropriate sign).
        precision (int): The number of decimal places to use for rounding.
            If None, this is estimated from the `step` paramenter.

    Yields:
        item (int|float): the next element of the sequence.

    Example:
        >>> list(sequence(0, 1, 0.1))
        [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        >>> list(sequence(0, 1, 0.3))
        [0.0, 0.3, 0.6, 0.9]
        >>> list(sequence(0, 10, 1))
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        >>> list(sequence(0.4, 4.6, 0.72))
        [0.4, 1.12, 1.84, 2.56, 3.28, 4.0]
        >>> list(sequence(0.4, 4.72, 0.72, 2))
        [0.4, 1.12, 1.84, 2.56, 3.28, 4.0, 4.72]
        >>> list(sequence(0.4, 4.72, 0.72, 4))
        [0.4, 1.12, 1.84, 2.56, 3.28, 4.0, 4.72]
        >>> list(sequence(0.4, 4.72, 0.72, 1))
        [0.4, 1.1, 1.8, 2.6, 3.3, 4.0, 4.7]
        >>> list(sequence(0.73, 5.29))
        [0.73, 1.73, 2.73, 3.73, 4.73]
        >>> list(sequence(-3.5, 3.5))
        [-3.5, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5]
        >>> list(sequence(3.5, -3.5))
        [3.5, 2.5, 1.5, 0.5, -0.5, -1.5, -2.5, -3.5]
        >>> list(sequence(10, 1, -1))
        [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]
        >>> list(sequence(10, 1, 1))
        []
        >>> list(sequence(10, 20, 10))
        [10, 20]
        >>> list(sequence(10, 20, 15))
        [10]
    """
    if step is None:
        step = 1 if stop > start else -1
    if precision is None:
        precision = guess_decimals(step)
    for i in range(int(round(stop - start, precision + 1) / step) + 1):
        item = start + i * step
        if precision:
            item = round(item, precision)
        yield item


# ======================================================================
def ssim(
        arr1,
        arr2,
        arr_interval=None,
        aa=(1, 1, 1),
        kk=(0.010, 0.030, 0.015)):
    """
    Calculate the structure similarity index, SSIM.

    This is defined as: SSIM = (lum ** alpha) * (con ** beta) * (sti ** gamma)
     - lum is a measure of the luminosity, with exp. weight alpha
     - con is a measure of the contrast, with exp. weight beta
     - sti is a measure of the structural information, with exp. weight gamma

    Args:
        arr1 (np.ndarray): The first input array.
        arr2 (np.ndarray): The second input array.
        arr_interval (tuple[float]): Minimum and maximum allowed values.
            The values of both arr1 and arr2 should be within this interval.
        aa (tuple[float]): The exponentiation weight factors. Must be 3.
            Modulate the relative weight of the three SSIM components
            (luminosity, contrast and structural information).
            If they are all equal to 1, the computation can be simplified.
        kk (tuple[float]): The ratio regularization constant factors. Must be 3.
            Determine the regularization constants as a factors of the total
            interval size (squared) for the three SSIM components
            (luminosity, contrast and structural information).
            Must be numbers much smaller than 1.

    Returns:
        ssim (float): The structure similarity index.

    See Also:
        Wang, Zhou, A. C. Bovik, H. R. Sheikh, and E. P. Simoncelli. “Image
        Quality Assessment: From Error Visibility to Structural Similarity.”
        IEEE Transactions on Image Processing 13, no. 4 (April 2004):
        600–612. doi:10.1109/TIP.2003.819861.
    """
    assert (arr1.shape == arr2.shape)
    if arr_interval is None:
        arr_interval = (
            min(np.min(arr1), np.min(arr2)), max(np.max(arr1), np.max(arr2)))
    interval_size = np.ptp(arr_interval)
    cc = [(k * interval_size) ** 2 for k in kk]
    mu1 = np.mean(arr1)
    mu2 = np.mean(arr2)
    sigma1 = np.std(arr1)
    sigma2 = np.std(arr2)
    sigma12 = np.sum((arr1 - mu1) * (arr2 - mu2)) / (arr1.size - 1)
    ff = [
        (2 * mu1 * mu2 + cc[0]) / (mu1 ** 2 + mu2 ** 2 + cc[0]),
        (2 * sigma1 * sigma2 + cc[1]) / (sigma1 ** 2 + sigma2 ** 2 + cc[1]),
        (sigma12 + cc[2]) / (sigma1 * sigma2 + cc[2])
    ]
    ssim = np.prod(np.array([f ** a for (f, a) in zip(ff, aa)]), 0)
    return ssim


# ======================================================================
def ssim_map(
        arr1,
        arr2,
        filter_sizes=5,
        sigmas=1.5,
        arr_interval=None,
        aa=(1, 1, 1),
        kk=(0.010, 0.030, 0.015)):
    """
    Calculate the local structure similarity index map.

    Args:
        arr1 (np.ndarray): The first input array.
        arr2 (np.ndarray): The second input array.
        filter_sizes (tuple[int]|int): The size of the filter in px.
            If a single value is given, is is assumed to be equal in all dims.
        sigmas (tuple[float]|float): The sigma of the gaussian kernel in px.
            If a single value is given, it is assumed to be equal in all dims.
        arr_interval (tuple[float]): Minimum and maximum allowed values.
            The values of both arr1 and arr2 should be within this interval.
        aa (tuple[float]): The exponentiation weight factors. Must be 3.
            Modulate the relative weight of the three SSIM components
            (luminosity, contrast and structural information).
            If they are all equal to 1, the computation can be simplified.
        kk (tuple[float]): The ratio regularization constant factors. Must be 3.
            Determine the regularization constants as a factors of the total
            interval size (squared) for the three SSIM components
            (luminosity, contrast and structural information).
            Must be numbers much smaller than 1.

    Returns:
        ssim_arr (np.ndarray): The local structure similarity index map
        ssim (float): The global (mean) structure similarity index.

    See Also:
        Wang, Zhou, A. C. Bovik, H. R. Sheikh, and E. P. Simoncelli. “Image
        Quality Assessment: From Error Visibility to Structural Similarity.”
        IEEE Transactions on Image Processing 13, no. 4 (April 2004):
        600–612. doi:10.1109/TIP.2003.819861.
    """
    assert (arr1.shape == arr2.shape)
    if arr_interval is None:
        arr_interval = (
            min(np.min(arr1), np.min(arr2)), max(np.max(arr1), np.max(arr2)))
    interval_size = np.ptp(arr_interval)
    ndim = arr1.ndim
    arr_filter = pmb.gaussian_nd(filter_sizes, sigmas, 0.5, ndim, True)
    convolve = scipy.signal.fftconvolve
    mu1 = convolve(arr1, arr_filter, 'same')
    mu2 = convolve(arr2, arr_filter, 'same')
    mu1_mu1 = mu1 ** 2
    mu2_mu2 = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    sg1_sg1 = convolve(arr1 ** 2, arr_filter, 'same') - mu1_mu1
    sg2_sg2 = convolve(arr2 ** 2, arr_filter, 'same') - mu2_mu2
    sg12 = convolve(arr1 * arr2, arr_filter, 'same') - mu1_mu2
    cc = [(k * interval_size) ** 2 for k in kk]
    # determine whether to use the simplified expression
    if all(aa) == 1 and 2 * cc[2] == cc[1]:
        ssim_arr = ((2 * mu1_mu2 + cc[0]) * (2 * sg12 + cc[1])) / (
            (mu1_mu1 + mu2_mu2 + cc[0]) * (sg1_sg1 + sg2_sg2 + cc[1]))
    else:
        sg1 = np.sqrt(np.abs(sg1_sg1))
        sg2 = np.sqrt(np.abs(sg2_sg2))
        ff = [
            (2 * mu1_mu2 + cc[0]) / (mu1_mu1 + mu2_mu2 + cc[0]),
            (2 * sg1 * sg2 + cc[1]) / (sg1_sg1 + sg2_sg2 + cc[1]),
            (sg12 + cc[2]) / (sg1 * sg2 + cc[2])
        ]
        ssim_arr = np.prod(np.array([f ** a for (f, a) in zip(ff, aa)]), 0)
    ssim = np.mean(ssim_arr)
    return ssim_arr, ssim


'''
# ======================================================================
def calc_averages(
       filepath_list,
       out_dirpath,
       threshold=0.05,
       rephasing=True,
       registration=False,
       limit_num=None,
       force=False,
       verbose=D_VERB_LVL):
   """
   Calculate the average of MR complex images.

   TODO: clean up code / fix documentation

   Parameters
   ==========
   """
   def _compute_regmat(par):
       """Multiprocessing-friendly version of 'compute_affine_fsl()'."""
       return compute_affine_fsl(*par)

   tmp_dirpath = os.path.join(out_dirpath, 'tmp')
   if not os.path.exists(tmp_dirpath):
       os.makedirs(tmp_dirpath)
   # sort by scan number
   get_num = lambda filepath: parse_filename(filepath)['num']
   filepath_list.sort(key=get_num)
   # generate output name
   sum_num, sum_avg = 0, 0
   for filepath in filepath_list:
       info = parse_filename(filepath)
       base, params = parse_series_name(info['name'])
       sum_num += info['num']
       if PARAM_ID['avg'] in params:
           sum_avg += params[PARAM_ID['avg']]
       else:
           sum_avg += 1
   params[PARAM_ID['avg']] = sum_avg // 2
   name = to_series_name(base, params)
   new_info = {
       'num': sum_num,
       'name': name,
       'img_type': TYPE_ID['temp'],
       'te_val': info['te_val']}
   out_filename = to_filename(new_info)
   out_tmp_filepath = os.path.join(out_dirpath, out_filename)
   out_mag_filepath = change_img_type(out_tmp_filepath, TYPE_ID['mag'])
   out_phs_filepath = change_img_type(out_tmp_filepath, TYPE_ID['phs'])
   out_filepath_list = [out_tmp_filepath, out_mag_filepath, out_phs_filepath]
   # perform calculation
  if pmb.check_redo(filepath_list, out_filepath_list, force) and sum_avg > 1:
       # stack multiple images together
       # assume every other file is a phase image, starting with magnitude
       img_tuple_list = []
       mag_filepath = phs_filepath = None
       for filepath in filepath_list:
           if verbose > VERB_LVL['none']:
               print('Source:\t{}'.format(os.path.basename(filepath)))
           img_type = parse_filename(filepath)['img_type']
           if img_type == TYPE_ID['mag'] or not mag_filepath:
               mag_filepath = filepath
           elif img_type == TYPE_ID['phs'] or not phs_filepath:
               phs_filepath = filepath
           else:
               raise RuntimeWarning('Filepath list not valid for averaging.')
           if mag_filepath and phs_filepath:
               img_tuple_list.append([mag_filepath, phs_filepath])
               mag_filepath = phs_filepath = None

#        # register images
#        regmat_filepath_list = [
#            os.path.join(
#            tmp_dirpath,
#            pmio.del_ext(os.path.basename(img_tuple[0])) +
#            pmb.add_extsep(pmb.EXT['txt']))
#            for img_tuple in img_tuple_list]
#        iter_param_list = [
#            (img_tuple[0], img_tuple_list[0][0], regmat)
#            for img_tuple, regmat in
#            zip(img_tuple_list, regmat_filepath_list)]
#        pool = multiprocessing.Pool(multiprocessing.cpu_count())
#        pool.map(_compute_regmat, iter_param_list)
#        reg_filepath_list = []
#        for idx, img_tuple in enumerate(img_tuple_list):
#            regmat = regmat_filepath_list[idx]
#            for filepath in img_tuple:
#                out_filepath = os.path.join(
#                    tmp_dirpath, os.path.basename(filepath))
#                apply_affine_fsl(
#                    filepath, img_tuple_list[0][0], out_filepath, regmat)
#                reg_filepath_list.append(out_filepath)
#        # combine all registered images together
#        img_tuple_list = []
#        for filepath in reg_filepath_list:
#            if img_type == TYPE_ID['mag'] or not mag_filepath:
#                mag_filepath = filepath
#            elif img_type == TYPE_ID['phs'] or not phs_filepath:
#                phs_filepath = filepath
#            else:
#               raise RuntimeWarning('Filepath list not valid for averaging.')
#            if mag_filepath and phs_filepath:
#                img_tuple_list.append([mag_filepath, phs_filepath])
#                mag_filepath = phs_filepath = None

       # create complex images and disregard inappropriate
       img_list = []
       avg_power = 0.0
       num = 0
       shape = None
       for img_tuple in img_tuple_list:
           mag_filepath, phs_filepath = img_tuple
           img_mag_nii = nib.load(mag_filepath)
           img_mag = img_mag_nii.get_data()
           img_phs_nii = nib.load(mag_filepath)
           img_phs = img_phs_nii.get_data()
           affine_nii = img_mag_nii.get_affine()
           if not shape:
               shape = img_mag.shape
           if avg_power:
               rel_power = np.abs(avg_power - np.sum(img_mag)) / avg_power
           if (not avg_power or rel_power < threshold) \
                   and shape == img_mag.shape:
               img_list.append(pmb.polar2complex(img_mag, img_phs))
               num += 1
               avg_power = (avg_power * (num - 1) + np.sum(img_mag)) / num
       out_mag_filepath = change_param_val(
           out_mag_filepath, PARAM_ID['avg'], num)

       # k-space constant phase correction
       img0 = img_list[0]
       ft_img0 = np.fft.fftshift(np.fft.fftn(img0))
       k0_max = np.unravel_index(np.argmax(ft_img0), ft_img0.shape)
       for idx, img in enumerate(img_list):
           ft_img = np.fft.fftshift(np.fft.fftn(img))
           k_max = np.unravel_index(np.argmax(ft_img), ft_img.shape)
           dephs = np.angle(ft_img0[k0_max] / ft_img[k_max])
           img = np.fft.ifftn(np.fft.ifftshift(ft_img * np.exp(1j * dephs)))
           img_list[idx] = img

       img = pmb.ndstack(img_list, -1)
       img = np.mean(img, -1)
       pmio.save(out_mag_filepath, np.abs(img), affine_nii)
#        pmio.save(out_phs_filepath, np.angle(img), affine_nii)

#        fixed = np.abs(img_list[0])
#        for idx, img in enumerate(img_list):
#            affine = pmb.affine_registration(np.abs(img), fixed, 'rigid')
#            img_list[idx] = pmb.apply_affine(img_list[idx], affine)
#        pmio.save(out_filepath, np.abs(img), affine_nii)
#        print(img.shape, img.nbytes / 1024 / 1024)  # DEBUG
#        # calculate the Fourier transform
#        for img in img_list:
#            fft_list.append(np.fft.fftshift(np.fft.fftn(img)))
#        fixed = np.abs(img[:, :, :, 0])
#        pmb.sample2d(fixed, -1)
#        tmp = tmp * np.exp(1j*0.5)
#        moving = sp.ndimage.shift(fixed, [1.0, 5.0, 0.0])
#        pmb.sample2d(moving, -1)

#        print(linear, shift)
#        moved = sp.ndimage.affine_transform(moving, linear, offset=-shift)
#        pmb.sample2d(moved, -1)
#        pmio.save(out_filepath, moving, affine)
#        pmio.save(mag_filepath, fixed, affine)
#        pmio.save(phs_filepath, moved-fixed, affine)
#        for idx in range(len(img_list)):
#            tmp_img = img[:, :, :, idx]
#            tmp_fft = fft[:, :, :, idx]
#            pmb.sample2d(np.real(tmp_fft), -1)
#            pmb.sample2d(np.imag(tmp_fft), -1)
#            pmb.sample2d(np.abs(img[:, :, :, idx]), -1)
#            pmb.sample2d(np.angle(img[:, :, :, idx]), -1)

       # calculate output
       if verbose > VERB_LVL['none']:
           print('Target:\t{}'.format(os.path.basename(out_mag_filepath)))
           print('Target:\t{}'.format(os.path.basename(out_phs_filepath)))
   return mag_filepath, phs_filepath
'''

# ======================================================================
if __name__ == '__main__':
    msg(__doc__.strip())
    doctest.testmod()
