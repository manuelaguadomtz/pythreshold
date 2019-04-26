# -*- coding:utf-8 -*-

import numpy as np

from skimage.util.shape import view_as_windows

__copyright__ = 'Copyright 2017'
__author__ = u'BSc. Manuel Aguado Martínez'


def bernsen_threshold(img, w_size=15, c_thr=30):
    """Runs the Bernsen thresholding algorithm
    
    Reference:
    Bernsen, J (1986), "Dynamic Thresholding of Grey-Level Images",
    Proc. of the 8th Int. Conf. on Pattern Recognition

    
    @param img: The input image. Must be a gray scale image
    @type img: ndarray
    @param w_size: The size of the local window to compute
        each pixel threshold. Should be an odd window.
    @type w_size: int
    @param c_thr: The threshold contrast to determine an
        homogeneous region
    @type c_thr: int
    
    @return: The estimated local threshold for each pixel
    @rtype: ndarray
    """
    thresholds = np.zeros(img.shape, np.uint8)

    # Obtaining windows
    hw_size = w_size // 2
    padded_img = np.ones((img.shape[0] + w_size - 1,
                          img.shape[1] + w_size - 1)) * np.nan
    padded_img[hw_size: -hw_size,
               hw_size: -hw_size] = img

    winds = view_as_windows(padded_img, (w_size, w_size))

    mins = np.nanmin(winds, axis=(2, 3))
    maxs = np.nanmax(winds, axis=(2, 3))

    # Calculating contrast and mid values
    contrast = maxs - mins
    mid_vals = (maxs + mins) / 2

    thresholds[contrast <= c_thr] = 128
    thresholds[contrast > c_thr] = mid_vals[contrast > c_thr]

    return thresholds
