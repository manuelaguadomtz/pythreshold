#!/usr/bin/env python
# -*- coding:utf-8 -*-

import numpy as np

from skimage.util.shape import view_as_windows

__copyright__ = 'Copyright 2015'
__author__ = u'Lic. Manuel Aguado Martínez'


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
    hw_size = w_size / 2
    padded_img_min = np.ones((img.shape[0] + w_size, img.shape[1] + w_size)) * 256
    padded_img_min[hw_size: -hw_size - 1, hw_size: -hw_size - 1] = img
    padded_img_max = np.ones((img.shape[0] + w_size, img.shape[1] + w_size)) * -1
    padded_img_max[hw_size: -hw_size - 1 , hw_size: -hw_size - 1] = img

    min_winds = view_as_windows(padded_img_min, (w_size, w_size))
    max_winds = view_as_windows(padded_img_max, (w_size, w_size))

    # Estimating maximums and minimums values
    mins = np.zeros_like(img)
    maxs = np.zeros_like(img)
    for i in np.arange(img.shape[0]):
        for j in np.arange(img.shape[1]):
            maxs[i, j] = np.max(max_winds[i, j])
            mins[i, j] = np.min(min_winds[i, j])

    # Calculating contrast and mid values
    contrast = maxs - mins
    mid_vals = (maxs + mins) / 2

    thresholds[contrast <= c_thr] = 128
    thresholds[contrast > c_thr] = mid_vals[contrast > c_thr]

    return thresholds
