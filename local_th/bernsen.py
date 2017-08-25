#!/usr/bin/env python
# -*- coding:utf-8 -*-

import numpy as np

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
    rows, cols = img.shape
    thresholds = np.zeros(img.shape, np.uint8).ravel()

    # Defining grid
    x, y = np.meshgrid(np.arange(0, rows), np.arange(0, cols))

    # Obtaining local coordinates
    hw_size = w_size / 2
    x1 = (x - hw_size).clip(0, cols).ravel()
    x2 = (x + hw_size).clip(0, cols).ravel()
    y1 = (y - hw_size).clip(0, rows).ravel()
    y2 = (y + hw_size).clip(0, rows).ravel()

    # Obtaining maximums and minimums
    mins = np.zeros_like(x1)
    maxs = np.zeros_like(x2)
    for i in np.arange(len(x1)):
        mins[i] = np.amin(img[y1[i]: y2[i] + 1, x1[i]: x2[i] + 1])
        maxs[i] = np.amax(img[y1[i]: y2[i] + 1, x1[i]: x2[i] + 1])

    # calculating contrast and mid values
    contrast = maxs - mins
    mid_vals = (maxs + mins) / 2

    thresholds[contrast <= c_thr] = 128
    thresholds[contrast > c_thr] = mid_vals[contrast > c_thr]

    return thresholds.reshape(img.shape)
