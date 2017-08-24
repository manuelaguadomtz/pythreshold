#!/usr/bin/env python
# -*- coding:utf-8 -*-

import numpy as np

__copyright__ = 'Copyright 2017'
__author__ = u'Lic. Manuel Aguado Martínez'


def lmean_threshold(img, w_size=15):
    """ Runs the local mean thresholding algorithm.

    Reference:

    Modifications: Using integral images to compute local mean.

    @param img: The input image
    @type img: ndarray
    @param w_size: The size of the local window to compute
        each pixel threshold. Should be and odd value
    @type w_size: int 

    @return: The estimated local threshold for each pixel
    @rtype: ndarray
    """
    # Obtaining rows and cols
    rows, cols = img.shape
    i_rows, i_cols = rows + 1, cols + 1

    # Computing integral image
    # Leaving first row and column in zero for convenience
    integral = np.zeros((i_rows, i_cols), np.float)

    integral[1:, 1:] = np.cumsum(np.cumsum(img, axis=1), axis=0)

    # Defining grid
    x, y = np.meshgrid(np.arange(1, i_rows), np.arange(1, i_cols))

    # Obtaining local coordinates
    hw_size = w_size / 2
    x1 = (x - hw_size).clip(1, cols)
    x2 = (x + hw_size).clip(1, cols)
    y1 = (y - hw_size).clip(1, rows)
    y2 = (y + hw_size).clip(1, rows)

    # Obtaining local areas size
    l_size = (x2 - x1 + 1) * (y2 - y1 + 1)

    # Computing sums
    sums = (integral[y2, x2] - integral[y2, x1 - 1] -
            integral[y1 - 1, x2] + integral[y1 - 1, x1 - 1])

    # Computing local means
    means = sums / l_size

    return means
