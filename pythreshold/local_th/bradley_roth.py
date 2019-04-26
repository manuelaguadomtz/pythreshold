# -*- coding:utf-8 -*-

import numpy as np

__copyright__ = 'Copyright 2017'
__author__ = u'BSc. Manuel Aguado Martínez'


def bradley_roth_threshold(img, w_size=15, w=0.15):
    """ Runs the Bradley-Roth thresholding algorithm.

    Reference:
    Bradley, D., & Roth, G. (2007). Adaptive thresholding
    using the integral image. Journal of Graphics Tools, 12(2), 13-21.

    @param img: The input image
    @type img: ndarray
    @param w_size: The size of the local window to compute
        each pixel threshold. Should be and odd value
    @type w_size: int 
    @param w: Used to verify is each pixel is 'w' percent lower than
        the local average. It should be a normalized value in the 
        range [0, 1].
    @type w: float

    @return: The estimated local threshold for each pixel
    @rtype: ndarray
    """
    # Obtaining rows and cols
    rows, cols = img.shape
    i_rows, i_cols = rows + 1, cols + 1

    # Computing integral image
    # Leaving first row and column in zero for convenience
    integ = np.zeros((i_rows, i_cols), np.float)

    integ[1:, 1:] = np.cumsum(np.cumsum(img.astype(np.float), axis=0), axis=1)

    # Defining grid
    x, y = np.meshgrid(np.arange(1, i_cols), np.arange(1, i_rows))

    # Obtaining local coordinates
    hw_size = w_size // 2
    x1 = (x - hw_size).clip(1, cols)
    x2 = (x + hw_size).clip(1, cols)
    y1 = (y - hw_size).clip(1, rows)
    y2 = (y + hw_size).clip(1, rows)

    # Obtaining local areas size
    l_size = (y2 - y1 + 1) * (x2 - x1 + 1)

    # Computing sums
    sums = (integ[y2, x2] - integ[y2, x1 - 1] -
            integ[y1 - 1, x2] + integ[y1 - 1, x1 - 1])

    # Computing local means
    means = sums / l_size

    return means * (1 - w)
