# -*- coding:utf-8 -*-

import numpy as np

__copyright__ = 'Copyright 2017'
__author__ = u'BSc. Manuel Aguado Martínez'


def nick_threshold(img, w_size=15, k=-0.2):
    """ Runs the NICK thresholding algorithm.

    Reference:
    Khurshid, K., Siddiqi, I., Faure, C., & Vincent, N.
    (2009, January). Comparison of Niblack inspired Binarization methods for
    ancient documents. In IS&T/SPIE Electronic Imaging (pp. 72470U-72470U).
    International Society for Optics and Photonics.

    Modifications: Using integral images to compute the local mean and the
    NICK variation of the Niblack standard deviation term

    @param img: The input image
    @type img: ndarray
    @param w_size: The size of the local window to compute
        each pixel threshold. Should be and odd value
    @type w_size: int
    @param k: Controls the value of the local threshold. Should lie in the
        interval [-0.2, -0.1]
    @type k: float

    @return: The estimated local threshold for each pixel
    @rtype: ndarray
    """
    # Obtaining rows and cols
    rows, cols = img.shape
    i_rows, i_cols = rows + 1, cols + 1

    # Computing integral images
    # Leaving first row and column in zero for convenience
    integ = np.zeros((i_rows, i_cols), np.float)
    sqr_integral = np.zeros((i_rows, i_cols), np.float)

    integ[1:, 1:] = np.cumsum(np.cumsum(img.astype(np.float), axis=0), axis=1)
    sqr_img = np.square(img.astype(np.float))
    sqr_integral[1:, 1:] = np.cumsum(np.cumsum(sqr_img, axis=0), axis=1)

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
    sqr_sums = (sqr_integral[y2, x2] - sqr_integral[y2, x1 - 1] -
                sqr_integral[y1 - 1, x2] + sqr_integral[y1 - 1, x1 - 1])

    # Computing local means
    means = sums / l_size

    # Computing NICK variation of the Niblack term corresponding
    # to the standard deviation
    nick_stds = np.sqrt((sqr_sums - np.square(means)) / l_size)

    # Computing thresholds
    thresholds = means + k * nick_stds

    return thresholds
