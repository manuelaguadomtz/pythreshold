# -*- coding:utf-8 -*-

import numpy as np

from skimage.util.shape import view_as_windows

__copyright__ = 'Copyright 2017'
__author__ = u'BSc. Manuel Aguado Martínez'


def feng_threshold(img, w_size1=15, w_size2=30,
                   k1=0.15, k2=0.01, alpha1=0.1):
    """ Runs the Feng's thresholding algorithm.

    Reference:
    Algorithm proposed in: Meng-Ling Feng and Yap-Peng Tan, “Contrast adaptive
    thresholding of low quality document images”, IEICE Electron. Express,
    Vol. 1, No. 16, pp.501-506, (2004).

    Modifications: Using integral images to compute the local mean and the
    standard deviation

    @param img: The input image. Must be a gray scale image
    @type img: ndarray
    @param w_size1: The size of the primary local window to compute
        each pixel threshold. Should be an odd window
    @type w_size1: int
    @param w_size2: The size of the secondary local window to compute
        the dynamic range standard deviation. Should be an odd window
    @type w_size2: int
    @param k1: Parameter value that lies in the interval [0.15, 0.25].
    @type k1: float
    @param k2: Parameter value that lies in the interval [0.01, 0.05].
    @type k2: float
    @param alpha1: Parameter value that lies in the interval [0.15, 0.25].
    @type alpha1: float

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
    hw_size = w_size1 // 2
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

    # Computing local standard deviation
    stds = np.sqrt(sqr_sums / l_size - np.square(means))

    # Obtaining windows
    padded_img = np.ones((rows + w_size1 - 1, cols + w_size1 - 1)) * np.nan
    padded_img[hw_size: -hw_size, hw_size: -hw_size] = img

    winds = view_as_windows(padded_img, (w_size1, w_size1))

    # Obtaining maximums and minimums
    mins = np.nanmin(winds, axis=(2, 3))

    # Obtaining local coordinates for std range calculations
    hw_size = w_size2 // 2
    x1 = (x - hw_size).clip(1, cols)
    x2 = (x + hw_size).clip(1, cols)
    y1 = (y - hw_size).clip(1, rows)
    y2 = (y + hw_size).clip(1, rows)

    # Obtaining local areas size
    l_size = (y2 - y1 + 2) * (x2 - x1 + 2)

    # Computing sums
    sums = (integ[y2, x2] - integ[y2, x1 - 1] -
            integ[y1 - 1, x2] + integ[y1 - 1, x1 - 1])
    sqr_sums = (sqr_integral[y2, x2] - sqr_integral[y2, x1 - 1] -
                sqr_integral[y1 - 1, x2] + sqr_integral[y1 - 1, x1 - 1])

    # Computing local means2
    means2 = sums / l_size

    # Computing standard deviation range
    std_ranges = np.sqrt(sqr_sums / l_size - np.square(means2))

    # Computing normalized standard deviations and extra alpha parameters
    n_stds = stds / std_ranges
    n_sqr_std = np.square(n_stds)
    alpha2 = k1 * n_sqr_std
    alpha3 = k2 * n_sqr_std

    thresholds = ((1 - alpha1) * means + alpha2 * n_stds
                  * (means - mins) + alpha3 * mins)

    return thresholds
