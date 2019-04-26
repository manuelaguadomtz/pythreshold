# -*- coding:utf-8 -*-

import numpy as np

from skimage.util.shape import view_as_windows

__copyright__ = 'Copyright 2017'
__author__ = u'BSc. Manuel Aguado Martínez'


def singh_threshold(img, w_size=15, k=0.85):  # 0.35
    """ Runs the Singh thresholding algorithm
    
    Reference:
    Singh, O. I., Sinam, T., James, O., & Singh, T. R. (2012). Local contrast
    and mean based thresholding technique in image binarization. International
    Journal of Computer Applications, 51, 5-10.

    Modifications: Using integral images to compute local mean
        and standard deviation

    @param img: The input image
    @type img: ndarray
    @param w_size: The size of the local window to compute
        each pixel threshold. Should be and odd value
    @type w_size: int 
    @param k: Controls the value of the local threshold. It lies in the
        interval [0, 1]
    @type k: float

    @return: The estimated local threshold for each pixel
    @rtype: ndarray
    """
    img = img.astype(np.float) / 255

    # Obtaining rows and cols
    rows, cols = img.shape
    i_rows, i_cols = rows + 1, cols + 1

    # Computing integral images
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

    # Obtaining windows
    padded_img = np.ones((rows + w_size - 1, cols + w_size - 1)) * np.nan
    padded_img[hw_size: -hw_size, hw_size: -hw_size] = img

    winds = view_as_windows(padded_img, (w_size, w_size))

    # Obtaining maximums and minimums
    mins = np.nanmin(winds, axis=(2, 3))
    maxs = np.nanmax(winds, axis=(2, 3))

    # Computing thresholds
    thresholds = k * (means + (maxs - mins) * (1 - img)) * 255

    return thresholds
