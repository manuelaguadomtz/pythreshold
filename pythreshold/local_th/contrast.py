# -*- coding:utf-8 -*-

import numpy as np

from skimage.util.shape import view_as_windows

__copyright__ = 'Copyright 2015'
__author__ = u'BSc. Manuel Aguado Martínez'


def contrast_threshold(img, w_size=15):
    """Runs the contrast thresholding algorithm

    Reference: 
    Parker, J. R. (2010). Algorithms for image processing and
    computer vision. John Wiley & Sons.

    @param img: The input image. Must be a gray scale image
    @type img: ndarray
    @param w_size: The size of the local window to compute
        each pixel threshold. Should be an odd window.
    @type w_size: int

    @return: The estimated local threshold for each pixel
    @rtype: ndarray
    """
    thresholds = np.zeros(img.shape)

    # Obtaining windows
    hw_size = w_size // 2
    padded_img = np.ones((img.shape[0] + w_size - 1,
                          img.shape[1] + w_size - 1)) * np.nan
    padded_img[hw_size: -hw_size, hw_size: -hw_size] = img

    winds = view_as_windows(padded_img, (w_size, w_size))

    # Obtaining maximums and minimums
    mins = np.nanmin(winds, axis=(2, 3))
    maxs = np.nanmax(winds, axis=(2, 3))

    min_dif = img - mins
    max_dif = maxs - img

    thresholds[min_dif <= max_dif] = 256
    thresholds[min_dif > max_dif] = 0

    return thresholds
