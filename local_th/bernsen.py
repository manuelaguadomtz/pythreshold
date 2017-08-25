#!/usr/bin/env python
# -*- coding:utf-8 -*-

import numpy as np

__copyright__ = 'Copyright 2015'
__author__ = u'Lic. Manuel Aguado Martínez'


def __window_threshold_and_mid_value(image, pos, window_size):
    """ Get the contrast and the contrast mid value

    Keyword Arguments:
    image -- The image to binarize
    pos -- A Point2D instance that represents the central
           position of the window
    window_size -- the size of the window

    returns: A tuple with the contrast value in the first position
        and the mid gray value in the second
    """
    slide = window_size / 2
    shape = image.shape
    right_x = min(shape[0] - 1, pos[0] + slide)
    left_x = max(0, pos[0] - slide)
    up_y = max(0, pos[1] - slide)
    down_y = min(shape[1] - 1, pos[1] + slide)

    block = image[left_x:right_x + 1, up_y:down_y + 1]
    maximum = int(np.amax(block))
    minimum = int(np.amin(block))

    return (maximum - minimum), (maximum + minimum) / 2


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
