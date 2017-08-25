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


def bernsen_threshold(image, window_size=15, contrast_threshold=30):
    """ Runs the Bernsen thresholding algorithm

    Bernsen, J (1986), "Dynamic Thresholding of Grey-Level Images",
    Proc. of the 8th Int. Conf. on Pattern Recognition

    Keywords:
    image -- The input image
        constraints:
            -- Must be a gray scale image
    window_size -- The size of the local window to compute
                   each pixel threshold
        constraints:
            -- Should be an odd window
    contrast_threshold -- The threshold contrast to determine an
        homogeneous region
    """

    result = np.zeros(image.shape, np.uint8)

    for i in xrange(0, image.shape[0]):
        for j in xrange(0, image.shape[1]):
            contrast, mid_gray = __window_threshold_and_mid_value(image,
                                                                  (i, j),
                                                                  window_size)
            if contrast <= contrast_threshold:
                value = 255 if image.item(i, j) >= 128 else 0
            else:
                value = 255 if image.item(i, j) >= mid_gray else 0

            result.itemset(i, j, value)

    return result
