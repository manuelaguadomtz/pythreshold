#!/usr/bin/env python
# -*- coding:utf-8 -*-

import numpy as np

__copyright__ = 'Copyright 2017'
__author__ = u'Lic. Manuel Aguado Martínez'


def kapur_threshold(image):
    """ Runs the Kapur's threshold algorithm.

    Reference:
    Kapur, J. N., P. K. Sahoo, and A. K. C.Wong. ‘‘A New Method for Gray-Level
    Picture Thresholding Using the Entropy of the Histogram,’’ Computer Vision,
    Graphics, and Image Processing 29, no. 3 (1985): 273–285.

    @param image: The input image
    @type image: ndarray

    @return: The estimated threshold
    @rtype: int
    """
    hist = np.histogram(image, range=(0, 255), bins=255, density=True)[0]
    c_hist = hist.cumsum()
    c_hist_i = 1.0 - c_hist

    # To avoid invalid operations regarding 0.
    c_hist[c_hist == 0] = 1
    c_hist_i[c_hist_i == 0] = 1

    c_entropy = (hist * np.log(hist + (hist == 0))).cumsum()
    b_entropy = -c_entropy / c_hist + np.log(c_hist)

    c_entropy_i = c_entropy[-1] - c_entropy
    f_entropy = -c_entropy_i / c_hist_i + np.log(c_hist_i)

    return np.argmax(b_entropy + f_entropy)
