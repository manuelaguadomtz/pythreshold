#!/usr/bin/env python
# -*- coding:utf-8 -*-

import numpy as np

__copyright__ = 'Copyright 2017'
__author__ = u'Lic. Manuel Aguado Martínez'


def pun_threshold(image):
    """ Runs the Pun's threshold algorithm.

    Reference:
    Pun, T. ‘‘A New Method for Grey-Level Picture Thresholding Using the
    Entropy of the Histogram,’’ Signal Processing 2, no. 3 (1980): 223–237.

    @param image: The input image
    @type image: ndarray
    
    @return: The estimated threshold
    @rtype: int
    """
    histogram = np.histogram(image, bins=256, normed=True)[0]
    entropy_cdf = np.cumsum(histogram * np.log(histogram + (histogram == 0)))
    cdf = np.cumsum(histogram)

    max_entropy = 0
    threshold = 0

    for t in xrange(len(histogram) - 1):
        black_max = np.max(histogram[:t + 1])
        white_max = np.max(histogram[t + 1:])
        if black_max * white_max > 0:
            entropy_ratio = entropy_cdf[t] / entropy_cdf[len(histogram) - 1]
            x = entropy_ratio * np.log(cdf[t])/np.log(black_max)
            y = 1.0 - entropy_ratio
            z = np.log(1 - cdf[t]) / np.log(white_max)
            entropy = x + y * z

            if max_entropy < entropy:
                max_entropy = entropy
                threshold = t

    return threshold
