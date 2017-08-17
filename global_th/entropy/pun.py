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
    # Calculating histogram
    histogram = np.histogram(image, bins=255, density=True)[0]

    # Calculating histogram cumulative sum
    hcs = np.cumsum(histogram)

    # Calculating inverted histogram cumulative sum
    i_hcs = 1.0 - hcs
    i_hcs[i_hcs == 0] = 1  # To avoid log(0) calculations

    # Calculating normed entropy cumulative sum
    ec_norm = np.cumsum(histogram * np.log(histogram + (histogram == 0)))
    ec_norm /= ec_norm[-1]

    max_entropy = 0
    threshold = 0
    for t in xrange(len(histogram) - 1):
        black_max = np.max(histogram[:t + 1])
        white_max = np.max(histogram[t + 1:])
        if black_max * white_max > 0:
            e_ratio = ec_norm[t]
            x = e_ratio * np.log(hcs[t]) / np.log(black_max)
            y = 1.0 - e_ratio
            z1 = np.log(i_hcs[t])
            z2 = np.log(white_max)
            z = z1 / z2
            entropy = x + y * z

            if max_entropy < entropy:
                max_entropy = entropy
                threshold = t

    return threshold
