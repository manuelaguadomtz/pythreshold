#!/usr/bin/env python
# -*- coding:utf-8 -*-

import numpy as np

__copyright__ = 'Copyright 2014'
__author__ = u'Lic. Manuel Aguado Martínez'


def __compute_entropy(histogram):
    """Compute the entropy cdf of a histogram
    """
    cdf = np.zeros(histogram.shape, dtype=np.float64)

    if histogram[0] > 0:
        cdf[0] = - histogram[0] * np.log(histogram[0])
    for i in xrange(1, len(histogram)):
        if histogram[i] > 0:
            cdf[i] = - histogram[i] * np.log(histogram[i])
        cdf[i] += cdf[i - 1]

    return cdf


def pun_threshold(image):
    """ Runs the Pun's threshold algorithm.

    Reference:
    Pun, T. ‘‘A New Method for Grey-Level Picture Thresholding Using the
    Entropy of the Histogram,’’ Signal Processing 2, no. 3 (1980): 223–237.

    Keyword Arguments:
    image -- The input image
    """
    histogram = np.histogram(image, bins=256, normed=True)[0]
    entropy_cdf = __compute_entropy(histogram)
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
