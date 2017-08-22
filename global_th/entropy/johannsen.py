#!/usr/bin/env python
# -*- coding:utf-8 -*-

import numpy as np

__copyright__ = 'Copyright 2017'
__author__ = u'Lic. Manuel Aguado Martínez'


def __compute_entropy(x):
    """"Compute the entropy of a gray value given his probability
    and cumulative probability
    """
    return 0.0 if x <= 0 else -x * np.log(x)


def __compute_total_entropy(histogram, cdf, threshold):
    """Compute the total entropy given a threshold

    Keywords:
    histogram -- The input image histogram
    cdf -- The cumulative histogram function
    threshold -- The threshold to distinguish between black and white pixels
    """
    if histogram[threshold] == 0:
        return 0

    sb = np.log(cdf[threshold]) + 1 / cdf[threshold] *\
        (__compute_entropy(histogram[threshold]) +
         __compute_entropy(cdf[threshold - 1]))

    white_pixels = cdf[len(histogram) - 1] - cdf[threshold - 1]
    sw = np.log(white_pixels) + 1 / white_pixels *\
        (__compute_entropy(histogram[threshold]) +
         __compute_entropy(white_pixels - histogram[threshold]))

    return 0 if sb * sw == 0 else sb + sw


def johannsen_threshold(image, truncate_hist=True):
    """ Runs the Johannsen's threshold algorithm.

    Reference:
    Johannsen, G., and J. Bille ‘‘A Threshold Selection Method Using
    Information Measures,’’ Proceedings of the Sixth International Conference
    on Pattern Recognition, Munich, Germany (1982): 140–143.

    @param image: The input image
    @type image: ndarray
    @param truncate_hist: If true the algorithm not considered as valid
        thresholds the first and the final values different of zero in the
        histogram.
    @type truncate_hist: bool

    @return: The estimated threshold
    @rtype: int
    """
    hist = np.histogram(image, range=(0, 255), bins=255, density=True)[0]
    cdf = hist.cumsum()

    start = 0
    end = len(hist) - 1
    if truncate_hist:
        while hist[start] == 0:
            start += 1
        start += 1
        while hist[end] == 0:
            end -= 1
        end -= 1

    min_entropy = 0
    threshold = -1

    for t in xrange(start, end + 1):
        entropy = __compute_total_entropy(hist, cdf, t)
        if entropy != 0 and (min_entropy > entropy or threshold == -1):
            min_entropy = entropy
            threshold = t

    return threshold
