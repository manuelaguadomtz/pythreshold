#!/usr/bin/env python
# -*- coding:utf-8 -*-

import numpy as np

__copyright__ = 'Copyright 2017'
__author__ = u'Lic. Manuel Aguado Martínez'


def __compute_entropy(probability, cum_probability):
    """"Compute the entropy of a gray value given his probability
    and cumulative probability
    """
    return 0.0 if probability <= 0 or cum_probability <= 0 else \
        -probability / cum_probability * np.log(probability / cum_probability)


def __compute_total_entropy(histogram, cdf, threshold):
    """Compute the total entropy given a threshold

    Keywords:
    histogram -- The input image histogram
    cdf -- The cumulative histogram function
    threshold -- The threshold to distinguish between black and white pixels
    """
    white_entropy = 0
    black_entropy = 0
    for i in xrange(0, len(histogram)):
        if i <= threshold:
            black_entropy += __compute_entropy(histogram[i], cdf[threshold])
        else:
            white_entropy += __compute_entropy(histogram[i],
                                               1.0 - cdf[threshold])

    return black_entropy + white_entropy


def kapur_threshold(image):
    """ Runs the Kapur's threshold algorithm.

    Reference:
    Kapur, J. N., P. K. Sahoo, and A. K. C.Wong. ‘‘A New Method for Gray-Level
    Picture Thresholding Using the Entropy of the Histogram,’’ Computer Vision,
    Graphics, and Image Processing 29, no. 3 (1985): 273–285.

    Keyword Arguments:
    image -- The input image
    """
    hist = np.histogram(image, range=(0, 255), bins=255, density=True)[0]
    cdf = hist.cumsum()

    max_entropy = 0
    threshold = 0

    for t in xrange(len(hist) - 1):
        entropy = __compute_total_entropy(hist, cdf, t)
        if max_entropy < entropy:
            max_entropy = entropy
            threshold = t

    return threshold
