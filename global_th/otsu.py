#!/usr/bin/env python
# -*- coding:utf-8 -*-

import numpy as np

__copyright__ = 'Copyright 2017'
__author__ = u'Lic. Manuel Aguado Martínez'


def otsu_threshold(image=None, hist=None):
    """ Runs the Otsu threshold algorithm.
    
    Reference:
    Otsu, Nobuyuki. "A threshold selection method from gray-level
    histograms." IEEE transactions on systems, man, and cybernetics
    9.1 (1979): 62-66.

    @param image: The input image
    @type image: numpy.ndarray
    @param hist: The input image histogram
    @type hist: numpy.ndarray
    
    @return: The Otsu threshold
    @rtype int
    """
    if image is None and hist is None:
        raise ValueError('You must pass as a parameter either'
                         'the input image or its histogram')

    if not hist:
        hist = np.histogram(image, bins=256)[0]

    n_pixels = image.shape[0] * image.shape[1]

    sum_a = np.sum(np.arange(len(hist)) * hist)
    sum_b = 0

    weight_background = 0
    max_variance = 0

    threshold = 0

    for t in xrange(len(hist)):
        weight_background += hist[t]
        if weight_background == 0:
            continue

        weight_foreground = n_pixels - weight_background
        if weight_foreground == 0:
            break

        sum_b += t * hist[t]

        # calulating the means
        mean_background = float(sum_b) / float(weight_background)
        mean_foreground = float(sum_a - sum_b) / float(weight_foreground)

        # calculating between class variance
        between_variance = weight_background * weight_foreground * \
            (mean_background - mean_foreground)**2

        if between_variance > max_variance:
            max_variance = between_variance
            threshold = t

    return threshold
