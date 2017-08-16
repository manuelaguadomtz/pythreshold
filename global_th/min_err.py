#!/usr/bin/env python
# -*- coding:utf-8 -*-

import numpy as np

__copyright__ = 'Copyright 2015'
__author__ = u'Lic. Manuel Aguado Martínez'


def min_err_threshold(image):
    """Runs the minimum error thresholding algorithm.

    Reference:
    Kittler, J. and J. Illingworth. ‘‘On Threshold Selection Using Clustering
    Criteria,’’ IEEE Transactions on Systems, Man, and Cybernetics 15, no. 5
    (1985): 652–655.
    
    @param image: The input image
    @type image: ndarray
    
    @return: The threshold that minimize the error
    @rtype: int
    """
    # Input image histogram
    hist = np.histogram(image, bins=256)[0]

    # The number of background pixels for each threshold
    w_backg = hist.cumsum()
    w_backg[w_backg == 0] = 1  # to avoid divisions by zero

    # The number of foreground pixels for each threshold
    w_foreg = w_backg[-1] - w_backg
    w_foreg[w_foreg == 0] = 1  # to avoid divisions by zero

    # Cumulative distribution function
    cdf = np.cumsum(hist * np.arange(len(hist)))

    # Means (Last term is to avoid divisions by zero)
    b_mean = cdf / (w_backg + (w_backg == 0))
    f_mean = (cdf[-1] - cdf) / (w_foreg + (w_foreg == 0))

    # Standard deviations
    b_std = ((np.arange(len(hist)) - b_mean)**2 * hist).cumsum() / w_backg
    f_std = ((np.arange(len(hist)) - f_mean) ** 2 * hist).cumsum()
    f_std = (f_std[-1] - f_std) / (w_foreg + (w_foreg == 0))

    threshold = 0
    min_error = -1

    for t in xrange(1, len(hist)):
        if w_backg[t] != 0 and w_foreg[t] != 0:
            backg_std = b_std[t]
            foreg_std = f_std[t]

            if foreg_std != 0 and backg_std != 0:
                error = 1 + 2 * (w_backg[t] * np.log(backg_std) + w_foreg[t] *
                                 np.log(foreg_std)) - 2 * (w_backg[t] *
                                                           np.log(w_backg[t])
                                                           + w_foreg[t] *
                                                           np.log(w_foreg[t]))
                if min_error == -1 or min_error > error:
                    min_error = error
                    threshold = t

    return threshold
