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

    # The number of foreground pixels for each threshold
    w_foreg = w_backg[-1] - w_backg

    # Cumulative distribution function
    cdf = np.cumsum(hist * np.arange(len(hist)))

    # Means
    b_mean = cdf * (w_backg != 0) / (w_backg + (w_backg == 0))
    w_mean = (cdf[-1] - cdf) * (w_foreg != 0) / (w_foreg + (w_foreg == 0))

    threshold = 0
    min_error = -1

    for t in xrange(1, len(hist)):
        if w_backg[t] != 0 and w_foreg[t] != 0:
            # TODO Compute these values outside the for loop
            black_std = (np.arange(t+1) - b_mean[t])**2 * hist[:t+1]
            black_std = black_std.sum() / w_backg[t]
            white_std = ((np.arange(t+1, len(hist)) - w_mean[t])**2 *
                         hist[t+1:len(hist)])
            white_std = white_std.sum() / w_foreg[t]

            if white_std != 0 and black_std != 0:
                error = 1 + 2 * (w_backg[t] * np.log(black_std) + w_foreg[t] *
                                 np.log(white_std)) - 2 * (w_backg[t] *
                                                           np.log(w_backg[t])
                                                           + w_foreg[t] *
                                                           np.log(w_foreg[t]))
                if min_error == -1 or min_error > error:
                    min_error = error
                    threshold = t

    return threshold
