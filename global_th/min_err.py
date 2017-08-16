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

    hist = np.histogram(image, bins=256)[0]
    c_hist = hist.cumsum()
    inv_c_histogram = c_hist[len(c_hist) - 1] - c_hist

    # TODO Improve the efficiency of the next calculus by using vectorize
    weight_cdf = (hist * np.arange(len(hist))).cumsum()
    black_mean = weight_cdf * (c_hist != 0) / (c_hist + (c_hist == 0))
    white_mean = (weight_cdf[len(weight_cdf) - 1] - weight_cdf) *\
                 (inv_c_histogram != 0) / (inv_c_histogram + (inv_c_histogram == 0))

    threshold = 0
    min_error = -1

    for t in xrange(1, len(hist)):
        if c_hist[t] != 0 and inv_c_histogram[t] != 0:
            # TODO Compute these values outside the for loop
            black_std = (np.arange(t+1) - black_mean[t])**2 * hist[:t+1]
            black_std = black_std.sum() / c_hist[t]
            white_std = ((np.arange(t+1, len(hist)) - white_mean[t])**2 *
                         hist[t+1:len(hist)])
            white_std = white_std.sum() / inv_c_histogram[t]

            if white_std != 0 and black_std != 0:
                error = 1 + 2 * (c_hist[t] * np.log(black_std) + inv_c_histogram[t] *
                                 np.log(white_std)) - 2 * (c_hist[t] *
                                                           np.log(c_hist[t])
                                                           + inv_c_histogram[t] *
                                                           np.log(inv_c_histogram[t]))
                if min_error == -1 or min_error > error:
                    min_error = error
                    threshold = t

    return threshold
