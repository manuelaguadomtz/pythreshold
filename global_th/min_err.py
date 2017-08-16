#!/usr/bin/env python
# -*- coding:utf-8 -*-

import numpy as np

__copyright__ = 'Copyright 2015'
__author__ = u'Lic. Manuel Aguado Martínez'


def min_err_threshold(image):
    """ Runs the Minimum Error Thresholding algorithm.

    Reference:
    Kittler, J. and J. Illingworth. ‘‘On Threshold Selection Using Clustering
    Criteria,’’ IEEE Transactions on Systems, Man, and Cybernetics 15, no. 5
    (1985): 652–655.

    Keyword Arguments:
    image -- The input image
    """

    hist = np.histogram(image, bins=256)[0]
    cdf = hist.cumsum()
    inv_cdf = cdf[len(cdf) - 1] - cdf

    # TODO Improve the efficiency of the next calculus by using vectorize
    weight_cdf = (hist * np.arange(len(hist))).cumsum()
    black_mean = weight_cdf * (cdf != 0) / (cdf + (cdf == 0))
    white_mean = (weight_cdf[len(weight_cdf) - 1] - weight_cdf) *\
                 (inv_cdf != 0) / (inv_cdf + (inv_cdf == 0))

    threshold = 0
    min_error = -1

    for t in xrange(1, len(hist)):
        if cdf[t] != 0 and inv_cdf[t] != 0:
            # TODO Compute these values outside the for loop
            black_std = (np.arange(t+1) - black_mean[t])**2 * hist[:t+1]
            black_std = black_std.sum() / cdf[t]
            white_std = ((np.arange(t+1, len(hist)) - white_mean[t])**2 *
                         hist[t+1:len(hist)])
            white_std = white_std.sum() / inv_cdf[t]

            if white_std != 0 and black_std != 0:
                error = 1 + 2 * (cdf[t] * np.log(black_std) + inv_cdf[t] *
                                 np.log(white_std)) - 2 * (cdf[t] *
                                                           np.log(cdf[t])
                                                           + inv_cdf[t] *
                                                           np.log(inv_cdf[t]))
                if min_error == -1 or min_error > error:
                    min_error = error
                    threshold = t

    return threshold
