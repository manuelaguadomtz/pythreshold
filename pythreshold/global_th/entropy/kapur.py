#!/usr/bin/env python
# -*- coding:utf-8 -*-

from itertools import combinations

import numpy as np

__copyright__ = 'Copyright 2020'
__author__ = u'BSc. Manuel Aguado Martínez'


def kapur_threshold(image):
    """ Runs the Kapur's threshold algorithm.

    Reference:
    Kapur, J. N., P. K. Sahoo, and A. K. C.Wong. ‘‘A New Method for Gray-Level
    Picture Thresholding Using the Entropy of the Histogram,’’ Computer Vision,
    Graphics, and Image Processing 29, no. 3 (1985): 273–285.

    @param image: The input image
    @type image: ndarray

    @return: The estimated threshold
    @rtype: int
    """
    hist, _ = np.histogram(image, bins=range(256), density=True)
    c_hist = hist.cumsum()
    c_hist_i = 1.0 - c_hist

    # To avoid invalid operations regarding 0 and negative values.
    c_hist[c_hist <= 0] = 1
    c_hist_i[c_hist_i <= 0] = 1

    c_entropy = (hist * np.log(hist + (hist <= 0))).cumsum()
    b_entropy = -c_entropy / c_hist + np.log(c_hist)

    c_entropy_i = c_entropy[-1] - c_entropy
    f_entropy = -c_entropy_i / c_hist_i + np.log(c_hist_i)

    return np.argmax(b_entropy + f_entropy)


def _get_regions_entropy(hist, c_hist, thresholds):
    """Get the total entropy of regions for a given set of thresholds"""

    total_entropy = 0
    for i in range(len(thresholds) - 1):
        # Thresholds
        t1 = thresholds[i] + 1
        t2 = thresholds[i + 1]

        # print(thresholds, t1, t2)

        # Cumulative histogram
        hc_val = c_hist[t2] - c_hist[t1 - 1]

        # Normalized histogram
        h_val = hist[t1:t2 + 1] / hc_val if hc_val > 0 else 1

        # entropy
        entropy = -(h_val * np.log(h_val + (h_val <= 0))).sum()

        # Updating total entropy
        total_entropy += entropy

    return total_entropy


def _get_thresholds(hist, c_hist, nthrs):
    """Get the thresholds that maximize the entropy of the regions

    @param hist: The normalized histogram of the image
    @type hist: ndarray
    @param c_hist: The cummuative normalized histogram of the image
    @type c_hist: ndarray
    @param nthrs: The number of thresholds
    @type nthrs: int
    """
    # Thresholds combinations
    thr_combinations = combinations(range(255), nthrs)

    max_entropy = 0
    opt_thresholds = None

    # Extending histograms for convenience
    # hist = np.append([0], hist)
    c_hist = np.append(c_hist, [0])

    for thresholds in thr_combinations:
        # Extending thresholds for convenience
        e_thresholds = [-1]
        e_thresholds.extend(thresholds)
        e_thresholds.extend([len(hist) - 1])

        # Computing regions entropy for the current combination of thresholds
        regions_entropy = _get_regions_entropy(hist, c_hist, e_thresholds)

        if regions_entropy > max_entropy:
            max_entropy = regions_entropy
            opt_thresholds = thresholds

    return opt_thresholds


def kapur_multithreshold(image, nthrs):
    """ Runs the Kapur's multi-threshold algorithm.

    Reference:
    Kapur, J. N., P. K. Sahoo, and A. K. C.Wong. ‘‘A New Method for Gray-Level
    Picture Thresholding Using the Entropy of the Histogram,’’ Computer Vision,
    Graphics, and Image Processing 29, no. 3 (1985): 273–285.

    @param image: The input image
    @type image: ndarray
    @param nthrs: The number of thresholds
    @type nthrs: int

    @return: The estimated threshold
    @rtype: int
    """
    # Histogran
    hist, _ = np.histogram(image, bins=range(256), density=True)

    # Cumulative histogram
    c_hist = hist.cumsum()

    return _get_thresholds(hist, c_hist, nthrs)
