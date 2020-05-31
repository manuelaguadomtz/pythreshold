# -*- coding:utf-8 -*-

from itertools import combinations

import numpy as np

__copyright__ = 'Copyright 2017'
__author__ = u'BSc. Manuel Aguado Martínez'


def otsu_threshold(image=None, hist=None):
    """ Runs the Otsu threshold algorithm.

    Reference:
    Otsu, Nobuyuki. "A threshold selection method from gray-level
    histograms." IEEE transactions on systems, man, and cybernetics
    9.1 (1979): 62-66.

    @param image: The input image
    @type image: ndarray
    @param hist: The input image histogram
    @type hist: ndarray

    @return: The Otsu threshold
    @rtype int
    """
    if image is None and hist is None:
        raise ValueError('You must pass as a parameter either'
                         'the input image or its histogram')

    # Calculating histogram
    if not hist:
        hist = np.histogram(image, bins=range(256))[0].astype(np.float)

    cdf_backg = np.cumsum(np.arange(len(hist)) * hist)
    w_backg = np.cumsum(hist)  # The number of background pixels
    w_backg[w_backg == 0] = 1  # To avoid divisions by zero
    m_backg = cdf_backg / w_backg  # The means

    cdf_foreg = cdf_backg[-1] - cdf_backg
    w_foreg = w_backg[-1] - w_backg  # The number of foreground pixels
    w_foreg[w_foreg == 0] = 1  # To avoid divisions by zero
    m_foreg = cdf_foreg / w_foreg  # The means

    var_between_classes = w_backg * w_foreg * (m_backg - m_foreg) ** 2

    return np.argmax(var_between_classes)


def _get_variance(hist, c_hist, cdf, thresholds):
    """Get the total entropy of regions for a given set of thresholds"""

    variance = 0

    for i in range(len(thresholds) - 1):
        # Thresholds
        t1 = thresholds[i] + 1
        t2 = thresholds[i + 1]

        # Cumulative histogram
        weight = c_hist[t2] - c_hist[t1 - 1]

        # Region CDF
        r_cdf = cdf[t2] - cdf[t1 - 1]

        # Region mean
        r_mean = r_cdf / weight if weight != 0 else 0

        variance += weight * r_mean ** 2

    return variance


def _get_thresholds(hist, c_hist, cdf, nthrs):
    """Get the thresholds that maximize the variance between regions

    @param hist: The normalized histogram of the image
    @type hist: ndarray
    @param c_hist: The normalized histogram of the image
    @type c_hist: ndarray
    @param cdf: The cummulative distribution function of the histogram
    @type cdf: ndarray
    @param nthrs: The number of thresholds
    @type nthrs: int
    """
    # Thresholds combinations
    thr_combinations = combinations(range(255), nthrs)

    max_var = 0
    opt_thresholds = None

    # Extending histograms for convenience
    c_hist = np.append(c_hist, [0])
    cdf = np.append(cdf, [0])

    for thresholds in thr_combinations:
        # Extending thresholds for convenience
        e_thresholds = [-1]
        e_thresholds.extend(thresholds)
        e_thresholds.extend([len(hist) - 1])

        # Computing variance for the current combination of thresholds
        regions_var = _get_variance(hist, c_hist, cdf, e_thresholds)

        if regions_var > max_var:
            max_var = regions_var
            opt_thresholds = thresholds

    return opt_thresholds


def otsu_multithreshold(image=None, hist=None, nthrs=2):
    """ Runs the Otsu's multi-threshold algorithm.

    Reference:
    Otsu, Nobuyuki. "A threshold selection method from gray-level
    histograms." IEEE transactions on systems, man, and cybernetics
    9.1 (1979): 62-66.

    Liao, Ping-Sung, Tse-Sheng Chen, and Pau-Choo Chung. "A fast algorithm
    for multilevel thresholding." J. Inf. Sci. Eng. 17.5 (2001): 713-727.

    @param image: The input image
    @type image: ndarray
    @param hist: The input image histogram
    @type hist: ndarray
    @param nthrs: The number of thresholds
    @type nthrs: int

    @return: The estimated thresholds
    @rtype: int
    """
    # Histogran
    if image is None and hist is None:
        raise ValueError('You must pass as a parameter either'
                         'the input image or its histogram')

    # Calculating histogram
    if not hist:
        hist = np.histogram(image, bins=range(256))[0].astype(np.float)

    # Cumulative histograms
    c_hist = np.cumsum(hist)
    cdf = np.cumsum(np.arange(len(hist)) * hist)

    return _get_thresholds(hist, c_hist, cdf, nthrs)
