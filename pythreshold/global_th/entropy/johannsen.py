# -*- coding:utf-8 -*-

import numpy as np

__copyright__ = 'Copyright 2017'
__author__ = u'BSc. Manuel Aguado Martínez'


def johannsen_threshold(image):
    """ Runs the Johannsen's threshold algorithm.

    Reference:
    Johannsen, G., and J. Bille ‘‘A Threshold Selection Method Using
    Information Measures,’’ Proceedings of the Sixth International Conference
    on Pattern Recognition, Munich, Germany (1982): 140–143.

    @param image: The input image
    @type image: ndarray

    @return: The estimated threshold
    @rtype: int
    """
    hist, _ = np.histogram(image, bins=range(256), density=True)

    # Knowing the number of leading zeros
    l_zeros = 0
    while hist[l_zeros] == 0:
        l_zeros += 1

    # Truncating histogram to ignore leading and trailing zeros
    hist = np.trim_zeros(hist)

    c_hist = hist.cumsum()
    ic_hist = 1.0 - c_hist

    # To avoid 0 invalid operations
    c_hist[c_hist <= 0] = 1
    hist[hist <= 0] = 1
    ic_hist[ic_hist <= 0] = 1

    # Obtaining shifted cumulative histograms
    sc_hist = np.ones_like(c_hist)
    sc_hist[1:] = c_hist[:-1]

    si_chist = np.ones_like(c_hist)
    si_chist[1:] = ic_hist[:-1]

    # Obtaining histogram entropy
    h_entropy = -hist * np.log(hist)

    # Background entropy
    b_entropy = h_entropy - sc_hist * np.log(sc_hist)
    s_backg = np.log(c_hist) + b_entropy / c_hist

    # Foreground entropy
    f_entropy = h_entropy - ic_hist * np.log(ic_hist)
    s_foreg = np.log(si_chist) + f_entropy / si_chist

    return np.argmin((s_foreg + s_backg)[1:-1]) + 1 + l_zeros
