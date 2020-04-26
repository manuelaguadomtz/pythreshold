# -*- coding:utf-8 -*-

import numpy as np

__copyright__ = 'Copyright 2017'
__author__ = u'BSc. Manuel Aguado Martínez'


def p_tile_threshold(image, pct):
    """Runs the p-tile threshold algorithm.

    Reference:
    Parker, J. R. (2010). Algorithms for image processing and
    computer vision. John Wiley & Sons.

    @param image: The input image
    @type image: ndarray
    @param pct: The percent of desired background pixels (black pixels).
        It must lie in the interval [0, 1]
    @type pct: float

    @return: The p-tile global threshold
    @rtype int
    """
    n_pixels = pct * image.shape[0] * image.shape[1]
    hist = np.histogram(image, bins=range(256))[0]
    hist = np.cumsum(hist)

    return np.argmin(np.abs(hist - n_pixels))
