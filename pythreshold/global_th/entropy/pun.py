# -*- coding:utf-8 -*-

import numpy as np

__copyright__ = 'Copyright 2017'
__author__ = u'BSc. Manuel Aguado Martínez'


def pun_threshold(image):
    """ Runs the Pun's threshold algorithm.

    Reference:
    Pun, T. ‘‘A New Method for Grey-Level Picture Thresholding Using the
    Entropy of the Histogram,’’ Signal Processing 2, no. 3 (1980): 223–237.

    @param image: The input image
    @type image: ndarray

    @return: The estimated threshold
    @rtype: int
    """
    # Calculating histogram
    hist, _ = np.histogram(image, bins=range(256), density=True)

    # Calculating histogram cumulative sum
    hcs = np.cumsum(hist)
    hcs[hcs <= 0] = 1  # To avoid log invalid calculations

    # Calculating inverted histogram cumulative sum
    i_hcs = 1.0 - hcs
    i_hcs[i_hcs <= 0] = 1  # To avoid log invalid calculations

    # Calculating normed entropy cumulative sum
    ecs_norm = np.cumsum(hist * np.log(hist + (hist <= 0)))
    ecs_norm /= ecs_norm[-1]

    max_entropy = 0
    threshold = 0
    for t in range(len(hist) - 1):
        black_max = np.max(hist[:t + 1])
        white_max = np.max(hist[t + 1:])
        if black_max * white_max != 0:
            x = ecs_norm[t] * np.log(hcs[t]) / np.log(black_max)
            y = 1.0 - ecs_norm[t]
            z = np.log(i_hcs[t]) / np.log(white_max)
            entropy = x + y * z

            if max_entropy < entropy:
                max_entropy = entropy
                threshold = t

    return threshold
