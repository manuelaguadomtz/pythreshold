#!/usr/bin/env python
# -*- coding:utf-8 -*-

import numpy as np

__copyright__ = 'Copyright 2014'
__author__ = u'Lic. Manuel Aguado Martínez'


def apply_threshold(img, threshold=128, wp_val=255):
    """Obtain a binary image based on a given global threshold or
    a set of local thresholds.
    
    :param img: The input image.
    :type img: numpy.ndarray
    :param threshold: The global or local thresholds corresponding 
        to each pixel of the image.
    :type threshold: Union[int, numpy.ndarray]
    :param wp_val: The value assigned to foreground pixels (white pixels).
    :type: int
    
    :return: A binary image.
    :rtype: numpy.ndarray
    """
    return ((img >= threshold) * wp_val).astype(np.uint8)
