# -*- coding:utf-8 -*-

from timeit import default_timer

import numpy as np
import matplotlib.pyplot as plt

from scipy.misc import face

# Importing global thresholding algorithms
from .global_th import otsu_threshold, p_tile_threshold,\
    two_peaks_threshold, min_err_threshold

# Importing global entropy thresholding algorithms
from .global_th.entropy import pun_threshold, kapur_threshold,\
    johannsen_threshold

# Importing local thresholding algorithms
from .local_th import sauvola_threshold, niblack_threshold, wolf_threshold,\
    nick_threshold, lmean_threshold, bradley_roth_threshold,\
    bernsen_threshold, contrast_threshold, singh_threshold, feng_threshold


__copyright__ = 'Copyright 2017'
__author__ = u'BSc. Manuel Aguado Martínez'


def apply_threshold(img, threshold=128, wp_val=255):
    """Obtain a binary image based on a given global threshold or
    a set of local thresholds.

    @param img: The input image.
    @type img: ndarray
    @param threshold: The global or local thresholds corresponding
        to each pixel of the image.
    @type threshold: Union[int, ndarray]
    @param wp_val: The value assigned to foreground pixels (white pixels).
    @type wp_val: int

    @return: A binary image.
    @rtype: ndarray
    """
    return ((img >= threshold) * wp_val).astype(np.uint8)


def test_thresholds(img=None):
    """Runs all the package thresholding algorithms on the input
    image with default parameters and plot the results.

    @param img: The input gray scale image
    @type img: ndarray
    """
    # Loading image if needed
    if img is None:
        img = face(gray=True)

    # Plotting test image
    plt.figure('image')
    plt.imshow(img, cmap='gray')

    # Plotting test image histogram
    plt.figure('Histogram')
    plt.hist(img.ravel(), range=(0, 255), bins=255)

    # Applying Otsu method
    start = default_timer()
    th = otsu_threshold(img)
    stop = default_timer()
    print('========Otsu==========')
    print('Threshold: {0}'.format(th))
    print('Execution time: {0}'.format(stop - start))
    print('====================================')

    # Plotting results
    plt.figure('Otsu method')
    plt.imshow(apply_threshold(img, th), cmap='gray')

    # Applying p_tile method
    start = default_timer()
    th = p_tile_threshold(img, 0.5)
    stop = default_timer()
    print('========P-tile [p=0.5]==========')
    print('Threshold: {0}'.format(th))
    print('Execution time: {0}'.format(stop - start))
    print('====================================')

    # Plotting results
    plt.figure('p_tile method [pct=0.5]')
    plt.imshow(apply_threshold(img, th), cmap='gray')

    # Applying two peaks method
    start = default_timer()
    th = two_peaks_threshold(img)
    stop = default_timer()
    print('========Two peaks==========')
    print('Threshold: {0}'.format(th))
    print('Execution time: {0}'.format(stop - start))
    print('====================================')

    # Plotting results
    plt.figure('Tow peaks method')
    plt.imshow(apply_threshold(img, th), cmap='gray')

    # Applying minimum error method
    start = default_timer()
    th = min_err_threshold(img)
    stop = default_timer()
    print('========Minimum Error==========')
    print('Threshold: {0}'.format(th))
    print('Execution time: {0}'.format(stop - start))
    print('====================================')

    # Plotting results
    plt.figure('Minimum error method')
    plt.imshow(apply_threshold(img, th), cmap='gray')

    # Applying global entropy Pun method
    start = default_timer()
    th = pun_threshold(img)
    stop = default_timer()
    print('========Global entropy Pun==========')
    print('Threshold: {0}'.format(th))
    print('Execution time: {0}'.format(stop - start))
    print('====================================')

    # Plotting results
    plt.figure('Global entropy Pun method')
    plt.imshow(apply_threshold(img, th), cmap='gray')

    # Applying global entropy Kapur method
    start = default_timer()
    th = kapur_threshold(img)
    stop = default_timer()
    print('========Global entropy Kapur==========')
    print('Threshold: {0}'.format(th))
    print('Execution time: {0}'.format(stop - start))
    print('====================================')

    # Plotting results
    plt.figure('Global entropy Kapur method')
    plt.imshow(apply_threshold(img, th), cmap='gray')

    # Applying global entropy Johannsen method
    start = default_timer()
    th = johannsen_threshold(img)
    stop = default_timer()
    print('========Global entropy Johannsen==========')
    print('Threshold: {0}'.format(th))
    print('Execution time: {0}'.format(stop - start))
    print('====================================')

    # Plotting results
    plt.figure('Global entropy Johannsen method')
    plt.imshow(apply_threshold(img, th), cmap='gray')

    # Applying local Sauvola method
    start = default_timer()
    th = sauvola_threshold(img)
    stop = default_timer()
    print('========Local Sauvola==========')
    print('Execution time: {0}'.format(stop - start))
    print('====================================')

    # Plotting results
    plt.figure('Local Sauvola method')
    plt.imshow(apply_threshold(img, th), cmap='gray')

    # Applying local Niblack method
    start = default_timer()
    th = niblack_threshold(img)
    stop = default_timer()
    print('========Local Niblack==========')
    print('Execution time: {0}'.format(stop - start))
    print('====================================')

    # Plotting results
    plt.figure('Local Niblack method')
    plt.imshow(apply_threshold(img, th), cmap='gray')

    # Applying local Wolf method
    start = default_timer()
    th = wolf_threshold(img)
    stop = default_timer()
    print('========Local Wolf==========')
    print('Execution time: {0}'.format(stop - start))
    print('====================================')

    # Plotting results
    plt.figure('Local Wolf method')
    plt.imshow(apply_threshold(img, th), cmap='gray')

    # Applying local NICK method
    start = default_timer()
    th = nick_threshold(img)
    stop = default_timer()
    print('========Local NICK==========')
    print('Execution time: {0}'.format(stop - start))
    print('====================================')

    # Plotting results
    plt.figure('Local NICK method')
    plt.imshow(apply_threshold(img, th), cmap='gray')

    # Applying local mean method
    start = default_timer()
    th = lmean_threshold(img)
    stop = default_timer()
    print('========Local mean==========')
    print('Execution time: {0}'.format(stop - start))
    print('====================================')

    # Plotting results
    plt.figure('Local mean method')
    plt.imshow(apply_threshold(img, th), cmap='gray')

    # Applying local Bradley-Roth method
    start = default_timer()
    th = bradley_roth_threshold(img)
    stop = default_timer()
    print('========Local Bradley-Roth==========')
    print('Execution time: {0}'.format(stop - start))
    print('====================================')

    # Plotting results
    plt.figure('Local Bradley-Roth method')
    plt.imshow(apply_threshold(img, th), cmap='gray')

    # Applying local Bernsen method
    start = default_timer()
    th = bernsen_threshold(img)
    stop = default_timer()
    print('========Local Bernsen==========')
    print('Execution time: {0}'.format(stop - start))
    print('====================================')

    # Plotting results
    plt.figure('Local Bernsen method')
    plt.imshow(apply_threshold(img, th), cmap='gray')

    # Applying local contrast method
    start = default_timer()
    th = contrast_threshold(img)
    stop = default_timer()
    print('========Local contrast==========')
    print('Execution time: {0}'.format(stop - start))
    print('====================================')

    # Plotting results
    plt.figure('Local contrast method')
    plt.imshow(apply_threshold(img, th), cmap='gray')

    # Applying local Singh method
    start = default_timer()
    th = singh_threshold(img)
    stop = default_timer()
    print('========Local Singh==========')
    print('Execution time: {0}'.format(stop - start))
    print('====================================')

    # Plotting results
    plt.figure('Local Singh method')
    plt.imshow(apply_threshold(img, th), cmap='gray')

    # Applying local Feng method
    start = default_timer()
    th = feng_threshold(img)
    stop = default_timer()
    print('========Local Feng==========')
    print('Execution time: {0}'.format(stop - start))
    print('====================================')

    # Plotting results
    plt.figure('Local Feng method')
    plt.imshow(apply_threshold(img, th), cmap='gray')

    # Showing plots
    plt.show()
