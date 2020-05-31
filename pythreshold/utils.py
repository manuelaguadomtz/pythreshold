# -*- coding:utf-8 -*-

from timeit import default_timer
from os.path import join

import numpy as np
import matplotlib.pyplot as plt
import cv2

from scipy.misc import face

# Importing global thresholding algorithms
from .global_th import (
    otsu_threshold,
    otsu_multithreshold,
    p_tile_threshold,
    two_peaks_threshold,
    min_err_threshold
)

# Importing global entropy thresholding algorithms
from .global_th.entropy import (
    pun_threshold,
    kapur_threshold,
    johannsen_threshold,
    kapur_multithreshold
)

# Importing local thresholding algorithms
from .local_th import (
    sauvola_threshold,
    niblack_threshold,
    wolf_threshold,
    nick_threshold,
    lmean_threshold,
    bradley_roth_threshold,
    bernsen_threshold,
    contrast_threshold,
    singh_threshold,
    feng_threshold
)


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


def apply_multithreshold(img, thresholds):
    """Obtain a binary image based on a given global threshold or
    a set of local thresholds.

    @param img: The input image.
    @type img: ndarray
    @param thresholds: Global multi-thresholds.
    @type threshold: iterable

    @return: The thresholded image.
    @rtype: ndarray
    """
    # Extending entropy and thresholds for convenience
    e_thresholds = [-1]
    e_thresholds.extend(thresholds)

    # Threshold image
    t_image = np.zeros_like(img)

    for i in range(1, len(e_thresholds)):
        t_image[img >= e_thresholds[i]] = i

    wp_val = 255 // len(thresholds)

    return t_image * wp_val


def test_thresholds_plt(img=None):
    """Runs all the package thresholding algorithms on the input
    image with default parameters and plot the results.

    @param img: The input gray scale image
    @type img: ndarray
    """
    # Loading image if needed
    if img is None:
        img = face(gray=True)

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
    print('')

    # Plotting results
    plt.figure('Otsu method')
    plt.imshow(apply_threshold(img, th), cmap='gray')

    # Applying Otsu multi-threshold method
    start = default_timer()
    th = otsu_multithreshold(img, nthrs=2)
    stop = default_timer()
    print('========Otsu multi-threshold==========')
    print('Thresholds: {0}'.format(th))
    print('Execution time: {0}'.format(stop - start))
    print('')

    # Plotting results
    plt.figure('Otsu multi-threshold method')
    plt.imshow(apply_multithreshold(img, th), cmap='gray')

    # Applying p_tile method
    start = default_timer()
    th = p_tile_threshold(img, 0.5)
    stop = default_timer()
    print('========P-tile [p=0.5]==========')
    print('Threshold: {0}'.format(th))
    print('Execution time: {0}'.format(stop - start))
    print('')

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
    print('')

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
    print('')

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
    print('')

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
    print('')

    # Plotting results
    plt.figure('Global entropy Kapur method')
    plt.imshow(apply_threshold(img, th), cmap='gray')

    # Applying global entropy Kapur multi-trehshold method
    start = default_timer()
    th = kapur_multithreshold(img, 2)
    stop = default_timer()
    print('========Global entropy Kapur multi-threshold==========')
    print('Threshold: {0}'.format(th))
    print('Execution time: {0}'.format(stop - start))
    print('')

    # Plotting results
    plt.figure('Global entropy Kapur multi-threshold method')
    plt.imshow(apply_multithreshold(img, th), cmap='gray')

    # Applying global entropy Johannsen method
    start = default_timer()
    th = johannsen_threshold(img)
    stop = default_timer()
    print('========Global entropy Johannsen==========')
    print('Threshold: {0}'.format(th))
    print('Execution time: {0}'.format(stop - start))
    print('')

    # Plotting results
    plt.figure('Global entropy Johannsen method')
    plt.imshow(apply_threshold(img, th), cmap='gray')

    # Applying local Sauvola method
    start = default_timer()
    th = sauvola_threshold(img)
    stop = default_timer()
    print('========Local Sauvola==========')
    print('Execution time: {0}'.format(stop - start))
    print('')

    # Plotting results
    plt.figure('Local Sauvola method')
    plt.imshow(apply_threshold(img, th), cmap='gray')

    # Applying local Niblack method
    start = default_timer()
    th = niblack_threshold(img)
    stop = default_timer()
    print('========Local Niblack==========')
    print('Execution time: {0}'.format(stop - start))
    print('')

    # Plotting results
    plt.figure('Local Niblack method')
    plt.imshow(apply_threshold(img, th), cmap='gray')

    # Applying local Wolf method
    start = default_timer()
    th = wolf_threshold(img)
    stop = default_timer()
    print('========Local Wolf==========')
    print('Execution time: {0}'.format(stop - start))
    print('')

    # Plotting results
    plt.figure('Local Wolf method')
    plt.imshow(apply_threshold(img, th), cmap='gray')

    # Applying local NICK method
    start = default_timer()
    th = nick_threshold(img)
    stop = default_timer()
    print('========Local NICK==========')
    print('Execution time: {0}'.format(stop - start))
    print('')

    # Plotting results
    plt.figure('Local NICK method')
    plt.imshow(apply_threshold(img, th), cmap='gray')

    # Applying local mean method
    start = default_timer()
    th = lmean_threshold(img)
    stop = default_timer()
    print('========Local mean==========')
    print('Execution time: {0}'.format(stop - start))
    print('')

    # Plotting results
    plt.figure('Local mean method')
    plt.imshow(apply_threshold(img, th), cmap='gray')

    # Applying local Bradley-Roth method
    start = default_timer()
    th = bradley_roth_threshold(img)
    stop = default_timer()
    print('========Local Bradley-Roth==========')
    print('Execution time: {0}'.format(stop - start))
    print('')

    # Plotting results
    plt.figure('Local Bradley-Roth method')
    plt.imshow(apply_threshold(img, th), cmap='gray')

    # Applying local Bernsen method
    start = default_timer()
    th = bernsen_threshold(img)
    stop = default_timer()
    print('========Local Bernsen==========')
    print('Execution time: {0}'.format(stop - start))
    print('')

    # Plotting results
    plt.figure('Local Bernsen method')
    plt.imshow(apply_threshold(img, th), cmap='gray')

    # Applying local contrast method
    start = default_timer()
    th = contrast_threshold(img)
    stop = default_timer()
    print('========Local contrast==========')
    print('Execution time: {0}'.format(stop - start))
    print('')

    # Plotting results
    plt.figure('Local contrast method')
    plt.imshow(apply_threshold(img, th), cmap='gray')

    # Applying local Singh method
    start = default_timer()
    th = singh_threshold(img)
    stop = default_timer()
    print('========Local Singh==========')
    print('Execution time: {0}'.format(stop - start))
    print('')

    # Plotting results
    plt.figure('Local Singh method')
    plt.imshow(apply_threshold(img, th), cmap='gray')

    # Applying local Feng method
    start = default_timer()
    th = feng_threshold(img)
    stop = default_timer()
    print('========Local Feng==========')
    print('Execution time: {0}'.format(stop - start))
    print('')

    # Plotting results
    plt.figure('Local Feng method')
    plt.imshow(apply_threshold(img, th), cmap='gray')

    # Showing plots
    plt.show()


def test_thresholds(img, odir, basename):
    """Runs all the package thresholding algorithms on the input
    image with default parameters and plot the results.

    @param img: The input gray scale image
    @type img: ndarray
    """
    # Applying Otsu method
    start = default_timer()
    th = otsu_threshold(img)
    stop = default_timer()
    print('========Otsu==========')
    print('Threshold: {0}'.format(th))
    print('Execution time: {0}'.format(stop - start))
    print('')
    fname = join(odir, "%s_Otsu.jpg" % basename)
    cv2.imwrite(fname, apply_threshold(img, th))

    # Applying Otsu multithreshold method
    start = default_timer()
    th = otsu_multithreshold(img, nthrs=2)
    stop = default_timer()
    print('========Otsu Multithreshold==========')
    print('Threshold: {0}'.format(th))
    print('Execution time: {0}'.format(stop - start))
    print('')
    fname = join(odir, "%s_OtsuMultiTh.jpg" % basename)
    cv2.imwrite(fname, apply_multithreshold(img, th))

    # Applying p_tile method
    start = default_timer()
    th = p_tile_threshold(img, 0.5)
    stop = default_timer()
    print('========P-tile [p=0.5]==========')
    print('Threshold: {0}'.format(th))
    print('Execution time: {0}'.format(stop - start))
    print('')
    fname = join(odir, "%s_p_tile.jpg" % basename)
    cv2.imwrite(fname, apply_threshold(img, th))

    # Applying two peaks method
    start = default_timer()
    th = two_peaks_threshold(img)
    stop = default_timer()
    print('========Two peaks==========')
    print('Threshold: {0}'.format(th))
    print('Execution time: {0}'.format(stop - start))
    print('')
    fname = join(odir, "%s_2peaks.jpg" % basename)
    cv2.imwrite(fname, apply_threshold(img, th))

    # Applying minimum error method
    start = default_timer()
    th = min_err_threshold(img)
    stop = default_timer()
    print('========Minimum Error==========')
    print('Threshold: {0}'.format(th))
    print('Execution time: {0}'.format(stop - start))
    print('')
    fname = join(odir, "%s_minError.jpg" % basename)
    cv2.imwrite(fname, apply_threshold(img, th))

    # Applying global entropy Pun method
    start = default_timer()
    th = pun_threshold(img)
    stop = default_timer()
    print('========Global entropy Pun==========')
    print('Threshold: {0}'.format(th))
    print('Execution time: {0}'.format(stop - start))
    print('')
    fname = join(odir, "%s_entropyPun.jpg" % basename)
    cv2.imwrite(fname, apply_threshold(img, th))

    # Applying global entropy Kapur method
    start = default_timer()
    th = kapur_threshold(img)
    stop = default_timer()
    print('========Global entropy Kapur==========')
    print('Threshold: {0}'.format(th))
    print('Execution time: {0}'.format(stop - start))
    print('')
    fname = join(odir, "%s_entropyKapur.jpg" % basename)
    cv2.imwrite(fname, apply_threshold(img, th))

    # Applying global entropy Kapur multi-trehshold method
    start = default_timer()
    th = kapur_multithreshold(img, 2)
    stop = default_timer()
    print('========Global entropy Kapur multi-trehshold==========')
    print('Threshold: {0}'.format(th))
    print('Execution time: {0}'.format(stop - start))
    print('')
    fname = join(odir, "%s_entropyKapurMultiTh.jpg" % basename)
    cv2.imwrite(fname, apply_multithreshold(img, th))

    # Applying global entropy Johannsen method
    start = default_timer()
    th = johannsen_threshold(img)
    stop = default_timer()
    print('========Global entropy Johannsen==========')
    print('Threshold: {0}'.format(th))
    print('Execution time: {0}'.format(stop - start))
    print('')
    fname = join(odir, "%s_entropyJohannsen.jpg" % basename)
    cv2.imwrite(fname, apply_threshold(img, th))

    # Applying local Sauvola method
    start = default_timer()
    th = sauvola_threshold(img)
    stop = default_timer()
    print('========Local Sauvola==========')
    print('Execution time: {0}'.format(stop - start))
    print('')
    fname = join(odir, "%s_sauvola.jpg" % basename)
    cv2.imwrite(fname, apply_threshold(img, th))

    # Applying local Niblack method
    start = default_timer()
    th = niblack_threshold(img)
    stop = default_timer()
    print('========Local Niblack==========')
    print('Execution time: {0}'.format(stop - start))
    print('')
    fname = join(odir, "%s_niblack.jpg" % basename)
    cv2.imwrite(fname, apply_threshold(img, th))

    # Applying local Wolf method
    start = default_timer()
    th = wolf_threshold(img)
    stop = default_timer()
    print('========Local Wolf==========')
    print('Execution time: {0}'.format(stop - start))
    print('')
    fname = join(odir, "%s_wolf.jpg" % basename)
    cv2.imwrite(fname, apply_threshold(img, th))

    # Applying local NICK method
    start = default_timer()
    th = nick_threshold(img)
    stop = default_timer()
    print('========Local NICK==========')
    print('Execution time: {0}'.format(stop - start))
    print('')
    fname = join(odir, "%s_nick.jpg" % basename)
    cv2.imwrite(fname, apply_threshold(img, th))

    # Applying local mean method
    start = default_timer()
    th = lmean_threshold(img)
    stop = default_timer()
    print('========Local mean==========')
    print('Execution time: {0}'.format(stop - start))
    print('')
    fname = join(odir, "%s_localMean.jpg" % basename)
    cv2.imwrite(fname, apply_threshold(img, th))

    # Applying local Bradley-Roth method
    start = default_timer()
    th = bradley_roth_threshold(img)
    stop = default_timer()
    print('========Local Bradley-Roth==========')
    print('Execution time: {0}'.format(stop - start))
    print('')
    fname = join(odir, "%s_bradleyRoth.jpg" % basename)
    cv2.imwrite(fname, apply_threshold(img, th))

    # Applying local Bernsen method
    start = default_timer()
    th = bernsen_threshold(img)
    stop = default_timer()
    print('========Local Bernsen==========')
    print('Execution time: {0}'.format(stop - start))
    print('')
    fname = join(odir, "%s_bernsen.jpg" % basename)
    cv2.imwrite(fname, apply_threshold(img, th))

    # Applying local contrast method
    start = default_timer()
    th = contrast_threshold(img)
    stop = default_timer()
    print('========Local contrast==========')
    print('Execution time: {0}'.format(stop - start))
    print('')
    fname = join(odir, "%s_localContrast.jpg" % basename)
    cv2.imwrite(fname, apply_threshold(img, th))

    # Applying local Singh method
    start = default_timer()
    th = singh_threshold(img)
    stop = default_timer()
    print('========Local Singh==========')
    print('Execution time: {0}'.format(stop - start))
    print('')
    fname = join(odir, "%s_singh.jpg" % basename)
    cv2.imwrite(fname, apply_threshold(img, th))

    # Applying local Feng method
    start = default_timer()
    th = feng_threshold(img)
    stop = default_timer()
    print('========Local Feng==========')
    print('Execution time: {0}'.format(stop - start))
    print('')
    fname = join(odir, "%s_feng.jpg" % basename)
    cv2.imwrite(fname, apply_threshold(img, th))
