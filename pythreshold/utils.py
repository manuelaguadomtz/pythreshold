# -*- coding:utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

# Importing thresholding algorithms
from global_th.otsu import otsu_threshold
from global_th.p_tile import p_tile_threshold
from global_th.two_peaks import two_peaks_threshold
from global_th.min_err import min_err_threshold
from global_th.entropy.pun import pun_threshold
from global_th.entropy.kapur import kapur_threshold
from global_th.entropy.johannsen import johannsen_threshold

from local_th.sauvola import sauvola_threshold
from local_th.niblack import niblack_threshold
from local_th.wolf import wolf_threshold
from local_th.nick import nick_threshold
from local_th.lmean import lmean_threshold
from local_th.bradley_roth import bradley_roth_threshold
from local_th.bernsen import bernsen_threshold
from local_th.contrast import contrast_threshold
from local_th.singh import singh_threshold
from local_th.feng import feng_threshold

__copyright__ = 'Copyright 2017'
__author__ = u'Lic. Manuel Aguado Martínez'


def apply_threshold(img, threshold=128, wp_val=255):
    """Obtain a binary image based on a given global threshold or
    a set of local thresholds.
    
    @param img: The input image.
    @type img: ndarray
    @param threshold: The global or local thresholds corresponding 
        to each pixel of the image.
    @type threshold: Union[int, ndarray]
    @param wp_val: The value assigned to foreground pixels (white pixels).
    @type: int
    
    @return: A binary image.
    @rtype: ndarray
    """
    return ((img >= threshold) * wp_val).astype(np.uint8)


def test_thresholds(img):
    # Plotting test image
    plt.figure('image')
    plt.imshow(img, cmap='gray')

    # Plotting test image histogram
    plt.figure('Histogram')
    plt.hist(img.ravel(), range=(0, 255), bins=255)

    # Applying Otsu method
    th = otsu_threshold(img)
    plt.figure('Otsu method')
    plt.imshow(apply_threshold(img, th), cmap='gray')

    # Applying p_tile method
    th = p_tile_threshold(img, 0.5)
    plt.figure('p_tile method [pct=0.5]')
    plt.imshow(apply_threshold(img, th), cmap='gray')

    # Applying two peaks method
    th = two_peaks_threshold(img)
    plt.figure('Tow peaks method')
    plt.imshow(apply_threshold(img, th), cmap='gray')

    # Applying minimum error method
    th = min_err_threshold(img)
    plt.figure('Minimum error method')
    plt.imshow(apply_threshold(img, th), cmap='gray')

    # Applying global entropy Pun method
    th = pun_threshold(img)
    plt.figure('Global entropy Pun method')
    plt.imshow(apply_threshold(img, th), cmap='gray')

    # Applying global entropy Kapur method
    th = kapur_threshold(img)
    plt.figure('Global entropy Kapur method')
    plt.imshow(apply_threshold(img, th), cmap='gray')

    # Applying global entropy Johannsen method
    th = johannsen_threshold(img)
    plt.figure('Global entropy Johannsen method')
    plt.imshow(apply_threshold(img, th), cmap='gray')

    # Applying local Sauvola method
    th = sauvola_threshold(img)
    plt.figure('Local Sauvola method')
    plt.imshow(apply_threshold(img, th), cmap='gray')

    # Applying local Niblack method
    th = niblack_threshold(img)
    plt.figure('Local Niblack method')
    plt.imshow(apply_threshold(img, th), cmap='gray')

    # Applying local Wolf method
    th = wolf_threshold(img)
    plt.figure('Local Wolf method')
    plt.imshow(apply_threshold(img, th), cmap='gray')

    # Applying local NICK method
    th = nick_threshold(img)
    plt.figure('Local NICK method')
    plt.imshow(apply_threshold(img, th), cmap='gray')

    # Applying local mean method
    th = lmean_threshold(img)
    plt.figure('Local mean method')
    plt.imshow(apply_threshold(img, th), cmap='gray')

    # Applying local Bradley-Roth method
    th = bradley_roth_threshold(img)
    plt.figure('Local Bradley-Roth method')
    plt.imshow(apply_threshold(img, th), cmap='gray')

    # Applying local Bernsen method
    th = bernsen_threshold(img)
    plt.figure('Local Bernsen method')
    plt.imshow(apply_threshold(img, th), cmap='gray')

    # Applying local contrast method
    th = contrast_threshold(img)
    plt.figure('Local contrast method')
    plt.imshow(apply_threshold(img, th), cmap='gray')

    # Applying local Singh method
    th = singh_threshold(img)
    plt.figure('Local Singh method')
    plt.imshow(apply_threshold(img, th), cmap='gray')

    # Applying local Feng method
    th = feng_threshold(img)
    plt.figure('Local Feng method')
    plt.imshow(apply_threshold(img, th), cmap='gray')

    # Showing plots
    plt.show()
