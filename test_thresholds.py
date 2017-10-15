# -*- coding:utf-8 -*-

import argparse

from timeit import default_timer

import matplotlib.pyplot as plt

from scipy.misc import face, imread

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

from utils import apply_threshold

__copyright__ = 'Copyright 2017'
__author__ = u'Lic. Manuel Aguado Martínez'


def threshold_and_plot(img):
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


if __name__ == '__main__':
    # Creating argument parser
    arg_p = argparse.ArgumentParser()
    arg_p.add_argument('-i', '--image', required=False, help='Input image')

    # Parsing arguments
    args = arg_p.parse_args()

    # Loading image
    if args.image:
        input_img = imread(args.image, flatten=True)
    else:
        input_img = face(gray=True)

    # Thresholding and plotting
    threshold_and_plot(input_img)
