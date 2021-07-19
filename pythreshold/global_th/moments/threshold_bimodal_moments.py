# -*- coding: utf-8 -*-
"""
Created on Thu Jul 23 18:15:14 2020

@author: zz19101
"""

import numpy as np
from PIL import Image
import os 
import shutil
import pandas as pd
import math
import gc


def threshold_bimodal_moments(stack_path):
    """
     
    This function calculates the threshold of a stack based on keeping the
    moments of the image before and after binarization.
    IMPORTANT: It provides the threshold for a bimodal histogram (foreground and background)
    See: 'Moment-Preserving Thresholding: A New Approach (Tsai)'
    This code has been inspired by the code written by G. Landini for the Fiji Plugin
    'https://github.com/fiji/Auto_Threshold/blob/master/src/main/java/fiji/threshold/Auto_Threshold.java'

    Parameters
    ----------
    stack_path : str
        Folder path where the set of 2D images (making up a 3D image like the
        ones resulting from a CT Scan) are stored. Images can be whatever usual format: PNG, TIFF, JPEG...

    Returns
    -------
    
    threshold : integer
        threshold resulting from the application of the moments conservation principle.
        
    dict_val : dictionary
        threshold resulting from the application of the moments conservation principle.
    
    """
    
    gc.collect()
    slices_list = sorted(os.listdir(stack_path)) # List of images in the folder
    
    #Define the histogram
    histogram = np.zeros((256,4))  # We generate the counts for each grey value, 3rd column probabilities and 4th column cumulative sum
    histogram[:,0] = np.arange(256)  # La primera columna contiene los valores de intensidad de los pixels
    
    # Convert the stack of 2D images into a 3D numpy array
    for x in slices_list:
        
        slice_path = os.path.join(stack_path, x)
        
        with Image.open(slice_path) as img:
            img = img.convert('L')
            img_array = np.array(img)
        
        hist_i = np.histogram(img_array, bins=256)
        counts = hist_i[0]
        histogram[:,1] = np.add(histogram[:,1], counts)
   
    row, col = img_array.shape
    nb_img = len(slices_list)
    total_voxels = row * col * nb_img
    
    histogram[:,2] = histogram[:,1]/total_voxels  # Relative frequency at each GV
    histogram[:,3] = np.cumsum(histogram[:,2])  # Cumulative of the relative frequency
    # 0 Moment = 1
    m0 = 1
    # 1st Moment 
    m1 = np.cumsum((histogram[:,0])*histogram[:,2])[-1]    
    
    # 2nd Moment
    m2 = np.cumsum((histogram[:,0]**2)*histogram[:,2])[-1]  # Take the last value
    
    # 3rd Moment
    m3 = np.cumsum((histogram[:,0]**3)*histogram[:,2])[-1]  # Take the last value
    
    # Now we must find the value in the binary image that preserves these moments
    # Min(abs(m2-m2*)) and Min(abs(min-min3*))
    
    # We solve the equalities --> For more info: https://github.com/pedrogalher/Auto_Threshold/blob/master/src/main/java/fiji/threshold/Auto_Threshold.java
    cd = m0 * m2 - m1 * m1
    c0 = (-m2*m2+m1*m3) / cd
    c1 = (m0 * (-m3) + m2 * m1) / cd
    z0 = 0.5 * (-c1 - math.sqrt(c1**2 - 4.0 * c0))
    z1 = 0.5 * (-c1 + math.sqrt(c1**2 - 4.0 * c0))
    p0 = (z1 - m1) / (z1 - z0)  # Fraction of the below-threshold pixels in the binary histogram
    dict_val = {
                'z0':z0,
                'z1':z1,
                'p0':p0,
                'p1':p0,
                'm0': m0,
                'm1': m1,
                'm2': m2,
                'm3': m3,
                'histogram':histogram,
                }
    threshold = 0
    dist = 10000000
    
    for i in range(256):
        
        # Select threshold --> closest to p0 from the normalized histogram
        print(i)
        p0_orig = histogram[i,3]  # Take the cumulative relative frequency
        dist_i = abs(p0-p0_orig)
        
        if p0_orig>p0:  # This one was used in the code written by G. Landini
            threshold = i
            break
        
        #if dist_i <= dist:  # This one was the one mentioned by Tsai ("Minimize the distance")
         #   print(dist_i)
          #  dist = dist_i
           # threshold = i
        
    return threshold, dict_val


if __name__ == "__main__":
    
    stack_path =  "\path.."
    
    threshold, dict_val = threshold_bimodal_moments(stack_path)
    
    # The histogram calculation has been double-checked with Fiji
        
        