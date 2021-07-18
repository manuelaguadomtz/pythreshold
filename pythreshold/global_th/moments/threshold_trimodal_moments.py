# -*- coding: utf-8 -*-
"""
Created on Thu Jul 23 18:15:14 2020

@author: zz19101
"""

import numpy as np
from PIL import Image
import os 
import shutil
import math
import gc
import cmath

def threshold_3d_multi_moments(stack_path):
    """ This function calculates the threshold of a stack based on keeping the
    moments of the image before and after binarization.
    See: Moment-Preserving Thresholding: A New Approach (Tsai)
    
    """
    gc.collect()
    slices_list = sorted(os.listdir(stack_path)) # 
    
    #Define the histogram
    bins_hist = list(range(0,257)) # Queremos llegar a tener 256 "papeleras", por eso tenemos que poner 257, para que haga de 0 hasta 256
    histogram = np.zeros((256,4))  # We generate the counts for each grey value, 3rd column probabilities and 4th column cumulative sum
    histogram[:,0] = np.arange(256)  # La primera columna contiene los valores de intensidad de los pixels
    
    for x in slices_list:
        
        slice_path = os.path.join(stack_path, x)
        
        with Image.open(slice_path) as img:
            img = img.convert('L')
            img_array = np.array(Image.open(slice_path).convert('L'))
        hist_i = np.histogram(img_array, bins=bins_hist)
        counts = hist_i[0]
        histogram[:,1] = np.add(histogram[:,1], counts)
   
    row, col = img_array.shape
    nb_img = len(slices_list)
    total_voxels = row * col * nb_img
    
    histogram[:,2] = histogram[:,1]/total_voxels  # Relative frequency at each GV
    histogram[:,3] = np.cumsum(histogram[:,2])  # Cumulative of the relative frequency
    
    
    ###  THRESHOLDS CALCULATION BEGIN ###
    
    # 0 Moment = 1
    m0 = 1.0
    
    # 1st Moment 
    m1 = np.cumsum((histogram[:,0])*histogram[:,2])[-1]    
    
    # 2nd Moment
    m2 = np.cumsum((histogram[:,0]**2)*histogram[:,2])[-1]  # Take the last value
    
    # 3rd Moment
    m3 = np.cumsum((histogram[:,0]**3)*histogram[:,2])[-1]  # Take the last value
    
    # 4th Moment
    m4 = np.cumsum((histogram[:,0]**4)*histogram[:,2])[-1]  # Take the last value
    
    # 5th Moment
    m5 = np.cumsum((histogram[:,0]**5)*histogram[:,2])[-1]  # Take the last value
    
    # Now we must find the value in the binary image that preserves these moments
    
    
    # We solve the equalities --> For solutions refer to Paper Annex A.2
    cd = (m0*m2*m4) + (m1*m3*m2) + (m1*m3*m2) - (m2*m2*m2) - (m1*m1*m4) - (m3*m3*m0)
    c0 = ((-m3*m2*m4) + (-m4*m3*m2) + (m1*m3*-m5) - (-m5*m2*m2) - (-m4*m1*m4) - (m3*m3*-m3)) / cd
    c1 = ((m0*-m4*m4) + (m1*-m5*m2) + (-m3*m3*m2) - (m2*-m4*m2) - (m1*-m3*m4) - (-m5*m3*m0)) / cd
    c2 = ((m0*m2*-m5) + (m1*m3*-m3) + (m1*-m4*m2) - (m2*m2*-m3) - (m1*m1*-m5) - (m3*-m4*m0)) /cd
    
    a1 = c0/2 - c1*c2/6 + (c2**3)/27
    a2 = (c0/2 - c1*c2/6 + (c2**3)/27)**2
    a3 = (c1/3 - (c2**2)/9)**3
    
    s = 1
    p = 1
    q = 1
    a = (a1 - cmath.sqrt(a2 + a3))**1/3 
   
    #s = cmath.sqrt(p)
    
    #q = a1-p
    #a = q**1/3
    
    b = -(c1/3 - (c2**2)/9)/a
    w1 = -0.5 + 1j * (math.sqrt(3)/2)
    w2 = -0.5 - 1j * (math.sqrt(3)/2)
    
    z0 = -c2/3 - a - b
    z1 = -c2/3 - w1*a - w2*b
    z2 = -c2/3 - w2*a - w1*b
   
    pd = (z1*z2**2) + (z2*z0**2) + (z0*z1**2) - (z0**2*z1) - (z0*z2**2) - (z1**2*z2)
    p0 = ((m0*z1*z2**2) + (m1*z1**2) + (z2*m2) - (m2*z1) - (m1*z2**2) - (z1**2*z2*m0)) /pd
    p1 = ((m1*z2**2) + (z0*m2) + (m0*z2*z0**2) - (z0**2*m1) - (z0*m0*z2**2) - (m2*z2)) / pd # Fraction of the below-threshold pixels in the binary histogram
    
    dict_val = {'cd':cd, 
                'c0':c0, 
                'c1':c1, 
                'c2':c2,
                'a1':a1,
                'a2':a2,
                'a3':a3,
                'a':a,
                'b':b,
                'w1':w1,
                'w2':w2,
                'z0':z0,
                'z1':z1,
                'z2':z2,
                'pd':pd,
                'p0':p0,
                'p1':p0,
                'p': p,
                'q': q,
                's': s,
                'm0': m0,
                'm1': m1,
                'm2': m2,
                'm3': m3,
                'm4': m4,
                'm5': m5}
    
    th1 = 0.0
    th2 = 0.0
    dist1 = 10000000
    dist2 = 10000000
    
    for i in range(254):
        for j in range(i+1, 255):
            # Select threshold --> closest to p0 from the normlaized histogram
            p0_orig = histogram[i,3]  # Take the cumulative relative frequency at the value p0
            p1_orig = histogram[j, 3] - histogram[i,3]
            dist_i = abs(p0 - p0_orig)
            dist_j = abs(p1 - p1_orig)
            
            #if p0_orig>p0:  # This one was used in the code written by G. Landini
               # threshold = i
               # break
            
            if dist_i < dist1 and dist_j < dist2:  # This one was the one mentioned by Tsai ("Minimize the distance")
                
                print(i,j,dist_i, dist_j)
                dist1 = dist_i
                dist2 = dist_j
                th1 = i
                th2 = j
            
            
    return img_array, histogram, th1, th2, dict_val


def moments_tsai_example():
    
    """ This function applies the following reasing from Tsai paper:
        A bilevel image consists of pixels with only two gray values z0 and z1, where 
        z0<z1
        The proposed moment-preserving thresholing is to select a threshold value 
        such that if all belo-threshold gray values in f are replaced by z1, 
        then the first three moemt of imafe f are preserved in the resulting
        bilevel image g.
        Apply the example described by tsai in the paper
        
    """
    
    img = np.array([[10, 8, 10, 9, 20, 21,32,30,40,41,41,40],
                   [12, 10, 11, 10, 19, 20, 30, 28, 38, 40, 40, 39],
                   [10, 9, 10, 8, 20, 21, 30, 29, 42, 40, 40, 39],
                   [11, 10, 9, 11, 19, 21, 31, 30, 40, 42,38, 40]])
    
    bins_hist = list(range(0,257)) # Queremos llegar a tener 256 "papeleras", por eso tenemos que poner 257, para que haga de 0 hasta 256
    histogram = np.zeros((256,4))  # We generate the counts for each grey value, 3rd column probabilities and 4th column cumulative sum
    histogram[:,0] = np.arange(256)  # La primera columna contiene los valores de intensidad de los pixels
    
    hist_i = np.histogram(img, bins=bins_hist)
    counts = hist_i[0]
    histogram[:,1] = np.add(histogram[:,1], counts)
   
    row, col = img.shape
    total_voxels = row * col
    
    histogram[:,2] = histogram[:,1]/total_voxels  # Relative frequency at each GV
    histogram[:,3] = np.cumsum(histogram[:,2])  # Cumulative of the relative frequency
    
    
    ###  THRESHOLDS CALCULATION BEGIN ###
    
    # 0 Moment = 1
    m0 = 1.0
    
    # 1st Moment 
    m1 = np.cumsum((histogram[:,0])*histogram[:,2])[-1]    
    
    # 2nd Moment
    m2 = np.cumsum((histogram[:,0]**2)*histogram[:,2])[-1]  # Take the last value
    
    # 3rd Moment
    m3 = np.cumsum((histogram[:,0]**3)*histogram[:,2])[-1]  # Take the last value
    
    # 4th Moment
    m4 = np.cumsum((histogram[:,0]**4)*histogram[:,2])[-1]  # Take the last value
    
    # 5th Moment
    m5 = np.cumsum((histogram[:,0]**5)*histogram[:,2])[-1]  # Take the last value
    
    # Now we must find the value in the binary image that preserves these moments
    
    
    # We solve the equalities --> For solutions refer to Paper Annex A.2
    cd = (m0*m2*m4) + (m1*m3*m2) + (m1*m3*m2) - (m2*m2*m2) - (m1*m1*m4) - (m3*m3*m0)
    c0 = ((-m3*m2*m4) + (-m4*m3*m2) + (m1*m3*-m5) - (-m5*m2*m2) - (-m4*m1*m4) - (m3*m3*-m3)) / cd
    c1 = ((m0*-m4*m4) + (m1*-m5*m2) + (-m3*m3*m2) - (m2*-m4*m2) - (m1*-m3*m4) - (-m5*m3*m0)) / cd
    c2 = ((m0*m2*-m5) + (m1*m3*-m3) + (m1*-m4*m2) - (m2*m2*-m3) - (m1*m1*-m5) - (m3*-m4*m0)) /cd
    
    a1 = c0/2 - c1*c2/6 + (c2**3)/27
    a2 = (c0/2 - c1*c2/6 + (c2**3)/27)**2
    a3 = (c1/3 - (c2**2)/9)**3
    
    p = a2 + a3 # Divide the number by -1 to be able to perform the square root and then multiply by sqrt(-1)
    s = cmath.sqrt(p)
    
    q = a1-s
    a = q**1/3
    
    b = -(c1/3 - (c2**2)/9)/a
    w1 = -0.5 + 1j * (math.sqrt(3)/2)
    w2 = -0.5 - 1j * (math.sqrt(3)/2)
    
    z0 = -c2/3 - a - b
    z1 = -c2/3 - w1*a - w2*b
    z2 = -c2/3 - w2*a - w1*b
   
    pd = (z1*z2**2) + (z2*z0**2) + (z0*z1**2) - (z0**2*z1) - (z0*z2**2) - (z1**2*z2)
    p0 = ((m0*z1*z2**2) + (m1*z1**2) + (z2*m2) - (m2*z1) - (m1*z2**2) - (z1**2*z2*m0)) /pd
    p1 = ((m1*z2**2) + (z0*m2) + (m0*z2*z0**2) - (z0**2*m1) - (z0*m0*z2**2) - (m2*z2)) / pd # Fraction of the below-threshold pixels in the binary histogram
    
    dict_val = {'cd':cd, 
                'c0':c0, 
                'c1':c1, 
                'c2':c2,
                'a1':a1,
                'a2':a2,
                'a3':a3,
                'a':a,
                'b':b,
                'w1':w1,
                'w2':w2,
                'z0':z0,
                'z1':z1,
                'z2':z2,
                'pd':pd,
                'p0':p0,
                'p1':p0,
                'p': p,
                'q': q,
                's': s,
                'm0': m0,
                'm1': m1,
                'm2': m2,
                'm3': m3,
                'm4': m4,
                'm5': m5}
    
    th1 = 0.0
    th2 = 0.0
    dist1 = 10000000
    dist2 = 10000000
    
    for i in range(254):
        for j in range(i+1, 255):
            # Select threshold --> closest to p0 from the normlaized histogram
            p0_orig = histogram[i,3]  # Take the cumulative relative frequency at the value p0
            p1_orig = histogram[j, 3] - histogram[i,3]
            dist_i = abs(p0 - p0_orig)
            dist_j = abs(p1 - p1_orig)
            
            #if p0_orig>p0:  # This one was used in the code written by G. Landini
               # threshold = i
               # break
            
            if dist_i < dist1 and dist_j < dist2:  # This one was the one mentioned by Tsai ("Minimize the distance")
                
                print(i,j,dist_i, dist_j)
                dist1 = dist_i
                dist2 = dist_j
                th1 = i
                th2 = j
            
            
    
    return th1, th2, dict_val


if __name__ == "__main__":
    
    stack_path =  "\path..."
    
    th1, th2, dict_val = threshold_3d_multi_moments(stack_path)
    
    
        
        