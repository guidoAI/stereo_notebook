# -*- coding: utf-8 -*-
"""
Created on Sun Jun 10 18:05:15 2018

@author: LocalAdmin
"""

import numpy as np
import cv2
from matplotlib import pyplot as plt

def read_stereo_image(im="stereo_image_explorer.bmp"):
    cv_im = cv2.imread(im);
    imgL = cv_im[0:96, 126:252, :];
    imgR = cv_im[0:96, 0:126, :];
    return [imgL, imgR]

def simple_stereo(imgL, imgR, max_disparity=30):
    
    W = imgL.shape[1];
    H = imgL.shape[0];
    
    # create the disparities image:
    Disparities = np.zeros([H, W]);
    
    # loop over the image
    for x in range(W):
        
        # in the left border of the left image, not all disparities can be investigated:
        max_disp = np.min([x, max_disparity]);
        disps = np.arange(0, max_disp+1);
        
        for y in range(H):
            # we can determine the differences in one go:
            differences = np.abs(imgL[y,x,0] - imgR[y, x-max_disp:x+1,0]);
            # the minimal difference determines the disparity
            disp_ind = np.argmin(differences);
            disparity = disps[disp_ind];
            Disparities[y, x] = disparity;
    
    return Disparities;

def get_differences_curve(imgL, imgR, x, y, max_disparity=30):
    
    # determine the disparities that will be investigated:
    max_disp = np.min([x, max_disparity]);
    disps = np.arange(0, max_disp+1);
    
    # we can determine the differences in one go:
    differences = np.abs(imgL[y,x,0] - imgR[y, x-max_disp:x+1,0]);
    # the minimal difference determines the disparity
    disp_ind = np.argmin(differences);
    disparity = disps[disp_ind];
    
    return [differences, disps, disp_ind];

def windowed_stereo(imgL, imgR, max_disparity=30, window_half_size=3):
    
    W = imgL.shape[1];
    H = imgL.shape[0];
    
    # create the disparities image:
    Disparities = np.zeros([H, W]);
    
    # loop over the image
    for x in np.arange(window_half_size, W-window_half_size):
        
        # in the left border of the left image, not all disparities can be investigated:
        max_disp = np.min([x-window_half_size, max_disparity]);
        if(max_disp >= 0):
            disps = np.arange(0, max_disp+1);
            differences = np.zeros([len(disps), 1]);
            
            for y in np.arange(window_half_size, H-window_half_size):
                
                window_left = imgL[y-window_half_size:y+window_half_size, x-window_half_size:x+window_half_size, 0];
                
                for d in disps:
                    window_right = imgR[y-window_half_size:y+window_half_size, x-d-window_half_size:x-d+window_half_size, 0];
                    differences[d] = np.sum(np.abs(window_left.astype(float) - window_right.astype(float)));
                
                # the minimal difference determines the disparity
                disp_ind = np.argmin(differences);
                disparity = disps[disp_ind];
                Disparities[y, x] = disparity;
    
    return Disparities;

def calculate_disparities(imgL, imgR, window_size=7, min_disp=0, num_disp=16):
    # semi-global matching:
    stereo = cv2.StereoSGBM_create(numDisparities = num_disp, blockSize = window_size);
    disp = stereo.compute(imgL, imgR).astype(np.float32) / 16.0;
    return disp; 

def plot_relation_disparity_depth(f = 300, T_X = 0.10, max_disp = 64):
    """ Focal length f is in pixels, T_X is in meters.
    """
    
    disparities = np.arange(1, max_disp+1, 1);
    depths = np.zeros(max_disp);
    for disp in disparities:
        depths[disp-1] = f * (T_X / disp);
    
    plt.figure();
    plt.plot(disparities, depths, 'o');
    plt.xlabel('Disparity [px]')
    plt.ylabel('Depth Z [m]')
    
plot_relation_disparity_depth(f = 140, T_X = 0.06, max_disp = 32)

#[imgL, imgR] = read_stereo_image();
#
#plt.figure();
#plt.subplot(121)
#plt.imshow(imgL);
#plt.title('Left');
#
#plt.subplot(122)
#plt.imshow(imgR);
#plt.title('Right');

#D = simple_stereo(imgL, imgR);
#plt.figure();
#plt.imshow(D, cmap='hot');
#plt.colorbar();
#plt.draw()
#           
#print('Image size, width = {}, height = {}'.format(imgL.shape[1], imgL.shape[0]))
#[differences, disps, disp_ind] = get_differences_curve(imgL, imgR, 48, 64); 
#plt.figure();
#plt.plot(disps, differences);
#plt.plot(disps[disp_ind], differences[disp_ind], 'x', markersize=10);
#plt.draw();
#
#D = windowed_stereo(imgL, imgR, max_disparity=30, window_half_size=3);
#plt.figure();
#plt.imshow(D, cmap='hot');
#plt.colorbar();
#plt.draw()
#
#D = calculate_disparities(imgL, imgR, window_size=7, min_disp=0, num_disp=16)
#plt.figure();
#plt.imshow(D, cmap='hot');
#plt.colorbar();
#plt.draw()
