# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 09:52:04 2024

@author: korol
"""

# Simple code to figure out how to create PNG file format
import numpy as np, imageio

# set dimensions

width = 256
height = 256

# create numpy matrix to hold png values

image = np.zeros((height, width, 3), dtype=np.uint8)

# modify the matrix to your desired image output

for i in range(len(image[1])-1):
    for j in range(len(image[2])-1):

        # create decimal values for RGB colors in each pixel

        r = np.double(i)/(width-1)
        g = np.double(j)/(height-1)
        b = 0.0

        # Create uint8 representation of colors

        ir = np.floor(255.999 * r)
        ig = np.floor(255.999 * g)
        ib = np.floor(255.999 * b)

        # set the image pixel

        image[i,j,:] = np.array([ir,ig,ib])

# write the image to a PNG format

imageio.imwrite('output.png', image)