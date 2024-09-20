# -*- coding: utf-8 -*-

import numpy as np

# this converts colors used in the code to colors we can display in a PNG
def create_color(pixel_color):

    r = pixel_color.x
    g = pixel_color.y
    b = pixel_color.z

    rbyte = np.floor(255.999 * r)
    gbyte = np.floor(255.999 * g)
    bbyte = np.floor(255.999 * b)

    return np.array([rbyte,gbyte,bbyte])



