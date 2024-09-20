from vec3 import vec3
from color import create_color

import numpy as np, imageio

# create aliases for vec3 to represent points and colors:

color = vec3
point3 = vec3

# set dimensions

width = 256
height = 256

# create numpy matrix to hold png values

image = np.zeros((height, width, 3), dtype=np.uint8)

# modify the matrix to your desired image output

for i in range(len(image[1])):
    for j in range(len(image[2])):

        # create decimal values for RGB colors in each pixel

        pixel_color = color([np.double(i)/(height-1),0.0,np.double(j)/(width-1)])

        image[i,j,:] = create_color(pixel_color)

# write the image to a PNG format

imageio.imwrite('output2.png', image)