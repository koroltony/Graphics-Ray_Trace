from vec3 import vec3
from color import create_color

import numpy as np, imageio

# create aliases for vec3 to represent points and colors:

color = vec3
point3 = vec3

# Create Ray Class:

class ray:

    # Set the parameters (origin and direction)
    def __init__(self,orig,direct):

        self.origin = orig
        self.direction = direct

    # return coordinates at step "t" from origin

    def at(self,t):
        return self.origin.e + t*self.direction.e

# create the example using Vec3 operations
'''
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
'''

# Create and add a sphere to the render:

def hit_sphere(center,radius,r):
    # using dot product to test if rays are hitting sphere surface:
    center_offset = center - r.origin
    a = r.direction.dot(r.direction)
    b = -2.0*r.direction.dot(center_offset)
    c = center_offset.dot(center_offset) - radius*radius

    disc = b*b - 4*a*c

    return disc >= 0


# Function for ray color (gradient background):

def ray_color(ray):

    if (hit_sphere(point3([0,0,-1]),0.5,ray)):
        return color([1,0,0])

    unit_direction = (ray.direction).unit_vector()

    # create gradient:

    a = 0.5*(unit_direction.y + 1.0)
    return (1.0-a)*color([1.0,1.0,1.0]) + a*color([0.5,0.7,1.0])

# set up image environment:

aspect_ratio = 16.0/9.0

image_width = 400

# get image height from aspect ratio

image_height = int(np.floor(image_width/aspect_ratio))

if image_height < 1:
    image_height = 1

# Create image matrix:

image = np.zeros((image_height, image_width, 3), dtype=np.uint8)

# set up the camera:

focal_length = 1.0
viewport_height = 2.0
# use image ratio to set up viewport aspect ratio
viewport_width = viewport_height * (image_width/image_height)
camera_center = point3([0,0,0])

# calculate the vectors that point in the x and y directions across the viewport

# x direction vector
viewport_u = vec3([viewport_width,0,0])
# y direction vector
viewport_v = vec3([0,-viewport_height,0])

# calculate the delta vectors in these directions for the image
# (viewport vector normalized to the coordinates of the image)

pixel_delta_u = viewport_u/image_width
pixel_delta_v = viewport_v/image_height

# calculate the location of the upper left pixel
# relative to the viewport, the image is smaller

# (ul: upper left)
# Relationship: Image_ul = viewport_ul + avg delta
# viewport_ul is offset from viewport by half its height and width

viewport_ul = camera_center - vec3([0,0,focal_length]) - viewport_u/2 - viewport_v/2

image_ul = viewport_ul + 0.5 * (pixel_delta_u + pixel_delta_v)

# Render the image:

for i in range(image_height):
    for j in range(image_width):
        # find the coordinates of the current pixel:

        pixel_center = image_ul + (i * pixel_delta_v) + (j * pixel_delta_u)
        ray_direction = pixel_center - camera_center

        # create ray for this pixel:

        r = ray(camera_center,ray_direction)

        pixel_color = ray_color(r)
        image[i,j,:] = create_color(pixel_color)

imageio.imwrite('output4.png', image)






