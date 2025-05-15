from vec3 import vec3
from color import create_color
import matplotlib.pyplot as plt

import numpy as np, imageio

# import the methods needed for abstract classes:

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

# create aliases for vec3 to represent points and colors:

color = vec3
point3 = vec3

# test commit again!

# Create Ray Class:

class ray:

    # Set the parameters (origin and direction)
    def __init__(self,orig,direct):

        self.origin = orig
        self.direction = direct

    # return coordinates at step "t" from origin

    def at(self,t):
        return vec3(self.origin.e + t*self.direction.e)


# Create abstract classes to define how hittable objects should be set up:

class HitRecord:
    # Set the parameters (origin and direction)
    def __init__(self,p,normal,t):

        self.p = tuple
        self.normal = tuple
        self.t = float
        self.frontFace = bool

    # determines the normal direction and orientation for the face
    # we want the normal direction to always be against the ray,
    # so for inner surfaces, the outward normal is flipped.

    def setFaceNormal(self,ray,outwardNormal):
        self.frontFace = ray.direction.dot(outwardNormal)<0
        if self.frontFace:
            self.normal = outwardNormal
        else:
            self.normal = -outwardNormal

class Hittable(ABC):
    """Abstract base class for hittable objects."""

    @abstractmethod
    def hit(self, ray: ray, ray_tmin: float, ray_tmax: float, rec: HitRecord) -> bool:

        pass

# Make a hittable sphere object (combines the hit_sphere stuff into one class with all attributes)

class Sphere(Hittable):
    def __init__(self, center: tuple, radius: float):
        self.center = center
        self.radius = max(0, radius)  # Ensure the radius is non-negative

    def hit(self, ray: ray, ray_tmin: float, ray_tmax: float, rec: HitRecord) -> bool:
        """
        Check if a ray intersects the sphere.

        Updates the provided HitRecord with intersection details.
        """
        # Compute the vector from the ray's origin to the sphere's center
        center_offset = self.center - ray.origin
        # Compute quadratic coefficients
        a = r.direction.len_squared()
        h = r.direction.dot(center_offset)
        c = center_offset.len_squared() - self.radius*self.radius

        discriminant = h * h - a * c
        if discriminant < 0:
            return False  # No intersection

        sqrtd = np.sqrt(discriminant)

        # Find the nearest root in the acceptable range
        root = (h - sqrtd) / a
        if root <= ray_tmin or ray_tmax <= root:
            root = (h + sqrtd) / a
            if root <= ray_tmin or ray_tmax <= root:
                return False

        # Use the recorded hit to get ray parameters
        rec.t = root
        rec.p = ray.at(rec.t)
        outwardNormal = (rec.p-self.center)/self.radius
        rec.setFaceNormal(ray, outwardNormal)

        return True

# make a list of "hittable" objects that will be in the scene

class hittableList:
    def __init__(self, obj=None):
        self.objects = []
        if obj is not None:
            self.add(obj)

    def clear(self):
        self.objects = []

    def add(self, obj):
        self.objects.append(obj)

    def hit(self, ray: ray, ray_tmin: float, ray_tmax: float, rec: HitRecord) -> bool:

        # temporary intersection at infinity to start off
        temp_rec = HitRecord(None, None, float('inf'))
        # hit flag
        hit_anything = False
        # start at the furthest bound of t
        closest_so_far = ray_tmax

        for obj in self.objects:
            # Check if the ray hits the current object within the current closest range.
            if obj.hit(ray, ray_tmin, closest_so_far, temp_rec):
                hit_anything = True
                closest_so_far = temp_rec.t
                rec.p = temp_rec.p
                rec.normal = temp_rec.normal
                rec.t = temp_rec.t
                rec.front_face = temp_rec.frontFace

        return hit_anything

# Function for ray color (gradient background):

def ray_color(ray,world):

    rec = HitRecord
    # if the ray hits something:
    if (world.hit(ray,0,np.inf,rec)):
        return 0.5*(rec.normal+color([1,1,1]))

    unit_direction = (ray.direction).unit_vector()

    # create gradient:

    a = 0.5*(unit_direction.y + 1.0)
    return (1.0-a)*color([1,1,1]) + a*color([1,0.4,0.5])


# set up image environment:

aspect_ratio = 16.0/9.0

image_width = 400

# get image height from aspect ratio

image_height = int(np.floor(image_width/aspect_ratio))

if image_height < 1:
    image_height = 1

# Create image matrix:

image = np.zeros((image_height, image_width, 3), dtype=np.uint8)

# set up world (the scene of objects to render):

world = hittableList()
# XYZ center coordinates and radius
world.add(Sphere(point3([0,0,-1]), 0.5))
world.add(Sphere(point3([0,-100.5,-1]), 100))

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

        pixel_color = ray_color(r,world)
        image[i,j,:] = create_color(pixel_color)

imageio.imwrite('output4.png', image)
plt.imshow(image)






