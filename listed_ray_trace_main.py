from vec3 import vec3
from color import create_color
import matplotlib.pyplot as plt

import numpy as np, imageio

# import the methods needed for abstract classes:

from abc import ABC, abstractmethod

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
        return vec3(self.origin.e + t*self.direction.e)
    
    
# Create interval class to keep track of the valid regions where we expect 
# to have intersections

class interval:
    
    def __init__(self,mini=np.inf,maxi=-np.inf):
        self.mini = mini
        self.maxi = maxi

    def size(self):
        return (self.maxi-self.mini)
    
    def contains(self,x):
        return (self.mini <= x <= self.maxi)
    
    def surrounds(self,x):
        return (self.mini < x < self.maxi)
    
    # Create default empty and infinite ranges for ease of use
    
    @classmethod
    def empty(cls):
        return cls(np.inf, -np.inf)
    
    @classmethod
    def universe(cls):
        return cls(-np.inf, np.inf)
    

# Create a camera class to instantiate camera variables, shoot rays, and render

class camera:
    
    def __init__(self,aspect_ratio=1.0,image_width=100):
        self.aspect_ratio = aspect_ratio
        self.image_width = image_width
        self.image_height = max(1, int(image_width / aspect_ratio))

        self.center = None
        self.pixel00_loc = None
        self.pixel_delta_u = None
        self.pixel_delta_v = None
        
    def initialize(self):
        self.image_height = max(1, int(self.image_width / self.aspect_ratio))
        self.center = point3([0, 0, 0])
    
        # Viewport
        focal_length = 1.0
        viewport_height = 2.0
        viewport_width = viewport_height * (self.image_width / self.image_height)
    
        viewport_u = vec3([viewport_width, 0, 0])
        viewport_v = vec3([0, -viewport_height, 0])
    
        self.pixel_delta_u = viewport_u / self.image_width
        self.pixel_delta_v = viewport_v / self.image_height
    
        viewport_upper_left = (
            self.center
            - vec3([0, 0, focal_length])
            - viewport_u / 2
            - viewport_v / 2
        )

        self.pixel00_loc = viewport_upper_left + 0.5 * (self.pixel_delta_u + self.pixel_delta_v)

    def render(self, world):
        self.initialize()
        
        image = np.zeros((self.image_height, self.image_width, 3), dtype=np.uint8)
    
        for i in range(self.image_height):
            for j in range(self.image_width):
                # find the coordinates of the current pixel:
    
                pixel_center = self.pixel00_loc + (i * self.pixel_delta_v) + (j * self.pixel_delta_u)
                ray_direction = pixel_center - self.center
    
                # create ray for this pixel:
    
                r = ray(self.center,ray_direction)
    
                pixel_color = self.ray_color(r,world)
                image[i,j,:] = create_color(pixel_color)

        imageio.imwrite('output4.png', image)
        plt.imshow(image)
    
    def ray_color(self,ray,world):

        rec = HitRecord
        # if the ray hits something:
        if (world.hit(ray,interval(0,np.inf),rec)):
            return 0.5*(rec.normal+color([1,1,1]))
        
        # If the ray doesn't hit anything (misses the spheres) draw a gradient:

        unit_direction = (ray.direction).unit_vector()

        # create gradient:

        a = 0.5*(unit_direction.y + 1.0)
        return (1.0-a)*color([1,1,1]) + a*color([0.4,0.4,1])
    
    

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
    """Abstract base class for hittable objects.
    
    This basically just makes sure that all objects that are introduced to the 
    scene follow the basic guidelines (include a hit function that defines intersections)
    """

    @abstractmethod
    def hit(self, ray: ray, ray_t: interval, rec: HitRecord) -> bool:

        pass

# Make a hittable sphere object (combines the hit_sphere stuff into one class with all attributes)

class Sphere(Hittable):
    def __init__(self, center: tuple, radius: float):
        self.center = center
        self.radius = max(0, radius)  # Ensure the radius is non-negative

    def hit(self, ray: ray, ray_t: interval, rec: HitRecord) -> bool:
        """
        Check if a ray intersects the sphere.

        Updates the provided HitRecord with intersection details.
        """
        # Compute the vector from the ray's origin to the sphere's center
        center_offset = self.center - ray.origin
        # Compute quadratic coefficients
        a = ray.direction.len_squared()
        h = ray.direction.dot(center_offset)
        c = center_offset.len_squared() - self.radius*self.radius

        discriminant = h * h - a * c
        if discriminant < 0:
            return False  # No intersection

        sqrtd = np.sqrt(discriminant)

        # Find the nearest root in the acceptable range
        root = (h - sqrtd) / a
        if not ray_t.surrounds(root):
            root = (h + sqrtd) / a
            if not ray_t.surrounds(root):
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

    def hit(self, ray: ray, ray_t: interval, rec: HitRecord) -> bool:

        # temporary intersection at infinity to start off
        temp_rec = HitRecord(None, None, float('inf'))
        # hit flag
        hit_anything = False
        # start at the furthest bound of t
        closest_so_far = ray_t.maxi

        for obj in self.objects:
            # Check if the ray hits the current object within the current closest range.
            if obj.hit(ray, interval(ray_t.mini, closest_so_far), temp_rec):
                hit_anything = True
                closest_so_far = temp_rec.t
                rec.p = temp_rec.p
                rec.normal = temp_rec.normal
                rec.t = temp_rec.t
                rec.front_face = temp_rec.frontFace

        return hit_anything

# ------------------------------ Main Function for Rendering ------------------

def main():
    cam = camera()
    cam.aspect_ratio = 16.0 / 9.0
    cam.image_width = 400

    # Set up world (the scene of objects to render)
    world = hittableList()
    world.add(Sphere(point3([0, 0, -1]), 0.2))
    world.add(Sphere(point3([0, -100.5, -1]), 100))

    cam.render(world)

if __name__ == "__main__":
    main()






