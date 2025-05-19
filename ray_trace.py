import numpy as np
import matplotlib.pyplot as plt
import imageio
import multiprocessing as mp
from abc import ABC, abstractmethod

# ----- Helper functions for things that np arrays can't do -------------------

def unit_vector(v):
    return v / np.linalg.norm(v)

def dot(u, v):
    return np.dot(u, v)

def length_squared(v):
    return np.dot(v, v)

# ---------- Classes used for ray tracing and rendering -----------------------

# Ray is a position in the scene parameterized by time coordinate 't'

class Ray:
    def __init__(self, origin, direction):
        self.origin = origin
        self.direction = direction

    def at(self, t):
        return self.origin + t * self.direction
    
# Interval is used to check whether a point is within a rendering area

class Interval:
    def __init__(self, mini=np.inf, maxi=-np.inf):
        self.mini = mini
        self.maxi = maxi

    def surrounds(self, x):
        return self.mini < x < self.maxi

    @classmethod
    def universe(cls):
        return cls(-np.inf, np.inf)

# HitRecord is used to store information about the point at which a hit occurs

class HitRecord:
    def __init__(self):
        self.p = None
        self.normal = None
        self.t = 0
        self.front_face = False

    def set_face_normal(self, ray, outward_normal):
        self.front_face = dot(ray.direction, outward_normal) < 0
        self.normal = outward_normal if self.front_face else -outward_normal
        
# Abstract class to make sure that any objects created include a hit function

class Hittable(ABC):
    @abstractmethod
    def hit(self, ray, ray_t, rec) -> bool:
        pass

# Defines sphere object and its hit function

class Sphere(Hittable):
    def __init__(self, center, radius):
        self.center = np.array(center)
        self.radius = radius

    def hit(self, ray, ray_t, rec):
        oc = self.center - ray.origin
        a = length_squared(ray.direction)
        h = dot(ray.direction, oc)
        c = length_squared(oc) - self.radius * self.radius
        discriminant = h * h - a * c

        if discriminant < 0:
            return False

        sqrtd = np.sqrt(discriminant)
        root = (h - sqrtd) / a
        if not ray_t.surrounds(root):
            root = (h + sqrtd) / a
            if not ray_t.surrounds(root):
                return False

        rec.t = root
        rec.p = ray.at(rec.t)
        outward_normal = (rec.p - self.center) / self.radius
        rec.set_face_normal(ray, outward_normal)
        return True


# Class for storing list of objects in the scene

class HittableList(Hittable):
    def __init__(self, objects=None):
        self.objects = objects if objects else []

    def add(self, obj):
        self.objects.append(obj)

    def hit(self, ray, ray_t, rec):
        temp_rec = HitRecord()
        hit_anything = False
        closest_so_far = ray_t.maxi

        for obj in self.objects:
            if obj.hit(ray, Interval(ray_t.mini, closest_so_far), temp_rec):
                hit_anything = True
                closest_so_far = temp_rec.t
                rec.p = temp_rec.p
                rec.normal = temp_rec.normal
                rec.t = temp_rec.t
                rec.front_face = temp_rec.front_face

        return hit_anything

# Camera class to set the camera location, aspect ratios, rendering and ray 
# creation, sampling for antialiasing
class Camera:
    def __init__(self, aspect_ratio=16/9, image_width=400, samples=10):
        self.aspect_ratio = aspect_ratio
        self.image_width = image_width
        self.samples = samples
        self.image_height = int(image_width / aspect_ratio)
        self.samples_scale = 1.0 / samples

    def initialize(self):
        self.center = np.array([0, 0, 0])
        focal_length = 1.0
        viewport_height = 2.0
        viewport_width = viewport_height * (self.image_width / self.image_height)

        u = np.array([viewport_width, 0, 0])
        v = np.array([0, -viewport_height, 0])
        self.pixel_delta_u = u / self.image_width
        self.pixel_delta_v = v / self.image_height

        viewport_upper_left = self.center - np.array([0, 0, focal_length]) - u / 2 - v / 2
        self.pixel00_loc = viewport_upper_left + 0.5 * (self.pixel_delta_u + self.pixel_delta_v)

    def get_ray(self, i, j):
        offset = np.random.rand(2) - 0.5
        pixel_center = self.pixel00_loc + (j + offset[0]) * self.pixel_delta_u + (i + offset[1]) * self.pixel_delta_v
        direction = pixel_center - self.center
        return Ray(self.center, direction)

    def ray_color(self, ray, world):
        rec = HitRecord()
        if world.hit(ray, Interval(0.001, np.inf), rec):
            return 0.5 * (rec.normal + np.array([1, 1, 1]))
        unit_dir = unit_vector(ray.direction)
        t = 0.5 * (unit_dir[1] + 1.0)
        return (1.0 - t) * np.array([1, 1, 1]) + t * np.array([0.4, 0.4, 1])

    # Parallelized rendering using render function defined below
    
    def render(self, world):
        self.initialize()
        image = np.zeros((self.image_height, self.image_width, 3), dtype=np.uint8)

        tasks = [(i, j, self.samples, self.pixel00_loc, self.pixel_delta_u, self.pixel_delta_v, self.center, world)
                 for i in range(self.image_height) for j in range(self.image_width)]

        with mp.Pool(mp.cpu_count()) as pool:
            for count, (i, j, color) in enumerate(pool.imap_unordered(_render_pixel, tasks, chunksize=32)):
                image[i, j] = color
                if count % self.image_width == 0:
                    print(f"{count // self.image_width} rows rendered (of {self.image_height})")

        return image

# ------ Rendering code optimized for parallel tasks---------------------------
def _render_pixel(args):
    i, j, samples, pixel00, du, dv, center, world = args
    accum = np.zeros(3)
    for _ in range(samples):
        offset = np.random.rand(2) - 0.5
        pixel = pixel00 + (j + offset[0]) * du + (i + offset[1]) * dv
        r = Ray(center, pixel - center)
        rec = HitRecord()
        if world.hit(r, Interval(0.001, np.inf), rec):
            color = 0.5 * (rec.normal + np.array([1, 1, 1]))
        else:
            unit_dir = unit_vector(r.direction)
            t = 0.5 * (unit_dir[1] + 1.0)
            color = (1.0 - t) * np.array([1, 1, 1]) + t * np.array([0.4, 0.4, 1])
        accum += color
    averaged = accum / samples
    return i, j, (np.clip(averaged, 0, 0.999) * 256).astype(np.uint8)

# ------------- Function for displaying images --------------------------------
def show_output(image, file='render_output.png'):
    imageio.imwrite(file, image)
    print(f"Image saved to {file}")
    plt.imshow(image)
    plt.axis('off')
    plt.show()



def main():
    cam = Camera(aspect_ratio=16/9, image_width=400, samples=50)
    world = HittableList()
    world.add(Sphere([0, 0, -1], 0.5))
    world.add(Sphere([0, -100.5, -1], 100))
    image = cam.render(world)
    show_output(image)

if __name__ == "__main__":
    main()
