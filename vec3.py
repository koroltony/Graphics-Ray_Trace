# -*- coding: utf-8 -*-

import numpy as np

# Creating a vector class with special overloaded operators for ray tracing

class vec3:

    # create an initialization for the vec3 class

    def __init__(self,e):

        # the vector itself
        self.e = np.array(e, dtype=np.double)

        # the individual coordinates of the vector

        self.x = e[0]
        self.y = e[1]
        self.z = e[2]

    # overload operators to get special functionality

    # negative sign operator flips sign of all coordinates
    def __neg__(self):
        return vec3([-self.e[0],-self.e[1],-self.e[2]])

    # create operator to index vec3
    def __getitem__(self,i):
        return self.e[i]

    # += operator adds new vec3 to self element-wise
    def __iadd__(self,v):
        self.e[0] += v.e[0]
        self.e[1] += v.e[1]
        self.e[2] += v.e[2]
        return self

    # *= multiplies integer to self element-wise

    def __imul__(self,t):
        self.e[0] *= t
        self.e[1] *= t
        self.e[2] *= t
        return self

    # /= divides integer to self element-wise

    def __itruediv__(self,t):
        self.e[0] /= t
        self.e[1] /= t
        self.e[2] /= t
        return self

    # extract length of the vector from the points using pythagorean distance

    def len_squared(self):
        return self.e[0]*self.e[0] + self.e[1]*self.e[1] + self.e[2]*self.e[2]

    def length(self):
        return np.sqrt(self.len_squared())

    # create further functions to flesh out arithmetic with vec3

    def __add__(self,u):
        return vec3([self.e[0]+u.e[0],self.e[1]+u.e[1],self.e[2]+u.e[2]])

    def __sub__(self,u):
        return vec3([self.e[0]-u.e[0],self.e[1]-u.e[1],self.e[2]-u.e[2]])

    def __mul__(self,obj):
        # the multiplication operation is overloaded differently for different types

        if np.logical_and(isinstance(self,vec3),isinstance(obj,vec3)):
            return vec3([self.e[0]*obj.e[0],self.e[1]*obj.e[1],self.e[2]*obj.e[2]])
        elif np.logical_and(isinstance(self,vec3),~isinstance(obj,vec3)):
            return vec3([self.e[0]*obj,self.e[1]*obj,self.e[2]*obj])

    # rmul allows the multiplication operand to go both ways (vec3 doesn't have to go first)

    def __rmul__(self,obj):
        if np.logical_and(isinstance(self,vec3),isinstance(obj,vec3)):
            return vec3([self.e[0]*obj.e[0],self.e[1]*obj.e[1],self.e[2]*obj.e[2]])
        elif np.logical_and(isinstance(self,vec3),~isinstance(obj,vec3)):
            return vec3([self.e[0]*obj,self.e[1]*obj,self.e[2]*obj])

    def __truediv__(self,t):
        # this uses the mul and rmul that we just defined! :)
        return (1/t) * self

    # Most importantly, we need the dot and cross products:

    def dot(self,v):
        return self.e[0]*v.e[0]+self.e[1]*v.e[1]+self.e[2]*v.e[2]

    def cross(self,v):
        x = self.e[1]*v.e[2]-self.e[2]*v.e[1]
        y = self.e[2]*v.e[0]-self.e[0]*v.e[2]
        z = self.e[0]*v.e[1]-self.e[1]*v.e[0]
        return vec3([x,y,z])

    def unit_vector(self):
        return vec3(self.e/self.length())

# Test cases for class functions:


# point = point3([1,2,3])
# point2 = point3([1,1,1])

# print(point.e)

# mult = 3*point
# div = mult/3
# dotted = mult.dot(point)
# crossed = point.cross(point2)

# print(mult.e)
# print(div.e)
# print(dotted)
# print(crossed.e)
# print(crossed.unit_vector())



