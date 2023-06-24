from __future__ import annotations
from dataclasses import dataclass
import math
from .approximate import EPS


@dataclass
class Circle:
    x: float
    y: float
    r: float

    def __init__(self, x: float, y: float, r: float):
        self.x = x
        self.y = y
        self.r = r
        self.O = Point2d(x, y)

    def __contains__(self, p: Point2d):
        dist = self.O.dist(p)
        judge = dist-self.r
        return judge < -EPS


@dataclass
class Point2d:
    x: float
    y: float

    def __iter__(self):
        return (self.x, self.y).__iter__()

    def __add__(self, other):
        return Point2d(self.x+other.x, self.y+other.y)

    def __sub__(self, other):
        return Point2d(self.x-other.x, self.y-other.y)

    def __truediv__(self, other):
        return Point2d(self.x/other, self.y/other)

    def __mul__(self, other):
        if(isinstance(other, Point2d)):
            # dot product
            return self.x*other.x+self.y*other.y
        else:
            return Point2d(self.x*other, self.y*other)
    def __rmul__(self, other):
        return self*other
    def dot(self, other):
        return self.x*other.x+self.y*other.y

    def __pow__(self, other):
        if(isinstance(other, Point2d)):
            return self.cross(other)
        else:
            return NotImplemented

    def cross(self, other):
        return self.x*other.y-self.y*other.x

    def ascomplex(self):
        return self.x+self.y*1j

    def rotate_by(self, ctr: Point2d, theta):
        return ctr+(self-ctr).rotate_by_angle(theta)

    def rotate_by_complex(self, other):
        ret = (self.x+self.y*1j)
        if(isinstance(other, complex)):
            mult = other
        elif(isinstance(other, Point2d)):
            mult = other.x+other.y*1j
        ret = ret*mult
        return Point2d(ret.real, ret.imag)

    def rotate_by_angle(self, theta):
        return self.rotate_by_complex(math.cos(theta)+math.sin(theta)*1j)

    def length(self):
        return math.sqrt(self.x*self.x+self.y*self.y)

    def dist(self, other):
        return (self-other).length()

    def dist2(self, other):
        return (self-other).sqrlength()

    def sqrlength(self):
        return self.x*self.x+self.y*self.y

    def unit(self):
        return self/self.length()
    @property
    def intxy(self):
        return int(self.x), int(self.y)
    @property
    def xy(self):
        return self.x, self.y
if(__name__ == "__main__"):
    A = Point2d(0, 2)
    B = Point2d(2, 0)
    print(A.dot(B))
    print(A.cross(B))
