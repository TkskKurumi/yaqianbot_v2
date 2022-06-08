from dataclasses import dataclass
import math


@dataclass
class point2d:
    x: float
    y: float
    def __iter__(self):
        return (self.x, self.y).__iter__()
    def __add__(self, other):
        return point2d(self.x+other.x, self.y+other.y)

    def __sub__(self, other):
        return point2d(self.x-other.x, self.y-other.y)

    def __truediv__(self, other):
        return point2d(self.x/other, self.y/other)

    def __mul__(self, other):
        if(isinstance(other, point2d)):
            # dot product
            return self.x*other.x+self.y*other.y
        else:
            return point2d(self.x*other, self.y*other)
    
    def dot(self, other):
        return self.x*other.x+self.y*other.y

    def cross(self, other):
        return self.x*other.y-self.y*other.x

    def ascomplex(self):
        return self.x+self.y*1j
    
    def rotate_by_complex(self, other):
        ret = (self.x+self.y*1j)
        if(isinstance(other, complex)):
            mult = other
        elif(isinstance(other, point2d)):
            mult = other.x+other.y*1j
        ret = ret*mult
        return point2d(ret.real, ret.imag)

    def rotate_by_angle(self, theta):
        return self.rotate_by_complex(math.cos(theta)+math.sin(theta)*1j)

    def length(self):
        return math.sqrt(self.x*self.x+self.y*self.y)
    def sqrlength(self):
        return self.x*self.x+self.y*self.y
    def unit(self):
        return self/self.length()


vec2d = point2d
vector2d = vec2d

if(__name__ == "__main__"):
    A = point2d(0, 2)
    B = point2d(2, 0)
    print(A.dot(B))
    print(A.cross(B))
