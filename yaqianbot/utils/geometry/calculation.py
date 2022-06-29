from .approximate import EPS, float_sign
from .elements import Point2d, Circle
import numpy as np



def does_segment_intersect(A, B, C, D):
    AB = B-A
    CD = D-C
    AC = C-A
    AD = D-A
    CA = A-C
    CB = B-C
    def diff_sign(x, y): 
        x = float_sign(x)
        y = float_sign(y)
        return x*y == -1

    cross1 = AB.cross(AC)
    cross2 = AB.cross(AD)
    judge1 = diff_sign(cross1, cross2)

    cross1 = CD.cross(CA)
    cross2 = CD.cross(CB)
    judge2 = diff_sign(cross1, cross2)

    return judge1 and judge2


def get_circle(A: Point2d, B: Point2d, C: Point2d):
    # https://blog.csdn.net/qingchunweiliang/article/details/9330927
    x1, y1 = A
    x2, y2 = B
    x3, y3 = C
    mat = np.array([[1, x1, y1], [1, x2, y2], [1, x3, y3]], np.float64)
    S = 1/2 * np.linalg.det(mat)

    sqr1 = A.sqrlength()
    sqr2 = B.sqrlength()
    sqr3 = C.sqrlength()
    mat = np.array([[1, sqr1, y1], [1, sqr2, y2], [1, sqr3, y3]], np.float64)
    X = 1/4 * np.linalg.det(mat) / S
    mat = np.array([[1, x1, sqr1], [1, x2, sqr2], [1, x3, sqr3]], np.float64)
    Y = 1/4 * np.linalg.det(mat) / S

    R = A.dist(Point2d(X, Y))

    return Circle(X, Y, R)
