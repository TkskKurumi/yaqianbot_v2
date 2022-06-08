from .point import point2d
import numpy as np
def get_circle(A:point2d, B:point2d, C:point2d):
    # https://blog.csdn.net/qingchunweiliang/article/ls/9330927
    x1, y1 = A
    x2, y2 = B
    x3, y3 = C
    mat = [[1, x1, y1], [1, x2, y2], [1, x3, y3]]
    S = 1/2 * np.linalg.det(mat)
    pass