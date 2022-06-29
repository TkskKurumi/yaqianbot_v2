from .elements import Point2d, Circle
from .calculation import get_circle, does_segment_intersect
A = Point2d(0, 0)
B = Point2d(0, 1)
C = Point2d(1, 0)
D = Point2d(1, 1)
O = get_circle(A, B, C)

print(A, B, C)
print(O)
print(does_segment_intersect(A, D, B, C))
print(does_segment_intersect(A, B, C, D))