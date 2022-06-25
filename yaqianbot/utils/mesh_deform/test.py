from ..image.colors import Color
from ..image.background import random_position
from ..geometry.elements import Point2d as Point
from .mesh import DelaunayMesh
from ..algorithms import kmeans
import random
from PIL import Image, ImageDraw
w, h=300, 300
points = []
n = 10
for i in range(n):
    for j in range(n):
        x = w/(n+1)*(i+1)
        y = h/(n+1)*(j+1)
        points.append(Point(x, y))
# kmeans_points = kmeans(points, n)
# points = list()
# for x, y in kmeans_points:
#     points.append(Point(x, y))
# # print(points)
mesh = DelaunayMesh(points, debug=False)
im = Image.new("RGB", (w, h), (255,)*3)
dr = ImageDraw.Draw(im)
for u, v in mesh.G.edges:
    point_u = points[u]
    point_v = points[v]
    c = Color.from_hsl(random.random()*360, 1, 0.5)
    dr.line((*point_u, *point_v), width=2, fill=c.get_rgb())
from ..image.print import image_show_terminal
image_show_terminal(im, rate = 0.8)