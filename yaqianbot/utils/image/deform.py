from ..geometry.elements import Point2d as Point
from PIL import Image

def make_int(x, *args):
    
    if(isinstance(x, int)):
        return x
    elif(isinstance(x, float)):
        return int(x)
    elif(isinstance(x, list)):
        return [int(i) for i in x]
    elif(isinstance(x, tuple)):
        return tuple([int(i) for i in x])
    elif(isinstance(x, Point)):
        return x.intxy
    else:
        raise NotImplemented(str(type(x)))

def quad_point(im: Image.Image, LU: Point, RU: Point, LL: Point, RL: Point, bg=None):
    left = min(LU, RU, LL, RL, key=lambda p: p.x).x
    right = max(LU, RU, LL, RL, key=lambda p: p.x).x
    upper = min(LU, RU, LL, RL, key=lambda p: p.y).y
    lower = max(LU, RU, LL, RL, key=lambda p: p.y).y

    width = int(right-left)
    height = int(lower-upper)
    offset = Point(left, upper)
    if(bg is None):
        if(im.mode=="RGBA"):
            bg = (0, )*4
        else:
            bg = im.getpixel((0, 0))
    ret = Image.new(im.mode, make_int((width, height)), bg)

    for x1 in range(width):
        for y1 in range(height):
            x = x1/width
            y = y1/height
            c = im.getpixel(make_int((x*im.width, y*im.height)))
            tar = (1-x) * (1-y) * LU\
                + x     * (1-y) * RU\
                + (1-x) * y     * LL\
                + x     * y     * RL\
                -offset
            x2, y2 = tar.intxy
            if(x2<width and y2<height):
                ret.putpixel(tar.intxy, c)
    return offset, ret
if(__name__=="__main__"):
    im = Image.open(r"C:\Users\TkskKurumi\Pictures\QQ图片20220312214042.png")
    offset, im = quad_point(im, Point(0, 0), Point(50, 10), Point(10, 80), Point(70, 40), bg=(0, )*4)
    im.show()
