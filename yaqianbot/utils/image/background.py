from PIL import Image, ImageDraw, ImageFilter
from math import pi
import numpy as np
from typing import List
from .colors import WHITE, Color, image_colors
import math
import random


def np_colormap(arr, colors):
    h, w = arr.shape
    mx, mn = np.max(arr), np.min(arr)
    arr = (arr-mn)/(mx-mn)
    n_color = len(colors)
    n_ch = len(colors[0])
    ret = np.zeros((h, w, n_ch), np.float16)
    for i in range(n_color-1):
        lo, hi = i/(n_color-1), (i+1)/(n_color-1)
        mask = (arr >= lo) & (arr < hi)
        mask = np.stack([mask]*n_ch, axis=-1)
        norm = (arr-lo)/(hi-lo)
        norm = np.stack([norm]*n_ch, axis=-1)
        color0 = np.array([[colors[i]]*w]*h, np.float16)
        color1 = np.array([[colors[i+1]]*w]*h, np.float16)

        add = color0*(1-norm)+color1*norm
        ret += add*mask

    mask = np.stack([arr >= 1]*n_ch, axis=-1)
    color = np.array([[colors[-1]]*w]*h, np.float16)
    ret += color*mask
    return ret


def colormap(arr: np.ndarray, colors: List):
    meow = np_colormap(arr, colors)
    return Image.fromarray(meow.astype(np.uint8))


def colorvec(arr: np.ndarray, colors: List):
    h, w = arr.shape[:2]
    n_ch = len(colors[0])
    n_color = len(colors)
    color_arr = np.array(colors, np.float16)
    ret = np.zeros((h, w, n_ch), np.float16)
    for y in range(h):
        for x in range(w):
            for i in range(n_color):
                ret[y, x] += color_arr[i]*arr[y, x, i]
            ret[y, x] /= np.sum(arr[y, x])
    return Image.fromarray(ret.astype(np.uint8))


def grids(w, h, color_h=None, color_v=None, angle=45, gap=None):
    if(color_h is None):
        color_h = [WHITE, Color(128, 128, 128)]
    if(color_v is None):
        color_v = color_h
    if(gap is None):
        gap = ((w*w+h*h)**0.5)/20

    def f(x, y, angle):
        nonlocal gap
        return (x*math.cos(angle/180*pi)+y*math.sin(angle/180*pi))/gap
    ret = Image.new("RGBA", (w, h))
    for x in range(w):
        for y in range(h):
            horizontal = f(x, y, angle)
            v = f(x, y, angle+90)
            h_idx = int(horizontal % len(color_h))
            v_idx = int(v % len(color_v))
            c = (color_h[h_idx]+color_v[v_idx])/2
            ret.putpixel((x, y), Color.aspil(c))
    return ret


def random_position(w, h):
    return random.randrange(w), random.randrange(h)


def frandrange(lo, hi):
    return lo+(hi-lo)*random.random()


def triangles(w, h, colors=None, n=None, size=None, m=None, strength=0.5, f_color=None):
    if(colors is None):
        if(isinstance(f_color, Image.Image)):
            colors = image_colors(f_color, 10)
        else:
            colors = [Color.from_any("PINK")]
    colors = [i if isinstance(i, Color) else Color(*i) for i in colors]
    ret = Image.new("RGBA", (w, h), colors[0].get_rgba())
    if(n is None):
        n = 10
    if(m is None):
        m = 10
    if(size is None):
        size = ((w*h)/n/m)**0.5
        size = size*1.2
    for i in range(m):
        layer = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        dr = ImageDraw.Draw(layer)
        for i in range(n):
            x, y = random_position(w, h)
            _size = size*frandrange(0.2, 1.8)
            if(not f_color):
                _alpha = int(random.random()*strength*255)
                _color = random.choice(colors).replace(A=_alpha)
                if(random.random() < 0.5):
                    _color = _color.lighten(random.random()**0.5)
                else:
                    _color = _color.darken(random.random()**0.5)
            elif(isinstance(f_color, Image.Image)):
                _color = Color(*f_color.getpixel((x, y)))
                _alpha = int(random.random()*strength*255)
                _color = _color.replace(A=_alpha)
            else:
                _color = Color(*f_color(x, y))
            dr.regular_polygon((x, y, _size), 3, fill=_color.get_rgba())
        ret = Image.alpha_composite(ret, layer)
    return ret


def normalize(arr, lo, hi):
    mn, mx = np.min(arr), np.max(arr)
    return (arr-mn).astype(np.float32)/(mx-mn)*(hi-lo)+lo


def random_stripe_mask(w, ratio=4, blur=2, rettype="image"):
    arr = np.random.normal(0, 1, (w//ratio, ))
    arr = normalize(arr, 0, 255)
    arr = np.stack([arr]*(w//ratio))
    im = Image.fromarray(arr.astype(np.uint8))
    im = im.resize((w, w), Image.Resampling.NEAREST)
    im = im.filter(ImageFilter.GaussianBlur(blur))
    if(rettype == "image"):
        return im
    else:
        return np.array(im)


def centric_mask(w, rettype="image"):
    arr = np.zeros((w, w), np.uint8)
    r = (w-1)/2
    for x in range(w):
        for y in range(w):
            dist = (x-r)*(x-r)+(y-r)*(y-r)
            dist = dist**0.5
            # y=kx+b; x=0, y=255; x=r, y=0
            alpha = max(-dist*255/r+255, 0)
            arr[x, y] = alpha
    if(rettype == "image"):
        return Image.fromarray(arr)
    else:
        return arr


def random_polygon_mask(w, n=16, rnd=0.2, rettype="image"):
    r = w/2
    points = []
    for i in range(n):
        angle = 2*pi/n*i
        _r = frandrange(rnd, 1)*r
        x, y = r+math.cos(angle)*_r, r+math.sin(angle)*_r
        points.append((x, y))
    ret = Image.new("L", (w, w), 0)
    dr = ImageDraw.Draw(ret)
    dr.polygon(points, fill=255)
    if(rettype=="image"):
        return ret
    else:
        return np.array(ret)

def unicorn(w, h, colora=None, colorb=None, colorc=None, colord=None):
    colora = colora or Color.from_any("lightpink").get_rgb()
    colorb = colorb or [253, 246, 237]
    colorc = colorc or Color.from_any("paleturquoise").get_rgb()
    colord = colord or Color.from_any("lightcyan").get_rgb()

    a, b, c, d, e, _f, g, _h = np.random.normal(0, 1, (8, ))
    o, p, q, r = np.abs(np.random.normal(0, 1, (4, )))+1
    arr = np.zeros((h, w, 4), np.float16)
    sz = math.sqrt(w*w+h*h)/13

    def f(x, y, rx, ry):
        return (x*rx+y*ry)/math.sqrt(rx*rx+ry*ry)
    for x in range(w):
        for y in range(h):
            wa = math.sin(f(x, y, a, e)/sz*o)+1
            wb = math.sin(f(x, y, b, _f)/sz*p)+1
            wc = math.sin(f(x, y, c, g)/sz*q)+1
            wd = math.sin(f(x, y, d, _h)/sz*r)+1
            arr[y, x] = [wa, wb, wc, wd]
    return colorvec(arr, [colora, colorb, colorc, colord])


if(__name__ == "__main__"):
    from .print import image_show_terminal
    a = random_polygon_mask(100, rettype="arr").astype(np.float32)
    a *= random_stripe_mask(100, ratio=2, blur=1, rettype="arr")/255
    a *= centric_mask(100, rettype="arr")/255
    im = Image.fromarray(a.astype(np.uint8))
    image_show_terminal(im)
