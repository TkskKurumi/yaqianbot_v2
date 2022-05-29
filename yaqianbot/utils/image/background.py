from PIL import Image, ImageDraw
import numpy as np
from typing import List
from .colors import color
import math


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


def grids(w, h, color_h = None, color_v = None, angle = 45):
    if(color_h is None):
        color(0,0,0).get_rgb()

    def f(x, y, angle):
        return x*math.cos(angle)+y*math.sin(angle)

def unicorn(w, h, colora = None, colorb = None, colorc= None,colord=None):
    colora = colora or color.from_any("lightpink").get_rgb()
    colorb = colorb or  [253, 246, 237]
    colorc = colorc or color.from_any("paleturquoise").get_rgb()
    colord = colord or color.from_any("lightcyan").get_rgb()

    a, b, c, d, e, _f, g, _h = np.random.normal(0, 1, (8, ))
    o ,p, q, r = np.abs(np.random.normal(0, 1, (4, )))+1
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
    
    im = unicorn(300, 200, colora = color.from_any("salmon").get_rgb(), colorb=color.from_any("darkorchid").get_rgb())
    im.save("./tmp.jpg")
    
    