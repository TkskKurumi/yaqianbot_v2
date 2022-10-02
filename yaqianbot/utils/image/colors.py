
from ..algorithms import kmeans
from dataclasses import dataclass
from PIL import ImageColor, Image
import random
import numpy as np


def _list_update_None(ls, *args):
    ret = list(ls)
    for newls in args:
        for idx, i in enumerate(newls):
            if(i is not None):
                ret[idx] = i
    return ret


def _dist(a, b):
    return (np.sum((a-b)**2))**0.5


def image_border_color(img, rettype = "color"):
    colors = []
    w, h = img.size
    arr = img.__array__()
    for x in [0, w-1]:
        for y in range(h):
            colors.append(arr[y, x])
    for x in range(w):
        for y in [0, h-1]:
            colors.append(arr[y, x])
    if(rettype == "color"):
        return Color(*np.mean(colors, axis=0))
    elif(rettype == "np"):
        return np.mean(colors, axis=0)
    else:
        raise ValueError(rettype)


def image_colors(img: Image.Image, k: int, weight_by_s = True, return_type = "color"):
    w, h = img.size
    colors = []
    weights = []
    for i in range(k*10):
        x, y = random.randrange(w), random.randrange(h)
        c = img.getpixel((x, y))
        colors.append(c)
        if(weight_by_s):
            hue, s, l = Color(*c).get_hsv()
            weights.append(s)
        else:
            weights = None
    ret = kmeans(colors, k, weights=weights)
    if(return_type == "color"):
        return [Color(*i) for i in ret]
    else:
        return np.array(ret)


@dataclass
class Color:
    R: int = 0
    G: int = 0
    B: int = 0
    A: int = 255

    def __iter__(self):
        return (self.R, self.G, self.B, self.A).__iter__()

    def __mul__(self, n):
        return Color(*[i*n for i in self])

    def __truediv__(self, n):
        return Color(*[i/n for i in self])

    def __add__(self, other):
        if(isinstance(other, Color)):
            ls1 = list(other)
            ls2 = list(self)
            return Color(*[ls1[i] + ls2[i] for i in range(4)])
        else:
            return NotImplemented

    def interpolate_hsl(self, other, n):
        h1, s1, l1 = self.get_hsl()
        h2, s2, l2 = other.get_hsl()
        s = s1*(1-n)+s2*n
        l = l1*(1-n)+l2*n
        h1, h2, nh = min(h1, h2), max(h1, h2), n if h1 < h2 else 1-n
        deltah = h2-h1
        if(deltah < 180):
            h = h1+deltah*n
        else:
            h = (h1+(deltah-360)*n) % 360
        return Color.from_hsl(h, s, l)

    def interpolate_rgb(self, other, n):
        return self*(1-n)+other*n

    def interpolate(self, other, n, mode="RGB"):
        if(mode == 'RGB'):
            return self.interpolate_rgb(other, n)
        else:
            return self.intrepolate_hsl(other, n)

    def get_rgb(self):
        """
            Returns tuple of R, G, B.
        """
        return int(self.R), int(self.G), int(self.B)

    def get_rgba(self):
        """
            Returns tuple of R, G, B, A.
        """
        return int(self.R), int(self.G), int(self.B), int(self.A)

    def get_hsl(self):
        """
            Returns tuple of H, S, L
        """
        rgb = self.get_rgb()
        R, G, B = rgb
        mx, mn = max(rgb), min(rgb)

        L = (mx+mn)/2
        if(mx == mn):
            S = 0
        else:
            if(L < 128):
                S = (mx-mn)/(mx+mn)
            else:
                S = (mx-mn)/(2*255-mx-mn)
        if(L == 0):
            H = 0
        elif(mx == R):
            H = 30*(G - B)/L
        elif(mx == G):
            H = 120 + 30*(B - R)/L
        else:
            H = 240 + 30*(R - G)/L
        return H % 360, S, L/255

    def get_hsv(self):
        """
            Returns tuple of H, S, V
        """
        V = max(self.get_rgb())
        denomina = V - min(self.get_rgb())
        if(V == 0):
            S = 0
        else:
            S = denomina/V
        R, G, B = self.get_rgb()
        if(denomina == 0):
            H = 0
        elif(V == R):
            H = 60*(G-B)/denomina
        elif(V == G):
            H = 120+60*(B-R)/denomina
        else:
            H = 240+60*(R-G)/denomina
        return H % 360, S, V/255

    def replace(self, R=None, G=None, B=None, H=None, S=None, L=None, A=None):
        isrgb = (R is not None) or (G is not None) or (B is not None)
        ishsl = (H is not None) or (S is not None) or (L is not None)
        if(isrgb and ishsl):
            raise ValueError("RGB or HSL conflicts")
        if(ishsl):
            H, S, L = _list_update_None(self.get_hsl(), (H, S, L))
            R, G, B = Color.from_hsl(H, S, L).get_rgb()
        else:
            R, G, B = _list_update_None(self.get_rgb(), (R, G, B))
        if(A is None):
            A = self.A
        return Color(R, G, B, A)

    @classmethod
    def from_hsv(cls, H, S, V, A=255):
        r, g, b = ImageColor.getrgb("hsv(%d,%d%%,%d%%)" % (H, S*100, V*100))
        return cls(r, g, b, A)

    @classmethod
    def from_hsl(cls, H, S, L, A=255):
        H = H%360
        r, g, b = ImageColor.getrgb("hsl(%d,%d%%,%d%%)" % (H, S*100, L*100))
        return cls(r, g, b, A)

    @classmethod
    def from_any(cls, x):
        if(isinstance(x, tuple)):
            return cls(*x)
        elif(isinstance(x, Color)):
            return x
        elif(isinstance(x, str)):
            return cls(*ImageColor.getrgb(x))

    def lighten(self, rate=0.5):
        H, S, L = self.get_hsl()
        L += (1-L)*rate
        return Color.from_hsl(H, S, L, self.A)

    def darken(self, rate=0.5):
        H, S, L = self.get_hsl()
        L += (0-L)*rate
        return Color.from_hsl(H, S, L, self.A)

    def weaken(self, rate=0.5):
        return self.replace(A=self.A*rate)

    def strengthen(self, rate=0.5):
        return self.replace(A=self.A*rate+255*(1-rate))

    def aspil(self, mode='RGB'):
        if(mode == "RGBA"):
            return tuple([int(i) for i in self.get_rgba()])
        else:
            return tuple([int(i) for i in self.get_rgb()])

    def borderof(img):
        w, h = img.size
        colors = []
        for x in [0, w-1]:
            for y in range(h):
                colors.append(img.getpixel((x, y)))
        for x in range(w):
            for y in [0, h-1]:
                colors.append(img.getpixel((x, y)))
        return Color(*np.mean(colors, axis=0))

    def hex(self):
        return "#%s%s%s%s" % tuple([hex(i)[2:] for i in self])

    def aspil(self):
        if(isinstance(self, Color)):
            return self.get_rgba()
        else:
            return tuple(self)

    def as_terminal_fg(self):
        return "\u001b[38;2;%d;%d;%dm" % self.get_rgb()

    def as_terminal_bg(self):
        return "\u001b[48;2;%d;%d;%dm" % self.get_rgb()
    def as_terminal_rst(self):
        return "\u001b[0m"
    def colored_terminal_str(self):
        if(sum(self.get_rgb()) < 3*255/2):
            fg = Color(255, 255, 255)
        else:
            fg = Color(0, 0, 0)
        return "%s%s%s%s"%(self.as_terminal_bg(), fg.as_terminal_fg(), self, self.as_terminal_rst())

BLACK = Color(0, 0, 0)
WHITE = Color(255, 255, 255)
# Color = Color
if(__name__ == "__main__"):
    # test
    from .print import print_colors
    from ..pyxyv import illust
    ill = illust.Illust(88125445)
    im = ill.get_pages(0, 1)[0]
    im = Image.open(im)
    print(image_colors(im, 1))
    print_colors(image_colors(im, 1)[0].replace(S=0.2, L=0.8))
    print_colors(image_border_color(im))
