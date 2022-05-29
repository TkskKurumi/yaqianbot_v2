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
from ..algorithms import kmeans

def image_colors(img:Image.Image, k:int):
    w, h=img.size
    colors = []
    weights = []
    for i in range(100):
        x, y=random.randrange(w),random.randrange(h)
        c = img.getpixel((x, y))
        hue, s, l = color(*c).get_hsl()
        colors.append(c)
        weights.append(s)
    return kmeans(colors, k, weights=weights)

@dataclass
class color:
    R: int = 0
    G: int = 0
    B: int = 0
    A: int = 255

    def __iter__(self):
        return tuple(self.get_rgba()).__iter__()
    
    def get_rgb(self):
        """
            Returns tuple of R, G, B.
        """
        return self.R, self.G, self.B

    def get_rgba(self):
        """
            Returns tuple of R, G, B, A.
        """
        return self.R, self.G, self.B, self.A

    def get_hsl(self):
        """
            Returns tuple of H, S, L
        """
        rgb = self.get_rgb()
        R, G, B = rgb
        mx, mn = max(rgb), min(rgb)

        L = (mx+mn)/2
        if(mx == mn):
            S=0
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
            H, S, L=_list_update_None(self.get_hsl(), (H, S, L))
            R, G, B = color.from_hsl(H, S, L).get_rgb()
        else:
            R, G, B = _list_update_None(self.get_rgb(), (R, G, B))
        if(A is None):
            A = self.A
        return color(R, G, B, A)

    @classmethod
    def from_hsv(cls, H, S, V, A=255):
        r, g, b = ImageColor.getrgb("hsv(%d,%d%%,%d%%)" % (H, S*100, V*100))
        return cls(r, g, b, A)

    @classmethod
    def from_hsl(cls, H, S, L, A=255):
        r, g, b = ImageColor.getrgb("hsl(%d,%d%%,%d%%)" % (H, S*100, L*100))
        return cls(r, g, b, A)
    @classmethod
    def from_any(cls, x):
        return cls(*ImageColor.getrgb(x))
    def lighten(self, rate=0.5):
        H, S, L = self.get_hsl()
        L = 1-rate + L*rate
        return color.from_hsl(H, S, L, self.A)
    def darken(self, rate=0.5):
        H, S, L = self.get_hsl()
        L = L*rate
        return color.from_hsl(H, S, L, self.A)
    def weaken(self, rate=0.5):
        return self.replace(A = self.A*rate)
    def strengthen(self, rate=0.5):
        return self.replace(A = self.A*rate+255*(1-rate))
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
        return color(*np.mean(colors, axis=0))
    def hex(self):
        return "#%s%s%s%s"%tuple([hex(i)[2:] for i in self])
if(__name__ == "__main__"):
    RED = color(255, 0, 0)
    print(RED.get_hsl())
    print(RED.get_hsv())

    print(color.from_hsl(0, 1, 0.5))
    print(color.from_hsv(0, 1, 1))
    print(ImageColor.getrgb("hsl(0,100%,50%)"))

    print(RED.lighten().get_rgb())
    print(RED.replace(H = 120))
    print(RED.replace(R = 233, G=120))
    print(color(114,233,111).replace(S=0.4, L=0.5).hex())
    print(color(233,111,223).replace(S=0.4, L=0.5).hex())