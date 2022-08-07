from PIL import Image
import numpy as np
from . import sizefit


def _hytk_LA(LDark, LLight, scale=255):
    A1 = 1-(LLight-LDark)/scale
    L = LDark/A1
    return L, A1*scale


def _hytk_RGBA(RGBDark, RGBLight):
    RGBA = []
    As = []
    for ch in range(3):
        LD = RGBDark[:, :, ch]
        LL = RGBLight[:, :, ch]
        L, A = _hytk_LA(LD, LL)
        RGBA.append(L)
        As.append(A)
    A = np.mean(As, axis=0)
    RGBA.append(A)
    return np.stack(RGBA, axis=-1)


def hytk(im_dark, im_light, brightness=0.5):
    arr_dark = np.array(im_dark.convert("RGB"))*brightness
    if(im_light.size != im_dark.size):
        im_light = sizefit.fit_crop(im_light, *im_dark.size)
    arr_light = np.array(im_light.convert("RGB"))*(1-brightness)+255*brightness
    arr = _hytk_RGBA(arr_dark, arr_light)
    arr = arr.astype(np.uint8)
    return Image.fromarray(arr)


if(__name__ == '__main__'):
    from ...plugins.plg_pixiv import rand_img
    from .print import image_show_terminal
    A = Image.open(rand_img())
    B = Image.open(rand_img())
    C = hytk(A, B)
    
    
    image_show_terminal(A, rate=0.2)
    image_show_terminal(B, rate=0.2)
    image_show_terminal(C, rate=0.2)
    
    Dark = Image.new("RGBA", C.size, (0,0,0,255))
    Dark.alpha_composite(C)
    Light = Image.new("RGBA", C.size, (255,255,255, 255))
    Light.alpha_composite(C)
    image_show_terminal(Dark, rate=0.2)
    image_show_terminal(Light, rate=0.2)