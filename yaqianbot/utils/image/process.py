from PIL import Image, ImageDraw, ImageFilter
import numpy as np
from .colors import Color, image_colors
from . import colors
from math import exp
from typing import Literal, Any
import random
from ..algorithms import kmeans

def shadow(img, radius=1, color: Any = Color.from_any("BLACK"), padding=True, sharpness=2, debug=False):
    w, h = img.size
    bgc = colors.image_border_color(img).get_rgba()
    if(padding):
        tmp = Image.new("RGBA", (w+int(radius*4), h +
                        int(radius*4)), bgc)
        tmp.paste(img, box=(int(radius*2), int(radius*2)))
        img = tmp
        w, h = img.size
    else:
        pass
        # ret = img.convert("RGBA")

    mask = Image.fromarray(np.array(img)[:, :, 3])

    mask = mask.filter(ImageFilter.GaussianBlur(radius/1.5))

    mask = np.array(mask).astype(np.float16)/255
    # print(np.min(mask), np.max(mask))
    # mask = np.maximum(mask-0.5, 0)*2
    if(sharpness > 0):
        # mask = np.array(mask).astype(np.float16)/255
        mask = mask**exp(-sharpness)
    mask = Image.fromarray((mask*255).astype(np.uint8))
    
    if(debug):
        from .print import image_show_terminal
        image_show_terminal(img, caption="shadow img")
    ret = Image.new("RGBA", (w, h), bgc)
    if(color is not None):
        color = Image.new("RGBA", (w, h), color.get_rgba())
    else:
        color = img.filter(ImageFilter.GaussianBlur(radius/1.5))
        color = adjust_A(color, 1)
    if(debug):
        from .print import image_show_terminal
        image_show_terminal(color, caption="shadow color")
        image_show_terminal(mask, caption="shadow mask")
    ret.paste(color, mask=mask)
    ret.alpha_composite(img)
    return ret


def adjust_A(im, adjust=-0.5):
    arr = im.convert("RGBA")
    arr = np.array(arr)
    alpha = arr[:, :, -1]
    if(adjust < 0):
        rate = -adjust
        alpha = alpha*(1-rate)
    else:
        alpha = alpha*(1-adjust)+255*adjust
    arr[:, :, -1] = alpha
    return Image.fromarray(arr)



def adjust_L(im, adjust=-0.9):
    alpha = int(abs(adjust)*255)
    if(adjust < 0):
        color = (0, 0, 0, alpha)
    else:
        color = (255, 255, 255, alpha)
    cover = Image.new("RGBA", im.size, color)
    ret = Image.alpha_composite(im.convert("RGBA"), cover)
    return ret.convert(im.mode)

def color_segmentation(im: Image.Image, k=3, seed=None, temperature=1, norm_area=False):
    if(seed):
        random.seed(seed)
        np.random.seed(seed)
    if(im.mode not in ["RGB", "RGBA"]):
        im = im.convert("RGBA")
    arr = np.array(im).astype(np.float32)
    h, w, ch = arr.shape
    colors = []
    for i in range(int((w*h)**0.5)*k):
        y, x = random.randrange(h), random.randrange(w)
        c = im.getpixel((x, y))
        colors.append(c)
    colors = np.array(kmeans(colors, k))
    print(colors)
    # color_avg = arr.mean(axis=0).mean(axis=0)
    color_avg = np.mean(colors, axis=0)
    colors = colors-color_avg
    # print(colors)
    rets = []
    for i in range(k):
        c = colors[i]
        # /np.linalg.norm(colors[i])
        print(c)
        # print(c, arr[0, 0], (arr[0, 0]*c).sum())
        dot = (arr-color_avg)*c
        # print(dot)
        dot = dot.sum(axis=-1)
        # print(dot)
        rets.append(dot)
    ret = np.stack(rets, axis=-1)
    ret = ret-ret.mean(axis=-1, keepdims=True)
    ret = ret/np.std(ret)*temperature
    # ret = ret/(np.std(ret, axis=-1, keepdims=True)+1e-9)*temperature
    
    print(ret.max(), ret.min())
    ex = np.exp(ret)
    if(norm_area):
        ex = ex/(ex.mean(axis=0, keepdims=True).mean(axis=1, keepdims=True))
    exsum = ex.sum(axis=-1, keepdims=True)
    return ex/exsum

    return ret



    


if(__name__ == "__main__"):
    im = Image.open(r"G:\setubot-iot\2654bfb199.jpg")
    
    arr = color_segmentation(im, temperature=2, k=3, seed=1)[:,:,:3]
    # print(arr)
    arr = (arr-arr.min())/(arr.max()-arr.min())*255

    im = Image.fromarray(arr.astype(np.uint8))

    im.show()
