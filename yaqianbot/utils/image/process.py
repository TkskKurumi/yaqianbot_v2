from PIL import Image, ImageDraw, ImageFilter
import numpy as np
from .colors import Color
from math import exp
from typing import Literal, Any


def shadow(img, radius=1, color: Any = Color.from_any("BLACK"), padding=True, sharpness=2, debug=False):
    w, h = img.size
    if(padding):
        tmp = Image.new("RGBA", (w+int(radius*4), h +
                        int(radius*4)), (0, 0, 0, 0))
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
    ret = Image.new("RGBA", (w, h), (0, 0, 0, 0))
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


if(__name__ == "__main__"):
    im = Image.new("RGB", (100, 100), (255, 100, 200))
    from .print import image_show_terminal
    image_show_terminal(adjust_A(im, -0.5))
