from PIL import Image, ImageDraw, ImageFilter
import numpy as np
from .colors import Color
from math import exp


def shadow(img, radius=1, color:Color=Color.from_any("BLACK"), padding=True, sharpness=2):
    w, h = img.size
    if(padding):
        tmp = Image.new("RGBA", (w+int(radius*4), h+int(radius*4)), (0, 0, 0, 0))
        tmp.paste(img, box=(int(radius*2), int(radius*2)))
        img = tmp
        w, h = img.size
    else:
        pass
        # ret = img.convert("RGBA")
    
    
    mask = Image.fromarray(np.array(img)[:,:,3])
    
    mask = mask.filter(ImageFilter.GaussianBlur(radius/1.5))
    
    mask = np.array(mask).astype(np.float16)/255
    # print(np.min(mask), np.max(mask))
    # mask = np.maximum(mask-0.5, 0)*2
    if(sharpness > 0):
        # mask = np.array(mask).astype(np.float16)/255
        mask = mask**exp(-sharpness)
    mask = Image.fromarray((mask*255).astype(np.uint8))
    # return mask
    ret = Image.new("RGBA", (w, h), (0,0,0,0))
    color = Image.new("RGBA", (w, h), color.get_rgba())
    # print(mask)
    ret.paste(color, mask=mask)
    ret.alpha_composite(img)
    return ret
if(__name__=="__main__"):
    from pil_functional_layout.widgets import Text
    t = Text("foobar", fill=(255,123,123), fontSize=36).render()
    from .print import image_show_terminal
    # image_show_terminal(t)
    # exit()
    t = shadow(t, color=Color.from_any("CYAN"))
    image_show_terminal(t)