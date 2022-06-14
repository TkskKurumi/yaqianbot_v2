import shutil
from .colors import Color
from .sizefit import fit_shrink, resize_ratio
from PIL import Image
import tempfile
from os import path
from .. import myhash
import numpy as np
def print_colors(colors, end="\n"):
    if(isinstance(colors, list)):
        print(end="[")
        for idx, c in enumerate(colors):
            # print(type(c), c)
            if(idx):
                print(end=", ")
            print_colors(c, end="")
        print("]", end=end)
    else:
        print(colors.colored_terminal_str(), end=end)
    # else:
    #     raise TypeError(type(colors))


def colored(text, fg: Color = None, bg=None):
    ret = []
    if(fg):
        ret.append(fg.as_terminal_fg())
    if(bg):
        ret.append(bg.as_terminal_bg())
    ret.append(text)
    ret.append("\u001b[0m")
    return "".join(ret)


def image_show_terminal(im, rate = 1):
    hashed = myhash.base32(np.array(im.resize((10, 10))))
    pth = path.join(tempfile.gettempdir(), "%s.png"%hashed)
    im.save(pth)
    ts = shutil.get_terminal_size()
    w, h = ts.columns-2, ts.lines*rate-2
    im = resize_ratio(im, rw=1, rh=0.4)
    im = fit_shrink(im, w, h).convert("RGB")

    w, h = im.size
    for y in range(h):
        for x in range(w):
            print(colored(" ", bg=Color(*im.getpixel((x, y)))), end="")
        print()
    
    
    print(pth)
if(__name__ == "__main__"):
    from .print import print_colors
    from ..pyxyv import illust
    ill = illust.Illust(88125445)
    im = Image.open(ill.get_pages(0, 1)[0])
    image_show_terminal(im)
