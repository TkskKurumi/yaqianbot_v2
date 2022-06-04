import shutil
from .colors import color
from .sizefit import fit_shrink, resize_ratio
from PIL import Image


def print_colors(colors, end="\n"):
    if(isinstance(colors, color)):
        print(colors.colored_terminal_str(), end=end)

    elif(isinstance(colors, list)):
        print(end="[")
        for idx, c in enumerate(colors):
            # print(type(c), c)
            if(idx):
                print(end=", ")
            print_colors(c, end="")
        print("]", end=end)
    else:
        raise TypeError(type(colors))


def colored(text, fg: color = None, bg=None):
    ret = []
    if(fg):
        ret.append(fg.as_terminal_fg())
    if(bg):
        ret.append(bg.as_terminal_bg())
    ret.append(text)
    ret.append("\u001b[0m")
    return "".join(ret)


def image_show_terminal(im):
    ts = shutil.get_terminal_size()
    w, h = ts.columns-2, ts.lines-2
    im = resize_ratio(im, rw=0.8, rh=0.4)
    im = fit_shrink(im, w, h).convert("RGB")

    w, h = im.size
    for y in range(h):
        for x in range(w):
            print(colored(" ", bg=color(*im.getpixel((x, y)))), end="")
        print()


if(__name__ == "__main__"):
    from .print import print_colors
    from ..pyxyv import illust
    ill = illust.Illust(88125445)
    im = Image.open(ill.get_pages(0, 1)[0])
    image_show_terminal(im)
