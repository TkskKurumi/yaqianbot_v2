from PIL import Image
from . import myhash
from typing import List
from .image.sizefit import _wh_fit_area
import numpy as np
import tempfile
from os import path
import os


def make_gif(frames: List[Image.Image], fps=24, area=None, frame_area_sum=None):
    print("making gif")
    if(len(frames)==1):
        frames = frames+frames
    w, h = frames[0].size
    if((frame_area_sum is not None) and (area is None)):
        area = frame_area_sum/len(frames)
    if(area is not None):
        w, h = _wh_fit_area(w, h, area)
    resized = list()
    hashed = 0
    hash_len = 50
    mask = (1 << hash_len)-1
    for idx, i in enumerate(frames):
        if(i.size != (w, h)):
            im = i.resize((w, h), Image.BILINEAR)
        else:
            im = i
        resized.append(im)
        tmp = np.array(im.resize((4, 4)))
        print(myhash.myhash(tmp))
        hashed = (hashed << 7) ^ myhash.myhash(tmp)
        hashed = (hashed & mask) ^ (hashed >> hash_len)
    hashed = myhash.base32(hashed)
    pth = path.join(tempfile.gettempdir(), "make_gif", hashed)
    if(path.exists(pth)):
        if(path.exists(path.join(pth, "out.gif"))):
            return path.join(pth, "out.gif")
    else:
        os.makedirs(pth)
    for idx, i in enumerate(resized):
        i.save(path.join(pth, "%03d.png" % idx))
    script = ["gifski", path.join(pth, "*.png")]
    script.extend(["--fps", "%d" % fps])
    script.extend(["--output", path.join(pth, "out.gif")])
    script = " ".join(script)
    p = os.popen(script)
    print(p.read())
    p.close()
    return path.join(pth, "out.gif")


if(__name__ == "__main__"):
    from .image import background
    a = np.array(background.unicorn(64, 64, colora=(255, 0, 0)))
    b = np.array(background.unicorn(64, 64, colora=(0, 0, 255)))
    frames = []
    for i in range(25):
        tmp = i/25
        arr = a*tmp + b*(1-tmp)
        frames.append(Image.fromarray(arr.astype(np.uint8)))
    print(make_gif(frames))
