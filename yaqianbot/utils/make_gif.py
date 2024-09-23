from PIL import Image
import time
from yaqianbot.backend.cqhttp.message import GREYSCALE
from . import myhash
from typing import List
from .image.sizefit import _wh_fit_area
import numpy as np
import tempfile
from os import path
import os
def gettempdir():
    usr = path.expanduser("~")
    return path.join(usr, ".tmp", "make_gif")
def make_mp4(frames, fps=24):
    # _ls = list(frames)+[fps]
    hashed = myhash.base32(frames+[fps], length = 10)
    pth = path.join(gettempdir(), "make_gif", hashed)
    outpth = path.join(pth, "out.mp4")
    if(path.exists(outpth)):
        return outpth
    elif(not path.exists(pth)):
        os.makedirs(pth)
    for idx, i in enumerate(frames):
        i.convert("RGB").save(path.join(pth, "%03d.jpg"%idx))
    scripts = ["ffmpeg", "-i", path.join(pth, "%03d.jpg"), "-c:v", "mpeg4", "-r", str(fps), outpth]
    scripts = " ".join(scripts)
    print(scripts)
    p = os.popen(scripts)
    log = p.read()
    return outpth


def make_gif_size(frames: List[Image.Image], fps=24, filesize_lim=1<<20):
    ratio = 1.0
    width, height = frames[0].size
    def resize(im: Image.Image):
        nonlocal width, height, ratio
        w, h = round(width*ratio), round(height*ratio)
        return im.resize((w, h), Image.Resampling.LANCZOS)
    def thumbnail(im):
        return im.resize((8, 8))
    thumbnailed = [thumbnail(i) for i in frames]
    hashname = myhash.base32([thumbnailed, int(time.time())])
    pth = path.join(gettempdir(), "make_gif", hashname)
    os.makedirs(pth)
    gifpth = path.join(pth, "out.gif")
    while (True):
        resized = [resize(i) for i in frames]
        for idx, i in enumerate(resized):
            i.save(path.join(pth, "%04d.png"%idx))
        script = ["gifski", path.join(pth, "*.png"), "--fps", "%d"%fps, "-o", gifpth]
        pipe = os.popen(" ".join(script))
        result = pipe.read()
        pipe.close()
        result_size = path.getsize(gifpth)
        print("GIF Size", result_size/1024/1024, "MB, %dx%d %d frames"%(width*ratio, height*ratio, len(frames)))
        if (result_size < filesize_lim):
            return gifpth
        else:
            ratio *= min(0.97, (filesize_lim/result_size)**0.5)**1.1
    

        
GREYSCALE = False
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
        if(GREYSCALE):
            im = im.convert("LA")
        resized.append(im)
        tmp = np.array(im.resize((4, 4)))
        # print(myhash.myhash(tmp))
        hashed = (hashed << 7) ^ myhash.myhash(tmp)
        hashed = (hashed & mask) ^ (hashed >> hash_len)
    hashed = myhash.base32(hashed)
    pth = path.join(gettempdir(), "make_gif", hashed)
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
    print(make_mp4(frames))
