from .background import *
from .colors import *
import numpy as np
def image_is_dark(img):
    arr = np.array(img.convert("L"))
    return np.mean(arr)<128

def gif_frames_duration(gifimg):
    frm = 0
    ret = []
    duration = 0
    while(True):
        try:
            gifimg.seek(frm)
            frm+=1
        except EOFError:
            break
        duration+=gifimg.info["duration"]
        ret.append(gifimg.convert("RGBA"))
    return ret, duration
def gif_frames_fps(gifimg):
    frames, dur = gif_frames_duration(gifimg)
    return frames, len(frames)/(dur/1000)