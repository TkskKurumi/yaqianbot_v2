from .background import *
from .process import adjust_A, adjust_L
from .colors import Color, image_colors
import numpy as np
from PIL import Image


def image_is_dark(img):
    arr = np.array(img.convert("L"))
    return np.mean(arr) < 128


def gif_frames_duration(gifimg):
    frm = 0
    ret = []
    duration = 0
    while(True):
        try:
            gifimg.seek(frm)
            frm += 1
        except EOFError:
            break
        duration += gifimg.info["duration"]
        ret.append(gifimg.convert("RGBA"))
    return ret, duration


def gif_frames_fps(gifimg):
    frames, dur = gif_frames_duration(gifimg)
    if(dur == 0):
        dur = 1000
    return frames, len(frames)/(dur/1000)

