from .background import *
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


def adjust_L(im, adjust=-0.9):
    alpha = int(abs(adjust)*255)
    if(adjust < 0):
        color = (0, 0, 0, alpha)
    else:
        color = (255, 255, 255, alpha)
    cover = Image.new("RGBA", im.size, color)
    ret = Image.alpha_composite(im.convert("RGBA"), cover)
    return ret.convert(im.mode)
