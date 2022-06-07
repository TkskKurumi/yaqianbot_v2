from ..backend import receiver, startswith
from ..backend import threading_run
from ..backend.cqhttp import CQMessage
import re
import random
from datetime import timedelta
import numpy as np
from PIL import Image
from ..utils import image
from ..utils.image.sizefit import fit_shrink, fix_width
from ..utils.image import gif_frames_duration
from pil_functional_layout.widgets import Column, RichText, Text
from ..utils.make_gif import make_gif
from ..utils import after_match
'''
@receiver
@threading_run
@startswith("/要我一直gif")
def cmd_ywyz_gif(message: CQMessage):
    img = message.get_sent_images()[0][1].convert("RGB")
    img = fit_shrink(img, 500, 500)
    w, h = img.size
    text1 = Text("要我一直", fontSize=36, bg=(255,)*4).render()
    small_image = fit_shrink(img, height=36)
    text2 = Text("吗", fontSize=36, bg=(255,)*4).render()
    w, h = max(img.size[0], sum([i.size[0]
               for i in [text1, small_image, text2]]))
    h = img.size[0]+max([i.size[1] for i in [text1, small_image, text2]])'''

def pic_ywyz(img):
    img = fix_width(img, 500)
    fontSize = 48
    rt = RichText(["要我一直", fit_shrink(img, height=fontSize),
                  "吗？"], fontSize=fontSize, width=500, autoSplit=False, dont_split=True)
    ret = Column([img, rt], bg=(255,)*3).render()
    return ret
@receiver
@threading_run
@startswith("/要我一直")
def cmd_ywyz(message: CQMessage):
    imgtype, img = message.get_sent_images()[0]
    if(imgtype == "image/gif"):
        frames, fps = image.gif_frames_fps(img)
        retframes = [pic_ywyz(i) for i in frames]
        gif = make_gif(retframes, fps=fps, frame_area_sum = 1e6)
        message.response_sync(gif)
    else:
        ret = pic_ywyz(img)
        message.response_sync(ret)
def img_filter(imgtype, img, filter):
    if(imgtype == "image/gif"):
        frames, fps = image.gif_frames_fps(img)
        retframes = [filter(frm) for frm in frames]
        gif = make_gif(retframes, fps=fps, frame_area_sum = 2e6)
        return gif
    else:
        return filter(img)

@receiver
@threading_run
@startswith("(/字幕)|(/加字)")
def cmd_face_caption(message: CQMessage):
    text = after_match("(/字幕)|(/加字)", message.plain_text).strip("\r\n ")
    def f(img):
        nonlocal text
        img = fix_width(img, 300)
        RT = RichText(text, fontSize = 36, width = 250, dont_split=False, autoSplit=False)
        COL = Column([img, RT], bg=(255,)*4, alignX=0.5)
        return COL.render()
    imgtype, img = message.get_sent_images()[0]
    im = img_filter(imgtype, img, f)
    message.response_sync(im)
@receiver
@threading_run
@startswith("/群青")
def cmd_gunjou(message: CQMessage):
    img = message.get_sent_images()[0][1].convert("RGB")
    arr = np.asarray(img)
    gray = arr.mean(axis=2, keepdims=False)
    colors = []
    colors.append((0, 0, 0))
    colors.append((0, 0, 255))
    colors.append((100, 155, 255))
    colors.append((255, 255, 255))
    out_arr = image.np_colormap(gray, colors)

    message.response_sync(Image.fromarray(out_arr.astype(np.uint8)))
