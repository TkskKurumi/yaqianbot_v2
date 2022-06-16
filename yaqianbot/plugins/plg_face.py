from ..backend.cqhttp.message import mes_str2arr
from ..backend.receiver_decos import on_exception_response, command
from ..backend import receiver, startswith
from ..backend import threading_run
from ..backend.cqhttp import CQMessage
import re
import random
from datetime import timedelta
import numpy as np
from PIL import Image, ImageDraw
from ..utils import image, algorithms
from ..utils.image.sizefit import fit_shrink, fix_width
from ..utils.image import sizefit, colors
from ..utils.image import gif_frames_duration, background
from pil_functional_layout.widgets import Column, RichText, Text

from pil_functional_layout.widgets import RichText
from ..utils.make_gif import make_gif
from ..utils import after_match
from ..utils.algorithms.kdt import kdt as KDT
from ..utils.algorithms.kdt import point as KDTPoint
from ..backend.requests import get_image
from ..utils.image import sizefit
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
        gif = make_gif(retframes, fps=fps, frame_area_sum=1e6)
        message.response_sync(gif)
    else:
        ret = pic_ywyz(img)
        message.response_sync(ret)


def img_filter(imgtype, img, filter):
    if(imgtype == "image/gif"):
        frames, fps = image.gif_frames_fps(img)
        retframes = [filter(frm) for frm in frames]
        gif = make_gif(retframes, fps=fps, frame_area_sum=2e6)
        return gif
    else:
        return filter(img)


@receiver
@threading_run
@on_exception_response
@startswith("/test")
def cmd_face_test(message: CQMessage):
    points = []

    imgtype, img = message.get_sent_images()[0]
    w, h = img.size
    mx = 5e4
    if(w*h > mx):
        img = sizefit.area(img, mx)
        w, h = img.size
    for i in range(100):
        x, y = random.randrange(w), random.randrange(h)
        points.append((x, y))
    kdt = KDT()
    kdt.build(points)

    def f(img):
        nonlocal w, h, kdt
        ret = Image.new(img.mode, img.size)
        for x in range(w):
            for y in range(h):
                p = KDTPoint((x, y))
                p1, dist = kdt.ann(p, recall_alpha=0.8)
                ox, oy = p1
                ret.putpixel((x, y), img.getpixel((ox, oy)))
        return ret
    message.response_sync(f(img.convert("RGB")))


@receiver
@threading_run
@startswith("/看扁")
def cmd_face_kanbian(message: CQMessage):
    foreground = Text("真的是被看扁了呢", fontSize=72, bg=(0, 0, 0, 0)).render()

    def f(img):
        img = img.resize(
            foreground.size, Image.Resampling.BILINEAR).convert("RGBA")
        return Image.alpha_composite(img, foreground)
    imgtype, img = message.get_sent_images()[0]
    ret = img_filter(imgtype, img, f)
    message.response_sync(ret)


@receiver
@threading_run
@startswith("(/8bit)|(/像素风)")
def cmd_face_8bit(message: CQMessage):
    text = after_match("(/8bit)|(/像素风)", message.plain_text).strip()
    if(text.isdigit()):
        n = min(max(int(text), 8), 256)
    else:
        n = 32

    imgtype, img = message.get_sent_images()[0]
    if(imgtype == "image/gif"):
        frms, fps = image.gif_frames_fps(img)
        w, h = frms[0].size
        cols = []
        for i in range(200):
            x, y = random.randrange(w), random.randrange(h)
            c = random.choice(frms).getpixel((x, y))
            cols.append(c)
        color16 = np.array(algorithms.kmeans(cols, n))
    else:
        color16 = None

    def f(img):
        nonlocal n, color16
        w, h = img.size
        if(w*h > 1e6):
            img = sizefit.area(img, 1e6)
            w, h = img.size
        w1, h1 = sizefit._rate(w, h, rate=n/w)
        if(color16 is None):
            color16 = colors.image_colors(img, n, return_type="array")
        img = img.resize((w1, h1), Image.Resampling.LANCZOS)

        for x in range(w1):
            for y in range(h1):
                c = img.getpixel((x, y))
                tmp = color16-c
                tmp = np.sum(tmp**2, axis=-1)
                idx = np.argmin(tmp)
                img.putpixel((x, y), tuple([int(i) for i in color16[idx]]))
        img = img.resize((w, h), Image.Resampling.NEAREST)

        return img
    ret = img_filter(imgtype, img, f)
    message.response_sync(ret)


@receiver
@threading_run
@startswith("(/字幕)|(/加字)")
def cmd_face_caption(message: CQMessage):
    text = after_match("(/字幕)|(/加字)", message.plain_text).strip("\r\n ")

    def f(img):
        nonlocal text
        img = fix_width(img, 300)
        RT = RichText(text, fontSize=36, width=250,
                      dont_split=False, autoSplit=False)
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


@receiver
@threading_run
@on_exception_response
@command("/三角形", opts={})
def cmd_triangle(message: CQMessage, *args, **kwargs):
    n = 44
    def f(img):
        nonlocal n
        ret = background.triangles(*img.size, f_color=img, n=n, m=n)
        return ret
    imgtype, img = message.get_sent_images()[0]
    ret = img_filter(imgtype, img, f)
    return message.response_sync(ret)


@receiver
@threading_run
@on_exception_response
@command("鲁迅说", opts={})
def cmd_luxunrt(message, *args, **kwargs):
    mes = message.raw.message
    if(isinstance(mes, str)):

        mes = mes_str2arr(mes)
        print(mes)
    content = []
    url = r"https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQHs3ZdlhN-jfFjxKVcLrkP4F3d-bE2qEL6jQ&usqp=CAU"

    luxun = get_image(url)[1].convert("RGBA")
    w, h = luxun.size
    print(mes)
    for i in mes:
        type = i["type"]
        data = i["data"]
        # print(i)
        if(type == "text"):
            t = data["text"]
            if(t.startswith("鲁迅说")):
                t = t[3:]
            content.append(t)
        elif(type == "image"):
            im = get_image(data["url"])[1]
            # im = sizefit.fit_shrink(im, w*0.9, h*0.5)
            content.append(im)
    content.append("\n--鲁迅")
    RT = RichText(content,
                  fill=(255,)*4,
                  fontSize=w//14,
                  width=int(w*0.9),
                  imageLimit=(int(w*0.9), int(h*0.25)),
                  alignX=1
                  )
    RT = RT.render()
    ww, hh = RT.size
    le = (w-ww)//2
    up = (h-hh)
    luxun.paste(RT, box=(le, up), mask=RT)
    message.response_sync(luxun)
