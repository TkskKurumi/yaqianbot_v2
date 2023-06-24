from functools import partial
from ..backend.cqhttp.message import mes_str2arr
from ..backend.receiver_decos import on_exception_response, command
from ..backend import receiver, startswith
from ..backend import threading_run
from ..backend.cqhttp import CQMessage
from ..utils import np_misc
import re
import random
from datetime import timedelta
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from ..utils import image, algorithms
from ..utils.image.sizefit import fit_shrink, fix_width
from math import pi as PI
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
from ..utils.image.colors import Color
import math
from ..utils.candy import simple_send
from ..utils.geometry.elements import Point2d
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
@command("/卡拉比丘", opts={"-k"})
def cmd_face_test(message: CQMessage, *args, **kwargs):
    from ..utils.image.process import color_segmentation
    from ..utils.image.deform import quad_point
    from ..utils.geometry.elements import Point2d as Point
    imgtype, img = message.get_sent_images()[0]

    def italy(im):
        s = (im.width**2 + im.height**2)**0.5
        _, ret = quad_point(im, Point(0, 0), Point(im.width, 0), Point(-s/10, im.height), Point(im.width-s/10, im.height))
        return ret
    
    ret = Image.new("RGBA", (700, 500), (255,)*4)
    dr = ImageDraw.Draw(ret)
    w, h = ret.size
    siz = (w*w+h*h)**0.5
    fs = siz/18
    CTR = Point(w/2, h/2)
    rad = siz/2

    ww, hh = img.size
    
    LL2RL = Point(1, -1.3).unit()
    UPWARD = Point(0, -1)
    LL = Point(w/10, h*0.9)
    RL = LL + LL2RL*siz/3
    neww = (RL-LL).length()
    LU = LL + UPWARD*neww/w*h*2
    RU = RL + UPWARD*neww/w*h*2*0.7

    offset, formed = quad_point(img.convert("RGBA"), LU, RU, LL, RL, bg=(0, )*4)

    for i in range(300):
        rnd_angle = random.random()*3.14/300
        rndc0 = random.random()*0.3+0.3
        rndc1 = random.random()
        A = Point(0, rad).rotate_by_angle(random.random()*3.14*2)
        B = A.rotate_by_angle(rnd_angle)
        C = (A*rndc1 + B*(1-rndc1))*rndc0

        dr.polygon(((A+CTR).xy, (B+CTR).xy, (C+CTR).xy), fill=(0, 0, 0, 255))

    ret.paste(formed, box=offset.intxy, mask=formed)

    klbq = italy(RichText(["卡拉比丘"], 512, fontSize=int(fs*1.5), bg=(0, )*4).render())
    texts = RichText(["我要成为\n", klbq, "\n高手"], 512, fontSize=int(fs), bg=(255, 255, 255, 50), alignX=0.5).render()
    texts = italy(texts)
    ret.paste(texts, box=(w-texts.width, 0), mask=texts)

    simple_send(ret)

    
    


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
@on_exception_response
@startswith("(/字幕)|(/加字)")
def cmd_face_caption(message: CQMessage):
    text = after_match("(/字幕)|(/加字)", message.plain_text).strip("\r\n ")

    def f(img):
        nonlocal text
        img = fix_width(img, 300)
        RT = RichText(text, fontSize=36, width=250,
                      dontSplit=False, autoSplit=False).render()
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


hanzi2color = {
    "黑": "BLACK",
    "白": "WHITE",
    "红": "RED",
    "蓝": "BLUE",
    "粉": "PINK",
    "千千": "rgb(255,153,164)",
    "绿": "GREEN",
    "橙": "ORANGE",
    "黄": "YELLOW",
    "透明": "rgba(0,0,0,0)",
    "MIKU": "rgb(57, 197, 187)",
    "青": "CYAN",
    "紫": "PURPLE",
    "灰": "GRAY",

}
match_hanzi_color = "|".join(hanzi2color)


@receiver
@threading_run
@startswith("(%s)+旗$" % match_hanzi_color)
@on_exception_response
def cmd_color_flag(message: CQMessage):
    colors = message.plain_text
    colors = re.findall("(%s)" % match_hanzi_color, colors)
    colors = [Color.from_any(hanzi2color[i]) for i in colors]
    w, h = 512, 288
    im = Image.new("RGB", (w, h))
    dr = ImageDraw.Draw(im)
    hs = []
    for i in range(len(colors)):
        hs.append(int(h*(i+1)/len(colors)))
    top = 0
    for idx, i in enumerate(colors):
        bot = hs[idx]
        dr.rectangle((0, top, w, bot), fill=i.get_rgb())
        top = bot
    simple_send(im)


@receiver
@threading_run
@startswith("(%s)底(%s)字" % (match_hanzi_color, match_hanzi_color))
def cmd_heidibaizi(message: CQMessage):
    pattern = "(%s)底(%s)字" % (match_hanzi_color, match_hanzi_color)
    string = message.plain_text
    match = re.match(pattern, string)
    bg, fg = match.groups()
    content = string[match.span()[1]:].strip("\n \r")
    if(content):
        bg = Color.from_any(hanzi2color[bg])
        fg = Color.from_any(hanzi2color[fg])
        RT = RichText(content, width=512, fontSize=48, bg=bg.get_rgba(
        ), fill=fg.get_rgba(), alignX=0.5, autoSplit=False, dontSplit=False)
        im = RT.render()
        message.response_sync(im)


@receiver
@threading_run
@on_exception_response
@command("/油画", opts={})
def cmd_oilpaint(message: CQMessage, *args, **kwargs):
    imgtype, img = message.get_sent_images()[0]
    n = 1000
    if(args):
        try:
            n = int(args[0])
        except Exception:
            pass
    w, h = img.size
    sz = max(int(((w*h)/n)**0.5)*2, 10)

    def f(img):
        nonlocal n, sz, w, h
        ret = Image.new(img.mode, img.size)
        for i in range(n):
            p = background.random_position(w, h)
            color = Color(*img.getpixel(p))
            # mask = background.random_polygon_mask(
            #    sz, rnd=0.5, rettype="arr").astype(np.float32)
            centmask = background.centric_mask(sz, rettype="arr")/255

            angle = background.frandrange(0, 180)
            strip = background.random_stripe_mask(
                sz, ratio=2, blur=1, rettype="arr")
            mask1 = strip*centmask
            mask1 = Image.fromarray(mask1.astype(np.uint8)).resize((sz, sz*3))
            mask1 = mask1.rotate(angle, expand=True, fillcolor=0)
            mask2 = (255-strip)*centmask
            mask2 = Image.fromarray(mask2.astype(np.uint8)).resize((sz, sz*3))
            mask2 = mask2.rotate(angle, expand=True, fillcolor=0)

            paste1 = Image.new(img.mode, mask1.size,
                               color.lighten(0.1).get_rgba())
            paste2 = Image.new(img.mode, mask1.size,
                               color.darken(0.1).get_rgba())
            _sz = paste1.size[0]
            le, up = p[0]-_sz//2, p[1]-_sz//2
            ret.paste(paste1, box=(le, up), mask=mask1)
            ret.paste(paste2, box=(le, up), mask=mask2)
        return ret
    ret = img_filter(imgtype, img, f)
    message.response_sync(ret)


@receiver
@threading_run
@on_exception_response
@command("/云", opts={})
def cmd_cloud(message, *args, **kwargs):

    border_color = None
    bluesky = None

    def f(img):
        nonlocal border_color, bluesky
        from ..utils.candy import log_header
        img = img.convert("RGB")
        arr = np.array(img)
        h, w, ch = arr.shape
        if(border_color is None):
            border_color = colors.image_border_color(img, rettype="np")
        if(bluesky is None):
            xys = background.arangexy(w, h)
            print(log_header(), "xys", xys.shape)
            weight_y0 = xys[:, :, 0]
            weight_y1 = h-weight_y0
            _arr = np.stack([weight_y0, weight_y1], axis=-1)
            print(log_header(), "_arr", _arr.shape)
            bluesky = background.colorvec1(
                _arr, [(121, 191, 246), (9, 71, 152)])
            print(log_header(), "blue sky", bluesky.size)
        diff = np_misc.vecs_l2dist(arr, border_color, keepdims=False)  # h x w
        # print(log_header(), diff.shape)
        diff = np_misc.normalize_range(diff, 0, 0.8)
        # print(log_header(), diff.shape)
        rnd = np.random.uniform(0, 1, diff.shape)
        mask = rnd < diff
        mask = mask.astype(np.uint8)*255
        # print(log_header(), mask.shape)
        mask = Image.fromarray(mask).filter(ImageFilter.GaussianBlur(3))
        white = Image.new("RGB", (w, h), (255,)*3)
        ret = bluesky.copy()
        ret.paste(white, mask=mask)
        return ret
    imgtype, img = message.get_sent_images()[0]
    ret = img_filter(imgtype, img, f)
    message.response_sync(ret)


@receiver
@threading_run
@on_exception_response
@command("/格子裙|/格子|/格裙", opts={"-h", "-hshift"})
def cmd_gridskirt(message, *args, **kwargs):
    imgtype, img = message.get_sent_images()[0]
    img = img.convert("RGB")
    size = 1280, 720
    w, h = size
    scale = ((w*w+h*h)**0.5)/80
    scale_big = scale*4
    scale_small = scale
    sin45 = math.sin(math.pi/4)
    freq_thread = scale_big*1.23
    scale_thread = freq_thread*0.08
    n_colors = 9
    cs = colors.image_colors(img, n_colors)
    if(kwargs.get("h") or kwargs.get("hshift")):
        hshift = int(kwargs.get("h") or kwargs.get("hshift"))
        _cs = []
        for i in cs:
            hue, s, l = i.get_hsl()
            hue = (hue+hshift) % 360
            _cs.append(colors.Color.from_hsl(hue, s, l))
        cs = _cs
    #colors_big_w = colors[:3]
    #colors_big_H = colors[3:6]
    #colors_small = colors[6:10]
    #color_thread = colors[10]
    xys = background.arangexy(*size)
    xs = xys[:, :, 1]
    ys = xys[:, :, 0]
    byx = xs.astype(np.float64)
    byy = ys.astype(np.float64)

    def grid_by_angle(scale, angle, k):
        _cos = math.cos(angle/180*math.pi)
        _sin = math.sin(angle/180*math.pi)
        ret = (byx*_cos+byy*_sin)/scale
        return (ret % k).astype(np.uint16)

    eye = np.eye(n_colors)

    eye_big_h = eye[grid_by_angle(scale_big, 0, 2)]
    eye_big_w = eye[grid_by_angle(scale_big, 90, 2)+2]

    # mask_small_w0 = byx%scale_small < (scale_small*0.1)
    # mask_small_w1 = ((byx+0.3*scale_small)%scale_small) < (scale_small*0.1)
    # mask = (mask_small_w0+mask_small_w1).reshape((h, w, 1))
    idx_small_w = grid_by_angle(scale_small, 0, 4)
    eye_small_w = eye[idx_small_w+4]  # *mask

    # mask_small_h0 = byy%scale_small < (scale_small*0.1)
    # mask_small_h1 = ((byy+0.3*scale_small)%scale_small) < (scale_small*0.1)
    # mask = (mask_small_h0+mask_small_h1).reshape((h, w, 1))
    idx_small_h = grid_by_angle(scale_small, 90, 4)
    eye_small_h = eye[idx_small_h+4]  # *mask

    mask_thread0 = (byy % freq_thread) < (freq_thread*0.08)
    mask_thread1 = (byx*sin45+byy*sin45) % scale_thread < (scale_thread*0.12)
    mask_thread = (mask_thread0 & mask_thread1).reshape((h, w, 1))
    eye_thread = [i == n_colors-1 for i in range(n_colors)]
    eye_thread = (np.zeros((h, w, n_colors))+eye_thread)*mask_thread

    alpha = 1.5
    arr = (eye_big_h+eye_big_w)*math.exp(alpha*1) +\
        (eye_small_w+eye_small_h)*math.exp(alpha*0) +\
        eye_thread*math.exp(alpha*1)
    img = background.colorvec1(arr, [tuple(i) for i in cs])
    message.response_sync(img)
    # img = Image.new("RGBA", (1, n_colors))
    # for i in range(n_colors):
    #     img.putpixel((0, i), cs[i].get_rgb())
    # message.response_sync(img.resize((233, 233), Image.Resampling.NEAREST))


@receiver
@threading_run
@on_exception_response
@command("/网格", opts={})
def cmd_grids(message, *args, **kwargs):
    if(args and args[0].isnumeric()):
        n = int(args[0])
        n = max(4, min(12, n))
    else:
        n = 4
    imgtype, img = message.get_sent_images()[0]
    img = img.convert("RGB")
    imcolors = colors.image_colors(img, n)
    colors1 = imcolors[:n//2]
    colors2 = imcolors[n//2:]
    img = background.grids1(*img.size, colors1, colors2)
    message.response_sync(img)


@receiver
@threading_run
@on_exception_response
@command("[!/]?星星", opts={})
def cmd_face_stars(mes, *args, **kwargs):
    star = RichText(["⭐"], width=233, fontSize=48).render()
    area = 1e5

    def angle_minus(a, b):
        ret = a-b
        while(ret > 180):
            ret -= 360
        while(ret < -180):
            ret += 360
        return ret

    def add_star(img, alpha, theta):
        nonlocal star, area
        img = sizefit.area(img, area)
        sizer = angle_minus(alpha, 90)
        sizer = 1.5 - abs(sizer/180)
        st = sizefit.resize_ratio(star, sizer)
        w, h = img.size
        ctr = Point2d(w/2, h/5)
        rot = Point2d(w/2/1.4*math.cos(alpha/180*PI), h/5/1.4 *
                      math.sin(alpha/180*PI)).rotate_by_angle(theta/180*PI)
        p = ctr+rot
        ll = (p-Point2d(*st.size)/2).intxy
        img.paste(st, box=ll, mask=st)
        return img
    imgtype, img = mes.get_sent_images()[0]
    if(imgtype == "image/gif"):
        frames, fps = image.gif_frames_fps(img)
    else:
        frames = [img]*20
        fps = 20
    frms = []
    nfrms = len(frames)
    duration = nfrms/fps
    nloops = int(max(1, duration*1.2))
    for idx, img in enumerate(frames):
        alpha = idx/nfrms*360*nloops
        frm = add_star(img, alpha, -15)
        frm = add_star(frm, alpha+120, -15)
        frm = add_star(frm, alpha+240, -15)
        frms.append(frm)
    gif = make_gif(frms, fps=fps, frame_area_sum=area*nfrms)
    # simple_send(gif)
    mes.response_sync(gif)


@receiver
@threading_run
@on_exception_response
@command("鲁迅说", opts={})
def cmd_luxunrt(message, *args, **kwargs):
    mes = message.raw.message
    if(isinstance(mes, str)):

        mes = mes_str2arr(mes)
        # print(mes)
    content = []
    url = r"https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQHs3ZdlhN-jfFjxKVcLrkP4F3d-bE2qEL6jQ&usqp=CAU"
    url = r"https://bkimg.cdn.bcebos.com/pic/7c1ed21b0ef41bd5ad6e6cee5a9496cb39dbb6fd97d1?x-bce-process=image/resize,m_lfit,w_536,limit_1"
    luxun = get_image(url)[1].convert("RGBA")
    w, h = luxun.size
    # print(mes)
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


@receiver
@threading_run
@on_exception_response
@command("/提取轮廓", opts={})
def cmd_contour(message, *args, **kwargs):
    # import numpy as np
    # from PIL import Image
    im = message.get_sent_images()[0][-1].convert("RGB")
    arr = np.asarray(im)
    h, w = arr.shape[:2]
    sz = 2
    arrs = []
    for x in range(sz):
        for y in range(sz):
            tmp = arr[y:y+h-sz, x:x+w-sz]
            arrs.append(tmp)
    arrs = np.array(arrs).astype(np.float32)
    avg = np.mean(arrs, axis=0)
    diff2 = (arrs-avg)**2
    diff = np.sum(diff2, axis=(0, -1))
    diff = np.sqrt(diff)
    mn, mx = np.min(diff), np.max(diff)
    norm = (diff-mn)/(mx-mn)
    arr_grey = (255-norm*255).astype(np.uint8)
    im = Image.fromarray(arr_grey)
    simple_send(im)

base_images = {}
@receiver
@threading_run
@on_exception_response
@command("/底图", opts={})
def cmd_face_set_base(message: CQMessage, *args, **kwargs):
    if(message.get_reply_image()):
        image = message.get_reply_image()
    else:
        imgtype, image = message.get_sent_images()[0]
    base_images[message.sender.id] = image
    simple_send(image)
@receiver
@threading_run
@on_exception_response
@command("/马赛克叠图", opts={})
def cmd_face_mosaic_compose(message: CQMessage, *args, **kwargs):
    img0 = base_images[message.sender.id]
    if(message.get_reply_image()):
        img1 = message.get_reply_image()
    else:
        imgtype, img1 = message.get_sent_images()[0]
    arr0 = np.array(img0.convert("RGB"))
    arr1 = np.array(img1.resize(img0.size).convert("RGB"))
    def fit_range(arr: np.ndarray, lo=0, hi=1):
        ret = (arr-arr.min())/(arr.max()-arr.min()+1e-10)
        return ret*(hi-lo)+lo
    def fit_mean(arr: np.ndarray, mean=0, mn=0, mx=255):
        ret = arr-arr.mean()
        # mn < ret + mean < mx
        # mn-mean < ret < mx - mean
        scale = np.maximum(1, np.maximum(ret+mean-mn, mx-mean-ret))
        ret = ret/scale+mean
        return ret

    def f(arr0: np.ndarray, arr1):
        h, w, ch=arr0.shape
        ratio = (1000/w/h)**0.5
        tile_h, tile_w = int(h*ratio), int(w*ratio)
        tile_size = min(h//tile_h, w//tile_w)
        ret = np.zeros((tile_h*tile_size, tile_w*tile_size, ch), np.float32)
        for x in range(tile_w):
            
            for y in range(tile_h):
                left = x*tile_size
                top = y*tile_size
                tile0 = arr0[top:top+tile_size, left:left+tile_size, :]
                tile1 = arr1[top:top+tile_size, left:left+tile_size, :]
                color = tile0.mean(axis=0, keepdims=True).mean(axis=1, keepdims=True)
                rge = np.minimum(color, 255-color)
                texture = fit_mean(tile1, 0, -rge, rge)
                tile = color+texture
                ret[top:top+tile_size, left:left+tile_size, :] = tile
        return ret
    arr = f(arr0, arr1)
    im = Image.fromarray(arr.astype(np.uint8))
    simple_send(im)
    


@receiver
@threading_run
@on_exception_response
@command("/鱼眼", opts={"-s"})
def cmd_yuyan(message: CQMessage, *args, **kwargs):
    
    from ..utils.image.background import arangexy
    import numpy as np
    _, im = message.get_sent_images()[0]
    w, h = im.size
    tmp = [0.5, 0.5]
    for idx, i in enumerate(args):
        if(idx<2):
            tmp[idx] = float(i)
    x, y = tmp
    cx, cy = w*x, h*y
    diam = (w*h)**0.5
    yxs = arangexy(w, h)
    dist = yxs-(cy, cx)
    dist = (dist**2).sum(axis=-1, keepdims = True)**0.5
    def sigmoid(x, np=np):
        exp = np.exp(-x)
        return 1/(1+exp)
    s = float(kwargs.get("s", 1))
    rate = sigmoid(dist/diam/s*3)*2-1

    # simple_send("%.3f, %.3f"%(rate.min(), rate.max()))
    yxs = (yxs*rate+np.array((cy, cx))*(1-rate)).astype(np.int32)
    ys, xs = yxs[:, : ,0], yxs[:, :, 1]
    def foo(y, x, arr):
        # print(y, x)
        return arr[y, x]
    imarr = np.array(im)
    channels = []
    for ch in range(imarr.shape[-1]):
        f = partial(foo, arr = imarr[:,:,ch])
        channels.append(np.vectorize(f)(ys, xs))
    meow = np.stack(channels, axis=-1)
    # simple_send("%s %s"%(meow.dtype, meow.shape))
    im = Image.fromarray(meow)
    simple_send(im)
