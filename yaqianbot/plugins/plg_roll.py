from ..backend import receiver, startswith
from ..backend import threading_run
from ..backend.cqhttp import CQMessage
import re
import random
from ..utils import after_match
from ..backend.receiver_decos import *
from ..utils.candy import simple_send
from ..utils.make_gif import make_gif
from PIL import Image, ImageDraw
from math import sin, cos
from pil_functional_layout.widgets import RichText
from ..utils.np_misc import np, smooth_frames

@receiver
@threading_run
@startswith("/coin")
def cmd_coin(message: CQMessage):
    fps = 20
    nframes = fps*4
    frames = []

    def f_rot(n):
        nonlocal nframes
        a = (nframes-1-n)/(nframes-1)
        a2 = a*a
        return a2*360*100

    def f_thick(n):
        nonlocal nframes
        a = (nframes-1-n)/(nframes-1)
        a2 = a*a
        shrink = 0.1**a
        s = (cos(a2*100)+1)/2
        return s*shrink
    w = 300
    text = after_match("/coin", message.plain_text).strip()
    if(len(text)<2):
        simple_send("正: 1\n反: 🌿")
        text = "1🌿"
    result = random.choice(text)
    
    T = RichText([result], width=512, fontSize=200, bg=(0,0,0,0), dontSplit=False, autoSplit=False).render()
    _, __ = T.size
    left, top = (w-_)//2, (w-__)//2
    print(left, top, T.size)
    for i in range(nframes):
        rot = f_rot(i)
        im = Image.new("RGB", (w, w), (255,)*3)
        dr = ImageDraw.Draw(im)
        rot, thick = f_rot(i), f_thick(i)
        dr.ellipse((0, w/2-w/2*thick, w, w/2+w/2*thick),
                   outline=(0, 0, 0), width=5, fill=(230,230,230))
        if(thick>0.8):
            im.paste(T, mask=T, box=(left, top))
        im = im.rotate(rot, fillcolor=(255, 255, 255))
        frames.append(np.asarray(im))
    last = [frames[-1]]
    frames.extend(last*fps)
    frames = smooth_frames(frames)
    frames = [Image.fromarray(i) for i in frames]
    gif = make_gif(frames, fps=fps)
    simple_send(gif)

@receiver
@threading_run
@on_exception_response
@command("问", opts={})
def cmd_roll_ask(message: CQMessage, *args, **kwargs):
    if(not args):
        return
    orig = message.plain_text
    ret = orig
    idx = 1
    while(idx<len(orig)-1):
        if(ret[idx]=="不"):
            if(ret[idx-1]==ret[idx+1]):
                le, mid, ri = ret[:idx-1], ret[idx-1:idx+2], ret[idx+2:]
                mid = random.choice([mid, mid[1:]])
                ret = "".join([le, mid, ri])
    if(ret!=orig):
        simple_send(ret)


@receiver
@startswith("/roll")
async def cmd_roll(message: CQMessage):
    print("/roll")
    text = after_match("/roll", message.plain_text)
    temp = text.strip()
    if(not(temp)):
        # simple_send(ctx,'您还没有输入要骰什么内容呐！')
        await message.response_async("还没有输入要骰什么内容呐！")
        return
    if(re.match(r'\d+d\d+$', temp)):
        m, n = re.findall('\d+', temp)
        m = int(m)
        n = int(n)
        mes = []
        for i in range(m):
            mes.append(random.randint(1, n))
        if(m > 1):
            sm = sum(mes)
            ret = '+'.join([str(_) for _ in mes])+'='+str(sm)
        else:
            ret = str(mes[0])

    else:
        _ = temp.split()
        ret = random.choice(_)
        ret = '当然是%s啦~' % ret
    await message.response_async(ret)
