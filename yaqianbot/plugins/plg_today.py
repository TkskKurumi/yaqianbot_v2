from ..backend import receiver, CQMessage
from ..backend.receiver_decos import threading_run, on_exception_response, command
from ..backend.paths import mainpth
from ..utils.lvldb import TypedLevelDB
from ..backend.bot_date import now
from ..utils.candy import simple_send
from ..utils.image.process import adjust_L
from ..utils.image import sizefit
from ..backend.configure import bot_config
from .plg_pixiv import rand_img
from PIL import Image
from pil_functional_layout.widgets import Column, Row, RichText, CompositeBG, AddBorder
from os import path
from ..utils.myhash import base32
import random
import os
from . import plg_pixiv
im_path = path.join(mainpth, "daily_fortune", "img")
db = TypedLevelDB.open(path.join(mainpth, "daily_fortune", "db"))


def generate(message: CQMessage):
    user = message.sender.id
    today = now().strftime("%Y%m%d")
    key = "%s-%s" % (user, today)

    good_or_bad = random.choice("吉凶")

    severity = random.choice(["", "大", "小"])
    if(good_or_bad == "吉"):
        hitokoto = random.choice([
            "可能会捡到钱钱",
            "会遇到好事",
            "小而幸福的愿望得到实现",
            "好运滚滚来",
            "打游戏总是赢",
            "猜测都能猜对",
            "直觉总是正确"
        ])
    else:
        hitokoto = random.choice([
            "走路千万不要看手机，看路",
            "请家人为你祝愿",
            "要小心翼翼, 小错误会被放大",
            "对爱你的人们要有所回应",
            "不要把客气当福气",
            "安全规则的制定都是有道理的, 请好好遵守"
        ])

    background = Image.open(rand_img())
    # background = adjust_L(background, 0.9)

    text = lambda **kwargs: [kwargs["text"]]
    rt = RichText(text, 512, autoSplit=False, dontSplit=False)

    gb = severity+good_or_bad
    gb = rt.render(text=gb, fontSize=72, bg=(255, 80, 120))
    hitokoto = rt.render(text=hitokoto, fontSize=48)
    row = Row([gb, hitokoto])
    im = CompositeBG(row, adjust_L(background, 0.8)).render()
    w = background.size[0]
    im = sizefit.fix_width(im, int(w*0.8))
    background.paste(im, (w//10, w//10))
    im = background
    nm = base32(im)
    os.makedirs(im_path, exist_ok=True)
    ret = path.join(im_path, nm+".png")
    im.save(ret)
    return ret


def gen(message: CQMessage):
    good_or_bad = random.choice("吉凶")
    severity = random.choice(["", "大", "小"])
    if(good_or_bad == "吉"):
        hitokoto = random.choice([
            "可能会捡到钱钱",
            "会遇到好事",
            "小而幸福的愿望得到实现",
            "好运滚滚来",
            "打游戏总是赢",
            "猜测都能猜对",
            "直觉总是正确"
        ])
    else:
        hitokoto = random.choice([
            "走路千万不要看手机，看路",
            "请家人为你祝愿",
            "要小心翼翼, 小错误会被放大",
            "对爱你的人们要有所回应",
            "不要把客气当福气",
            "安全规则的制定都是有道理的, 请好好遵守"
        ])
    good_or_bad = severity + good_or_bad

    rendered_gb = RichText([good_or_bad], fontSize=72, width=720)
    

    ftext = lambda **kwargs: [kwargs["text"]]
    H1 = RichText(ftext, fontSize=48, width=720)
    H2 = RichText(ftext, fontSize=36, width=720)
    username = message.sender.name
    rows = []
    setu = Image.open(plg_pixiv.rand_landscape())
    rows.append(sizefit.fix_width(setu, 720))
    rows.append(H1.render(text="%s的今日运势:" % username))
    rows.append(rendered_gb)
    rows.append(H2.render(text="小贴士:\n    "+hitokoto))
    im = Column(rows, bg=(255, 255, 255), alignX=0.01).render()
    nm = base32(im)
    os.makedirs(im_path, exist_ok=True)
    ret = path.join(im_path, nm+".png")
    im.save(ret)
    return ret


@receiver
@threading_run
@on_exception_response
@command("/今日运势", opts={"-debug"})
def cmd_today_fortune(message: CQMessage, *args, debug=False, **kwargs):
    user = message.sender.id
    today = now().strftime("%Y%m%d")
    key = "%s-%s" % (user, today)
    if(debug == True):
        uid = message.sender.id
        sus = bot_config.get("SUPERUSERS", "").split()
        if(str(uid) in sus):
            tmp = gen(message)
            simple_send(["Test", tmp])
    if(key in db):
        ret = db[key]
    else:
        ret = gen(message)
        db[key] = ret
    simple_send(ret)
