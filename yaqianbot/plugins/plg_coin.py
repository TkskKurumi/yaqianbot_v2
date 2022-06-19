from ..backend.receiver_decos import *
from ..backend import receiver, startswith
from ..backend.requests import get_avatar
from ..backend import threading_run, mainpth
from ..backend import requests
from ..backend.cqhttp import CQMessage
from ..backend.configure import bot_config
from os import path
from ..utils.jsondb import jsondb
from datetime import datetime, timezone, timedelta
from ..utils.pyxyv import rand_img
from pil_functional_layout.widgets import Column, Row, AvatarCircle, Text, ProgressBar, Pill, CompositeBG, RichText
from PIL import Image
import time
from ..utils.image import colors, image_is_dark
pth = path.join(mainpth, "coin")


def method(x): return str(x)[:3]


coins = jsondb(path.join(pth, "coin"), method=method)
dailybonus = jsondb(path.join(pth, "dailybonus"), method)
impression = jsondb(path.join(pth, "impression"), method)
nickname = jsondb(path.join(pth, "nickname"), method)

dailybonus_ts = jsondb(path.join(pth, "daily_ts"))


def now():
    tz = timezone(timedelta(hours=int(bot_config.get("tz",  0))))
    ret = datetime.now().astimezone(tz)
    return ret


def fromtimestamp(ts):
    tz = timezone(timedelta(hours=int(bot_config.get("tz",  0))))
    ret = datetime.fromtimestamp(ts).astimezone(tz)
    return ret
def today_string():
    meow = now()
    return meow.strftime("%Y%m%d")


def hitokoto():
    r = requests.sess.request(
        "GET", "https://v1.hitokoto.cn/?c=a&encode=json", expire_after=0)
    j = r.json()

    return j["hitokoto"]+"\n---"+j["from"]




def earn_coin(uid, cnt):
    coins[uid] = coins.get(uid, 0)+cnt


def earn_impression(uid, cnt):
    impression[uid] = impression.get(uid, 0)+cnt

def status_image(user_id, user_name, coin_delta=0, impression_delta=0):
    bg = Image.open(rand_img())
    size = 1280
    golden_rate = (5**0.5-1)/2
    avatar_size = size//5
    uname_size = int(avatar_size/2)
    coin_size = int(uname_size*(golden_rate**2.5))
    hitokoto_size = coin_size
    border = avatar_size//50

    avatar_img = get_avatar(user_id)

    # dark_theme = not image_is_dark(avatar_img)
    dark_theme = True
    im_avatar = AvatarCircle(avatar_img, size=avatar_size)
    bg = Image.open(rand_img())
    cols = colors.image_colors(avatar_img, 3)
    color_a, color_b, color_c = [colors.Color(*i).replace(S=0.5, L=0.5, A=255) for i in cols]  # nopep8
    light_a, light_b, light_c = color_a.replace(L=0.9), color_b.replace(L=0.9), color_c.replace(L=0.9)  # nopep8
    dark_a, dark_b, dark_c = color_a.replace(
        L=0.1), color_b.replace(L=0.1), color_c.replace(L=0.1)
    im_username = Text(user_name, fontSize=uname_size,
                       fill=light_c.get_rgba() if dark_theme else dark_c.get_rgba())

    t_coin_label = Text("金币", fontSize=coin_size,
                        fill=light_a.get_rgba(), bg=(0, 0, 0, 0))
    if(coin_delta):
        t_coin_num = "%.1f (+%.1f)" % (coins.get(user_id, 0), coin_delta)
    else:
        t_coin_num = "%.1f" % (coins.get(user_id, 0),)
    t_coin_num = Text(t_coin_num, fontSize=coin_size, fill=dark_b.get_rgba(), bg=(0, 0, 0, 0))
    im_coins = Pill(t_coin_label, t_coin_num, colorBorder=color_c.get_rgba(
    ), colorA=dark_a.get_rgba(), colorB=light_b.get_rgba())
    t_imp_label = Text("好感度", fontSize=coin_size,
                       fill=light_a.get_rgba(), bg=(0, 0, 0, 0))
    if(impression_delta):
        t_imp_num = "%.1f (+%.1f)" % (impression.get(user_id, 0), impression_delta)
    else:
        t_imp_num = "%.1f" % impression.get(user_id, 0)

    t_imp_num = Text(t_imp_num,
                     fontSize=coin_size, fill=dark_b.get_rgba(), bg=(0, 0, 0, 0))
    im_imp = Pill(t_imp_label, t_imp_num, colorBorder=color_c.get_rgba(
    ), colorA=dark_a.get_rgba(), colorB=light_b.get_rgba())

    row_status = Row([im_coins, im_imp], borderWidth=hitokoto_size)
    t_hitokoto = RichText(
        hitokoto(), fontSize=hitokoto_size, bg=(0,)*4, fill=color_c.replace(S=0.2), width=int(size*golden_rate), autoSplit=False, alignX=1)
    col = Column([im_username, row_status], alignX=0.05, borderWidth=border)
    if(dark_theme):
        bgc = dark_c.replace(A=180).get_rgba()
    else:
        bgc = light_c.replace(A=180).get_rgba()
    ret = Row([im_avatar, col], borderWidth=border)
    ret = Column([ret, t_hitokoto], borderWidth=border, bg=bgc, alignX = 0.9)
    ret = CompositeBG(ret, bg=bg)
    return ret.render()
@ receiver
@ threading_run
@ on_exception_response
@ startswith("/签到")
def cmd_qiandao(message: CQMessage):
    coin_earn = int(bot_config.get("DAILY_BONUS_COIN_EARN", 10))
    impression_earn = int(bot_config.get("DAILY_BONUS_COIN_EARN", 10))
    uid = message.sender.id
    uname = message.sender.name
    today = today_string()
    dailybonus[today] = dailybonus.get(today, dict())
    dailybonus_ts[today] = dailybonus_ts.get(today, dict())
    if(uid not in dailybonus[today]):
        dailybonus[today][uid] = True
        dailybonus_ts[today][uid] = time.time()
        dailybonus._save(today)
        dailybonus_ts._save(today)
        earn_coin(uid, coin_earn)
        earn_impression(uid, impression_earn)
    img = status_image(uid, uname, coin_earn, impression_earn)
    message.response_sync(img)
@ receiver
@ threading_run
@ startswith("/状态")
def cmd_mystatus(message: CQMessage):
    uid = message.sender.id
    uname = message.sender.name
    img = status_image(uid, uname)
    message.response_sync(img)
