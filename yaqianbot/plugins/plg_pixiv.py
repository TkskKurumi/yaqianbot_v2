from ..backend.receiver_decos import on_exception_response, command
from ..backend.cqhttp.message import prepare_message
from ..backend import receiver, startswith
from ..backend import threading_run
from ..backend.cqhttp import CQMessage
from ..backend.cqhttp import _bot
from ..utils.pyxyv.illust import Illust, Ranking, _getRankingToday, get_ranking
from ..utils.pyxyv.visualize import illust_listing
from ..utils.pyxyv import rand_illust
import re
import random
from datetime import timedelta
from .plg_help import *
from .plg_admin import link
from ..utils.candy import simple_send
import numpy as np
from PIL import Image
from ..utils.make_gif import make_gif
from ..utils.image import hytk
last_illust = dict()

def make_safe(image):
    image1 = Image.open(rand_img())
    return hytk.hytk(image, image1)

def show_illust(message, ill):
    imgs = ill.get_pages(quality="regular")
    if(not ill.is_safe):
        imgs = [make_safe(Image.open(i)) for i in imgs]
    last_illust[message.sender] = ill
    mes = []
    if(len(imgs)>20):
        for i in imgs:
            mes.append(message.contruct_forward(i))
        message.send_forward(mes)
    else:
        message.response_sync(imgs)
def link_show_illust(ill: Illust):
    def inner(message):
        show_illust(message, ill)
    nm = "pixiv %s"%(ill.id)
    return link(nm, inner)
def rand_img(message: CQMessage=None):
    today = _getRankingToday()
    delta = abs(random.normalvariate(0, 700))
    delta = timedelta(days=delta)
    page = random.randint(1, 2)
    ranking = Ranking(today-delta, mode="monthly", page=page)
    id = random.choice(ranking.ids)
    ill = Illust(id)
    imgs = ill.get_pages(quality="regular")
    if(message):
        last_illust[message.sender] = ill
    return random.choice(imgs)
def rand_landscape(message: CQMessage=None):
    target_ratio = 16/9
    def f_ratio(r):
        nonlocal target_ratio
        if(r>target_ratio):
            return target_ratio/r
        else:
            return r/target_ratio
    candidates = []
    for i in range(10):
        illust = rand_illust()
        page = random.choice(illust.get_pages(quality = "regular"))
        w, h = Image.open(page).size
        r = w/h
        score = f_ratio(r)
        candidates.append((score, page, illust))
    best = max(candidates)
    score, page, ill = best
    if(message):
        last_illust[message.sender] = ill
    return page

    

def cmd_pixiv_rank(message, *args, **kwargs):
    page = kwargs.get("page", None) or kwargs.get("p", None) or 1
    page = int(page) - 1
    st = page*20
    ed = (page+1)*20

    mode = kwargs.get("mode") or kwargs.get("m") or "weekly"
    date = kwargs.get("date") or kwargs.get("d") or 0
    date = int(date)
    rnk = get_ranking(date, mode, start = st, end = ed)
    def f_extra(ill, RT):
        lnk = link_show_illust(ill)
        text = "输入%s查看"%lnk
        return RT.render(texts = [text])
    img = illust_listing(rnk, func_extra_caption=f_extra)
    message.response_sync(img)

# @receiver
@threading_run
@on_exception_response
def cmd_pixiv_view(message: CQMessage, *args, **kwargs):
    if(not args):
        simple_send("请给出pixiv id")
    else:
        id = args[0]
        ill = Illust(id)
        show_illust(message, ill)
@receiver
@threading_run
@on_exception_response
@command("/pixiv", opts ={"-mode", "-page", "-date", "-m", "-d", "-p"})
def cmd_pixiv(message: CQMessage, *args, **kwargs):
    if(args):
        arg0 = args[0]
        if(arg0.startswith("rank")):
            return cmd_pixiv_rank(message, *args, **kwargs)
        elif(arg0.startswith("v")):
            return cmd_pixiv_view(message, *args[1:], **kwargs)
            




@receiver
@threading_run
@startswith("/pix$|/色图")
def cmd_pixiv_(message: CQMessage):
    today = _getRankingToday()
    delta = abs(random.normalvariate(0, 700))
    delta = timedelta(days=delta)
    page = random.randint(1, 2)
    ranking = Ranking(today-delta, mode="monthly", page=page)
    id = random.choice(ranking.ids)
    ill = Illust(id)
    imgs = ill.get_pages(quality="regular")
    message.response_sync(imgs+["%s by %s" % (ill.title, ill.author)])
@receiver
@threading_run
@startswith("/每.色图")
def cmd_pixiv_daily(message):
    def f(id):
        ret=dict()
        data=dict()
        ret["type"]="node"
        data["uin"]=message.raw["self_id"]
        data["name"]="每日色图"
        imgs=Illust(id).get_pages(quality="regular")
        mes=prepare_message(imgs)
        data["content"]=mes
        ret["data"]=data
        return ret
    mode="daily"
    if("月" in message.plain_text):
        mode="monthly"
    elif("周" in message.plain_text):
        mode="weekly"
    ranking=Ranking(mode=mode)
    meow=[f(i) for i in ranking.ids[:10]]
    _bot.sync.send_group_forward_msg(self_id=message.raw["self_id"],messages=meow,group_id=message.raw["group_id"])

@receiver
@threading_run
@startswith("/色图time")
def cmd_setutime(message: CQMessage):
    def f():
        nonlocal message
        ret = dict()
        data = dict()
        ret["type"] = "node"
        data['uin'] = message.raw['self_id']
        data["name"] = "色图time"
        data["content"] = prepare_message(rand_img(message))
        ret["data"] = data
        return ret
    mes = list()
    for i in range(50):
        try:
            mes.append(f())
        except Exception:
            import traceback
            traceback.print_exc()
            
    # mes = [f() for i in range(50)]
    _bot.sync.send_group_forward_msg(self_id = message.raw["self_id"], messages=mes, group_id = message.raw["group_id"])

plg = plugin(__name__, "PIXIV")
func_ranking = plugin_func("/pixiv rank")
opt_mode = plugin_func_option("-opt", "排行周期 daily/weekly/monthly")
func_ranking.append(opt_mode)
plg.append(func_ranking)
