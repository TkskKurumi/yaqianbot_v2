from ..backend import receiver
from ..backend.receiver_decos import command, threading_run, on_exception_response
from ..backend import threading_run, mainpth
from ..backend.cqhttp import CQMessage
from os import path
from ..utils.jsondb import jsondb
from ..utils import np_misc
from ..utils.algorithms.lcs import lcs
from ..utils.osu.mania_difficulty import Chart as ManiaChart
# from ..utils.tetrio import User, Illust
from ..utils.osu import user, illust
from ..utils.make_gif import make_gif, make_mp4
import numpy as np
from PIL import Image
from .plg_help import *
qq2osu = jsondb(path.join(mainpth, "osu", "qq2osu"),
                method=lambda x: str(x)[:3])


def mania_gif(score, t=None, duration=10):
    bm = score["beatmap"]
    is_dt = "DT" in score["mods"]
    assert bm["mode"] == "mania"
    chart = ManiaChart.from_osu_id(bm["id"], dt=is_dt)
    if(t is None):
        bytime, UNUSED = chart.calc_all()
        times = bytime["Time"]
        overall = bytime["Overall"]
        overalls = sorted(list(enumerate(overall)), key=lambda x: x[1])
        n = len(overalls)
        idx = overalls[int(n*0.95)][0]
        t = times[idx]
    tm = t
    frames = []
    fps = 24
    for i in range(duration*fps):
        tmm = tm+i/fps
        frm = chart.render(tmm*1000, width=300)
        frames.append(frm)
    gif = make_gif(frames, fps=fps, frame_area_sum = 1e7) #, frame_area_sum=5e6)
    return gif


def osu_bm(message, *args, **kwargs):
    # mode = kwargs.get("mode") or kwargs.get("m")
    osuid = kwargs.get("user") or kwargs.get("u")
    if(kwargs.get("debug")):
        message.response_sync(str([args, kwargs]))
    if(osuid is None):
        qqid = message.sender.id
        if(qqid in qq2osu):
            osuid = qq2osu[qqid]
        else:
            message.response_sync("请用-user/-u选项指定要查询的用户名喵")
            return
    score_type = args[0].lower()  # should be bm/beatmap/bid
    assert score_type in "bid beatmap".split()
    u = user.User(osuid)
    bid = args[1]
    if(not bid.isnumeric()):
        bid = " ".join(args[1:])
        scores = u.get_scores("best")+u.get_scores("recent")
        ls = []
        for score in scores:
            bm = score["beatmap"]
            bmset = score["beatmapset"]
            meow0 = bmset["title"]+bm['version']
            meow1 = bmset["title_unicode"]+bm['version']
            lcs0 = lcs(bid, meow0)
            lcs1 = lcs(bid, meow1)
            common = max(lcs0.common_ratio_a, lcs1.common_ratio_a)
            
            ls.append((common, score))
        score = max(ls, key=lambda x:x[0])[1]
    else:
        score = u.get_beatmap_score(bid)
        if("error" in score):
            message.response_sync("无法获取分数")
            return
        score = score["score"]

    # score = score["score"]
    im = illust.illust_score_detail(score)
    message.response_sync(im)
    bm = score["beatmap"]
    if(bm["mode"] == "mania"):
        gif = mania_gif(score)
        message.response_sync(gif)


def osu_recent(message, *args, **kwargs):
    mode = kwargs.get("mode") or kwargs.get("m")
    osuid = kwargs.get("user") or kwargs.get("u")
    if(kwargs.get("debug")):
        message.response_sync(str([args, kwargs]))
    if(osuid is None):
        qqid = message.sender.id
        if(qqid in qq2osu):
            osuid = qq2osu[qqid]
        else:
            message.response_sync("请用-user/-u选项指定要查询的用户名喵")
            return
    score_type = args[0].lower()
    if(len(args) > 1):
        index = max(int(args[1]), 1)
    else:
        index = 1
    if(score_type not in "r recent b best".split()):
        message.response_sync("未知的成绩类型%s" % score_type)
        return
    for i in "recent best".split():
        if(i.startswith(score_type)):
            score_type = i
    u = user.User(osuid)
    print(score_type, mode)
    try:
        scores = u.get_scores(type=score_type, mode=mode)
    except Exception as e:
        message.response_sync("无法获取%s %s成绩" % (score_type, mode))
        import sys
        import traceback
        if("yaqianbot.plugins.plg_admin" in sys.modules):
            plg_admin = sys.modules["yaqianbot.plugins.plg_admin"]
            exc = traceback.format_exc()
            lnk = plg_admin.link_print_exc(exc)
            message.response_sync("输入%s查看完整" % lnk)
        return None
    if(index > len(scores)):
        message.response_sync("没有%s成绩" % (" ".join(args), ))
        return
    else:
        score = scores[index-1]
        im = illust.illust_score_detail(score)
        message.response_sync(im)
        bm = score["beatmap"]
        if(bm["mode"] == "mania"):
            gif = mania_gif(score)
            message.response_sync(gif)


@receiver
@threading_run
@on_exception_response
@command("/osu", {"-set", "-m", "-mode", "-user", "-u", "-debug"}, bool_opts={"-set", "-debug"})
def cmd_osu(message: CQMessage, *args, **kwargs):
    if(args):
        if(args[0] in "r recent b best".split()):
            return osu_recent(message, *args, **kwargs)
        elif(args[0] in "bid beatmap".split()):
            return osu_bm(message, *args, **kwargs)
    if(kwargs.get("debug")):
        message.response_sync(str([args, kwargs]))
    set_osuid = kwargs.get("set", False)
    mode = kwargs.get("mode") or kwargs.get("m")
    qqid = message.sender.id
    if(not args):
        if(qqid in qq2osu):
            osuid = qq2osu[qqid]
        else:
            message.response_sync("未知您的OSU!id !!")
            return
    else:
        osuid = " ".join(args)
        if(qqid not in qq2osu):
            set_osuid = True
    if(set_osuid):
        qq2osu[qqid] = osuid
        message.response_sync("为QQ %s 绑定到osu %s" % (qqid, osuid))
    u = user.User(osuid)
    illu = illust.illust_user(u, size=1280, mode=mode).convert("RGB")
    message.response_sync(illu)


plg = plugin(__name__, "OSU")
opt_mode = plugin_func_option(
    "-m或-mode", "指定模式(osu/mania/taiko/fruits)", arg_desc="模式名")
func_osu = plugin_func("/osu [id]或best或recent")
func_osu.append(plugin_func_option("-set", "更新OSU!id", type=OPT_NOARGS))
func_osu.append(opt_mode)
func_osu_recent = plugin_func("/osu recent或best [序号]或默认序号“1”")
func_osu_recent.append(plugin_func_option("-u", "指定用户", arg_desc="用户名"))
func_osu_recent.append(opt_mode)
func_osu_bid = plugin_func("/osu bid或beatmap [beatmap id]")
func_osu_bid.append(plugin_func_option("-u", "指定用户", arg_desc="用户名"))
plg.append(func_osu)
plg.append(func_osu_recent)
plg.append(func_osu_bid)
