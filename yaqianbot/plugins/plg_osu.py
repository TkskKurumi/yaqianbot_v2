from ..backend import receiver
from ..backend.receiver_decos import command, threading_run, on_exception_response
from ..backend import threading_run, mainpth
from ..backend.cqhttp import CQMessage
from os import path
from ..utils.jsondb import jsondb
# from ..utils.tetrio import User, Illust
from ..utils.osu import user, illust
qq2osu = jsondb(path.join(mainpth, "osu", "qq2osu"), method=lambda x:str(x)[:3])

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
    if(len(args)>1):
        index = max(int(args[1]), 1)
    else:
        index = 1
    if(score_type not in "r recent b best".split()):
        message.response_sync("未知的成绩类型%s"%score_type)
        return
    for i in "recent best".split():
        if(i.startswith(score_type)):
            score_type = i
    u = user.User(osuid)
    print(score_type, mode)
    try:
        scores = u.get_scores(type=score_type, mode=mode)
    except Exception as e:
        message.response_sync("无法获取%s %s成绩"%(score_type, mode))
        import sys, traceback
        if("yaqianbot.plugins.plg_admin" in sys.modules):
            plg_admin = sys.modules["yaqianbot.plugins.plg_admin"]
            exc= traceback.format_exc()
            lnk = plg_admin.link_print_exc(exc)
            message.response_sync("输入%s查看完整"%lnk)
        return None
    if(index > len(scores)):
        message.response_sync("没有%s成绩"%(" ".join(args), ))
        return
    else:
        score = scores[index-1]
        im = illust.illust_score_detail(score)
        message.response_sync(im)
@receiver
@threading_run
@on_exception_response
@command("/osu", {"-set", "-m", "-mode", "-user", "-u", "-debug"}, bool_opts={"-set", "-debug"})
def cmd_osu(message: CQMessage, *args, **kwargs):
    if(args):
        if(args[0] in "r recent b best".split()):
            return osu_recent(message, *args, **kwargs)
    
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
        message.response_sync("为QQ %s 绑定到osu %s"%(qqid, osuid))
    u = user.User(osuid)
    illu = illust.illust_user(u, size=1280, mode=mode).convert("RGB")
    message.response_sync(illu)