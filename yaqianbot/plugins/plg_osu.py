from ..backend import receiver
from ..backend.receiver_decos import command, threading_run, on_exception_response
from ..backend import threading_run, mainpth
from ..backend.cqhttp import CQMessage
from os import path
from ..utils.jsondb import jsondb
# from ..utils.tetrio import User, Illust
from ..utils.osu import user, illust
qq2osu = jsondb(path.join(mainpth, "osu", "qq2osu"), method=lambda x:str(x)[:3])

@receiver
@threading_run
@on_exception_response
@command("/osu", {"-set", "-m", "-mode"}, bool_opts={"-set"})
def cmd_osu(message: CQMessage, *args, **kwargs):
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
        osuid = args[0]
        if(qqid not in qq2osu):
            set_osuid = True
    if(set_osuid):
        qq2osu[qqid] = osuid
        message.response_sync("为QQ %s 绑定到osu %s"%(qqid, osuid))
    u = user.User(osuid)
    illu = illust.illust_user(u, size=1280, mode=mode).convert("RGB")
    message.response_sync(illu)