from ..backend import receiver
from ..backend.receiver_decos import command, threading_run, on_exception_response
from ..backend import threading_run, mainpth
from ..backend.cqhttp import CQMessage
from os import path
from ..utils.jsondb import jsondb
from ..utils.tetrio import User, Illust
qq2io = jsondb(path.join(mainpth, "tetrio", "qq2io"), method=lambda x:str(x)[:3])

@receiver
@threading_run
@on_exception_response
@command("/io", {"-set"}, bool_opts={"-set"})
def cmd_tetrio(message: CQMessage, *args, **kwargs):
    set_ioid = kwargs.get("set", False)
    qqid = message.sender.id
    if(not args):
        if(qqid in qq2io):
            ioid = qq2io[qqid]
        else:
            message.response_sync("未知您的tetr.io id!!!")
            return
    else:
        ioid = args[0]
        if(qqid not in qq2io):
            set_ioid = True
    if(set_ioid):
        qq2io[qqid] = ioid
        message.response_sync("为QQ %s 绑定到io %s"%(qqid, ioid))
    user = User.User(ioid)
    illu = Illust.profile(user)
    message.response_sync(illu)