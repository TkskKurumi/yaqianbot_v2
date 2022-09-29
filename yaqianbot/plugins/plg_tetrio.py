from ..backend import receiver
from ..backend.receiver_decos import command, threading_run, on_exception_response
from ..backend import threading_run, mainpth
from ..backend.cqhttp import CQMessage
from os import path
from ..utils.jsondb import jsondb
from ..utils.tetrio import User, Illust
from ..utils.candy import simple_send
from .plg_help import plugin, plugin_func, plugin_func_option, OPT_NOARGS
qq2io = jsondb(path.join(mainpth, "tetrio", "qq2io"), method=lambda x:str(x)[:3])

@receiver
@threading_run
@on_exception_response
@command("/io", {"-set", "-u"}, bool_opts={"-set"})
def cmd_tetrio(message: CQMessage, *args, **kwargs):
    set_ioid = kwargs.get("set", False)
    qqid = message.sender.id
    ioid = None
    if(args):
        ioid = args[0]
    if(kwargs.get("u")):
        ioid = kwargs["u"]
    if(ioid is None):
        if(qqid in qq2io):
            ioid = qq2io[qqid]
        else:
            message.response_sync("未知您的tetr.io id!!!")
            return
    else:
        # ioid = args[0]
        if(qqid not in qq2io):
            set_ioid = True
    if(set_ioid):
        qq2io[qqid] = ioid
        message.response_sync("为QQ %s 绑定到io %s"%(qqid, ioid))
    simple_send("正在获取%s的io数据.."%ioid)
    user = User.User(ioid)
    illu = Illust.profile(user)
    message.response_sync(illu)
plg = plugin(__name__, "tetrio")
func_io = plugin_func("/io [id]", desc = "查看tetr.io成就、TETRA LEAGUE ")
func_io.append(plugin_func_option("-set", "更新QQ关联的tetr.io id", type=OPT_NOARGS))
plg.append(func_io)