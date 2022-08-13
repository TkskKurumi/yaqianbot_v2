from ..backend import *
from ..backend import base_message as Message
from ..backend.receiver_decos import *
from ..utils import after_match
from ..utils.myhash import base32
from ..utils.candy import simple_send
from pil_functional_layout.widgets import RichText
import sys
import importlib
lnks = dict()


def link(name, func):
    nm = "/"+base32(name, 3)
    lnks[nm] = func
    return nm


def link_print_exc(content):
    def inner(message: CQMessage):
        nonlocal content
        RT = RichText([content], width=512, fontSize=14, fill=(
            0, 0, 0, 255), bg=(255,)*4, autoSplit=False)
        message.response_async(RT.render())
    return link(str(content), inner)


def link_send_content(content):
    def inner(message: CQMessage):
        nonlocal content
        message.response_sync(content)
    return link(str(content), inner)


@receiver
@threading_run
def cmd_handle_lnk(message: Message):
    if(message.plain_text in lnks):
        lnks[message.plain_text](message)


@receiver
@threading_run
@on_exception_response
@is_su
@command("/reload", opts = {})
def cmd_admin_reload(message:Message, *args, **kwargs):
    for plg in args:
        for name, module in list(sys.modules.items()):
            if(name.endswith(plg)):
                importlib.reload(module)
                f = getattr(module, "__file__", "unknownfile")
                simple_send("reloaded %s (%s)"%(name, f))
    
@receiver
@threading_run
@startswith("/exec")
@on_exception_response
@is_su
def cmd_exec(message: CQMessage):
    text = message.plain_text
    commands = after_match("/exec", text).strip()
    exec(commands)


@receiver
@threading_run
@command("/test ", {"-func", "-meow", "-foo"})
def cmd_test(message, *args, **kwargs):
    ret = "arg: %s, kwargs: %s" % (args, kwargs)
    message.response_sync(ret)
