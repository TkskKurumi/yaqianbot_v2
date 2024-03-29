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
@command("/reload", opts = {"-no_callback"}, bool_opts={"-no_callback"})
def cmd_admin_reload(message:Message, *args, **kwargs):
    mes = []
    nc = kwargs.get("no_callback")
    for plg in args:
        for name, module in list(sys.modules.items()):
            if(name.endswith(plg)):
                if((not nc) and hasattr(module, "_on_reload") and callable(module._on_reload)):
                    has_cb = True
                    reload_args, reload_kwargs = module._on_reload()
                else:
                    has_cb = False
                importlib.reload(module)
                if(has_cb):
                    module._on_after_reload(*reload_args, **reload_kwargs)
                f = getattr(module, "__file__", "unknownfile")
                mes.append("reloaded %s (%s)"%(name, f))
    simple_send("\n".join(mes))
    
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
