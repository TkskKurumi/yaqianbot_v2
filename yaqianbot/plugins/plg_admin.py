from ..backend import *
from ..backend import base_message as Message
from ..backend.receiver_decos import *
from ..utils import after_match
from ..utils.myhash import base32
lnks = dict()

def link(name, func):
    nm = "/"+base32(name, 4)
    lnks[nm] = func
    return nm

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
