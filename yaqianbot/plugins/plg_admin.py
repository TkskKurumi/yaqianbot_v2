from ..backend import *
from ..backend.receiver_decos import *
from ..utils import after_match
@receiver
@threading_run
@startswith("/exec")
@is_su
def cmd_exec(message: CQMessage):
    text = message.plain_text
    commands = after_match("/exec", text).strip()
    exec(commands)
@receiver
@threading_run
@command("/test ", {"-func", "-meow", "-foo"})
def cmd_test(message, *args, **kwargs):
    ret = "arg: %s, kwargs: %s"%(args, kwargs)
    message.response_sync(ret)