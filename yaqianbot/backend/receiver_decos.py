import inspect
from .bot_threading import threading_run
from functools import wraps
from .base_message import Message
import re
from ..utils.trace import __FILE__, __FUNC__, __LINE__
from .configure import bot_config
from ..utils.parse_args import parse_args
import sys, traceback
# from . import receiver
def is_ated(func):
    @wraps(func)
    def inner(message: Message):
        if(message.is_ated):
            return func(message)
        return None
    return inner
def is_su(func):
    @wraps(func)
    def inner(message: Message):
        uid = message.sender.id
        sus = bot_config.get("SUPERUSERS", "").split()
        if(str(uid) in sus):
            return func(message)
        else:
            pass
    return inner
def on_exception_response(func):
    @wraps(func)
    def inner(message, *args, **kwargs):
        try:
            ret = func(message, *args, **kwargs)
            return ret
        except Exception as e:
            mes = "Error: %s"%e
            if(len(mes)>192):
                mes = mes[:192]+"..."
            message.response_sync(mes)
            if("yaqianbot.plugins.plg_admin" in sys.modules):
                plg_admin = sys.modules["yaqianbot.plugins.plg_admin"]
                    
                exc = traceback.format_exc()
                lnk = plg_admin.link_print_exc(exc)
                # lnk = plg_admin.link_send_content(traceback.format_exc())
                message.response_sync("输入%s查看完整"%lnk)
            raise e
    return inner
all_commands = {}
def command(pattern, opts, bool_opts=None, ls_opts=None):
    _re = re.compile(pattern)
    def deco(func):
        global all_commands
        all_commands[func.__name__] = (pattern, func)
        @wraps(func)
        def inner(message: Message):
            text = message.plain_text.strip(" ")
            match = _re.match(text)
            if(match):
                rest = text[match.span()[1]:]
                args, kwargs = parse_args(rest, opts, bool_opts=bool_opts, ls_opts=ls_opts)
                kwa = {i[1:]:j for i, j in kwargs.items()}
                return func(message, *args, **kwa)
        return inner
    return deco
def startswith(pattern):
    _re = re.compile(pattern)
    
    def deco(func, _re=_re):
        @wraps(func)
        def inner(message: Message):
            match = _re.match(message.plain_text)
            if(match):
                ret = func(message)
                return ret
        return inner
    return deco
