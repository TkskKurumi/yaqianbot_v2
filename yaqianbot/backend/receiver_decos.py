import inspect
from .bot_threading import threading_run
from functools import wraps
from .base_message import Message
import re
from ..utils.trace import __FILE__, __FUNC__, __LINE__
from .configure import bot_config
from ..utils.parse_args import parse_args

def is_su(func):
    @wraps(func)
    def inner(message: Message):
        uid = message.sender.id
        sus = bot_config.get("SUPERUSERS", "").split()
        if(str(uid) in sus):
            func(message)
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
            message.response_sync("Error: %s"%e)
            raise e
    return inner
def command(pattern, opts, bool_opts=None):
    _re = re.compile(pattern)
    def deco(func):
        @wraps(func)
        def inner(message: Message):
            text = message.plain_text
            match = _re.match(text)
            if(match):
                rest = text[match.span()[1]:]
                args, kwargs = parse_args(rest, opts, bool_opts)
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
