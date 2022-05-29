import inspect
from .bot_threading import threading_run
from functools import wraps
from .base_message import Message
import re
from ..utils.trace import __FILE__, __FUNC__, __LINE__
from .configure import bot_config
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
