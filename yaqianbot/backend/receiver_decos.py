import inspect
from .bot_threading import threading_run
from functools import wraps
from .base_message import Message
import re
from ..utils.trace import __FILE__, __FUNC__, __LINE__


def startswith(pattern):
    _re = re.compile(pattern)

    def deco(func, _re=_re):
        def inner(message: Message):
            match = _re.match(message.plain_text)
            if(match):
                ret = func(message)
                return ret
        return inner
    return deco
