import os
from .bot_threading import *
from .configure import bot_config
from os import path
from .receiver_decos import startswith
from .paths import temppth, cachepth, mainpth
# qq backend
if(bot_config.get("BACKEND", "cqhttp") == "cqhttp"):
    from .cqhttp.cqhttp import *
    from .cqhttp import cqhttp as backend
