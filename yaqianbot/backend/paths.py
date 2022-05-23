from os import path
from .configure import bot_config
mainpth = \
    bot_config.get("DIRECTORY") or \
    bot_config.get("PATH") or \
    path.join(path.dirname(__file__), "files")
temppth = path.join(mainpth, '.tmp')
cachepth = path.join(mainpth, '.cache')
