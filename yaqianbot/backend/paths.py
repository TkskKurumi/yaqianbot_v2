from os import path
from .configure import bot_config
import os
mainpth = \
    bot_config.get("DIRECTORY") or \
    bot_config.get("PATH") or \
    path.join(path.expanduser("~"), ".yaqianbot", "files")
temppth = path.join(mainpth, '.tmp')
cachepth = path.join(mainpth, '.cache')

def ensure_directory(pth):
    dir = path.dirname(pth)
    if(not path.exists(dir)):
        os.makedirs(dir)
    return pth