from ..backend import receiver, startswith
from ..backend.requests import get_avatar
from ..backend import threading_run, mainpth
from ..backend import requests
from ..backend.cqhttp import CQMessage
from ..backend.configure import bot_config
from os import path
from ..utils.jsondb import jsondb
from datetime import datetime, timezone, timedelta
from ..utils.pyxyv import rand_img
from pil_functional_layout.widgets import Column, Row, AvatarCircle, Text, ProgressBar, Pill, CompositeBG, RichText
from PIL import Image
import time
from ..utils.image import colors, image_is_dark

qq2io = jsondb(path.join(mainpth, "tetrio", "qq2io"), method=lambda x:str(x)[:3])
