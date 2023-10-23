import logging
from .configure import bot_config
logger = logging.getLogger("yaqianbot")
channel = logging.StreamHandler()
logger.addHandler(channel)
formatter = logging.Formatter('%(levelname)s:%(asctime)s - %(name)s - %(message)s')
channel.setFormatter(formatter)

def str2level(s):
    if (s in logging.__dict__):
        return logging.__dict__.get(s)
    else:
        print("unknown logging level", s)
        return logging.NOTSET
level = str2level(bot_config.get("LOGLEVEL", "INFO"))
logger.setLevel(level)