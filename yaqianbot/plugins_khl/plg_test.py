from ..backend import receiver
from ..backend.kook import KHLMessage
from ..backend.receiver_decos import command, threading_run
from ..backend import log
import random
from PIL import Image

@receiver
@threading_run
@command("/roll", {})
def cmd_roll(message: KHLMessage, *args):
    if(args):
        message.response_sync("当然是%s啦~"%random.choice(args))
    else:
        pass

@receiver
@threading_run
@command("/img", {})
def cmd_image(message: KHLMessage, *args):
    message.response_sync(["bocchi"])
    message.response_sync([Image.open(r"E:\Pics\bocchi.png")])