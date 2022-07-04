from aiocqhttp import CQHttp, Event
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
from ..bot_threading import threading_run
import asyncio
import inspect
import traceback
from ..configure import bot_config
from .message import CQMessage, CQUser
receivers = {}
poke_receivers = {}
_backend_type = "cqhttp"


def receiver(func):
    receivers[func.__name__] = func
    return func
def poke_receiver(func):
    poke_receivers[func.__name__] = func
    return func
async def message_receiver(event: Event):
    print("received", event.raw_message)
    mes = CQMessage.from_cq(event)
    for name, func in receivers.items():
        # print(name, func, inspect.isawaitable(func))
        ret = func(mes)
        if(inspect.isawaitable(ret)):
            await ret
async def _poke_receiver(event: Event):
    print("received poke", event)
    mes = await CQMessage.from_cqpoke(event)
    print("received poke meow", event)
    for name, func in poke_receivers.items():
        ret = func(mes)
        if(inspect.isawaitable(ret)):
            await ret

async def debugger(event):

    try:
        # print(event)
        if(event.message.startswith("exec")):
            if(str(event.user_id) in bot_config.get("SUPERUSERS", "").split()):
                exec(event.message)
    except Exception:
        traceback.print_exc()
        print(event)
_bot = CQHttp()
_bot.on_message(message_receiver)
_bot.on("notice.notify.poke")(_poke_receiver)
if(bot_config.get("DEBUG", "false").lower() == "true"):
    print("debug")
    _bot.on_message(debugger)
else:
    print("debug:", bot_config.get("DEBUG", "false"))


def run(host="127.0.0.1", port=8008):
    print("run at", bot_config)
    # loop = asyncio.new_event_loop()
    # loop.create_task(_bot.run_task(host=host, port=port))
    # loop.run_forever()
    asyncio.run(_bot.run_task(host=host, port=port))
    print("meow")
    return None
