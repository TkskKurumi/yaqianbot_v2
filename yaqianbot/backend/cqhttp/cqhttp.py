from aiocqhttp import CQHttp, Event
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
from ..bot_threading import threading_run
import asyncio
import inspect
import traceback, time
import schedule
from ..configure import bot_config
from .message import CQMessage, CQUser
receivers = {}
poke_receivers = {}
scheduled_jobs = {}
_backend_type = "cqhttp"


def receiver(func):
    receivers[func.__name__] = func
    return func
def scheduled(func):
    scheduled_jobs[func.__name__]=func
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
_is_running = False
def bot_is_running():
    global _is_running
    return _is_running
@threading_run
def timer():
    while(bot_is_running()):
        # time.sleep(10)
        # schedule.run_pending()
        print("Timer 30 seconds")
        time.sleep(30)
def run(host="127.0.0.1", port=8008):
    global _is_running
    print("run at", bot_config)
    # loop = asyncio.new_event_loop()
    # loop.create_task(_bot.run_task(host=host, port=port))
    # loop.run_forever()
    _is_running = True
    timer()
    asyncio.run(_bot.run_task(host=host, port=port))
    print("meow")
    _is_running = False
    return None
