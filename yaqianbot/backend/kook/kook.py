from khl import Bot
from khl import Message as RawKHLMessage
from ..configure import bot_config
from typing import Dict, Callable
from .message import KHLMessage
from ..log import logger
from ...utils.candy import dump_obj
import inspect, asyncio
from functools import partial


_backend_type = "kook"

class FuncRegister:
    def __init__(self):
        self.funcs: Dict[str, Callable] = {}
    def __call__(self, func):
        key = func.__name__
        self.funcs[key] = func
    def __iter__(self):
        return iter(self.funcs)
    def items(self):
        return self.funcs.items()
    
receiver = FuncRegister()

async def message_handler(bot, msg: RawKHLMessage):
    wrapped_msg = KHLMessage.from_khl(bot, msg)
    logger.info("%s", wrapped_msg)
    logger.debug("%s", dump_obj(msg))
    for k, foo in receiver.items():
        ret = foo(wrapped_msg)
        if(inspect.isawaitable(ret)):
            await ret


def run(token):
    bot = Bot(token=token)
    bot.on_message()(partial(message_handler, bot))

    @bot.command(name='hello')
    async def world(msg: RawKHLMessage):
        await msg.reply('world!')
    
    # asyncio.run(bot.start())
    bot.run()

    

