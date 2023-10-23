from khl import Message as RawKHLMessage
from khl import Client as RawKHLClient
from khl import MessageTypes as KHLSendTypes
from khl import Bot as RawKHLBot
from khl.card import CardMessage as KHLCardMessage, Card as KHLCardCard, Module as KHLCardModule, Element as KHLCardElement
from ..base_message import Message, User
from ..bot_threading import threading_run
from .misc import acreate_asset
from ..requests import get_image
from ..log import logger
from PIL import Image
import asyncio
import traceback

class KHLUser(User):
    @classmethod
    def from_khl(cls, bot: RawKHLBot, msg: RawKHLMessage):
        usr = msg.author
        name = usr.username
        from_group = msg.extra.get("guild_id", "UNKNOWN")
        uid = str(msg.author_id)
        ret = cls(id=uid, name=name, from_group=from_group)
        ret._raw = msg.author
        return ret
    def get_avatar(self):
        raise NotImplementedError("WIP: KOOK User Avatar")




class KHLMessage(Message):
    _bot: RawKHLBot
    _client: RawKHLClient
    _msg: RawKHLMessage
    def __repr__(self):
        return "<KHLMessage user.name=%s, user.uid=%s>"%(self.sender.name, self.sender.id)
    

    @classmethod
    def _from_khl_type2pic(cls, bot: RawKHLBot, msg: RawKHLMessage):
        user = KHLUser.from_khl(bot, msg)
        img = get_image(msg.content)
        ated = False
        self_id = "UNKNOWN" # WIP
        ret = cls(sender=user, pics=[img], ated=ated, plain_text="[图片]", group=user.from_group, raw=msg, self_id=self_id)
        ret.update_rpics()
        ret._bot = bot
        ret._client = bot.client
        ret._msg = msg
        return ret

    @classmethod
    def from_khl(cls, bot: RawKHLBot, msg: RawKHLMessage):
        if (msg._type==2):
            return KHLMessage._from_khl_type2pic(bot, msg)
        user = KHLUser.from_khl(bot, msg)
        ated = False # WIP
        self_id = "UNKNOWN" # WIP
        ret = cls(sender=user, pics=[], ated=ated, plain_text = msg.content, group=user.from_group, raw=msg, self_id=self_id)
        ret.update_rpics()
        ret._bot = bot
        ret._client = bot.client
        ret._msg = msg
        return ret

    
    async def response_async(self, message):
        if (not isinstance(message, list)):
            message = [message]
        
        
        texts = []
        images = []
        async def do_send():
            nonlocal texts, images
            card_msg = KHLCardMessage()
            card_card = KHLCardCard()
            if(images):
                elements = []
                for i in images:
                    elements.append(KHLCardElement.Image(src=i))
                card_card.append(KHLCardModule.ImageGroup(*elements))
            if (texts):
                card_card.append(KHLCardModule.Section("\n".join(texts)))
            card_msg.append(card_card)
            ret = await self._msg.reply(card_msg)
            return ret

        for i in message:
            if(isinstance(i, Image.Image)):
                try:
                    url = await acreate_asset(self._client, i)
                    # gather.append(self._msg.reply(url, type=KHLSendTypes.IMG))
                    images.append(url)
                except Exception:
                    traceback.print_exc()
            elif(isinstance(i, str)):
                # gather.append(self._msg.reply(i))
                texts.append(i)
            else:
                await do_send()
                raise TypeError(type(i))
        
        return await do_send()
    @threading_run
    def response_sync(self, message):
        try:
            loop = asyncio.get_running_loop()
            future = asyncio.ensure_future(self.response_async(message))
        except RuntimeError as e:
            asyncio.run_coroutine_threadsafe(self.response_async(message), self._bot.loop)
        
    def get_one_pic(self):
        if (not self.recent_pics):
            raise Exception("没有发送过图片")
        else:
            return self.recent_pics[0][1]