from typing import Dict
from aiocqhttp import Event
import re

from ..bot_threading import threading_run
from ..base_message import User, Message
from .. import requests
from . import cqhttp

def mes_str2arr(message):
    pattern = r"(\[CQ:(.+?),(.+?)\])"
    all_cqcode = re.findall(pattern, message)
    all_plain = re.split(pattern, message)
    print(all_cqcode)
    print(all_plain)
    ret = []
    for idx, cqmatch in enumerate(all_cqcode):
        cqfull, cqtype, cqparams = cqmatch
        i = dict()
        data = dict()
        i['type'] = cqtype
        for j in cqparams.split(","):
            ls = j.split("=")
            key = ls[0]
            value = "=".join(ls[1:])
            data[key] = value
        i['data'] = data
        if(all_plain[idx]):
            ret.append({"data": {"text": all_plain[idx]}, "type": "text"})
        ret.append(i)
    if(all_plain[-1]):
        ret.append({"data": {"text": all_plain[-1]}, "type": "text"})
    return ret


class CQUser(User):
    @classmethod
    def from_cq(cls, sender: Dict, from_group: str):
        name = sender.get("nickname") or sender.get(
            "user_name") or "Unknown-User"
        id = str(sender.get("user_id"))
        return cls(id=id, name=name, from_group=str(from_group))


class CQMessage(Message):
    @classmethod
    def from_cq(cls, event):
        if("group_id" in event):
            group = str(event.group_id)
            sender = CQUser.from_cq(event.sender, event.group_id)
        else:
            group = "private"
            sender = CQUser.from_cq(event.sender, "private")

        mes = event.message
        if(isinstance(mes, str)):
            mes = mes_str2arr(mes)
        pics = list()
        ated = list()
        for i in mes:
            type = i.get("type")
            data = i.get("data", dict())
            if(type == "image"):
                pics.append(i)
            elif(type == "at"):
                ated.append(data.get("qq"))
        ret = cls(sender=sender, pics=pics, ated=ated, plain_text = event.raw_message, group=group, raw = event)
        ret.update_rpics()
        return ret

    def get_sent_images(self):
        rpics = self.recent_pics
        ret = []
        for i in rpics:
            data = i['data']
            url = data['url']
            im = requests.get_image(url)
            ret.append(im)
        return ret
    @threading_run
    def response_sync(self, message, at=False, reply=False):
        args = dict()
        def prepare(**kwargs):
            args.update(kwargs)
        for key in ["message_type","group_id","user_id","self_id"]:
            if(key in self.raw):
                args[key]=self.raw[key]
        prepare(message = message)
        print(args)
        ret = cqhttp._bot.sync.send_msg(**args)
        
    async def response_async(self, message, at=False, reply=False):
        args = dict()
        def prepare(**kwargs):
            args.update(kwargs)
        for key in ["message_type","group_id","user_id","self_id"]:
            if(key in self.raw):
                args[key]=self.raw[key]
        prepare(message = message)
        print(args)
        ret = await cqhttp._bot.send_msg(**args)
        return ret