import html
from typing import Dict
from aiocqhttp import Event
import re
from aiocqhttp.message import MessageSegment as MSEG
from ..bot_threading import threading_run
from ..base_message import User, Message
from .. import requests
from . import cqhttp
from os import path
from PIL import Image
from math import sqrt
from ..paths import temppth
import base64
from io import BytesIO


def img_b64(im, limit_size=(2 << 20)):

    if("A" in im.mode):
        format = "PNG"
    else:
        format = "JPEG"
    w, h = im.size
    original_w, original_h = im.size
    bio, cur_size = None, None
    original_size = None

    def _save():
        nonlocal bio, im, format, cur_size, original_size
        bio = BytesIO()
        im.save(bio, format)
        cur_size = bio.tell()
        original_size = original_size or cur_size
        return cur_size
    while(_save() > limit_size):
        r = 0.85*min(1, sqrt(limit_size/cur_size))
        w, h = int(w*r), int(h*r)
    if((w, h) != (original_w, original_h)):
        print("Compress image %dx%d %d bytes to %dx%d %d bytes" %
              (
                  original_w, original_h, original_size,
                  w, h, cur_size)
              )
    bio.seek(0)
    bytes = bio.read()
    bio.close()
    return base64.b64encode(bytes).decode("ascii")


def img_file_b64(filename, limit_size=(2 << 20)):
    im = Image.open(filename)
    if(im.mode not in ["RGB", "RGBA"]):
        with open(filename, "rb") as f:
            ret = base64.b64encode(f.read()).decode("ascii")
        return ret
    else:
        return img_b64(im, limit_size)


def prepare_message(mes):
    if(not isinstance(mes, list)):
        mes = [mes]
    ret = []
    for i in mes:
        if(isinstance(i, str)):
            if(path.exists(i)):
                ext = path.splitext(i)[-1].lower()
                if(ext in [".jpg", ".gif", ".bmp", ".png"]):
                    file = "base64://"+img_file_b64(i)
                    mes = MSEG.image(file)
                    ret.append(mes)
                elif(ext in [".wav", ".mp3", ".ogg"]):
                    mes = MSEG.record(i)
                    ret.append(mes)
                else:
                    ret.append(MSEG.text(i))
            else:
                ret.append(MSEG.text(i))
        elif(isinstance(i, Image.Image)):
            file = "base64://"+img_b64(i)
            ret.append(MSEG.image(file))
        elif(isinstance(i, MSEG) or isinstance(i, dict)):
            ret.append(i)
        else:
            ret.append(MSEG.text(str(i)))
    return ret


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
        plain_text = html.unescape(event.raw_message)
        ret = cls(sender=sender, pics=pics, ated=ated,
                  plain_text=plain_text, group=group, raw=event)
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
        for key in ["message_type", "group_id", "user_id", "self_id"]:
            if(key in self.raw):
                args[key] = self.raw[key]
        prepare(message=prepare_message(message))
        # print(args)
        ret = cqhttp._bot.sync.send_msg(**args)

    async def response_async(self, message, at=False, reply=False):
        args = dict()

        def prepare(**kwargs):
            args.update(kwargs)
        for key in ["message_type", "group_id", "user_id", "self_id"]:
            if(key in self.raw):
                args[key] = self.raw[key]
        prepare(message=prepare_message(message))
        # print(args)
        ret = await cqhttp._bot.send_msg(**args)
        return ret
