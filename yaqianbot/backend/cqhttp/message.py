import html
from typing import Dict
from aiocqhttp import Event
import re
from aiocqhttp.message import MessageSegment as MSEG
from ..bot_threading import threading_run
from ..base_message import User, Message
from .. import requests
from ...utils.candy import print_time, log_header
from . import cqhttp
from os import path
from PIL import Image
from math import sqrt
from ..paths import temppth
import base64
from io import BytesIO


def img_b64(im, limit_size=1e6):

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
        im.resize((w, h)).save(bio, format)
        cur_size = bio.tell()

        if((w, h) != (original_w, original_h)):
            print("Compress image %dx%d %d bytes to %dx%d %d bytes" %
                  (
                      original_w, original_h, original_size,
                      w, h, cur_size)
                  )

        original_size = original_size or cur_size
        return cur_size
    while(_save() > limit_size):
        r = 0.85*min(1, sqrt(limit_size/cur_size))
        w, h = int(w*r), int(h*r)

    bio.seek(0)
    bytes = bio.read()
    bio.close()

    with print_time("b64 encode image"):
        ret = base64.b64encode(bytes).decode("ascii")
    return ret


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
    all_plain = re.split(pattern, message)[::4]
    # print(log_header(), all_cqcode)
    # print(log_header(), all_plain)
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
    def construct_forward(self, message, uin=None, name=None):
        if(uin is None):
            uin = self.raw["self_id"]
        if(name is None):
            _ = cqhttp._bot.sync.get_login_info(self_id=self.raw["self_id"])
            name = _.get("nickname", "菜菜")
        data = dict()
        data["uin"] = uin
        data["name"] = name
        data["content"] = prepare_message(message)
        return {"type": "node", "data": data}
    def send_forward_message(self, messages):
        raw = self.raw
        kwargs = {}
        kwargs["self_id"] = raw["self_id"]
        if("group_id" in raw):
            kwargs["group_id"] = raw["group_id"]
        else:
            raise Exception("only supports send_foward_message for group")
        kwargs["messages"] = messages
        return cqhttp._bot.sync.send_group_forward_msg(**kwargs)
    def get_rich_array(self):
        mes = self.raw.message
        if(isinstance(mes, str)):
            mes = mes_str2arr(html.unescape(mes))
        ret = []
        for i in mes:
            type = i["type"]
            data = i["data"]
            # print(i)
            if(type == "text"):
                t = data["text"]
                ret.append(t)
            elif(type == "image"):
                im = requests.get_image(data["url"])[1]
                # im = sizefit.fit_shrink(im, w*0.9, h*0.5)
                ret.append(im)
        return ret
        # return super().get_rich_array()
    @classmethod
    async def from_cqpoke(cls, event):
        user_id = event["user_id"]
        self_id = event["self_id"]
        user_info = await cqhttp._bot.get_stranger_info(self_id = self_id, user_id = user_id)
        from_group = event.get("group_id", "private")
        sender = User(str(user_id), user_info.get("nickname", "Unknown User"), from_group)
        raw = event
        self_id = str(self_id)
        return cls(raw = raw, self_id = self_id, sender = sender)

    @classmethod
    def from_cq(cls, event):

        ated = list()
        self_id = str(event.get("self_id"))
        if("group_id" in event):
            group = str(event.group_id)
            sender = CQUser.from_cq(event.sender, event.group_id)
        else:
            group = "private"
            sender = CQUser.from_cq(event.sender, "private")

        mes = event.message
        if(isinstance(mes, str)):
            mes = mes_str2arr(html.unescape(mes))
        pics = list()
        plain_texts = []
        for i in mes:
            type = i.get("type")
            data = i.get("data", dict())
            if(type == "image"):
                pics.append(i)
                plain_texts.append("[图片]")
            elif(type == "at"):
                ated.append(str(data.get("qq")))
            elif(type == "text"):
                plain_texts.append(data["text"])
            else:
                pass
        # plain_text = html.unescape(event.raw_message)
        plain_text = "".join(plain_texts)
        # import inspect
        # print(inspect.signature(cls.__init__).parameters.items())
        ret = cls(sender=sender, pics=pics, ated=ated,
                  plain_text=plain_text, group=group, raw=event, self_id=self_id)
        ret.update_rpics()
        return ret

    def get_sent_images(self, rettype="image", **kwargs):
        rpics = self.recent_pics
        ret = []
        if(not rpics):
            raise Exception("未发送图片！")
        for i in rpics:
            data = i['data']
            url = data['url']
            if(rettype == "image"):
                im = requests.get_image(url)
            else:
                im = requests.get_file(url, **kwargs)
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
        if("message_type" not in args):
            if(self.sender.from_group == "private"):
                args["message_type"] = "private"
            else:
                args["message_type"] = "group"
        prepare(message=prepare_message(message))
        # print(args)
        ret = cqhttp._bot.sync.send_msg(**args)
        return ret

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
