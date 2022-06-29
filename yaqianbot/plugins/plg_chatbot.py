from ast import Index
from collections import defaultdict
from ..backend.cqhttp.message import CQMessage
from ..backend import base_message as Message
from ..backend.receiver_decos import is_su, threading_run, on_exception_response, command, is_ated
from ..backend import receiver
from ..backend.paths import mainpth
from ..utils import after_match
from ..utils.myhash import base32
from ..utils.jsondb import jsondb
from ..utils.algorithms.lcs import lcs as LCS
from os import path
from dataclasses import dataclass, asdict
from ..utils.myhash import base32
import random
from ..utils.algorithms import lower_bound
from .plg_admin import link

@dataclass
class Chat:
    query: str
    response: list

    def hashed(self):
        return base32(self.asdict())

    def asdict(self):
        return asdict(self)

    @classmethod
    def fromdict(cls, D):
        return cls(**D)

    def calc(self, query, debug=False):
        lcs = LCS(query, self.query)
        if(debug):
            print("%.2f" % lcs.common_ratio, *lcs.color_common())
        return lcs.common_ratio

    def item(self):
        return self.hashed(), self.asdict()

    def savetodb(self, db):
        key, value = self.item()
        db[key] = value

chatbot_img_path = path.join(mainpth, "chatbot", "images")
saved_chats = jsondb(path.join(mainpth, "chatbot", "chat"),
                     method=lambda x: str(x)[:3])
pending_chats = jsondb(path.join(mainpth, "chatbot", "pending_chat"),
                     method=lambda x: str(x)[:3])

def get_chat(query, db=saved_chats, debug=False, EPS=0.25):
    chats = []
    weights = []
    for k, v in db.items():
        chat = Chat.fromdict(v)
        score = chat.calc(query, debug=debug)
        if(score > EPS):
            chats.append(chat)
            weights.append(score**3)
    if(not chats):
        return None
    return weighted_choice(chats, weights)


def strip_rich(rich, st, strip_start=True, strip_end=False):
    ret = []
    for idx, i in enumerate(rich):
        if(isinstance(i, str)):
            if(idx == 0 and strip_start):
                if(i.startswith(st)):
                    _i = i[len(st):]
                else:
                    _i = i
            elif(idx == len(rich)-1 and strip_end):
                if(i.endswith(st)):
                    _i = i[:-len(st)]
                else:
                    _i = i
            else:
                _i = i
        else:
            _i = i
        ret.append(_i)
    return ret


def weighted_choice(choices, weights):
    prefix_sum = []
    weight_sum = 0
    for i in weights:
        weight_sum += i
        prefix_sum.append(weight_sum)
    rnd = random.random()*weight_sum
    idx = lower_bound(prefix_sum, rnd)
    return choices[idx]
def replace(message_ls, *args):
    ret = []
    for idx, i in enumerate(message_ls):
        for j in range(0, len(args), 2):
            src = args[j]
            dst = args[j+1]
            i = i.replace(src, dst)
        ret.append(i)
    return ret
def show_pending(message, key=None):
    if(key is None):
        ls = list(pending_chats)
        key = random.choice(ls)
    chat = pending_chats[key]
    chat = Chat.fromdict(chat)
    mes = ["问:\n"]
    if(isinstance(chat.query,str)):
        mes.append(chat.query)
    else:
        mes.append(chat.query)
    mes.append("\n答\n")
    mes.extend(chat.response)
    mes = replace(mes, "%chatbot_img_path%", chatbot_img_path)
    @is_su
    def accept(message: Message):
        nonlocal key
        if(key in pending_chats):
            pending_chats.pop(key)
            chat.savetodb(saved_chats)
            message.response_sync("问答被收录！")
        else:
            message.response_sync("没有此项问答")
    @is_su
    def decline(message: Message):
        nonlocal key
        if(key in pending_chats):
            pending_chats.pop(key)
            # chat.savetodb(saved_chats)
            message.response_sync("问答被拒绝！")
        else:
            message.response_sync("没有此项问答")
    accept_link = link("accept "+key, accept)
    decline_link = link("decline "+key, decline)
    meow = ["\n输入%s以确认加入问答; 输入%s以取消加入问答"%(accept_link, decline_link)]
    message.response_sync(mes+meow)
@receiver
@threading_run
@on_exception_response
@is_ated
def response_ated(message):
    if(message.plain_text):
        chat = get_chat(message.plain_text)
        # message.response_sync(message.plain_text)
        if(chat is not None):
            mes = chat.response
            mes = replace(mes, "%chatbot_img_path%", chatbot_img_path)
            message.response_sync(mes)
        else:
            chat = get_chat("没有回答")
            if(chat is not None):
                mes = chat.response
                mes = replace(mes, "%chatbot_img_path%", chatbot_img_path)
                message.response_sync(mes)
@receiver
@threading_run
@on_exception_response
@command("/审核问答", opts={})
def cmd_view_pending_chat(message):
    show_pending(message)
@receiver
@threading_run
@on_exception_response
@command("/添加问答", opts={})
def cmd_add_custom_chat(message:CQMessage, *args, **kwargs):
    texts = []
    response = []
    for i in args:
        if(i == "带图"):
            ls = message.get_sent_images(rettype = "file", savepath = chatbot_img_path)
            for idx, i in enumerate(ls):
                ls[idx]=path.join("%chatbot_img_path%", path.basename(i))
            response.extend(ls)
        else:
            texts.append(i)
    query = texts[0]
    response.extend(texts[1:])
    if(not response):
        message.response_sync("没有输入答句")
        return
    chat = Chat(query, response)
    chat.savetodb(pending_chats)
    show_pending(message, chat.hashed())

if(__name__ == '__main__'):
    import tempfile
    tmp = tempfile.gettempdir()+"/mamoaimdsoijaosid"
    tmp = jsondb(tmp)
    Chat("被戳的反应", "别戳了").savetodb(tmp)
    Chat("今天天气怎么样", "不好不坏").savetodb(tmp)
    Chat("戳你", "别戳我").savetodb(tmp)
    Chat("你是笨蛋", "我不是笨蛋").savetodb(tmp)
    Chat("嗯喵", "阿喵喵").savetodb(tmp)
    Chat("阿喵喵", "只要你懂阿喵喵，阿喵喵就会帮你").savetodb(tmp)
    Chat("喵你", "只要你懂阿喵喵，阿喵喵就会帮你").savetodb(tmp)
    a = input("喵？")
    while(a):
        print(get_chat(a, tmp, debug=True))
        a = input("喵？")
