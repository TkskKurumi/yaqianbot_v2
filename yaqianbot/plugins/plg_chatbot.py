
from collections import defaultdict
from ..backend.cqhttp.message import CQMessage
from ..backend import base_message as Message
from ..backend.receiver_decos import is_su, threading_run, on_exception_response, command, is_ated
from ..backend import receiver
from ..backend.paths import mainpth
from ..utils import after_match
from ..utils.myhash import base32
from ..utils.jsondb import jsondb
from ..utils.lvldb import TypedLevelDB
from ..utils.algorithms.lcs import lcs as LCS
from os import path
from dataclasses import dataclass, asdict
from ..utils.myhash import base32
import random
from ..utils.algorithms import lower_bound
from .plg_admin import link
from . import plg_chatbot_record
from ..utils.candy import simple_send, print_time
import time


@dataclass
class Chat:
    query: str
    response: list

    def __hash__(self):
        return self.hashed()

    def __eq__(self, other):
        return self.query == other.query and self.response == other.query

    def hashed(self):
        return base32(self.asdict())

    def asdict(self):
        return asdict(self)

    @classmethod
    def from_any(cls, D):
        if(isinstance(D, Chat)):
            return D
        else:
            return cls.fromdict(D)

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


def get_chat(query, db=saved_chats, debug=False, EPS=0.1):
    chats = []
    if(debug):
        meow = []
    weights = []
    for k, v in db.items():
        chat = Chat.from_any(v)
        score = chat.calc(query)
        if(score > EPS):
            chats.append(chat)
            weights.append(score**3)
        if(debug):
            meow.append((score, chat))
    if(debug):
        meow.sort(key=lambda x:-x[0])
        print(meow[:10])
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
            if(isinstance(i, str)):
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
    if(isinstance(chat.query, str)):
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
    meow = ["\n输入%s以确认加入问答; 输入%s以取消加入问答" % (accept_link, decline_link)]
    message.response_sync(mes+meow)


def response_when(message, when):
    chat = get_chat(when)
    if(chat is not None):
        mes = chat.response
        mes = replace(mes, "%chatbot_img_path%", chatbot_img_path)
        message.response_sync(mes)


def send_chat(message, chat):
    mes = chat.response
    mes = replace(mes, "%chatbot_img_path%", chatbot_img_path)
    simple_send(mes)


@receiver
@threading_run
@is_ated
@on_exception_response
def response_ated(message):
    def ask(prompt, db=saved_chats):
        chat = get_chat(prompt, db=db, debug=False)
        return chat
    pt = message.plain_text.strip()
    if(pt):
        database = {}
        database.update(saved_chats._asdict())
        database.update(get_db_from_group(message.group, max_nocache = 250))
        
        ls = list(plg_chatbot_record.db)
        random.shuffle(ls)
        for idx, g in enumerate(ls):
            database.update(get_db_from_group(g, max_nocache=25, exclude_type="reply;at,image"))
            if(idx>5):
                chat = ask(pt, database)
                if(chat is not None):
                    break
        chat = ask(pt, database)
        
        # message.response_sync(message.plain_text)
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


record_chat_cache = {}


def mes_pair_as_chat(mesi, mesj, exclude_type="reply;at"):
    key = (mesi.raw.get("message_seq"), mesj.raw.get("message_seq"))
    if(key in record_chat_cache):
        return 1, key, record_chat_cache[key]
    else:
        q = mesi.plain_text
        r = mesj.get_mseg_list()
        _r = []
        for i in r:
            type = i.get("type")
            if(type in exclude_type):continue
            _r.append(i)
        r = _r
        if(not r):
            return None
        ret = Chat(q, r)
        record_chat_cache[key] = ret
        return 0, key, ret


def get_db_from_group(group, max_nocache=100, exclude_type="at,reply"):
    with print_time("get message record db"):
        com, buf = plg_chatbot_record.get_mes_record(group)
        messages = com+buf
        messages = messages[::-1]
        prune = max_nocache
        ret = {}
        
        for jdx, j in enumerate(messages[:-2]):
            # j = messages[idx+1]
            i = messages[jdx+1]
            mesi = CQMessage.from_cq(i)
            if(not mesi.plain_text):
                continue
            mesj = CQMessage.from_cq(j)
            j_text = mesj.plain_text
            if(len(j_text)>300):
                continue
            chat = mes_pair_as_chat(mesi, mesj, exclude_type=exclude_type)
            if(not chat):
                continue
            is_cached, key, chat = chat
            ret[key] = chat
            prune -= 1-is_cached
            if(prune <= 0):
                print("Info: Lazy processing Chat record for group %s, %d/%d is processed"%(group, jdx, len(messages)-1))
                break
        return ret


@receiver
@threading_run
@on_exception_response
@is_su
@command("/测试问答", opts={"-g"})
def cmd_chatbot_test_record_chat(message: CQMessage, *args, **kwargs):
    contents = " ".join(args)
    if("g" in kwargs):
        group = kwargs["g"]
    else:
        group = message.group
    ret_messages = []

    st = time.time()
    dummy_db = get_db_from_group(group)
    ed = time.time()
    elapsed = ed-st
    a, b = len(dummy_db), elapsed
    c = a/b
    ret_messages.append(
        "Loaded %.1f message records in %.1f seconds, %.1f/sec" % (a, b, c))

    st = time.time()

    chat = get_chat(contents, db=dummy_db, EPS=0.1, debug = True)
    if(chat is None):
        simple_send("没有回答")
    else:
        send_chat(message, chat)

    ed = time.time()
    elapsed = ed-st
    a, b = len(dummy_db), elapsed
    c = a/b
    ret_messages.append(
        "\nCaculated %.1f message records in %.1f seconds, %.1f/sec" % (a, b, c))
    simple_send(ret_messages)


@receiver
@threading_run
@on_exception_response
@command("/添加问答", opts={})
def cmd_add_custom_chat(message: CQMessage, *args, **kwargs):
    texts = []
    response = []
    for i in args:
        if(i == "带图"):
            ls = message.get_sent_images(
                rettype="file", savepath=chatbot_img_path)
            for idx, i in enumerate(ls):
                ls[idx] = path.join("%chatbot_img_path%", path.basename(i))
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
