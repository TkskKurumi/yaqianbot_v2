from ..backend.cqhttp.message import CQMessage
from ..backend.receiver_decos import *
from ..backend import receiver
from ..utils.candy import lockedmethod, simple_send
from urllib.parse import urlencode
import requests
HOST = "http://localhost:8001"
API_PATH = "/cont"

mes_from_state = {}
mes_state = {}
remember = {}
user_last = {}
cont_args = {}
chat_args = {}

form_args = lambda **kwargs: kwargs

def do_cont(message: CQMessage, prompt, from_state="new", recall=None):
    global cont_args
    data = form_args(
        feed=prompt,
        stop_at_eot=True,
        recall=recall,
        length=250
    )
    url = HOST+API_PATH+"/"+from_state
    r = requests.post(url, json=data)
    j = r.json()
    if(j["status"]!=0):
        simple_send(j["message"])
        return
    data = j["data"]
    contents = data["contents"]
    t = simple_send(contents)
    sent = t.result()
    message_id = str(sent.get("message_id"))
    if(message_id):
        uid = message.sender.id
        mes_state[message_id] = data["state"]
        cont_args[uid] = cont_args[message_id] = form_args(
            from_state=data["state"],
            recall=recall
        )

def do_chat(message: CQMessage, prompt, from_state="new", username="User", botname="Bot", sep=":"):
    global chat_args, cont_args
    if(from_state=="new"):
        feed_prefix = f"""
The following is a coherent verbose detailed conversation between {botname} and her friend {username}. \
{botname} is very intelligent, creative and friendly. \
{botname} likes to tell {username} a lot about herself and her opinions. \
{botname} usually gives {username} kind, helpful and informative advices.
{username}{sep} 你好，你是谁？
{botname}{sep} 我是{botname}。很高兴认识你。
{username}{sep} 我可以问你几个问题吗？
{botname}{sep} 当然，乐意效劳。
{username}{sep} 今天天气如何？
{botname}{sep} 我不知道您处在哪个城市？或许您可以自行上网查询一下当地天气。
"""
    else:
        feed_prefix=""
    data = form_args(
        feed = feed_prefix+f"""
{username}{sep} {prompt}
{botname}{sep} 
""",
        recall=['请总结以上对话的主要内容', "# Response:", "请总结以上对话内容"],
        stop_before=[f"{username}{sep}"],
        stop_at_eot=True,
        length=250
    )
    print(data["feed"])
    url = HOST+API_PATH+"/"+from_state
    r = requests.post(url, json=data)
    j = r.json()
    if(j["status"]!=0):
        simple_send(j["message"])
        return
    data = j["data"]
    contents = data["contents"]
    t = simple_send(f"{botname}{sep} {contents}")
    sent = t.result()
    message_id = str(sent.get("message_id"))
    if(message_id):
        uid = message.sender.id
        mes_state[message_id] = data["state"]
        chat_args[uid] = chat_args[message_id] = form_args(
            from_state=data["state"],
            username=username,
            botname=botname,
            sep=sep
        )
        
        cont_args[uid] = cont_args[message_id] = form_args(
            from_state=data["state"]
        )


@receiver
@threading_run
@on_exception_response
@command("/cont", {"-recall"}, ls_opts={"-recall"})
def cmd_rwkv_cont(message: CQMessage, *args, **kwargs):
    prompt = " ".join(args)
    kwa = {}
    if(message.reply_mes_id):
        mid = str(message.reply_mes_id)
        if(mid in cont_args):
            kwa.update(cont_args[mid])
    for k, v in kwargs.items():
        if(k=="recall"):
            kwa[k]=v
    do_cont(message, prompt, **kwa)

@receiver
@threading_run
@on_exception_response
@command("/chat", {"-me", "-bot", "-reset"}, bool_opts={"-reset"})
def cmd_rwkv_chat(message: CQMessage, *args, **kwargs):
    prompt = " ".join(args)
    kwa = {}
    if(message.reply_mes_id):
        mid = str(message.reply_mes_id)
        if(mid in chat_args):
            kwa.update(chat_args[mid])
    else:
        if(not kwargs.get("reset")):
            uid = message.sender.id
            if(uid in chat_args):
                kwa.update(chat_args[uid])
    for k, v in kwargs.items():
        if(k=="me"):
            kwa["username"] = v
        elif(k=="bot"):
            kwa["botname"] = v
        
    do_chat(message, prompt, **kwa)