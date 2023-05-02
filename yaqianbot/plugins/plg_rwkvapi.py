from ..backend.cqhttp.message import CQMessage
from ..backend.receiver_decos import *
from ..backend import receiver
from ..utils.candy import lockedmethod, simple_send
from pil_functional_layout import RichText, Keyword, Row, Column
from math import sqrt
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
    t = send(message, r)
    sent = t.result()
    message_id = str(sent.get("message_id"))
    if(message_id):
        uid = message.sender.id
        mes_state[message_id] = data["state"]
        cont_args[uid] = cont_args[message_id] = form_args(
            from_state=data["state"],
            recall=recall
        )
def send_pic_if_long(message, contents):
    if(len(contents)>100):
        im = RichText([contents], width=512, bg=(255,)*4).render()
        return simple_send(im)
    else:
        return simple_send(contents)
def send(message, response):
    
    j = response.json()
    data = j["data"]
    contents = data["contents"]
    full = data["full_history"]
    prev = full[:-len(contents)]
    RT = RichText(Keyword("contents"), width=512, bg=(255,)*4)
    FS = 36
    if(not prev):
        B = RT.render(contents=[contents], fill=(0, 0, 0, 255), fontSize=FS)
        return simple_send(B)
    prev_fs = int(min(FS, max(8, FS*sqrt(len(contents)/len(prev)))))

    
    A = RT.render(contents=[prev], fill=(200, 200, 200, 255), fontSize=prev_fs)
    B = RT.render(contents=[contents], fill=(0, 0, 0, 255), fontSize=FS)
    ret = Column([A, B]).render()
    return simple_send(ret)
def do_instruct(message: CQMessage, prompt, from_state="new"):
    global cont_args
    feed = f"""
Below is an instruction that describes a task. Write a response that appropriately completes the request.

# Instruction:
{prompt}

# Response:
"""
    data = form_args(
        feed=feed,
        length=500
    )
    url = HOST+API_PATH+"/"+from_state
    r = requests.post(url, json=data)
    j = r.json()
    if(j["status"]!=0):
        mes = [j["message"]]
        if(j["status"] == -404):
            mes.append("服务端返回会话不存在, 请在指令后加入 -reset 来取消使用上一次对话的状态。")
        simple_send("\n".join(mes))
        return
    data = j["data"]
    contents = data["contents"]
    t = send(message, r)
    
    sent = t.result()
    cont_arg = form_args(
        from_state = data["state"]
    )
    uid = message.sender.id
    cont_args[uid] = cont_arg
    mid = message_id = str(sent.get("message_id", ""))
    if(message_id):
        cont_args[mid] = cont_arg

chat_template_original = lambda username, botname, sep:f"""
The following is a coherent verbose detailed conversation between a Chinese girl named {botname} and her friend {username}. \
{botname} is very intelligent, creative and friendly. \
{botname} likes to tell {username} a lot about herself and her opinions. \
{botname} usually gives {username} kind, helpful and informative advices.

{username}{sep} lhc

{botname}{sep} LHC是指大型强子对撞机（Large Hadron Collider），是世界最大最强的粒子加速器，由欧洲核子中心（CERN）在瑞士日内瓦地下建造。LHC的原理是加速质子（氢离子）并让它们相撞，让科学家研究基本粒子和它们之间的相互作用，并在2012年证实了希格斯玻色子的存在。

{username}{sep} 企鹅会飞吗

{botname}{sep} 企鹅是不会飞的。企鹅的翅膀短而扁平，更像是游泳时的一对桨。企鹅的身体结构和羽毛密度也更适合在水中游泳，而不是飞行。
"""

chat_template_qq = lambda username, botname, sep: f"""一下是一段{username}和{botname}之间的对话。
"""

def do_chat(message: CQMessage, prompt, from_state="new", username="Bob", botname="Alice", sep=":", template="original", n_lf=2):
    global chat_args, cont_args
    if(from_state=="new"):
        if(template == "original"):
            feed_prefix = chat_template_original(username, botname, sep)
        elif(template=="qq"):
            feed_prefix = chat_template_qq(username, botname, sep)
        else:
            feed_prefix = ""
    else:
        feed_prefix=""
    lf = "\n"*n_lf
    data = form_args(
        feed = feed_prefix+f"""{username}{sep} {prompt}{lf}{botname}{sep} """,
        recall=['请总结以上对话的主要内容', "# Response:", "请总结以上对话内容"],
        stop_before=[f"{username}{sep}"],
        stop_at_eot=True,
        adjust = {f"{username}{sep}":-0.3},
        length=500
    )
    print(data["feed"])
    url = HOST+API_PATH+"/"+from_state
    r = requests.post(url, json=data)
    j = r.json()
    if(j["status"]!=0):
        mes = [j["message"]]
        if(j["status"] == -404):
            mes.append("服务端返回会话不存在, 请在指令后加入 -reset 来取消使用上一次对话的状态。")
        simple_send("\n".join(mes))
        return
    data = j["data"]
    contents = data["contents"]
    t = send(message, r)
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

def process_prompt(p: str) -> str:
    def process_lf(p: str) -> str:
        return p.replace("\\n", "\n")
    def process_slash(p: str) -> str:
        return "\\".join([process_lf(i) for i in p.split("\\\\")])
    return process_slash(p)

@receiver
@threading_run
@on_exception_response
@command("/instruct", {})
def cmd_rwkv_instruct(message: CQMessage, *args, **kwargs):
    prompt = process_prompt(" ".join(args))
    do_instruct(message, prompt)

@receiver
@threading_run
@on_exception_response
@command("/cont", {"-recall", '-reset', "-stop_before"}, ls_opts={"-recall", "-stop_before"}, bool_opts = {'-reset'})
def cmd_rwkv_cont(message: CQMessage, *args, **kwargs):
    uid = message.sender.id
    is_reset = kwargs.get("reset")
    prompt = process_prompt(" ".join(args))
    kwa = {}
    if(not is_reset):
        if(message.reply_mes_id):
            mid = str(message.reply_mes_id)
            if(mid in cont_args):
                kwa.update(cont_args[mid])
        elif(uid in cont_args):
            kwa.update(cont_args[uid])
    for k, v in kwargs.items():
        if(k=="recall"):
            kwa[k]=v
    do_cont(message, prompt, **kwa)

@receiver
@threading_run
@on_exception_response
@command("/chat", {"-me", "-bot", "-reset", "-template"}, bool_opts={"-reset"})
def cmd_rwkv_chat(message: CQMessage, *args, **kwargs):
    prompt = process_prompt(" ".join(args))
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
        elif(k=="template"):
            kwa["template"] = v
        
    do_chat(message, prompt, **kwa)