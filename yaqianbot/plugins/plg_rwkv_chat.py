from ..backend.cqhttp.message import CQMessage
from ..plugins.plg_chatbot_record import get_mes_record
from ..backend.receiver_decos import *
from ..backend import receiver
from ..utils.candy import lockedmethod, simple_send
from pil_functional_layout import RichText, Keyword, Row, Column
from math import sqrt
from urllib.parse import urlencode
import requests
from ..utils.rwkv_client import Client as RWKVClient
from ..backend.cqhttp.cqhttp import receivers



for name, module in list(sys.modules.items()):
    if(name.endswith("plg_chatbot")):
        if(hasattr(module, "response_ated")):
            f = module.response_ated
            found = None
            for k, v in receivers.items():
                if(v is f):
                    found = k
            if(found is not None):
                receivers.pop(found)

HOST = bot_config.get("RWKV_HOST", "http://localhost:8001")


clients = {}

form_args = lambda **kwargs: kwargs

def do_chat(message, prompt, retry=1, uname="User", bname="菜菜"):
    uid = message.sender.id
    
    sb = [
        f"{uname}:",
        f"{uname}：",
        f"{bname}:",
        f"{bname}：",
        f"以下是",
        f"Bot:",
        f"Bot：",
        f"用户：",
        f"Human:",
        f"Assistant:",
        f"\n\n"
    ]

    if(uid in clients):
        cl = clients[uid]
    else:
        cl = RWKVClient(stop_before=sb, host=HOST, ignore_occurrence=sb)
    resp, cl = cl.cont(f"{uname}: {prompt}\n{bname}:", length=200)
    if(resp.status_code!=200):
        if(retry > 0):
            clients.pop(uid)
            return do_chat(message, prompt, retry=retry-1, uname=uname, bname=bname)
        else:
            resp.raise_for_status()
    else:
        data = resp.json()["data"]
        contents = data["contents"]
        clients[uid] = cl
        simple_send([contents])


@receiver
@threading_run
@on_exception_response
@command("/reset_rwkv_chat", opts={})
def cmd_reset_rwkv_chat(message: CQMessage, *args, **kwargs):
    uid = message.sender.id
    clients.pop(uid, None)
    simple_send("「菜菜现在不记得之前和%s的对话了…」"%(message.sender.name))

@receiver
@threading_run
@is_ated
@on_exception_response
def rwkv_response_ated(message: CQMessage):
    pt = message.plain_text.strip()
    do_chat(message, pt)