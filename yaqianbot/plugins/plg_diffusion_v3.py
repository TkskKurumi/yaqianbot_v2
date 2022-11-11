import random
import re
from threading import Lock
from typing import Dict, List, Tuple, Union, Literal
from PIL import Image, ImageFilter
import numpy as np
from io import BytesIO
from urllib.parse import urlencode
from pil_functional_layout.widgets import RichText, Grid, Column
from pil_functional_layout import Keyword
from ..backend.cqhttp.message import CQMessage
from ..backend.receiver_decos import *
from ..backend import receiver
from ..backend.configure import bot_config
from ..utils.candy import lockedmethod, simple_send
from ..utils.make_gif import make_gif
from ..utils.lvldb import TypedLevelDB
from ..utils.image import sizefit
from ..utils.image.colors import Color
import requests
from ..backend.paths import mainpth
from os import path
from ..utils.algorithms.lcs import lcs as LCS
from .plg_help import plugin_func, plugin, plugin_func_option, OPT_OPTIONAL
from ..utils.myhash import base32
import math
import time
user_tag_db = TypedLevelDB.open(
    path.join(mainpth, "saves", "plg_diffusion_v3", "user_tags"))
sent_image = {}
last_prompt = {}
if("DIFFUSION_HOST_v3" in bot_config):
    HOST = bot_config.get("DIFFUSION_HOST_v3").strip("[/~～]")
else:
    HOST = "http://localhost:8002"

class SimpleWeightedThrottle:
    def __init__(self, n, period):
        self.called = []
        self.period = period
        self.n = n
        self.lck = Lock()
    def _clear(self):
        tm = time.time()
        period = self.period
        idx = 0
        for idx, i in enumerate(self.called):
            called_tm, n = i
            if(called_tm+period>=tm):
                break
        self.called = self.called[idx:]
    def _sm(self):
        self._clear()
        ret = 0
        for called_tm, n in self.called:
            ret += n
        return ret
    @lockedmethod
    def wait_time(self):
        sm = self._sm()
        if(sm <= self.n):
            return 0
        tm = time.time()
        for called_tm, n in self.called:
            sm -= n
            if(sm <= self.n):
                return called_tm+self.period-tm
        return self.period
    @lockedmethod
    def add(self, n=1):
        tm = time.time()
        self.called.append((tm, n))
class SimpleThrottle:
    def __init__(self, times, period):
        self.called = []
        self.period = period
        self.times = times
    def wait_time(self):
        tm = time.time()
        while(self.called and self.called[0]<tm-self.period):
            self.called = self.called[1:]
        if(len(self.called)<self.times):
            return 0
        t = self.called[-self.times]
        return t+self.period-tm
    def add(self):
        self.called.append(time.time())
throttles = {}
def get_throttle(message):
    id = message.sender.id
    if(id in throttles):
        thr = throttles[id]
    else:
        thr = SimpleWeightedThrottle(120, 120)
        throttles[id] = thr
    return thr
class SimpleImageIndexing:
    def __init__(self, max_n = 128):
        self.keys = []
        self.values = []
        self.max_n = max_n
    def asarray(self, im):
        im = im.convert("RGB").resize((10, 10))
        arr = np.asarray(im)
        return arr
    def __setitem__(self, key, value):
        key = self.asarray(key)
        self.keys.append(key)
        self.values.append(value)
        if(len(self.keys)>self.max_n):
            self.keys = self.keys[1:]
            self.values = self.values[1:]
    def __getitem__(self, key):
        keys = np.array(self.keys).astype(np.float16)
        key = self.asarray(key)
        diff = keys-key
        diff = diff.reshape((diff.shape[0], -1))
        diff = (diff**2).sum(axis=-1)
        idx = diff.argmin()
        return self.values[idx]
        
img_prompt = SimpleImageIndexing()


def _on_reload() -> Tuple[Tuple, Dict]:
    return tuple(), {"sent_image": sent_image, "img_prompt": img_prompt}


def _on_after_reload(*args, **kwargs):
    global sent_image, img_prompt
    if("sent_image" in kwargs):
        sent_image = kwargs["sent_image"]
    if("img_prompt" in kwargs):
        img_prompt = kwargs["img_prompt"]
user_tickets = {}
def _hsl2rgb(h, s, l, a=255):
    return Color.from_hsl(h, s, l, a).get_rgba()
def img2bio(img):
    bio = BytesIO()
    img.convert("RGB").save(bio, "JPEG")
    bio.seek(0)
    return bio
class DiffuserFastAPITicket:
    def __init__(self, purpose):
        r = requests.get(HOST+"/ticket/create/%s" % purpose)
        self.ticket_id = r.json()["data"]["id"]
        self.ticket_url = HOST+"/ticket/"+self.ticket_id
        self.params = {}
    def __getitem__(self, key):
        return self.params[key]
    def param(self, **kwargs):
        self.params.update(kwargs)
        data = {"data": kwargs}
        r = requests.post(self.ticket_url+"/param", json=kwargs)
        return r.json()
    def upload_image(self, image, name="orig_image"):
        url = self.ticket_url+"/upload_image?fn="+name
        bio = img2bio(image)
        return requests.post(url, files={"data":bio}).json()

    @property
    def result(self):
        r = requests.get(self.ticket_url+"/result")
        return r.json()

    def submit(self):
        r = requests.get(self.ticket_url+"/submit")
        return r.json()

    def get_image(self):
        result = self.result
        data = result["data"]
        data_type = data["type"]
        if(data_type == "image"):
            image = data["image"]
        elif(data_type == "image_seq"):
            image = data["images"][0]
        else:
            raise TypeError(data_type)
        r = requests.get(HOST+"/images/"+image)
        bio = BytesIO()
        bio.write(r.content)
        bio.seek(0)
        im = Image.open(bio)
        return im
    def get_image_seq(self):
        result = self.result
        data = result["data"]
        data_type = data["type"]
        assert data_type == "image_sequence", "Response Data is not Image Sequence"
        ret = []
        for i in data["images"]:
            r = requests.get(HOST+"/images/"+i)
            bio = BytesIO()
            bio.write(r.content)
            bio.seek(0)
            im = Image.open(bio)
            ret.append(im)
        return ret


def process_prompt(message: CQMessage, prompt):
    SEP = ", "
    uid = message.sender.id
    tags = {}
    for k, v in user_tag_db.items():
        tags.update(v)
    tags.update(user_tag_db.get(uid, {}))

    kvs = list(tags.items())
    kvs.sort(key=lambda x: -len(x[0]))  # sort by key length
    ret = prompt
    replaced = {}
    remain = prompt
    for k, v in kvs:
        if(k in ret):
            ret = ret.replace(k, SEP+v+SEP)
            remain = ret.replace(k, "")
            replaced[k] = v
    kvs.sort(key=lambda x: -len(x[1]))  # sort by value length
    for k, v in kvs:
        if(v in remain):
            remain = remain.replace(v, "")
            replaced[k] = v
    return


class PromptProcessor:
    def _process_symbols(prompt: str, trim_bracket=False) -> str:
        if(trim_bracket):
            prompt = re.sub(r"[{}]/\*|\*/", "", prompt)
        prompt = prompt.replace("\uff0c", ',')
        prompt = re.sub(", +, *", ", ", prompt)
        prompt = re.sub(" +", " ", prompt)
        prompt = prompt.strip(", ")
        return prompt
    def _process_weights(prompt, return_dict=False, return_dict_cat=", "):
        pattern = r"[{}]|/\*|\*/"
        ret = []
        operators = re.findall(pattern, prompt)
        segments = re.split(pattern, prompt)
        curr_weight = 1
        for idx, seg in enumerate(segments):
            if(seg.strip(" ")):
                ret.append((curr_weight, seg))
            if (idx < len(operators)):
                op = operators[idx]
                if (op == "{"):
                    curr_weight *= 1.1
                elif (op == "}"):
                    curr_weight /= 1.1
                else:
                    curr_weight *= -1
        if(return_dict):
            ret_dict = {}
            for w, s in ret:
                if(w in ret_dict):
                    ret_dict[w] += return_dict_cat+s
                else:
                    ret_dict[w] = s
            for w, s in ret_dict.items():
                s = PromptProcessor._process_symbols(s)
                ret_dict[w] = s
            return ret_dict
        return ret
    def _process_translation(prompt: str, entries:List[Tuple[str, str]]) -> Tuple[str, str, dict]:
        SEP = ", "
        ret = prompt
        remain = ret
        replacement = dict()
        entries.sort(key=lambda x: -len(x[0]))
        for k, v in entries:
            if(k in ret):
                ret = ret.replace(k, SEP+v+SEP)
                remain = remain.replace(k, "")
                replacement[k] = v
        entries.sort(key=lambda x:-len(x[1]))
        for k, v in entries:
            if(v in remain):
                remain = remain.replace(v, "")
                replacement[k] = v
        ret = PromptProcessor._process_symbols(ret)
        remain = PromptProcessor._process_symbols(remain, trim_bracket=True)
        return ret, remain, replacement
    def __init__(self, prompt, entries):
        raw, remain, replacements = PromptProcessor._process_translation(prompt, entries)
        friendly_weighted = PromptProcessor._process_weights(raw, return_dict=True)
        for w, s in friendly_weighted.items():
            _entries = [(k, v) for k, v in entries if(len(v)<0.95*len(s))]
            # 免除一个词解释等于没解释的情况？
            _, _remain, _replacement = PromptProcessor._process_translation(s, _entries)
            s = [_remain] +list(_replacement)
            s = ", ".join(s)
            s = PromptProcessor._process_symbols(s)
            friendly_weighted[w] = s
        self.friendly_weighted = friendly_weighted
        fraw = []
        for w, s in friendly_weighted.items():
            try:
                n = math.log(abs(w))/math.log(1.1)
            except Exception:
                raise Exception("Math Error when log(abs(%s))/log(1.1)"%w)
            n = round(n)
            if(n>0):
                s="{"*n+s+"}"*n
            elif(n<0):
                s = "}"*(-n)+s+"{"*(-n)
            if(w<0):
                s = "/*"+s+"*/"
            fraw.append(s)
        self.friendly_raw = ", ".join(fraw)

        self.raw = raw
        self.remain = remain
        self.replacements = replacements
        
    def illust(self, **kwargs):
        RT = RichText(Keyword("texts"), width=512, fontSize=30)
        ws = list(self.friendly_weighted)
        minw, maxw = min(ws), max(ws)
        if(minw<0):
            minh, maxh = 0, 120
        else:
            minh, maxh = 60, 120
        def get_weight_color(w):
            nonlocal minw, minh, maxw, maxh
            x = (w-minw)/(maxw-minw+1e-8)
            H = minh+x*(maxh-minh)
            return Color.from_hsl(H, 1, 0.2).get_rgb()
        itms = []
        for k, v in kwargs.items():
            if(isinstance(v, str)):
                itms.append(RT.render(texts=["%s: %s"%k, v]))
            elif(isinstance(v, Image.Image)):
                itms.append(v)
        fill = _hsl2rgb(180, 0.1, 0.15)
        bg = _hsl2rgb(180, 0.8, 0.9)
        itms.append(RT.render(texts=["结果: %s"%self.raw], fill=fill, bg=bg, fontSize=24))

        if(self.remain):
            if(re.findall("[^\x00-\x7F]", self.remain)):
                fill = _hsl2rgb(0, 0.2, 0.55)
                bg = _hsl2rgb(0, 0.8, 0.5)
            else:
                fill = _hsl2rgb(0, 0.2, 0.15)
                bg = _hsl2rgb(0, 0.8, 0.9)
            itms.append(RT.render(texts=["未翻译: %s"%self.remain], fill=fill, bg=bg))
    
    
        for w in sorted(ws):
            s = self.friendly_weighted[w]
            prompt = "权重%.3f: "%w+s
            itms.append(RT.render(texts=[prompt], fill=get_weight_color(w), bg=(255,)*3))
        
        COL = Column(itms, bg=(255,)*3, borderWidth=20, outer_border=True)
        return COL.render()
_Entry = Tuple[str, str]
_Entries = List[_Entry]
def get_user_entries_dict(uid):
    all_entries = {}
    self_entries = {}
    for id, v in user_tag_db.items():
        if(id == uid):
            self_entries.update(v)
        else:
            all_entries.update(v)
    return self_entries, all_entries
def get_user_entries(uid: str) -> Tuple[_Entries, _Entries]:
    all_entries = []
    self_entries = []
    for id, v in user_tag_db.items():
        if(id == uid):
            self_entries.extend(v.items())
        else:
            all_entries.extend(v.items())
    return self_entries, all_entries
def roll_prompt(uid, orig_prompt = "", max_length = 512):
    s, o = get_user_entries(uid)
    so = s+o
    values = []
    keys = []
    addable = True
    while(addable):
        random.shuffle(so)
        addable = False
        for k, v in so:
            temp = values+[v]
            temp = ", ".join(temp)
            if(len(temp)<max_length-len(orig_prompt)):
                values.append(v)
                keys.append(k)
                addable = True
    
    return ", ".join(keys), ", ".join(values)
def illust_entries(self_e, other_e):
    RT = RichText(Keyword("texts"), width=512, fontSize=36)
    itms = []
    color_k = Color.from_hsl(240, 1, 0.2).get_rgba()
    color_v = Color.from_hsl(240, 0.5, 0.8).get_rgba()
    for k, v in other_e:
        k_illust = RT.render(texts=[k], fill=color_k)
        v_illust = RT.render(texts=["(%s)"%v], fill=color_v)
        itms.append(RT.render(texts=[k_illust, v_illust]))
    color_k = Color.from_hsl(120, 1, 0.2).get_rgba()
    color_v = Color.from_hsl(120, 0.5, 0.8).get_rgba()
    for k, v in self_e:
        k_illust = RT.render(texts=[k], fill=(40, 80, 0))
        v_illust = RT.render(texts=["(%s)"%v], fill=(40, 180, 100))
        itms.append(RT.render(texts=[k_illust, v_illust]))
    
    return Column(itms, bg=(255,)*3).render()
@receiver
@threading_run
@on_exception_response
@command("/测试", opts={})
def cmd_test(message: CQMessage, *args, **kwargs):
    im = message.get_reply_image()
    if(im):
        prompt = img_prompt[im]
        uid = message.sender.id
        udb = user_tag_db.get(uid, {})
        key = "标签"+base32(prompt, length=2)
        udb[key] = prompt
        simple_send("将此图标签简化为了「%s」"%key)
        user_tag_db[uid] = udb
    else:
        simple_send("无法获取此条消息的图片")
def do_search_tag(message: CQMessage, kwd):
    uid = message.sender.id
    sd, od = get_user_entries_dict(uid)
    s, o = get_user_entries(uid)
    candidates = []
    for key, val in sd.items():
        score1 = LCS(kwd, key).get_common_ratio(0.8)
        score2 = LCS(kwd, val).get_common_ratio(0.8)
        score = score1+score2/3
        cand = (-score, key, val, -1)
        candidates.append(cand)
    for key, val in od.items():
        score1 = LCS(kwd, key).get_common_ratio(0.8)
        score2 = LCS(kwd, val).get_common_ratio(0.8)
        score = score1+score2/3
        cand = (score, key, val, 0)
        candidates.append(cand)
    mes = []
    candidates = sorted(candidates)
    for i in candidates[:10]:
        score, key, val, isself = i
        val_ill = PromptProcessor(val, s+o).friendly_raw
        mes.append(message.construct_forward([key, " ", val_ill]))
    message.send_forward_message(mes)
@receiver
@threading_run
@on_exception_response
@command("[/~～]标签简写", opts={"-pop"}, bool_opts={"-pop"})
def cmd_tag_alias_v3(message: CQMessage, *args, **kwargs):
    uid = message.sender.id
    im = message.get_reply_image()
    if(im):
        s, o = get_user_entries(uid)
        prompt = img_prompt[im]
        uid = message.sender.id
        udb = user_tag_db.get(uid, {})
        if(args):
            key = "".join(args)
        else:
            key = "临时"+base32(prompt, length=2)
        udb[key] = prompt
        PP = PromptProcessor(prompt, s+o)
        simple_send("将此图标签简化为了%s"%key)
        forw = message.construct_forward(PP.friendly_raw)
        message.send_forward_message([forw])
        user_tag_db[uid] = udb
        return
    def do_show():
        nonlocal message, uid
        s, o = get_user_entries(uid)
        if(not (s+o)):
            simple_send("没有任何标签简写")
        else:
            ill= illust_entries(s, o)
            simple_send(ill)
    if(args):
        k = args[0]
        udb = user_tag_db.get(uid, {})
        if(len(args)>1):
            s, o = get_user_entries(uid)
            v = " ".join(args[1:])
            PP = PromptProcessor(v, s+o)
            v = PP.raw
            udb[k] = v
            user_tag_db[uid] = udb
        else:
            sd, od = get_user_entries_dict(uid)
            s, o = get_user_entries(uid)
            if("pop" in kwargs):
                udb = user_tag_db.get(uid, {})
                v = udb.pop(k, None)
                user_tag_db[uid] = udb
                sd, od = get_user_entries_dict(uid)
                assert (k not in sd), "删除失败？？？"
                simple_send("已删除%s -> %s"%(k, v))
            return do_search_tag(message, k)
    do_search_tag(message, " ".join(args))

def do_img2img(message, orig_image, *args, **kwargs):
    uid = message.sender.id
    t = DiffuserFastAPITicket("img2img")
    # imgtype, orig_image = message.get_sent_images()[0]

    if(not args):
        if(uid in last_prompt):
            prompt = last_prompt[uid]
        else:
            return simple_send("请输入prompt")
    else:
        prompt = " ".join(args)

    strength = kwargs.get("s") or kwargs.get("strength")
    if(strength is None):
        strength = 0.68
    else:
        strength = float(strength)

    g = kwargs.get("guidance") or kwargs.get("g")
    if(g):
        g = float(g)
    else:
        g = 10/strength

    ddim_noise = kwargs.get("noise")
    if(ddim_noise is not None):
        ddim_noise = float(ddim_noise)
    else:
        ddim_noise = 0

    s, o=get_user_entries(message.sender.id)
    PP = PromptProcessor(prompt, s+o)
    last_prompt[uid] = PP.raw
    params = {}
    params["alpha"] = strength
    params["prompt"] = PP.raw
    params["guidance"] = g
    params["ddim_noise"] = ddim_noise

    tickets = []
    num = int(kwargs.get("n", 1))
    thr = get_throttle(message)
    if(thr.wait_time()):
        simple_send("过于频繁，请再过%.3f秒"%thr.wait_time())
        return
    


    for i in range(num):
        t = DiffuserFastAPITicket("img2img")
        t.param(**params)
        t.upload_image(orig_image)
        tickets.append(t)
    PP_ill = PP.illust(orig_image = orig_image)
    for idx, t in enumerate(tickets):
        submit = t.submit()
        eta = submit["data"]["eta"]
        eta_this = submit["data"]["eta_this"]
        thr.add(eta_this)
        
        
        if(not idx):
            mes = [PP_ill, "预计%.1f秒"%(eta*(num-idx))]
            simple_send(mes)
        img = t.get_image()
        sent_image[message.sender.id] = img
        if(num>1):
            mes = [img]
            if(idx != num-1):
                mes.append("%d/%d, 预计%.1f秒"%(idx+1, num, eta*(num-1-idx)))
            else:
                mes.append("%d/%d"%(num, num))
            simple_send(mes)
        else:
            simple_send(img)


@receiver
@threading_run
@on_exception_response
@command("[/~～]插值画图", opts={"-from", "-to", "-a"},ls_opts={"-from","-to"})
def cmd_interp_v3(message: CQMessage, *args, **kwargs):
    base_prompt = " ".join(args)
    from_prompt = base_prompt+" ".join(kwargs.get("from", ""))
    to_prompt = base_prompt+" ".join(kwargs.get("to", ""))
    
    s, o=get_user_entries(message.sender.id)
    FPP = PromptProcessor(from_prompt, s+o)
    TPP = PromptProcessor(to_prompt, s+o)
    from_raw = FPP.raw
    to_raw = TPP.raw
    if(FPP.raw == TPP.raw):
        simple_send("文本相同")
        return
    a = float(kwargs.get('a', 9/16))

    t = DiffuserFastAPITicket("txt2img_interp")
    params = {"aspect":a, "prompt":from_raw, "prompt1":to_raw}
    t.param(**params)
    submit = t.submit()
    eta_this = submit["data"]["eta_this"]
    get_throttle(message).add(eta_this)
    if(submit.get("data", {}).get("eta")):
        eta = submit.get("data", {}).get("eta")
        simple_send("预计用时%.1f秒"%eta)

    imgs = t.get_image_seq()
    gif = make_gif(imgs, fps=3, frame_area_sum=1e7)
    fw = []
    for i in imgs:
        fw.append(message.construct_forward(i))
    simple_send(gif)
    message.send_forward_message(fw)
    
@receiver
@threading_run
@on_exception_response
@command("[/~～]完善画图", opts={"-guidance", "-g", "-strength", "-s"})
def cmd_continue_img2img_v3(message: CQMessage, *args, **kwargs):
    if(message.get_reply_image()):
        orig_image = message.get_reply_image()
    else:
        orig_image = sent_image[message.sender.id]
    return do_img2img(message, orig_image, *args, **kwargs)
@receiver
@threading_run
@on_exception_response
@command("[/~～]以图画图", opts={"-guidance", "-g", "-strength", "-s", "-noise", "-n"})
def cmd_img2img_v3(message: CQMessage, *args, **kwargs):
    if(message.get_reply_image()):
        orig_image = message.get_reply_image()
    else:
        imgtype, orig_image = message.get_sent_images()[0]
    n = kwargs.get("n")
    if(n is not None):
        n = max(1, min(int(n), 4))
    else:
        n = 1
    kwargs["n"] = n
    do_img2img(message, orig_image, *args,**kwargs)
    
    
@receiver
@threading_run
@on_exception_response
@command("[/~～]画图", opts={'-guidance', "-g", "-aspect", "-a", "-n", "-roll"}, bool_opts={"-roll"})
def cmd_aidraw_v3(message: CQMessage, *args, **kwargs):
    
    uid = message.sender.id
    
    
    if(not args):
        if(uid in last_prompt):
            prompt = last_prompt[uid]
        elif(not kwargs.get("roll")):
            return simple_send("请输入prompt")
        else:
            prompt=""
    else:
        prompt = " ".join(args)
    
    s, o=get_user_entries(message.sender.id)
    PP = PromptProcessor(prompt, s+o)
    last_prompt[uid] = PP.raw
    
    task_kwargs = {"prompt": PP.raw}
    g = kwargs.get("guidance") or kwargs.get("g")
    if(g):
        task_kwargs["guidance"] = float(g)
    a = kwargs.get("aspect") or kwargs.get("a")
    if(a):
        task_kwargs["aspect"] = float(a)
    n = kwargs.get("n")
    
    if(n):
        n = min(max(1, int(n)), 10)
    else:
        n = 1
    thr = get_throttle(message)
    if(thr.wait_time()):
        simple_send("过于频繁，请再过%.3f秒"%thr.wait_time())
        return
    tickets = []
    rolled = []
    for i in range(n):    
        t = DiffuserFastAPITicket("txt2img")
        
        if(kwargs.get("roll")):
            k, v = roll_prompt(uid, PP.raw)
            rolled.append(k)
            task_kwargs["prompt"] = PP.raw+", "+v

        param_result = t.param(**task_kwargs)
        user_tickets[message.sender.id] = t
        if(i == 0):
            
            submit = t.submit()
            eta_this = submit["data"]["eta_this"]
            thr.add(eta_this)
        tickets.append(t)
    if(rolled):
        mes = ["roll到了"]+["   "+i for i in rolled]
        simple_send("\n".join(mes))
    submit_mes = [PP.illust()]
    if(submit.get("data", {}).get("eta")):
        eta = submit.get("data", {}).get("eta")*n
        submit_mes.append("预计用时%.1f秒"%eta)
    simple_send(submit_mes)
    for idx, t in enumerate(tickets):
        mes = []

        if(n>1):
            mes.append("%d/%d"%(idx+1, n))
            submit = t.submit()
            eta_this = submit["data"]["eta_this"]
            thr.add(eta_this)
            if(submit.get("data", {}).get("eta")):
                eta = submit.get("data", {}).get("eta")*(n-idx-1)
                if(eta):
                    mes.append(", 预计用时%.1f秒"%eta)

        img = t.get_image()
        img_prompt[img] = t["prompt"]
        mes.append(img)
        sent_image[uid] = img
        simple_send(mes)
@receiver
@threading_run
@on_exception_response
@command("[/~～]重发", opts={})
def cmd_resent_v3(message:CQMessage, *args,**kwargs):
    simple_send(sent_image[message.sender.id])




def do_process(message, orig_image: Image.Image, expand=False, gamma=1, **kwargs):
    ret = orig_image.copy()
    orig_w, orig_h = ret.size
    if(expand):
        if(kwargs.get("rate")):
            exp = float(kwargs["rate"])*orig_h/orig_w-1
            kwargs["left"] = exp/2
            kwargs["right"] = exp/2
        if(kwargs.get("left")):
            left = float(kwargs["left"])
            w, h = ret.size
            w0, new_w = int(orig_w*0.1), int(w+orig_w*left)
            center = ret.crop((w0, 0, w, h))
            w1 = center.size[0]
            left = ret.crop((0, 0, w0, h)).resize((new_w-w1, h))
            _ret = Image.new(ret.mode, (new_w, h))
            _ret.paste(left, (0, 0))
            _ret.paste(center, (new_w-w1, 0))
            ret = _ret
        if(kwargs.get("right")):
            right = float(kwargs["right"])
            w, h = ret.size
            w0, new_w = int(orig_w*0.1), int(w+orig_w*right)
            
            center = ret.crop((0, 0, w-w0, h))
            w1 = w-w0
            right = ret.crop((w1, 0, w, h)).resize((new_w-w1, h))
            _ret = Image.new(ret.mode, (new_w, h))
            _ret.paste(center, (0, 0))
            _ret.paste(right, (w1, 0))
            ret = _ret

    sent_image[message.sender.id] = ret
    simple_send(ret)
    return ret
process_options = {"opts":{"-expand", "-left", "-right", "-rate"}, "bool_opts":{"-expand"}}
@receiver
@threading_run
@on_exception_response
@command("[/~～]处理", **process_options)
def cmd_process_v3(message: CQMessage, *args, **kwargs):
    if(message.get_reply_image()):
        img = message.get_reply_image()
    else:
        
        _, img = message.get_sent_images()[0]
    return do_process(message, img.convert("RGB"), **kwargs)

@receiver
@threading_run
@on_exception_response
@command("[/~～]重发", **process_options)
def cmd_resent_v3(message:CQMessage, *args,**kwargs):
    if(message.get_reply_image()):
        img = message.get_reply_image()
    else:
        img = sent_image[message.sender.id]
    w, h = img.size
    x, y =random.randrange(w), random.randrange(h)
    img.putpixel((x, y), (114, 51, 4))
    # simple_send(img)
    do_process(message, img.convert("RGB"), **kwargs)

@receiver
@threading_run
@on_exception_response
@command("[/~～]inpaint", opts={"-p"}, ls_opts={"-p"})
def cmd_inpaint_v3(message:CQMessage, *args, **kwargs):
    s, o=get_user_entries(message.sender.id)
    PP = PromptProcessor(" ".join(args), s+o)
    PP_ill = PP.illust()
    last_prompt[message.sender.id] = PP.raw

    W=100
    mask = np.zeros((W, W), np.float16)
    
    for p in kwargs.get("p", list()):
        params = [0, 1, 0, 1, 1, 0.95]
        if(p=="center"):
            p = "0.3:0.7:0.3:0.7"
        elif(p=="lower"):
            p = "0:1:0.5:1"
        elif(p=="upper"):
            p = "0:1:0:0.5"
        elif(p=="left"):
            p = "0:0.5"
        elif(p=="right"):
            p = "0.5:1"
        for idx, i in enumerate(p.split(":")):
            i=float(i.strip())
            params[idx]=i
        le, ri, up, lo, x, s = params
        le, ri, up, lo = map(lambda x:int(x*W), [le, ri, up, lo])
        mask[le:ri, up:lo] = mask[le:ri, up:lo]*(1-s) + s*x
    mask = (mask.swapaxes(0, 1)*255).astype(np.uint8)
    if(not np.any(mask)):
        simple_send("没有指定绘画区域")
        return
    imgtype, orig_image = message.get_sent_images()[0]
    mask = Image.fromarray(mask).resize(orig_image.size)
    w, h=mask.size
    rad = (w*h)**0.5
    mask=mask.filter(ImageFilter.GaussianBlur(rad/15))
    orig_invert = np.array(orig_image)
    orig_invert = 255*(orig_invert<128)
    orig_invert = Image.fromarray(orig_invert.astype(np.uint8))
    orig_shown = orig_image.copy()
    orig_shown.paste(orig_invert, mask=mask)

    
    t = DiffuserFastAPITicket("inpaint")
    t.param(prompt=PP.raw, mode=1)
    t.upload_image(orig_image)
    t.upload_image(mask, "mask_image")
    sub = t.submit()
    eta = sub["data"]["eta"]
    
    eta_this = sub["data"]["eta_this"]
    thr = get_throttle(message)
    thr.add(eta_this)
    
    mes = [PP.illust(mask=orig_shown), "预计%.1f秒"%eta]
    simple_send(mes)
    im = t.get_image()
    sent_image[message.sender.id] = im
    simple_send(im)

