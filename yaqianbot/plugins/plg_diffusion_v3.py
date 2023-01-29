import random
from ..backend.bot_date import now
from ..utils.image.hytk import hytk
import re
from threading import Lock
from typing import Dict, List, Tuple, Union, Literal
from PIL import Image, ImageFilter
import numpy as np
from io import BytesIO
from urllib.parse import urlencode
from pil_functional_layout.widgets import RichText, Grid, Column, Row
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
from ..utils.algorithms.lcs import _lcs
from .plg_help import plugin_func, plugin, plugin_func_option, OPT_OPTIONAL
from ..utils.myhash import base32
import math
import time
user_tag_db = TypedLevelDB.open(
    path.join(mainpth, "saves", "plg_diffusion_v3", "user_tags"))
user_today_db = TypedLevelDB.open(
    path.join(mainpth, "saves", "plg_diffusion_v3", "today")
)
sent_image = {}
last_prompt = {}
if("DIFFUSION_HOST_V3" in bot_config):
    HOST = bot_config.get("DIFFUSION_HOST_V3").strip("[/~～]")
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
        prompt = re.sub("[ ,]*,[ ,]*", ", ", prompt)
        prompt = re.sub(" +", " ", prompt)
        prompt = re.sub(r"\(, \)+", ", ", prompt)
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
    def _process_exclude(prompt, entries, excludes: Union[Tuple, List, Literal[None]] = None):
        if(excludes):
            if(isinstance(excludes, str)):
                excludes = PromptProcessor(excludes, entries).ordered
                excludes = excludes.split(",")
            excludes = [i.strip() for i in excludes]
            tags = re.split(r"/\*|\*/|,|[\{\}]", prompt)
            print("exc", excludes)
            print("tags", [i.strip() for i in tags])
            operators = re.findall(r"/\*|\*/|,|[\{\}]", prompt)
            _ = []
            for idx, i in enumerate(tags):
                if(i.strip() not in excludes):
                    _.append(i.strip())
                if(idx < len(operators)):
                    _.append(operators[idx])
            prompt = "".join(_)
        return prompt
    def __init__(self, prompt, entries, excludes: Union[Tuple, List, Literal[None]] = None):
        raw, remain, replacements = PromptProcessor._process_translation(prompt, entries)
        raw = PromptProcessor._process_exclude(raw, entries, excludes)
        friendly_weighted = PromptProcessor._process_weights(raw, return_dict=True)
        self.raw_weighted = dict(friendly_weighted)
        ordered = ""
        for w, s in friendly_weighted.items():
            _entries = [(k, v) for k, v in entries if(len(v)<0.95*len(s))]
            # 免除一个词解释等于没解释的情况？
            try:
                n = math.log(abs(w))/math.log(1.1)
            except Exception:
                raise Exception("Math Error when log(abs(%s))/log(1.1)"%w)
            n = round(n)
            o = s
            if(n>0):
                o = "{"*n+o+"}"*n
            elif(n<0):
                o = "}"*(-n)+o+"{"*(-n)
            if(w<0):
                o = "/*" + o +"*/"
            ordered+=o
            
            _, _remain, _replacement = PromptProcessor._process_translation(s, _entries)
            s = [_remain] +list(_replacement)
            s = ", ".join(s)
            s = PromptProcessor._process_symbols(s)
            friendly_weighted[w] = s
        self.ordered = ordered
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
                itms.append(RT.render(texts=["%s: %s"%(k, v)]))
            elif(isinstance(v, Image.Image)):
                itms.append(v)
        fill = _hsl2rgb(180, 0.1, 0.15)
        bg = _hsl2rgb(180, 0.8, 0.9)
        itms.append(RT.render(texts=["结果: %s"%self.ordered], fill=fill, bg=bg, fontSize=24))

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
def roll_prompt(uid, orig_prompt = "", max_length = 768):
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
def do_search_tag(message: CQMessage, kwd, topk = 10, **kwargs):
    uid = message.sender.id
    sd, od = get_user_entries_dict(uid)
    s, o = get_user_entries(uid)
    rank = list(sd)+list(od)+[kwd]
    rank = sorted(rank)
    rank = {i: idx for idx, i in enumerate(rank)}
    
    candidates = []
    for key, val in sd.items():
        score1 = LCS(kwd, key).get_common_ratio(0.8)
        score2 = LCS(kwd, val).get_common_ratio(0.8)
        score = score1+score2
        if(kwargs.get("rnd")):
            score*=random.random()
        rnk = abs(rank[kwd]-rank[key])
        cand = (-score, rnk, key, val, -1)
        candidates.append(cand)
    for key, val in od.items():
        score1 = LCS(kwd, key).get_common_ratio(0.8)
        score2 = LCS(kwd, val).get_common_ratio(0.8)
        score = score1+score2
        if(kwargs.get("rnd")):
            score*=random.random()
        rnk = abs(rank[kwd]-rank[key])
        cand = (-score, rnk, key, val, 0)
        candidates.append(cand)
    mes = []
    mes_raw = []
    PPs = []
    candidates = sorted(candidates)
    for i in candidates[:topk]:
        score, _, key, val, _ = i
        PP = PromptProcessor(val, s+o)
        PPs.append(PP)
        if(kwargs.get("raw")):
            v = PP.ordered
        elif(kwargs.get("simple")):
            v = ""
        else:
            v = PP.friendly_raw
        mes.append(message.construct_forward([key, " ", v]))
        mes_raw.append([PP.illust(**{"标签": key})])
    try:
        message.send_forward_message(mes)
    except Exception:
        mes = []
        for i in mes_raw[:4]:
            mes.extend(i)
        simple_send(mes)
        if(kwargs.get("raw")):
            simple_send(PPs[0].ordered)
    
# @receiver
@threading_run
@on_exception_response
@command("[/~～]标签简写", opts={"-pop", "-k", "-raw", "-simple", "-rnd"}, bool_opts={"-pop", "-raw", "-simple", "-rnd"})
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
        user_tag_db[uid] = udb
#         message.send_forward_message([forw])

        return
    def do_show():
        nonlocal message, uid
        s, o = get_user_entries(uid)
        if(not (s+o)):
            simple_send("没有任何标签简写")
        else:
            ill= illust_entries(s, o)
            simple_send(ill)
    search_key = " ".join(args)
    if(args):
        k = args[0]
        search_key = k
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
                PP = PromptProcessor(v, s+o)
                user_tag_db[uid] = udb
                sd, od = get_user_entries_dict(uid)
                assert (k not in sd), "删除失败？？？"
                simple_send(["已删除%s ->"%k, PP.illust()])
            
    if("k" in kwargs):
        kwargs["topk"] = int(kwargs["k"])
    do_search_tag(message, search_key, **kwargs)

def do_img2img(message, orig_image, *args, **kwargs):
    uid = message.sender.id
    t = DiffuserFastAPITicket("img2img")
    # imgtype, orig_image = message.get_sent_images()[0]

    if(not args):
        if(kwargs.get("roll") or kwargs.get("roll_from")):
            prompt = ""
        elif(uid in last_prompt):
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
    

    rolled = []
    for i in range(num):
        t = DiffuserFastAPITicket("img2img")
        if(kwargs.get("roll")):
            k, v = roll_prompt(message.sender.id, PP.raw)
            params["prompt"] = PP.raw + v
            rolled.append(k)
        elif(kwargs.get("roll_from")):
            k, v = roll_prompt_from(uid, kwargs["roll_from"])
            params["prompt"] = PP.raw+", "+v
            rolled.append(k)
        t.param(**params)
        t.upload_image(orig_image)
        tickets.append(t)
        new_pp = PromptProcessor(params["prompt"], s+o)
    rolled = "\n".join(["    "+i for i in rolled])
    if(PP.raw):
        PP_ill = PP.illust(orig_image = orig_image, roll = rolled)
    else:
        PP_ill = new_pp.illust(orig_image = orig_image, roll = rolled)
    for idx, t in enumerate(tickets):
        submit = t.submit()
        eta = submit["data"]["eta"]
        eta_this = submit["data"]["eta_this"]
        thr.add(eta_this)
        
        
        if(not idx):
            mes = [PP_ill, "预计%.1f秒"%(eta*(num-idx))]
            simple_send(mes)
        img = t.get_image()
        if(kwargs.get("hytk")):
            simple_send(hytk(img, orig_image))
        img_prompt[img] = t["prompt"]
        sent_image[message.sender.id] = img
        if(num>1):
            mes = [img]
            
            if(idx != num-1):
                mes.append("%d/%d, 预计%.1f秒"%(idx+1, num, eta*(num-1-idx)))
            else:
                mes.append("%d/%d"%(num, num))
            simple_send(mes, force_png = True)
        else:
            simple_send(img, force_png = True)
    

@receiver
@threading_run
@on_exception_response
@command("[/~～]插值画图", opts={"-from", "-to", "-a", "-n"},ls_opts={"-from","-to"})
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
    nframes = kwargs.get("n", 8)
    nframes = int(nframes)
    params = {"aspect":a, "prompt":from_raw, "prompt1":to_raw}
    params["nframes"] = nframes
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
@command("[/~～]以图画图", opts={"-guidance", "-g", "-strength", "-s", "-noise", "-n", "-debug", "-roll", "-roll_from", "-hytk"}, bool_opts = {"-roll", "-debug", "-hytk"}, ls_opts = {"-roll_from"})
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
def kwa(**kwargs):
    return kwargs
def roll_prompt_from(uid, args):
    n = int(args[-1])
    n = min(max(1, n), 200)
    

    s, o=get_user_entries(uid)
    pr = ", ".join(args[:-1])
    if(not pr):
        keys = [k for k, v in s] + [k for k, v in o]
        pr = ", ".join(keys)
    pr = PromptProcessor(pr, s+o).raw
    ops = ""
    pr_aswhole = re.findall("<[\s\S].+?>", pr)
    pr_aswhole = [i.strip("<>") for i in pr_aswhole]
    pr = re.sub("<[\s\S].+?>", "", pr)
    pr_noneg = re.sub(r"/\*[\s\S]+?\*/", "", pr).split(",")
    _pr_negs = re.findall(r"/\*[\s\S]+?\*/", pr)
    pr_negs = []
    for i in _pr_negs:
        for j in i.strip("/*").split(","):
            pr_negs.append("/*"+j+"*/")
    
    prs = pr_noneg+pr_negs+pr_aswhole


    ret = []
    for i in range(n):
        ret.append(random.choice(prs))
    ret = ", ".join(ret)
    ret = ret.replace("  ", " ")
    k = PromptProcessor(ret, s+o).friendly_raw
    return k, ret
def do_txt2img(message: CQMessage, *args, roll=False, **kwargs):
    prompt = " ".join(args)
    uid = message.sender.id
    s, o=get_user_entries(message.sender.id)
    exc = " ".join(kwargs.get("exc", list()))
    PP = PromptProcessor(prompt, s+o, excludes=exc)
    
    n = int(kwargs.get("n", 1))
    n = max(min(n, 8), 1)

    guidance = kwargs.get("g") or kwargs.get("guidance") or 12
    guidance = float(guidance)

    aspect = float(kwargs.get("a", 9/16))

    params = kwa(prompt=PP.raw, aspect=aspect, guidance=guidance)

    tickets = []
    rolled = []
    for i in range(n):
        t = DiffuserFastAPITicket("txt2img")
        if(roll):
            k, v = roll_prompt(uid, PP.raw)
            params["prompt"] = PP.raw+", "+v
            rolled.append(k)
        elif(kwargs.get("roll_from")):
            k, v = roll_prompt_from(uid, kwargs["roll_from"])
            params["prompt"] = PP.raw+", "+v
            rolled.append(k)
        t.param(**params)
        tickets.append(t)
    if(rolled):
        im = PP.illust(roll="\n".join(rolled))
    else:
        im = PP.illust()
    for idx, t in enumerate(tickets):
        submit = t.submit()
        eta = submit["data"]["eta"]*(n-idx)
        mes = [im, "%d/%d预计剩余%.2f秒"%(idx, n, eta)]
        simple_send(mes)
        im = t.get_image()
        img_prompt[im] = t["prompt"]
    simple_send(im)

def do_diff(message: CQMessage, p0, p1, *args, send=True):
    _p0, _p1 = p0, p1
    n = 15
    if(len(_p0)>n):
        _p0 = _p0[:n-3]+"..."
    elif(len(_p1)>n):
        _p1 = _p1[:n-3]+"..."
    s, o=get_user_entries(message.sender.id)
    p0 = PromptProcessor(p0, s+o).raw
    p1 = PromptProcessor(p1, s+o).raw
    p0_ls = []
    weights = {}
    for w, i in PromptProcessor._process_weights(p0, return_dict=True).items():
        ls = i.split(",")
        ls = [(w, j.strip()) for j in ls if j.strip()]
        weights.update({j: w for w, j in ls})
        p0_ls.extend(ls)
    p1_ls = []
    for w, i in PromptProcessor._process_weights(p1, return_dict=True).items():
        ls = i.split(",")
        ls = [(w, j.strip()) for j in ls if j.strip()]
        weights.update({j: w for w, j in ls})
        p1_ls.extend(ls)
    p0_ls = [j for i, j in sorted(p0_ls)]
    p1_ls = [j for i, j in sorted(p1_ls)]
    def f_similarity(x, y):
        ret = LCS(x, y).get_common_ratio(0.5)
        if(ret<0.7):
            ret = 0
        return ret
    lcs = _lcs.calc1(p0_ls, p1_ls, f_similarity=f_similarity, donot_trim=True)
    if(not send):
        return lcs.get_common_ratio(0.5)
    i, j = 0, 0
    RT = RichText(Keyword("texts"), 1000, fontSize = 26, autoSplit=False)
    
    contents = []
    BLACK = (0, 0, 0, 255)
    WHITE = (255, )*4
    D_GRAY = (30, 30, 30)
    L_GRAY = (222, 222, 222)
    while(i<len(p0_ls) or j<len(p1_ls)):
        x, y = None, None
        if(i<len(p0_ls) and (not lcs.a_matched[i])):
            x = p0_ls[i]
            if(weights[x]<0):
                x = "- " + x
            x = RT.render(texts=[x], bg=(255, 200, 220), fill=(128, 32, 64))
            i += 1
        elif(j<len(p1_ls) and (not lcs.b_matched[j])):
            y = p1_ls[j]
            if(weights[y]<0):
                y = y + " -"
            y = RT.render(texts=[y], bg=(200, 255, 220), fill=(32, 128, 64))
            j += 1
        elif(i<len(p0_ls) and j<len(p1_ls)): # matched
            x = p0_ls[i]
            y = p1_ls[j]
            mid = "=" if x==y else "?"
            t = " ".join([x, mid, y])
            if(weights[x]<0):
                t = "- %s -"%t
            if (x!=y):
                t = RT.render(texts=[t], bg=L_GRAY, fill=D_GRAY)
            else:
                t = RT.render(texts=[t], bg=WHITE, fill=BLACK)
            
            contents.append(t)
            i += 1
            j += 1
            continue
        else:
            if(i<len(p0_ls)):
                x = p0_ls[i]
                if(weights[x]<0):
                    x = "- " + x
                x = RT.render(texts=[x], bg=(255, )*3, fill=(0, 0, 0, 255))
                i += 1
            if(j<len(p1_ls)):
                y = p1_ls[j]
                if(weights[y]<0):
                    y = y + " -"
                y = RT.render(texts=[y], bg=(255, )*3, fill=(0, 0, 0, 255))
                j += 1
        if(x is None):
            x = Image.new("RGBA", y.size, (0, )*4)
        if(y is None):
            y = Image.new("RGBA", x.size, (0, )*4)
        contents.append(Row([x, y]))
    im = Column(contents, bg=WHITE).render()
    mes = [im, "%s <-> %s = %.3f%%"%(_p0, _p1, lcs.get_common_ratio(0.5)*100)]
    mes.extend(args)
    simple_send(mes)
    return lcs.get_common_ratio(0.5)




def do_merge(message: CQMessage, prompts):
    s, o = get_user_entries(message.sender.id)
    weights = {}
    cnt = {}
    for p in prompts:
        PP = PromptProcessor(p, s+o)
        praw = PP.raw
        _cnt = {}
        for w, i in PromptProcessor._process_weights(praw, return_dict=True).items():
            ls = i.split(",")
            for tag in ls:
                tag = tag.strip()
                if(not tag):continue
                _cnt[tag] = _cnt.get(tag, 0) + 1
                weights[tag] = w
        for k, v in _cnt.items():
            cnt[k] = max(cnt.get(k, 0), v)
    ret_pos = []
    ret_neg = []
    for tag, n in cnt.items():
        w = weights[tag]
        if(w>0):
            ret_pos.extend([tag]*n)
        else:
            ret_neg.extend([tag]*n)
    ret = ", ".join(ret_pos) + "/*" + ", ".join(ret_neg) + "*/"
    simple_send(ret)
def do_count(message: CQMessage, *args, **kwargs):
    st = "regex:"
    le = len(st)
    s, o = get_user_entries(message.sender.id)
    prompt = " ".join(args)
    if(prompt.startswith(st)):
        prompts = []
        pattern = re.compile(prompt[le:])
        for k, v in s+o:
            if(pattern.match(k)):
                prompts.append(k)
    elif(not prompt):
        prompts = [k for k, v in s+o]
    else:
        prompts = [prompt]
    cnt = {}
    for p in prompts:
        PP = PromptProcessor(p, s+o)
        for w, i in PromptProcessor._process_weights(PP.raw):
            for j in i.split(","):
                j = j.strip()
                if(not j):continue
                key = (w, j)
                cnt[key] = cnt.get(key, 0) + 1
    contents = []
    RT = RichText(Keyword('texts'), width = 720, fontSize=36)
    cnt = sorted([(v, k) for k, v in cnt.items()])[::-1]
    tot = 0
    presum = []
    tags = []
    le = int(kwargs.get("k", 50))
    for v, k in cnt[:le*2]:
        w, tag = k
        w = "pos" if w>0 else "neg"
        t = "(%s)%d: %s"%(w, v, tag)
        contents.append(RT.render(texts=[t], fill=(0, 0, 0, 255), bg=(255,)*4))
        tot += v
        presum.append(tot)
        tags.append(k)
    G = Grid(contents, autoAspectRatio=0.1, bg=(255,)*4)
    simple_send(G.render())
    presum = [i/tot for i in presum]
    
    pos, neg = list(), list()
    j = 0
    
    for i in range(le):
        while(j<len(presum)-1 and presum[j]<i/le):
            j += 1
        w, tag = tags[j]
        if(w<0):
            neg.append(tag)
        else:
            pos.append(tag)
    prompt = "%s/*%s*/"%(", ".join(pos), ", ".join(neg))
    uid = message.sender.id
    udb = user_tag_db.get(uid, {})
    if(kwargs.get("name")):
        key = kwargs["name"]
    else:
        key = "临时合并"+base32(prompt, length=2)
    udb[key] = prompt
    user_tag_db[uid] = udb
    m = "采样/原长度/压缩率: %d/%d/%.2f%%"%(le, tot, le/tot*100)
    do_set_tag(message, kwargs.get("name"), prompt, mes=[m])
    

def do_display_tags_list(message: CQMessage, key, show_raw=False):
    uid = message.sender.id
    s, o = get_user_entries(uid)
    val = PromptProcessor(key, s+o).raw
    search_detail = True
    st = time.time()
    entries = []
    rank = sorted(s+o+[(key, val)])
    rank = {i[0]: idx for idx, i in enumerate(rank)}
    for idx, i in enumerate(s):
        k, v = i
        score = LCS(key, k).get_common_ratio(0.5)
        rankscore = abs(rank[k] - rank[key])
        if(search_detail):
            score1 = do_diff(message, val, v, send=False)
            score = max(score, score1)
            if(idx%15==0):
                search_detail = (time.time()-st) < 5
        entries.append((-score, 1, rankscore, k, v))

    for idx, i in enumerate(o):
        k, v = i
        score = LCS(key, k).get_common_ratio(0.5)
        rankscore = abs(rank[k] - rank[key])
        if(search_detail):
            score1 = do_diff(message, val, v, send=False)
            score = max(score, score1)
            if(idx%15==0):
                search_detail = (time.time()-st) < 5
        entries.append((-score, 0, rankscore, k, v))


    entries = sorted(entries)
    mes_forward = []
    mes_image = []
    for idx, i in enumerate(entries[:10]):
        score, is_self, rankscore, k, v = i
        if(show_raw):
            vv = PromptProcessor(v, []).ordered
            mes_forward.append(message.construct_forward(k+" "+vv))
        else:
            mes_forward.append(message.construct_forward(k))
        if(idx<5):
            mes_image.append(PromptProcessor(k, s+o).illust(name=k))
    try:
        message.send_forward_message(mes_forward)
    except Exception:
        if(show_raw):
            v = entries[0][-1]
            simple_send(PromptProcessor(v, []).ordered)
        simple_send(mes_image)
def do_set_tag(message: CQMessage, key, value, mes=None, do_backup=True):
    uid = message.sender.id
    sd = user_tag_db.get(uid, {})
    if(mes is None):
        mes = []
    if(len(value)>15):
        vtrim = value[:12]+"..."
    else:
        vtrim = value

    if(not key):
        key = "临时**"
    
    if(key.count("*")):
        b32 = base32(value, key.count("*"))
        for idx, i in enumerate(b32):
            key = key.replace("*", i, 1)
    
    if(do_backup and (key in sd)):
        if(sd[key] != value):
            backkey = "回收站" + base32(sd[key], 1)
            sd[backkey] = sd[key]
            mes.append("将「%s」原值备份为「%s」"%(key, backkey))
    sd[key] = value
    user_tag_db[uid] = sd
    mes.append("将「%s」简化为「%s」"%(vtrim, key))
    simple_send(", ".join(mes))
    do_display_tags_list(message, key)
def do_pop(message: CQMessage, *args, **kwargs):
    uid = message.sender.id
    sd = user_tag_db.get(uid, {})
    print(args, kwargs)
    tags = kwargs["pop"]
    for tag in tags:
        if(tag in sd):
            sd.pop(tag)
    user_tag_db[uid] = sd
    tag_ = ", ".join(tags)
    simple_send("已删除%s"%tag_)
    do_display_tags_list(message, tag_)
def gcd(x, y):
    if(y==0):return x
    return gcd(y, x%y)
def do_most_similar(message: CQMessage, xs, ys, *args, **kwargs):
    xs = list(set(xs))
    ys = list(set(ys))
    
    memo_sim = {}
    def get_sim(x, y):
        nonlocal memo_sim
        if(x<y):
            key = (x, y)
        else:
            key = (y, x)
        if(key in memo_sim):
            return memo_sim[key]
        ret = do_diff(message, x, y, send=False)
        memo_sim[key] = ret
        return ret
    cnt = 0
    tot = len(xs)*len(ys)
    best = None
    last_report = 0
    def partial_report():
        nonlocal cnt, tot, best, last_report
        last_report = cnt
        
        s, x, y = best
        mes = "\n已搜索%d/%d=%.2f%%"%(cnt, tot, cnt*100/tot)
        do_diff(message, x, y, mes)
    st = time.time()
    lastt = st
    nxt = 10
    for x in xs:
        for y in ys:
            cnt += 1
            
            
            if(x==y):continue
            
            score = get_sim(x, y)
            if(score>0.98):continue
            score = (score, x, y)
            if(best is None):
                best = score
            elif(score > best):
                best = score
            
            if(cnt%15==0):
                if(best[0]>0.85):
                    break
                tm = time.time()
                if(tm-lastt > nxt):
                    nxt *= 2
                    lastt = tm
                    partial_report()
                else:
                    print("%s <-> %s %d/%d %.2f%%, next report %.1f"%(x, y, cnt, tot, cnt/tot*100, nxt+lastt-tm))

    if(last_report!=cnt):
        partial_report()

@receiver
@threading_run
@on_exception_response
@command("[/~～]列出标签", opts={"-p"})
def cmd_show_all_tags(message: CQMessage, *args, **kwargs):
    key = " ".join(args)
    if(not key):
        key = "组合"
    uid = message.sender.id
    s, o = get_user_entries(uid)
    val = PromptProcessor(key, s+o).raw
    search_detail = True
    st = time.time()
    entries = []
    rank = sorted(s+o+[(key, val)])
    rank = {i[0]: idx for idx, i in enumerate(rank)}
    for idx, i in enumerate(s):
        k, v = i
        score = LCS(key, k).get_common_ratio(0.5)
        rankscore = abs(rank[k] - rank[key])
        if(search_detail):
            score1 = do_diff(message, val, v, send=False)
            score = max(score, score1)
            if(idx%15==0):
                search_detail = (time.time()-st) < 5
        entries.append((-score, 1, rankscore, k, v))

    for idx, i in enumerate(o):
        k, v = i
        score = LCS(key, k).get_common_ratio(0.5)
        rankscore = abs(rank[k] - rank[key])
        if(search_detail):
            score1 = do_diff(message, val, v, send=False)
            score = max(score, score1)
            if(idx%15==0):
                search_detail = (time.time()-st) < 5
        entries.append((-score, 0, rankscore, k, v))


    entries = sorted(entries)
    mes_forward = []
    mes_image = []
    WHITE = (255, )*4
    BLACK = (0, 0, 0, 255)
    RT = RichText(Keyword("texts"), 1080, bg=WHITE, fill=BLACK, fontSize=32)
    contents = []
    p = int(kwargs.get("p", 1))
    fr = (p-1)*50
    to = p*50
    for idx, i in enumerate(entries[fr:to]):
        score, is_self, rankscore, k, v = i
        contents.append(RT.render(texts=[k]))
    G = Grid(contents, autoAspectRatio=0.2, bg=WHITE)
    title = RT.render(texts=["正在显示第%d-%d/共%d个结果"%(fr+1, to, len(entries))], fontSize=48)
    Col = Column([G, title], bg=WHITE)
    simple_send(Col.render())

@receiver
@threading_run
@on_exception_response
@command("[/~～]标签", opts={"-diff", "-most_similar", "-regex", "-xs", "-ys", "-merge", "-cnt", "-k", "-name", "-pop", "-raw"}, ls_opts={"-diff", "-xs", "-ys", "-pop"}, bool_opts={"-most_similar", "-regex", "-cnt", "-merge", "-raw"})
def cmd_prompt_process(message: CQMessage, *args, **kwargs):
    if(kwargs.get("diff")):
        prompt0 = " ".join(args)
        prompt1 = " ".join(kwargs["diff"])
        return do_diff(message, prompt0, prompt1)
    elif(True and kwargs.get("most_similar")):
        s, o = get_user_entries(message.sender.id)
        s = [i for i, j in s]
        xs, ys = s, s
        if(args):
            xs = [" ".join(args)]
        elif(kwargs.get("xs")):
            st = "regex:"
            le = len(st)
            xs = " ".join(kwargs["xs"])
            if(xs.startswith(st)):
                pattern = re.compile(xs[le:])
                xs = [i for i in s if pattern.match(i)]
            else:
                xs = [xs]
        if(kwargs.get("ys")):
            st = "regex:"
            le = len(st)
            ys = " ".join(kwargs["ys"])
            if(ys.startswith(st)):
                pattern = re.compile(ys[le:])
                ys = [i for i in s if pattern.match(i)]
            else:
                ys = [ys]
        xs = list(set(xs))
        ys = list(set(ys))
        print("DEBUG: most_sim:1179")
        do_most_similar(message, xs, ys)
    elif(False and kwargs.get("most_similar")):
        s, o = get_user_entries(message.sender.id)
        s = [i for i, j in s]
        xs, ys = s, s
        if(args):
            xs = [" ".join(args)]
        elif(kwargs.get("xs")):
            st = "regex:"
            le = len(st)
            xs = " ".join(kwargs["xs"])
            if(xs.startswith(st)):
                pattern = re.compile(xs[le:])
                xs = [i for i in s if pattern.match(i)]
            else:
                xs = [xs]
        if(kwargs.get("ys")):
            st = "regex:"
            le = len(st)
            ys = " ".join(kwargs["ys"])
            if(ys.startswith(st)):
                pattern = re.compile(ys[le:])
                ys = [i for i in s if pattern.match(i)]
            else:
                ys = [ys]
        xs = list(set(xs))
        ys = list(set(ys))
        searched = set()
        best = None
        num = 0
        t = time.time()
        idx = 0
        ps = [3, 5, 7, 11, 13, 131, 739, 5623, 131071, 1742527]
        tot = len(xs)*len(ys)
        p = random.choice(ps)
        while(gcd(tot, p)!=1):
            p = random.choice(ps)
        idx = random.randrange(tot)
        _idx = idx
        while(True):
            num = len(searched)
            if(num == len(xs)*len(ys)):
                break
            i = xs[idx%len(xs)]
            j = ys[(idx//len(xs)) % len(ys)]
            idx += p
            if((i, j) in searched):
                continue
            searched.add((i, j))
            if(i == j):continue
            score = do_diff(message, i, j, send=False)
            if(score == 1):continue
            score = (score, i, j)
            if((best is None) or (score>best)):
                best = score
            if(num % 50):
                if(time.time()-t > 25):
                    break
            if(best[0] > 0.8):
                break
        print(idx-_idx, num)
        s, i, j = best
        tot = len(xs)*len(ys)
        do_diff(message, i, j, "\n已搜索%d/%d=%.2f%%"%(num, tot, num*100/tot))
    # elif(kwargs.get("merge")):
        # return do_merge(message, kwargs["merge"])
    elif(kwargs.get("cnt") or kwargs.get("merge")):
        return do_count(message, *args, **kwargs)
    elif(kwargs.get('pop')):
        return do_pop(message, *args, **kwargs)
    elif("=" in args):
        args0 = []
        for idx, i in enumerate(args):
            if(i == "="):
                args1 = args[idx+1:]
                break
            else:
                args0.append(i)
        key = " ".join(args0)
        s, o = get_user_entries(message.sender.id)
        sd, od = get_user_entries_dict(message.sender.id)
        value = PromptProcessor(" ".join(args1), s+o).raw
        do_set_tag(message, key, value)
    else:
        im = message.get_reply_image()
        if(im):
            val = img_prompt[im]
            key = " ".join(args)
            do_set_tag(message, key, val)
        elif(args):
            do_display_tags_list(message, " ".join(args), show_raw = kwargs.get("raw"))
        else:
            simple_send("不知道要干什么")


@receiver
@threading_run
@on_exception_response
@command("[/~～]画图", opts={'-guidance', "-g", "-aspect", "-a", "-n", "-roll", "-roll_from", "-exc"}, bool_opts={"-roll"}, ls_opts = {"-roll_from", "-exc"})
def cmd_aidraw_v3(message: CQMessage, *args, **kwargs):
    return do_txt2img(message, *args, **kwargs)
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
        simple_send(mes, force_png = True)
@receiver
@threading_run
@on_exception_response
@command("[/~～]重发", opts={"-debug"}, bool_opts={"-debug"})
def cmd_resent_v3(message:CQMessage, *args, **kwargs):
    print("meow")
    if(kwargs.get("debug")):
        sent_image[message.sender.id].save(path.join(mainpth, "test.png"))
        print(path.join(mainpth, "test.png"))
        simple_send("saved to"+path.join(mainpth, "test.png"))
    simple_send(sent_image[message.sender.id])




def do_process(message, orig_image: Image.Image, expand=False, **kwargs):
    ret = orig_image.copy()
    orig_w, orig_h = ret.size
    for k, v in kwargs.items():
        try:
            v = float(v)
            kwargs[k] = v
        except Exception:
            pass
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
    arr = np.array(ret)
    for idx, ch in enumerate("rgb"):
        key = "gamma_"+ch
        if(key in kwargs):
            v = kwargs[key]
            charr = arr[:, :, idx]/255
            charr = (charr**(1/v))*255
            arr[:, :, idx] = charr.astype(np.uint8)
    ret = Image.fromarray(arr)
    sent_image[message.sender.id] = ret
    simple_send(ret)
    return ret
process_options = {"opts":{"-expand", "-left", "-right", "-rate", "-gamma_r", "-gamma_g", "-gamma_b", "-debug"}, "bool_opts":{"-expand", "-debug"}}
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
    if(kwargs.get("debug")):
        img.save(path.join(mainpth, "test.png"))
        simple_send("saved"+path.join(mainpth, "test.png"))
    w, h = img.size
    x, y =random.randrange(w), random.randrange(h)
    img.putpixel((x, y), (114, 51, 4))
    # simple_send(img)
    do_process(message, img.convert("RGB"), **kwargs)
@receiver
@threading_run
@on_exception_response
@command("[/~～]超分", opts={"-downscale"})
def cmd_upscale(message:CQMessage, *args, **kwargs):
    if(message.get_reply_image()):
        orig_image = message.get_reply_image()
    else:
        imgtype, orig_image = message.get_sent_images()[0]
    def normalize_resolution(w, h, resolution=512*512, mo=64):
        if(resolution is not None):
            rate = (resolution/w/h)**0.5
        else:
            rate = 1
        w = int(w*rate)
        h = int(h*rate)
        w -= w%mo
        h -= h%mo
        return w, h
    if(kwargs.get("downscale")):
        res = float(kwargs.get("downscale"))
        orig_image = orig_image.resize(normalize_resolution(*orig_image.size, resolution=res, mo=1), Image.Resampling.BICUBIC)

    url = HOST+"/upscale"
    bio = img2bio(orig_image)
    r = requests.post(url, files={"data":bio})
    bio = BytesIO()
    bio.write(r.content)
    bio.seek(0)
    im = Image.open(bio)
    simple_send([im, "%dx%d"%im.size])

@receiver
@threading_run
@on_exception_response
@command("[/~～]今日老婆", opts = {})
def cmd_today_aidraw(message:CQMessage, *args, **kwargs):
    user = message.sender.id
    today = now().strftime("%Y%m%d")
    key = "%s-%s" % (user, today)
    if(key in user_today_db):
        prompt = user_today_db
    else:
        prompt = roll_prompt_from(user, [70])[1]
    user_today_db[key] = prompt
    s, o=get_user_entries(message.sender.id)
    PP = PromptProcessor(prompt, s+o)
    t = DiffuserFastAPITicket("txt2img")
    t.param(prompt=PP.ordered)
    simple_send(t.get_image())

@receiver
@threading_run
@on_exception_response
@command("[/~～]inpaint", opts={"-p", "-smooth"}, ls_opts={"-p"})
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

    if(message.get_reply_image()):
        orig_image = message.get_reply_image()
    else:
        imgtype, orig_image = message.get_sent_images()[0]

    mask = Image.fromarray(mask).resize(orig_image.size)
    w, h=mask.size
    rad = (w*h)**0.5
    if(kwargs.get("smooth")):
        smooth_mask = float(kwargs.get("smooth"))
    else:
        smooth_mask = 1
    mask=mask.filter(ImageFilter.GaussianBlur(rad/40*smooth_mask))
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

