
from typing import Dict, Tuple, Union, Literal
from PIL import Image

from io import BytesIO
from urllib.parse import urlencode
from pil_functional_layout.widgets import RichText, Grid, Column
from pil_functional_layout import Keyword
from ..backend.cqhttp.message import CQMessage
from ..backend.receiver_decos import *
from ..backend import receiver
from ..backend.configure import bot_config
from ..utils.candy import simple_send
from ..utils.make_gif import make_gif
from ..utils.lvldb import TypedLevelDB
import requests
from ..backend.paths import mainpth
from os import path
from ..utils.algorithms.lcs import lcs as LCS
user_tag_db = TypedLevelDB.open(
    path.join(mainpth, "saves", "plg_diffusion", "user_tags"))

if("DIFFUSION_HOST" in bot_config):
    HOST = bot_config.get("DIFFUSION_HOST").strip("/")
else:
    HOST = "http://localhost:8000"
print("using", HOST, "as diffusion backend")


def compose_url(*args, **kwargs):
    paths = "/"+"/".join(args)
    query = "?"+urlencode(kwargs) if kwargs else ""
    host = HOST
    return "".join([host, paths, query])


def check_backend_ready():
    url = compose_url("ping")
    try:
        r = requests.get(url)
    except Exception:
        return False, "EXC"
    if(r.text == "Pong!"):
        return True, None
    if(r.status_code == 404):
        return False, "HOST/ping returned 404, maybe not correct backend version"
    return False, r.status_code


def get_kemomimi_translations():
    ret = dict()

    def add(*args):
        nonlocal ret
        for arg in args[:-1]:
            ret[arg] = args[-1]
    add("猫耳", "cat ears")
    add("犬耳", "dog ears")
    return ret


def get_misc_translations():
    ret = dict()

    def add(*args):
        nonlocal ret
        for arg in args[:-1]:
            ret[arg] = args[-1]

    add("大师之作", "masterpiece")
    add("高画质", "masterpiece, best quality, high quality, cinematic lighting, highres, absurdres")
    add("壁纸", "wallpaper")
    add("高细节", "detailed")
    add("高质量", "high quality")
    add("景深", "depth of field")
    add("全身像", "full body")
    add("抱枕", "dakimakura")
    add("精致美丽", "美丽精致", "extremely delicate and beautiful")
    add("原创", "original")
    add("高分辨率", "highres")
    add("赛博朋克", "cyberpunk, city lights, neon lights")
    add("一位男生", "1boy")
    add("一位女生", "1girl")
    add("船锚符号", "anchor symbol")
    add("福瑞", "furry")
    add("虎牙", "fang")
    add("伸脚", "presenting foot")
    add("脚趾", "toes")
    add("蒸汽", "雾气", "steam")
    add("看看笔", "presenting pussy")
    add("城市景观", "cityscape")
    add("花田", "flower fields")
    add("草原", "grasslands")
    return ret


def get_pose_translations():
    ret = dict()

    def add(*args):
        nonlocal ret
        for arg in args[:-1]:
            ret[arg] = args[-1]
    add("叉腰", "hand on own hips")
    add("撑脸", "hand on own cheek")
    add("捂裤裆", "hand on own crotch")
    add("看你", "looking at viewer")
    add("看旁边", "looking to the side")
    add("看别人", "looking at another")
    add("回头看", "looking back")
    add("抚胸", "hand on own chest")
    add("摸胸", "hand on breast")
    add("持武器", "holding weapon")
    add("持食物", "holding food")
    add("从下面", "from below")
    add("弓身", "bent over")
    add("顽皮表情", "捉弄表情", "naughty face")
    add("笑", "smiling")
    add("厌恶", "disgust")
    add("张嘴", "mouth open")
    add("闭眼", "eyes closed")
    add("从腿到头视角", "cowboy shot")
    add("摸摸奈子", "breast grab")
    add("调整泳装", "adjusting swimsuit")
    add("调整衣装", "adjusting clothes")
    add("调整发型", "adjusting hair")
    add("调整帽子", "扶帽子", "adjusting clothes")
    add("调整胖次", "在穿胖次", "adjusting panties")
    add("调整袜子", "在穿袜子", "adjusting legwear")
    add("扶眼睛", "adjusting eyewear")
    return ret


def get_body_translations():
    ret = dict()

    def add(*args):
        nonlocal ret
        for arg in args[:-1]:
            ret[arg] = args[-1]
    add("大腿", "thighs")
    add("乳头", "nipples")
    add("长发", "long hair")
    add("义肢", "prosthesis")
    add("露肩", "bare shoulders")
    add("露背", "bare back")
    add("裸体", "nude, naked")
    add("肚子", "stomach, navel")
    add("露出手臂", "bare arms")
    add("露肩", "bare shoulders")
    add("搭肩", "hand on own shoulder")
    add("搭别人的肩", "hand on another's shoulder")
    add("露腰", "bare hips")
    add("裸足", "bare foot")
    add("裸腿", "bare legs")
    add("腹股沟", "大腿根", "大腿根儿", "groin")
    add("秀发遮胸", "hair over breasts")
    return ret


def get_clothes_translations():
    ret = dict()

    def add(*args):
        nonlocal ret
        for arg in args[:-1]:
            ret[arg] = args[-1]
    add("卫衣", "hoodie")
    add("泳装", "swimsuit")
    add("比基尼", "bikini")
    add("蕾丝文胸", "lace bra")
    add("蕾丝内裤", "蕾丝胖次", "lace panties")
    add("蕾丝内衣", "lace panties, lace bra")
    add("只穿内衣", "underwear only")
    add("花嫁", "wedding dress, lace-trimed headdress")
    add("围裙", "apron")
    add("裸体围裙", "naked apron")
    add("白色围裙", "white apron")
    add("女仆头饰", "maid headdress")
    add("女仆装", "maid apron, maid headdress")
    add("猫耳", "cat ears")
    add("黑丝袜", "黑丝", "black legwear")
    add("白丝袜", "白丝", "white legwear")
    add("黑过膝袜", "black stockings")
    add("过膝袜", "stockings")
    add("裤袜", "pantyhose")
    add("旗袍", "china dress, chinese clothes")
    add("机甲娘", "mecha musume")
    add("机娘", "mecha musume, prosthesis, mechanical weapon, holding weapon")
    add("机械手", "artificial arms, mechanical arms, prosthetic arms")
    add("短裙", "skirt")
    add("分体袖子", "detached sleeves")
    add("百褶裙", "pleated skirt")
    add("校服", "school uniform")
    add("水手领", "sailor collar")
    add("黑色水手领", "black sailor collar")
    add("水手服", "serafuku, sailor collar")
    add("JK制服", "high school uniform")
    add("衬衫", "shirt")
    add("红色披肩", "red shawl")
    add("披肩", "shawl")
    add("领巾", "neckerchief")
    add("红领巾", "red neckerchief")
    add("敞开衣服", "open clothes")
    add("敞开外套", "open coat")
    add("敞开衬衫", "open shirts")
    add("腿环", "thigh strap")
    add("臂环", "arm strp")
    add("护腕", "wrist guards")
    add("勒肉", "skindentation")
    add("晒黑", "tan")
    add("晒痕", "tanlines")
    add("连体泳装晒痕", "bikini tan")
    add("比基尼晒痕", "one-piece tan")
    add("拉开比基尼", "bikini pull")
    add("脱胖次", "panty pull")
    add("脱裤袜", "pantyhose pull")
    add("朱唇微启", "parted lips")
    add("膝枕", "lap pillow")
    add("洛丽塔", "lolita fashion")
    return ret


def get_breasts_translations():
    ret = dict()

    def add(*args):
        nonlocal ret
        for arg in args[:-1]:
            ret[arg] = args[-1]
    add("平胸", "flat chest")
    add("微乳", "贫乳", "small breasts")
    add("美乳", "medium breasts")
    add("巨乳", "big breasts")
    add("极巨乳", "极乳", "huge breasts")
    add("乳沟", "cleavage")
    add("锁骨", "collarone")
    return ret


def get_items_translations():
    ret = dict()

    def add(*args):
        nonlocal ret
        for arg in args[:-1]:
            ret[arg] = args[-1]

    add("塑料杯", "disposable cup")
    add("炸鸡", "fried chicken")
    add("耳机", "headphone")
    add("车辆", "vehicle")
    add("霓虹灯", "neon lights")
    return ret


def get_chara_translations():
    ret = dict()

    def add(*args):
        nonlocal ret
        for arg in args[:-1]:
            ret[arg] = args[-1]
    add("阿夸", "pink hair, pink hair, aqua hair, multicolored hair, two-tone hair, pink eyes, streaked hair, ahoge, braid, hololive, minato aqua")
    add("狼王", "crown, throne, jewelry, gray hair, shawl, wolf ears")
    add("むら", "silver hair, blue eyes, short twintails, ahoge")
    add("C54", "C.54", "golden eyes, black hair, short twintail, red shawl, two-tone hair, red hair, multicolored hair, fang, red streaked hair")
    add("佐久间魔鸳", "golden hair, yellow hair, small breasts, red eyes, demon horn, succubus, succubus tail")
    add("埃尔克拉夫特", "blue eyes, flat chest, light bonde hair, brown hair, gradient hair, blunt bangs, bob cut")

    return ret


def get_hairstyle_translations():
    ret = dict()

    def add(*args):
        nonlocal ret
        for arg in args[:-1]:
            ret[arg] = args[-1]
    add("双马尾", "twintails")
    add("短双马尾", "short twintail")
    add("马尾", "ponytail")
    add("侧马尾", "side ponytail")
    add("长发", "long hair")
    add("超长发", "absurdly long hair")
    add("编织头发", "麻花辫", "braid")
    add("刘海", "bangs")
    add("发饰", "hair ornament")
    add("叉叉发饰", "x hair ornament, cross hair ornament")
    add("头饰", "headdress")
    add("头戴装备", "head gear")
    add("小熊发饰", "bear hair ornament")
    add("猫咪发饰", "cat hair ornament")
    add("鱼发饰", "fish hair ornament")
    add("发夹", "hairclip")
    add("发髻", "hair bun")
    add("渐变头发", "gradient hair")
    add("彩色头发", "multicolored hair")
    add("挑染头发", "streaked hair")
    return ret


def get_nsfw_translations():
    ret = dict()

    def add(*args):
        nonlocal ret
        for arg in args[:-1]:
            ret[arg] = args[-1]
    ret["裸体"] = "naked, nude"
    ret["自慰"] = "female masturbation"
    add("妹汁", "pussy juice")
    add("妹汁成丝", "妹汁拉丝", "pussy juice trail")
    add("妹汁成湖", "妹汁成河", "满地妹汁", "pussy juice puddle")
    add("手指自慰", "fingering")
    add("口交", "blowjob, cum in mouth, fellatio")
    add("射嘴里", "cum in mouth")
    add("勃起", "erection")
    add("高潮", "orgasm")
    add("女性高潮", "female orgasm")
    add("女性潮吹", "female ejaculation")
    add("精液", "semen")
    add("内射", "cum in pussy")
    add("射屁股里", "cum in ass")
    add("射屁股上", "cum on ass")
    add("射身上", "cum on body")
    add("射脸上", "颜射", "cum on face, facial")
    add("吐口水", "spitroast")
    add("吐舌头", "tongue out")
    add("口水", "drooling")
    add("骆驼趾", "cameltoe")
    add("淫纹", "pubic tattoo, stomach tattoo")
    add("写正字", "tally")
    add("色色装置", "sex machine")
    add("异物插入", "object insertion")
    add("强制高潮", "forced orgasm")
    return ret


def get_prompt_translations(message: Union[Literal[None], CQMessage] = None):
    ret = dict()

    def add(*args):
        nonlocal ret
        for arg in args[:-1]:
            ret[arg] = args[-1]

    add("千千", "pink hair, heterochromia, red eyes, blue eyes, cat ears")
    add("黄金黑箱", "black hair, yellow eyes, golden eyes, elf, ahoge")
    add("黑箱", "black hair, yellow eyes, elf, ahoge")
    colors = "pink 粉,blue 蓝,red 红,yellow 黄,golden 金,brown 棕,light brown 浅棕,purple 紫,gray 灰,black 黑".split(
        ",")
    for i in colors:
        spl = i.split(" ")
        eng = " ".join(spl[:-1])
        chn = spl[-1]
        add(chn+"发", eng+" hair")
        add(chn+"眼", eng+" eyes")
    add("萝莉", "小萝莉", "loli")
    add("小女孩", "little girl")
    add("御姐", "mature female")
    ret.update(get_clothes_translations())
    ret.update(get_pose_translations())
    ret.update(get_body_translations())
    ret.update(get_misc_translations())
    ret.update(get_breasts_translations())
    ret.update(get_items_translations())
    ret.update(get_hairstyle_translations())
    ret.update(get_kemomimi_translations())
    ret.update(get_nsfw_translations())
    ret.update(get_chara_translations())
    if(message):
        custom = user_tag_db.get(message.sender.id, {})
        ret.update(custom)
    return ret


def process_prompt(prompt, message: Union[Literal[None], CQMessage] = None):

    def process_symbols(prompt):
        prompt = prompt.replace("，", ",")
        prompt = prompt.replace(",", ", ")
        prompt = prompt.replace("  ", " ")
        prompt = prompt.replace(", ,", ", ")
        return prompt.strip(", ")

    def process_translation(prompt, trans):
        replaced = {}
        remain = prompt
        for k in sorted(trans, key=lambda x: -len(x)):
            v = trans[k]
            if(k in prompt):
                prompt = prompt.replace(k, v+", ")
                remain = remain.replace(k, "")
                replaced[k] = v
            if(v in remain):
                remain = remain.replace(v, "")
                replaced[k] = v
        prompt = process_symbols(prompt)
        remain = process_symbols(remain.replace("{", "").replace("}", ""))
        return prompt, replaced, remain
    prompt = process_symbols(prompt)

    prompt, replaced, remain = process_translation(
        prompt, get_prompt_translations(message=message))
    return prompt, replaced, remain


def illust_prompt(prompt, replaced, remain, include_translation=True, message: Union[Literal[None], CQMessage] = None):
    tmp = []
    RT = RichText(Keyword("texts"), width=620, fontSize=None,
                  autoSplit=False, imageLimit=(600, 400))
    for i in replaced:
        chn = RT.render(texts=["<"+i+">"], fontSize=36,
                        fill=(0, 30, 60), bg=(180, 223, 223))
        if(include_translation):
            eng = RT.render(texts=["\n("+replaced[i]+")"],
                            fill=(180, 120, 150), bg=(255,)*3, fontSize=18)
            tmp.append(Column([chn, eng], borderWidth=3).render())
        else:
            tmp.append(chn)
    if(remain):
        texts = ["没有翻译："+remain]
        all_translations = get_prompt_translations(message=message)
        mx = None
        for k, v in all_translations.items():
            score = LCS(k, remain).common_ratio, k
            if((mx is None) or (score > mx)):
                mx = score
            score = LCS(v, remain).common_ratio, k
            if((mx is None) or (score > mx)):
                mx = score
        s, k = mx
        v = all_translations[k]
        texts.append("你想要的可能是: %s(%s), 或者尚未收录"%(k, v))
        tmp.append(RT.render(texts=texts, fontSize=30, bg=(255, 180, 200), fill=(128, 0, 64)))
    if(prompt):
        tmp.append(RT.render(texts=[prompt], fontSize=30))
    tmp.sort(key=lambda x: x.size[1])
    gr = Grid(tmp, bg=(255,)*3, borderWidth=5, autoAspectRatio=0.05)
    return gr.render()


def invoke_api(message, *args, files=None, **kwargs):
    ready, reason = check_backend_ready()
    if(not ready):
        if(message):
            simple_send("后端未就绪%s" % reason)
        return False, reason
    url = compose_url(*args, **kwargs)
    try:
        if(files):
            r = requests.post(url, files=files)
        else:
            r = requests.get(url)
    except Exception as e:
        if(message):
            simple_send("与后端连接失败")
        return False, "Connection Fail"
    if(r.status_code != 200):
        if(message):
            simple_send("后端返回状态码%s" % r.status_code)
        return False, "HTTP%s" % r.status_code
    return True, r


def img2bio(img: Image.Image):
    bio = BytesIO()
    # bio.write()
    img.convert("RGB").save(bio, "JPEG")
    bio.seek(0)
    return bio


def bytes2img(bytes):
    bio = BytesIO()
    bio.write(bytes)
    bio.seek(0)
    im = Image.open(bio)
    return im


def get_diffusion_eta(n=1):
    success, response = invoke_api(None, "performance")
    if(success):
        data = response.json()["data"]

        if("Diffusion-Inference" in data.get("runtime", {})):
            _data = data["runtime"]["Diffusion-Inference"]
            if("ETA" in _data):
                if("Recent.Speed" in _data):
                    return success, _data["ETA"]+n/_data["Recent.Speed"]
                else:
                    return False, "no speed info"
            else:
                return False, "no ETA info"
        else:
            return False, "no diffusion info"
    else:
        return False, "api fail"


def run_report_performance(message: CQMessage):
    success, response = invoke_api(message, "performance")

    if(success):
        data = response.json()["data"]
        mes = []
        if("Diffusion-Inference" in data.get("runtime", {})):
            _data = data["runtime"]["Diffusion-Inference"]
            mes.append("图片生成:\n")
            if("ETA" in _data):
                mes.append("    预计排队时间：%.3f秒\n" % _data["ETA"])
            if("Recent.Speed" in _data):
                spd = _data["Recent.Speed"]
                _ = "    速度："
                if(spd > 1):
                    _ += "%.1f图片/秒\n" % spd
                else:
                    _ += "%.1f秒/图片\n" % (1/spd)
                mes.append(_)
            if("Avg.Load" in _data):
                load = _data["Avg.Load"]
                mes.append("    平均负载：%.2f%%\n" % (load*100))
            if("currently_running" in _data):
                load = _data["currently_running"]
                mes.append("    当前负载：%.2f%%\n" % (load*100))
        mes[-1] = mes[-1].strip("\n")
        simple_send(mes)


sent_image = dict()


def _on_reload() -> Tuple[Tuple, Dict]:
    return tuple(), {"sent_image": sent_image}


def _on_after_reload(*args, **kwargs):
    global sent_image
    if("sent_image" in kwargs):
        sent_image = kwargs["sent_image"]


def show_partition_for_inpaint(orig_image, selected=None):
    pass


def run_inpaint(message, prompt, guidance=None, strength=None, parts=None):
    if(strength is None):
        strength = 0.65
    else:
        strength = float(strength)
        strength = max(0, min(1, strength))
    if(guidance is None):
        guidance = 8/strength
    else:
        guidance = float(guidance)
    imgtype, orig_image = message.get_sent_images()[0]
    if(parts is None):
        simple_send("请指定分区")
    else:
        pass


def run_img2img(message: CQMessage, prompt, guidance=None, strength=None, orig_image=None):
    if(strength is None):
        strength = 0.65
    else:
        strength = float(strength)
        strength = max(0, min(1, strength))
    if(guidance is None):
        guidance = 8/strength
    else:
        guidance = float(guidance)
    if(orig_image is None):
        imgtype, orig_image = message.get_sent_images()[0]

    bio = img2bio(orig_image)
    files = {"data": bio}
    prompt, replaced, remain = process_prompt(prompt, message=message)

    success, eta = get_diffusion_eta(n=strength)
    if(success):
        pr_img = illust_prompt(prompt, replaced, remain)
        simple_send(["正在生成...预计需要%.2f秒" % eta, pr_img])
    else:
        print("Cannot get eta because", eta)

    success, response = invoke_api(
        message, "img2img", prompt, files=files, guidance=guidance, strength=strength)
    if(success):
        img = bytes2img(response.content)
        sent_image[message.sender.id] = img
        simple_send([img,
                     illust_prompt(prompt, replaced, remain), message.construct_reply()])
    else:
        simple_send("失败%s" % response)


def run_interp(message: CQMessage, prompt0, prompt1, aspect_ratio=None, guidance=None):
    if(aspect_ratio is None):
        aspect_ratio = 3/4
    else:
        aspect_ratio = float(aspect_ratio)
    if(guidance is None):
        guidance = 15
    else:
        guidance = float(guidance)
    prompt0, replaced0, remain0 = process_prompt(prompt0, message=message)
    prompt1, replaced1, remain1 = process_prompt(prompt1, message=message)
    multiplier = 2.5
    success, eta = get_diffusion_eta(n=multiplier)
    if(success):
        pr_img0 = illust_prompt(prompt0, replaced0, remain0)
        pr_img1 = illust_prompt(prompt1, replaced1, remain1)
        simple_send(["正在生成...预计需要%.2f秒" % eta, pr_img0, pr_img1])
    else:
        print("Cannot get eta because", eta)
    success, response = invoke_api(
        message, "interp_prompts", prompt0, prompt1=prompt1, guidance=guidance, aspect=aspect_ratio)
    if(success):
        data = response.json()["data"]
        imgs = []
        for_messages = []
        for id in data["ids"]:
            suc, r = invoke_api(message, "storaged_imgs", id)
            if(suc):
                im = bytes2img(r.content)
                imgs.append(im)
                for_messages.append(message.construct_forward(im))
            else:
                simple_send("获取图片失败")
                return
        n = len(imgs)
        gif = make_gif(imgs, fps=max(1, int(n/2.5)), frame_area_sum=1e7)
        message.send_forward_message(for_messages)
        simple_send(gif)
    else:
        simple_send("失败%s" % response)


def run_txt2img(message: CQMessage, prompt, aspect_ratio=None, guidance=None, batch=None, ex_neg=None):
    if(aspect_ratio is None):
        aspect_ratio = 9/16
    else:
        aspect_ratio = float(aspect_ratio)
    if(guidance is None):
        guidance = 12
    if(batch is None):
        batch = 1
    else:
        batch = int(batch)
    prompt, replaced, remain = process_prompt(prompt, message=message)
    success, eta = get_diffusion_eta(n=batch)
    if(success):
        pr_img = illust_prompt(prompt, replaced, remain)
        simple_send(["正在生成...预计需要%.2f秒" % eta, pr_img])
    else:
        print("Cannot get eta because", eta)
    imgs = []
    if(batch > 4):
        batch = 4
    imgs = []
    response = "batch<=0"
    for i in range(batch):
        success, response = invoke_api(
            message, "txt2img", prompt, aspect=aspect_ratio, guidance=guidance, ex_neg=ex_neg)
        if(success):
            im = bytes2img(response.content)
            imgs.append(im)
            sent_image[message.sender.id] = im
    if(imgs):
        simple_send(imgs+[message.construct_reply()])
    else:
        simple_send("生成失败：%s" % response)


@receiver
@threading_run
@on_exception_response
@command("/可用标签", opts={})
def cmd_availale_tags(message, *args, **kwargs):
    ill = illust_prompt("", get_prompt_translations(),
                        "", include_translation=True)
    simple_send(ill)


@receiver
@threading_run
@on_exception_response
@command("/AI负载", opts={})
def cmd_diffusion_load(message: CQMessage, *args, **kwargs):

    run_report_performance(message)


@receiver
@threading_run
@on_exception_response
@command("/以图画图", opts={"-guidance", "-strength", "-s", "-g"})
def cmd_img2img(message: CQMessage, *args, **kwargs):
    strength = kwargs.get("s") or kwargs.get("strength")
    guidance = kwargs.get("g") or kwargs.get("guidance")
    return run_img2img(message, " ".join(args), strength=strength, guidance=guidance)


@receiver
@threading_run
@on_exception_response
@command("/完善画图", opts={"-guidance", "-strength", "-s", "-g"})
def cmd_enhance_img(message: CQMessage, *args, **kwargs):
    if(message.sender.id in sent_image):
        orig_image = sent_image[message.sender.id]
    else:
        simple_send("还未画过图哟")
        return
    strength = kwargs.get("s") or kwargs.get("strength")
    guidance = kwargs.get("g") or kwargs.get("guidance")

    return run_img2img(message, " ".join(args), strength=strength, guidance=guidance, orig_image=orig_image)


@receiver
@threading_run
@on_exception_response
@command("/插值画图", opts={"-a", "-g"})
def cmd_interp(message: CQMessage, *args, **kwargs):
    current = "COMMON"
    pro0 = []
    pro1 = []
    for i in args:

        if(i.startswith("-c")):
            current = "COMMON"
        elif(i.startswith("A:") or i.startswith("甲：")):
            current = "0"
            i = i[2:]
        elif(i.startswith("B:") or i.startswith("乙：")):
            current = "1"
            i = i[2:]
        if(i):
            if(current == "COMMON"):
                pro0.append(i)
                pro1.append(i)
            elif(current == "0"):
                pro0.append(i)
            else:
                pro1.append(i)
    prompt0 = " ".join(pro0)
    prompt1 = " ".join(pro1)
    a = kwargs.get("a")
    g = kwargs.get("g")
    if(prompt0 == prompt1):
        simple_send("文本相同")
    else:
        run_interp(message, prompt0, prompt1, aspect_ratio=a, guidance=g)


@receiver
@threading_run
@on_exception_response
@command("/画图debug ", opts={"-neg", "-g", "-a"}, ls_opts={"-neg"})
def cmd_txt2img1(message: CQMessage, *args, **kwargs):
    prompts = " ".join(args)
    simple_send("kwargs:" + str(kwargs))
    neg_prompts = " ".join(kwargs.get("neg", []))
    aspect_ratio = kwargs.get("a")
    guidance = kwargs.get("guidance") or kwargs.get("g")
    # batch = kwargs.get("n") or kwargs.get("batch")
    return run_txt2img(message, prompts, aspect_ratio=aspect_ratio, guidance=guidance, ex_neg=neg_prompts)


@receiver
@threading_run
@on_exception_response
@command("/自定义标签|/标签简写|/标签简化|/简化标签", opts={})
def cmd_custom_tags(message: CQMessage, *args, **kwargs):
    uid = message.sender.id
    custom_translations = user_tag_db.get(uid, {})
    if(args):
        key = args[0]
        value = " ".join(args[1:])
        custom_translations[key] = value
    user_tag_db[uid] = custom_translations
    im = illust_prompt("", custom_translations, "")
    simple_send(im)


@receiver
@threading_run
@on_exception_response
@command("/画图 ", opts={"-aspect_ratio", "-a", "-aspect", "-guidance", "-g", "-宽", "-n", "-batch", "-neg"}, ls_opts={"-neg"})
def cmd_txt2img(message: CQMessage, *args, **kwargs):
    aspect_ratio = kwargs.get("a") or kwargs.get(
        "aspect") or kwargs.get("aspect_ratio") or kwargs.get("宽")
    guidance = kwargs.get("guidance") or kwargs.get("g")
    batch = kwargs.get("n") or kwargs.get("batch")
    ex_neg = kwargs.get("neg")
    if(ex_neg is not None):
        ex_neg = ", ".join(ex_neg)
    return run_txt2img(message, " ".join(args), aspect_ratio=aspect_ratio, guidance=guidance, batch=batch, ex_neg=ex_neg)
