from ..utils.saucenao import get_sauce
from ..backend.receiver_decos import on_exception_response, command, threading_run
from ..backend import receiver
from ..backend.cqhttp.message import CQMessage
from ..backend import requests
from pil_functional_layout.widgets import Text, RichText, Column, Row
from .plg_admin import link_send_content
from ..utils.image import sizefit
from ..utils.image.colors import Color


def renderitem(sauce, size=512):
    head = sauce["header"]
    data = sauce["data"]
    thumb = head["thumbnail"]

    
    f = lambda text, **kwargs: text
    highlight_fill = Color.from_hsl(0, 1, 0.8).get_rgba()
    highlight_bg = Color.from_hsl(0, 1, 0.2).get_rgba()
    T = Text(f, fill=highlight_fill, bg=highlight_bg, fontSize=size//20)  # nopep8
    TR = RichText(f, fill=(0, 0, 0, 255), fontSize=size//15, width=size, autoSplit=False, dontSplit=False)  # nopep8

    try:
        img = requests.get_image(thumb)[1]
    except Exception:
        img = T.render(text="无法获取图片")
    img = sizefit.area(img, 5000)
    img = sizefit.fix_width(img, size)


    lines = [img]
    lines.append(TR.render(text="相似度: %s%%"%head["similarity"]))
    for k, v in data.items():
        if(k == "ext_urls"):
            for url in v:
                lnk = link_send_content(url)
                text = ['发送"', T.render(text=lnk), '"获取链接']
                lines.append(TR.render(text=text))
        elif(k == "creator"):
            if(isinstance(v, list)):
                text = "作者: "+", ".join(v)
            else:
                text = "作者: "+str(v)
            lines.append(TR.render(text=text))
        elif(k in ["eng_name", "jp_name"]):
            text = "标题: "+v
            lines.append(TR.render(text=text))
            lnk = link_send_content(v)
            text = ['发送"', T.render(text=lnk), '"获取标题']
            lines.append(TR.render(text=text))
        else:
            text = "%s: %s" % (k, v)
            lines.append(TR.render(text=text))
    ret = Column(lines, alignX=0, bg=(255,)*4)
    return ret.render()


@receiver
@threading_run
@on_exception_response
@command("(/sauce)|(/识图)", opts={})
def cmd_sauce(message: CQMessage, *args, **kwargs):
    imgtype, img = message.get_sent_images()[0]
    sauce = get_sauce(img)

    sauces = sauce["results"]
    sauces = sorted(sauces, key=lambda x: -float(x["header"]["similarity"]))
    mes = []
    for i in sauces[:5]:
        if(float(i["header"]["similarity"])<70):
            break
        mes.append(renderitem(i))
    if(mes):
        mes = Column(mes).render()
        message.response_sync(mes)
    else:
        message.response_sync("居然..没有搜索结果！！")
