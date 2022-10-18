from pil_functional_layout.widgets import Column, RichText, CompositeBG
from ..image import sizefit, process
from .fetch import Torrent
import re
from .requests import get_image
from PIL import Image
from datetime import timedelta
from os import path
from .paths import illust_cache_pth, ensure_directory
img_expire_after = timedelta(days = 180)
def kwget(key, default = None):
    def f(**kwargs):
        nonlocal key, default
        return kwargs.get(key, default)
    return f
def illust_torrent(t: Torrent, size=384, style="light", extra = None):
    if("_id" in t):
        cache_key = "illut_torrent-%s-%s-%s"%(t._id, style, size)
    else:
        cache_key = None
    if(cache_key is not None):
        _cache_pth = path.join(illust_cache_pth, cache_key+".png")
        if(path.exists(_cache_pth)):
            return Image.open(_cache_pth)
    global img_expire_after
    intro = t.introduction
    is_dark = style == "dark"
    
    
    columns = []
    
    pattern = r'<img src="(.+?)"'
    imgs = re.findall(pattern, intro)
    img = None
    if(imgs):
        try:
            img = get_image(imgs[0], expire_after = img_expire_after)  
        except Exception:
            img = None
    if(img is None):
        if(is_dark):
            BG = Image.new("RGBA", (32, 32), (0, 0, 0, 255))
        else:
            BG = Image.new("RGBA", (32, 32), (255,)*3)
    else:
        BG = process.adjust_L(img, -0.7 if is_dark else 0.7)
        columns.append(sizefit.fix_width(img, size))
            
        
    if(is_dark):
        font_fill = (255,)*4
    else:
        font_fill = (0, 0, 0, 255)
    RT = RichText(kwget("text"), fontSize = int(size/15), autoSplit=False,dontSplit=False,fill=None, bg = None, width=size)
    
    columns.append(RT.render(text=[t.title], fill=font_fill))
    if(extra):
        ex = extra(t, RT, style)
        columns.append(ex)
    ret = Column(columns)
    ret = CompositeBG(ret, BG)
    ret = ret.render()
    ensure_directory(_cache_pth)
    ret.save(_cache_pth)
    return ret


if(__name__ == "__main__"):
    from .search import search
    from .fetch import TorrentPage
    from ..image.print import image_show_terminal
    ls = TorrentPage.from_page_idx(1)
    t = ls.torrents[0]
    def extra(t:Torrent, RT):
        return RT.render(text = [t.magnet], fill=(0,255,255),bg=(128, 233,123 ,255))
    image_show_terminal(illust_torrent(t, extra=extra))
