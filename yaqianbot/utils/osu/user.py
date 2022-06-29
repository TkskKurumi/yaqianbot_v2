import requests
from . import credentials
from urllib.parse import urlencode
from ..io import savejson
import tempfile
from os import path
from ...backend import requests as cached_request
from ..image import colors, adjust_L
from PIL import Image, ImageDraw
endpoint = 'https://osu.ppy.sh/api/v2'




# def illust_score(score, style="dark"):
#     bmset = score["beatmapset"]
#     cover = requests.get_image(bmset["covers"]["slimcover"])[1]

#     color_main = colors.image_colors(cover)

#     if(style == "dark"):
#         color_back = color_main.replace(L=0.15)
#         color_fore = color_main.replace(L=0.95)
#         cover_back = adjust_L(cover, -0.9)
#     else:
#         color_back = color_main.replace(L=0.85)
#         color_fore = color_main.replace(L=0.05)
#         cover_back = adjust_L(cover, 0.9)

class AttrDict(dict):
    
    def __getattr__(self, __name: str):
        if(__name in self):
            ret = self[__name]
            if(isinstance(ret, dict)):
                return AttrDict(ret)
            return ret
        raise AttributeError(__name)
class User:
    def __init__(self, id):
        self.id = id
        # self.info = AttrDict(self.get_info())
        self.get_info()
        self.id_int = self.info.id
    def get_info(self, mode=None):
        ls = [endpoint, "users", self.id]
        if(mode is not None):
            ls.append(mode)
        url = "/".join(ls)
        r = requests.get(url, headers=credentials.get_header())
        self.info = AttrDict(r.json())
        return self.info


    def get_scores(self, type="best", params=None, mode=None):
        if(params is None):
            params = dict()
        if(mode is not None):
            params["mode"] = mode
        params["limit"]=256
        url = "/".join([endpoint, "users", str(self.id_int), "scores", type])
        if(params):
            url += "?"+urlencode(params)
        r = requests.get(url, headers=credentials.get_header())
        j = r.json()
        for idx, i in enumerate(j):
            i["type"] = type
            i["type_idx"] = idx+1
            j[idx] = i
        return j


if(__name__ == "__main__"):
    u = User("[Crz]Rumia")
    # print(u.info)
    print("len_scores", len(u.get_scores()))
    pth = path.join(tempfile.gettempdir(), "userscores.json")
    
    savejson(pth, u.get_scores())
    print(pth)

    pth = path.join(tempfile.gettempdir(), "user.json")
    savejson(pth, u.get_info("osu"))
    print(pth)
    
    pth = path.join(tempfile.gettempdir(), "user1.json")
    savejson(pth, u.info)
    print(pth)

    # print(u.info.statistics.pp)
    u.get_info(mode = "fruits")
    pth = path.join(tempfile.gettempdir(), "user_afk.json")
    savejson(pth, u.info)
    print(pth)