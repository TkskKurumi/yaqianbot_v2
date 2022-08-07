import os
from urllib.parse import urlencode
import requests
import tempfile
from os import path
from io import BytesIO
from .jsondb import jsondb
from PIL import Image
from .myhash import base32
import time
tmp = path.expanduser("~")
cache = jsondb(path.join(tmp, "tmp", "saucenao", "cache"), method=lambda x:str(x)[:3])
sz = 256


def get_api_key():
    ret = os.environ.get("SAUCENAO_API_KEY")
    if(ret is None):
        raise Exception("SAUCENAO API_KEY missing.")
    return ret


def get_sauce(img):
    if(isinstance(img, str) and path.exists(img)):
        img = Image.open(img)
    
    # hashed = imghash(img)
    hashed = base32(img)
    if(hashed in cache):
        ret = cache[hashed]
        if(ret["expire"] < time.time()):
            return ret["data"]

    thumb = img.convert("RGB")
    thumb = thumb.resize((sz, sz), Image.Resampling.LANCZOS)
    bio = BytesIO()
    thumb.save(bio, format="JPEG")
    file = ("%s.jpg" % hashed, bio.getvalue())
    file = {"file": file}
    url = r"http://saucenao.com/search.php"
    params = {"output_type": 2, "api_key": get_api_key()}
    url = url+"?"+urlencode(params)

    r = requests.post(url, files=file)
    ret = r.json()
    if(r.status_code == 200):
        saved = {"expire": time.time()+3600, "data": ret}
        cache[hashed] = saved
    return ret


if(__name__ == "__main__"):
    # from .pyxyv import illust
    # ill = illust.Illust()
    # im = ill.get_pages()[0]
    print(get_sauce("/home/TkskKurumi/tmp/1918396-0003.jpg"))
