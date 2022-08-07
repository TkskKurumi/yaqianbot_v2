import requests_cache
from requests_cache.backends.sqlite import SQLiteCache
from datetime import timedelta
from .paths import cachepth
import os, tempfile
from os import path
from PIL import Image
from io import BytesIO
cache_backend = SQLiteCache(cachepth, cache_control=True)
sess = requests_cache.CachedSession(
    'CachedSession',
    backend=cache_backend,
    expire_after=timedelta(minutes=20)
)
request_kwargs = {}
if(os.environ.get("PIXIV_PROXY")):
    proxies = {
        "http": os.environ.get("PIXIV_PROXY"),
        "https": os.environ.get("PIXIV_PROXY")
    }
    request_kwargs["proxies"] = proxies
# def request(url)
def get(url, *args, **kwargs):
    kwa = dict(request_kwargs) # copy to preserve
    kwa.update(kwargs)
    return sess.get(url, *args, **kwargs)
def get_image(url, *args, **kwargs):
    r = request("GET", url, *args, **kwargs)
    bio = BytesIO()
    bio.write(r.content)
    bio.seek(0)
    im = Image.open(bio)
    return im
def request(method, *args, **kwargs):
    kwa = dict(request_kwargs)
    kwa.update(kwargs)
    return sess.request(method, *args, **kwargs)