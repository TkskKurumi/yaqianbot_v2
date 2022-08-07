import requests_cache
from requests_cache.backends.sqlite import SQLiteCache
from datetime import timedelta
from .paths import pth as work_pth
from os import path
import os
cache_backend = SQLiteCache(path.join(work_pth, "OSU_CACHE"), cache_control=True)
sess = requests_cache.CachedSession(
    'CachedSession',
    backend=cache_backend,
    expire_after=timedelta(minutes=20)
)

request_kwargs = {}

if(os.environ.get("OSU_PROXY")):
    proxies = {
        "http": os.environ.get("OSU_PROXY"),
        "https": os.environ.get("OSU_PROXY")
    }
    request_kwargs["proxies"] = proxies


def get(url, *args, **kwargs):
    kwa = dict(request_kwargs)  # copy to preserve
    kwa.update(kwargs)
    return sess.get(url, *args, **kwargs)

def get_image(url, *args, **kwargs):
    from io import BytesIO
    from PIL import Image
    r = sess.get(url, *args, **kwargs)
    content = r.content
    bio = BytesIO()
    bio.write(content)
    bio.seek(0)
    im = Image.open(bio)
    return [r.headers.get('Content-Type'), im]