import requests_cache
from requests_cache.backends.sqlite import SQLiteCache
from datetime import timedelta
from .paths import cachepth, mime2ext, base32, temppth, ensure_directory
import os, tempfile
from os import path
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
    request_kwargs.update(proxies)

def get(url, *args, **kwargs):
    kwa = dict(request_kwargs) # copy to preserve
    kwa.update(kwargs)
    return sess.get(url, *args, **kwargs)

def get_file(url, *args, **kwargs):
    r = get(url, *args, **kwargs)
    
    type = r.headers.get('Content-Type')
    if(type is not None):
        type = type.split(";")[0]
    ext = mime2ext.get(type,'.tmp')

    name = base32([url,args,kwargs])
    pth = path.join(temppth, name+ext)
    ensure_directory(pth)
    with open(pth,"wb") as f:
        f.write(r.content)
    return pth
def post(url, *args, **kwargs):
    kwa = dict(request_kwargs) # copy to preserve
    kwa.update(kwargs)
    return sess.post(url, *args, **kwargs)
def head(url, *args, **kwargs):
    kwa = dict(request_kwargs) # copy to preserve
    kwa.update(kwargs)
    return sess.head(url, *args, **kwargs)

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
