import requests_cache
from requests_cache.backends.sqlite import SQLiteCache
from datetime import timedelta
from .paths import cachepth, temppth, ensure_directory
from io import BytesIO
from PIL import Image
from ..utils.myhash import base32
from ..utils.candy import print_time
from os import path
cache_backend = SQLiteCache(cachepth, cache_control=True)
sess = requests_cache.CachedSession(
    'CachedSession',
    backend=cache_backend,
    expire_after=timedelta(minutes=20)
)
with print_time("clear cache"):
    sess.cache.remove_expired_responses()

def get_file(*args, savepath=None, **kwargs):
    mime2ext = {
        "image/jpeg": ".jpg",
        "image/png": ".png",
        "application/json": ".json",
        "application/javascript": ".js",
        "text/html": ".html",
        "text/plain": ".txt",
        "image/gif": ".gif"
    }
    r = sess.get(*args, **kwargs)
    ext = mime2ext.get(r.headers.get('Content-Type'), '.tmp')
    name = base32(r.content)+ext
    if(savepath is None):
        savepath = temppth
    savepath = path.join(savepath, name)
    ensure_directory(savepath)
    with open(savepath, "wb") as f:
        f.write(r.content)
    return savepath


def get_image(*args, **kwargs):
    r = sess.get(*args, **kwargs)
    content = r.content
    bio = BytesIO()
    bio.write(content)
    bio.seek(0)
    try:
        im = Image.open(bio)
    except Exception as e:
        print("Cannot get image", args, kwargs)
        raise e
    return [r.headers.get('Content-Type'), im]


def get_avatar(uid):
    url = "https://q.qlogo.cn/headimg_dl?dst_uin=%s&img_type=jpg&spec=640" % uid
    im = get_image(url)[1]
    return im


get = sess.get
post = sess.post
head = sess.head
