import requests_cache
from requests_cache.backends.sqlite import SQLiteCache
from datetime import timedelta
from .paths import cachepth

cache_backend = SQLiteCache(cachepth, cache_control=True)
sess = requests_cache.CachedSession(
    'CachedSession',
    backend=cache_backend,
    expire_after=timedelta(minutes=20)
)


def get_image(*args, **kwargs):
    from io import BytesIO
    from PIL import Image
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
    url = "https://q.qlogo.cn/headimg_dl?dst_uin=%s&img_type=jpg&spec=640"%uid
    im = get_image(url)[1]
    return im

get = sess.get
post = sess.post
head = sess.head