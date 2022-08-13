# import requests_cache
# from requests_cache.backends.sqlite import SQLiteCache
import json
from datetime import timedelta
import os, tempfile
from os import path
from PIL import Image
from io import BytesIO
import cloudscraper
work_path = path.join(path.expanduser("~"), "tmp", "tetrio")
# cache_backend = SQLiteCache(cachepth, cache_control=True)
# sess = requests_cache.CachedSession(
#     'CachedSession',
#     backend=cache_backend,
#     expire_after=timedelta(minutes=20)
# )
sess = cloudscraper.create_scraper(browser = "chrome")
request_kwargs = {}
if(os.environ.get("TETRIO_PROXY")):
    proxies = {
        "http": os.environ.get("TETRIO_PROXY"),
        "https": os.environ.get("TETRIO_PROXY")
    }
    request_kwargs["proxies"] = proxies
if(path.exists(path.join(work_path, "cookies.json"))):
    with open(path.join(work_path, "cookies.json"), "r") as f:
        j = json.load(f)
    cookies = {}
    for idx, i in enumerate(j):
        cookies[i["name"]] = i["value"]
    request_kwargs["cookies"] = cookies
    print("loaded tetrio cookies")
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