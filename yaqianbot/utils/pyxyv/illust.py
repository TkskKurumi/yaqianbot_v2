from . import requests
import re
from dataclasses import dataclass
import json
from .executor import create_task
from typing import List
from datetime import datetime, timedelta
from urllib.parse import urlencode
import json
from .paths import temppth, ensure_directory
from os import path


class Illust:
    def __init__(self, id=59580629):
        if(isinstance(id, str)):
            if(id.isdigit()):
                # ok
                pass
            elif("artworks" in id):
                id = re.findall(r"artworks/(\d+)", id)
            else:
                pass
                # what?
        elif(isinstance(id, int)):
            # ok
            pass
        else:
            raise TypeError(id)
        url = r"https://www.pixiv.net/artworks/%s" % id

        html = requests.get(url).text

        start_str = "<meta name=\"preload-data\" id=\"meta-preload-data\" content='"
        end_str = "'>\n<script async src="
        start = html.find(start_str)
        end = html.find(end_str, start)
        j = html[start + len(start_str):end]
        j = json.loads(j)["illust"]
        j = j[str(id)]

        self.id = j['id']
        self.title = j['title']
        self.urls = j['urls']
        self.author = j['userName']
        self.author_id = j['userId']
        self.page_count = j["pageCount"]

    def __repr__(self):
        return '<Pixiv Illust title="%s", author="%s", id=%s>' % (self.title, self.author, self.id)

    def get_pages(self, start=None, end=None, quality="regular"):
        if(start is None):
            start = 0
        if(end is None):
            end = self.page_count
        ret = []

        for i in range(start, end):
            url = self.urls[quality].replace("_p0_", "_p%d_" % i)
            referer = "https://www.pixiv.net/artworks/%s" % self.id
            headers = {"referer": referer}
            task = create_task(requests.get_file, url, headers=headers)
            # file = requests.get_file(url, headers = headers)
            ret.append(task)
        return [i.result() for i in ret]


class BaseListing:
    def __init__(self, ids=None, items=None):
        self.ids = ids or list()
        self.items = items or list()


class ListingElement:
    def __init__(self, id, preview=None, title=None):

        _ = None

        def info():
            nonlocal _
            if(_ is not None):
                return _
            _ = Illust(id)
            return _
        if(title is None):
            title = info().title
        if(preview is None):
            preview = info().urls["thumb"]

        self.id = id
        self.preview = preview
        self.title = title
        self.preview_referer = "https://www.pixiv.net/artworks/%s" % id

    def get_preview(self):
        return requests.get_file(self.preview, headers={"referer": self.preview_referer})


def _getRankingToday():
    # connection_throttle.acquire()
    t = requests.get(r'https://www.pixiv.net/ranking.php?mode=daily').text
    f = re.findall(
        r'<link rel="canonical" href="https://www.pixiv.net/ranking.php\?mode=daily&amp;date=(\d{8})">', t)[0]
    return datetime(year=int(f[:4]), month=int(f[4:6]), day=int(f[6:]))


def get_ranking(date=None, mode="weekly", start=0, end=20):
    ret = BaseListing()
    pages = dict()

    def get_idx(i):
        pagen = (i//50)+1
        remainder = i%50
        if(pagen in pages):
            page = pages[pagen]
        else:
            page = Ranking(date, mode, pagen)
            pages[pagen] = page
        return page.items[remainder], page.ids[remainder]
    for i in range(start, end):
        item, id = get_idx(i)
        ret.items.append(item)
        ret.ids.append(id)
    return ret

class Ranking(BaseListing):
    def __init__(self, date=None, mode="weekly", page=1):
        if(date is None):
            date = _getRankingToday()
        elif(isinstance(date, int)):
            date = _getRankingToday()+timedelta(days=date)
        if(isinstance(date, datetime)):
            date = date.strftime("%Y%m%d")

        params = {
            "mode": mode,
            "content": "illust",
            "format": "json",
            "p": page,
            "data": date
        }
        url = 'https://www.pixiv.net/ranking.php?'+urlencode(params)
        # print(url)
        r = requests.get(url)
        j = r.json()

        self.ids = []
        self.items = []
        for i in j["contents"]:
            item = ListingElement(
                id=i['illust_id'], title=i['title'], preview=i['url'])
            self.items.append(item)
            self.ids.append(i["illust_id"])


if(__name__ == "__main__"):
    ill = Illust(99213489)
    print(ill.title)
    print(ill.urls)
    print(ill.get_pages(quality="small"))
