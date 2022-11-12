from . import requests
import re
from dataclasses import dataclass
import json
from .executor import create_task, create_pool
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
        end_str = "'>\n<"
        start = html.find(start_str)
        end = html.find(end_str, start)
        j = html[start + len(start_str):end]
        try:
            j = json.loads(j)["illust"]
        except Exception as e:
            import os
            os.makedirs(temppth, exist_ok=True)
            debugpth = path.join(temppth, "debug.html")
            with open(debugpth, "w", encoding="utf-8") as f:
                f.write(html)
            raise Exception("Error parsing pixiv html %s, debug with %s"%(url, debugpth))
        j = j[str(id)]

        self.id = j['id']
        self.title = j['title']
        self.urls = j['urls']
        self.author = j['userName']
        self.author_id = j['userId']
        self.page_count = j["pageCount"]
        plain_tags = [i['tag'] for i in j["tags"]['tags']]
        plain_tags.append("user: "+self.author)
        plain_tags.append("userId: "+self.author_id)
        self.plain_tags = plain_tags
        self.raw = j
    @property
    def is_safe(self):
        if("R-18" in self.plain_tags):
            return False
        if("R-18G" in self.plain_tags):
            return False
        return True
    def __repr__(self):
        return '<Pixiv Illust title="%s", author="%s", id=%s>' % (self.title, self.author, self.id)

    def get_pages(self, start=None, end=None, quality="regular"):
        if(start is None):
            start = 0
        if(end is None):
            end = self.page_count
        ret = []

        for i in range(start, end):
            url = self.urls[quality].replace("_p0", "_p%d" % i)
            referer = "https://www.pixiv.net/artworks/%s" % self.id
            headers = {"referer": referer}
            task = create_task(requests.get_file, url, headers=headers)
            # file = requests.get_file(url, headers = headers)
            ret.append(task)
        rets = [i.result() for i in ret]
        return rets


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
        remainder = i % 50
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

class UserIllusts(BaseListing):
    def __init__(self, user_id):
        url = "https://www.pixiv.net/ajax/user/%s/profile/all"%user_id
        r = requests.get(url)
        j = r.json()
        if(j["error"]):
            raise Exception(j["message"])
        body = j["body"]
        illusts = body["illusts"]
        self.ids = list(illusts)
        self.items = []
        for id in self.ids:
            item = ListingElement(id)
            self.items.append(item)
class Related(BaseListing):
    def __init__(self, orig_id):
        headers = {"referer": "https://www.pixiv.net/artworks/%s"%orig_id}
        url=r"https://www.pixiv.net/ajax/illust/%s/recommend/init?limit=%d&lang=zh"%(orig_id, 18)
        
        j = requests.get(url, headers=headers).json()
        print(url, j)
        self.ids = j['body']['nextIds']
        self.items = []
        for id in self.ids:
            item = ListingElement(id)
            self.items.append(item)
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
            "date": date
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
    ill = Illust(99389974)
    print(ill.title)
    print(ill.urls)
    print(ill.get_pages(quality="small"))
    print(ill.plain_tags)
    from ..io import savejson
    savejson("/tmp/tmp.json", ill.raw)
    print("/tmp/tmp.json")
