from . import requests
from dateutil.parser import parse as parse_time
from typing import List
from .database import torrents as torrents_db
class AttrDict(dict):
    def __getattr__(self, __name: str):
        if(__name in self):
            ret = self[__name]
            if(isinstance(ret, dict)):
                return AttrDict(ret)
            return ret
        raise AttributeError(__name)

class Torrent(AttrDict):
    @property
    def timestamp(self):
        pub_time = self.publish_time
        t = parse_time(pub_time)
        return t.timestamp()
class TorrentListing:
    def __init__(self, torrents: List[Torrent]):
        self.torrents = torrents
        for torrent in torrents:
            if(torrent._id not in torrents_db):
                torrents_db[torrent._id] = torrent
    @property
    def time_min(self):
        mn = None
        for torrent in self.torrents:
            t = torrent.timestamp
            if((mn is None) or (t<mn)):
                mn = t
        return mn
    @property
    def time_max(self):
        mx = None
        for torrent in self.torrents:
            t = torrent.timestamp
            if((mx is None) or (t>mx)):
                mx = t
        return mx
    @property
    def time_range(self):
        return self.time_min, self.time_max
def _torrent_page(idx):
    url = r"https://bangumi.moe/api/torrent/page/%s"%idx
    # r = requests.sess.request("GET", url, expire_after = 120)
    r = requests.request("GET", url, expire_after = 300)
    return AttrDict(r.json())
class TorrentPage(TorrentListing):
    @classmethod
    def page_count(cls):
        page_json = _torrent_page(1)
        return page_json.page_count
    @classmethod
    def from_page_idx(cls, idx):
        page_json = _torrent_page(idx)
        torrents = [Torrent(t) for t in page_json.torrents]
        print("Get bangumi.moe page %s"%idx)
        return cls(torrents)
    @classmethod
    def around_timestamp(cls, t):
        page_json = _torrent_page(1)
        page_cnt = page_json.page_count
        l = 1
        r = page_cnt
        while(l<=r): # 
            mid = (l+r)>>1
            page = cls.from_page_idx(mid)
            t_l, t_r = page.time_range
            if(t_l<=t and t<=t_r):
                return page
            elif(t<t_l):
                l = mid+1
            elif(t>t_r):
                r = mid-1
        return page
if(__name__=="__main__"):
    import time
    p = TorrentPage.from_page_idx(1)

    print(p.torrents[0])
    print(p.time_range)
    
    t = time.time()-3600*24
    p = TorrentPage.around_timestamp(t)
    t_l, t_r = p.time_range
    print("%d <= %d <= %d"%(t_l, t, t_r))