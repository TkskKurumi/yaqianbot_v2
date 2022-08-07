from .database import torrents as torrents_db
from .fetch import Torrent, TorrentListing
from ..algorithms.lcs import lcs as LCS
import time
from ..candy import print_time

cache = dict()
def torrent_lcs(torrent, search_key):
    if("_id" in torrent):
        _id = torrent._id
        cache_key = (_id, search_key)
    else:
        cache_key = None
    if(cache_key is None):
        ret = LCS(torrent.title, search_key)
        return ret
    elif(cache_key in cache):
        return cache[cache_key]
    else:
        ret = LCS(torrent.title, search_key)
        cache[cache_key] = ret
        return ret
def search(title, progress_call_back=None):
    ret = []
    with print_time("get bangumi db"):
        itm = torrents_db.items()
    start = time.time()
    n = len(itm)
    has_cb =callable(progress_call_back) 
    for idx, i in enumerate(itm):
        if(has_cb):
            progress_call_back(idx, n, start)
        k, v = i
        v = Torrent(v)
        # lcs = LCS(title, v.title)
        lcs = torrent_lcs(v, title)
        v.search_score = lcs.common_ratio
        ret.append(v)
    elapsed = time.time()-start
    print("search %d torrent in %.3f seconds" % (len(itm), elapsed))
    ret.sort(key=lambda x: -x.search_score)
    return TorrentListing(ret)


if(__name__ == "__main__"):
    ls = search("Lycoris Recoil [03]")
    for i in ls.torrents[:3]:
        print(i.title)
