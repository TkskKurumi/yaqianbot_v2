from .database import torrents as torrents_db
from .fetch import Torrent, TorrentListing
from ..algorithms.lcs import lcs as LCS
import time
from ..candy import print_time
import re
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


def search_tagged(title, progress_call_back=None):
    ret = []
    tags = dict()
    pat = re.compile("[\[\]\(\)]")
    with print_time("get bangumi db"):
        itm = torrents_db.items()
    # with print_time("search tagged"):
    itm = list(itm)
    st = time.time()
    has_cb = callable(progress_call_back)
    n = len(list(itm))
    with print_time("search_tagged"):
        for idx, i in enumerate(itm):
            if(has_cb):
                progress_call_back(idx, n, st)
            k, v = i
            v = Torrent(v)
            tor_title = v.title
            tor_tags = pat.split(tor_title)
            tor_score = 0
            for tag in tor_tags:
                if(not tag):
                    continue
                if(tag in tags):
                    tag_score = tags[tag]
                else:
                    tag_score = LCS(title, tag).common_ratio
                    tags[tag] = tag_score
                tor_score += tag_score*len(tag)
            v.search_score = tor_score
            ret.append(v)
    ret.sort(key=lambda x: -x.search_score)
    return TorrentListing(ret)


def search(title, progress_call_back=None):
    ret = []
    with print_time("get bangumi db"):
        itm = torrents_db.items()
    start = time.time()
    itm = list(itm)
    n = len(itm)
    # print(n, type(itm))
    has_cb = callable(progress_call_back)
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
    print("search %d torrent in %.3f seconds" % (n, elapsed))
    ret.sort(key=lambda x: -x.search_score)
    return TorrentListing(ret)


if(__name__ == "__main__"):
    def cb(idx, n, st):
        if(idx % 50 == 49):
            print("%.1f%%" % (idx/n*100,), end="\r")
    from .fetch import TorrentPage
    for i in range(1, 11):
        TorrentPage.from_page_idx(i)
    ls1 = search_tagged("Lycoris Recoil", progress_call_back = cb)
    print(ls1.torrents[0])
    