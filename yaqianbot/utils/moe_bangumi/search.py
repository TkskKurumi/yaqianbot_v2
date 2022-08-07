from .database import torrents as torrents_db
from .fetch import Torrent, TorrentListing
from ..algorithms.lcs import lcs as LCS
import time
from ..candy import print_time


def search(title, progress_call_back=None):
    ret = []
    with print_time("get bangumi db"):
        itm = torrents_db.items()
    start = time.time()
    n = len(itm)
    for idx, i in enumerate(itm):
        if(callable(progress_call_back)):
            progress_call_back(idx, n, start)
        k, v = i
        v = Torrent(v)
        lcs = LCS(title, v.title)
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
