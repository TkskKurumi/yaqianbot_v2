from ..backend import Message
from ..backend import scheduled, receiver
from ..backend.receiver_decos import threading_run, on_exception_response, command
from ..utils.moe_bangumi.search import search as moe_search
from ..utils.moe_bangumi.fetch import TorrentPage, Torrent
from ..utils.moe_bangumi.illust import illust_torrent
from .plg_admin import link_send_content
from ..utils.candy import simple_send, print_time
import random
import time
from pil_functional_layout.widgets import Grid

fetch_idx = random.randrange(TorrentPage.page_count())
last_fetch = time.time()
@scheduled
@threading_run
def update_db():
    global fetch_idx, last_fetch
    # print("update db for bangumi-moe")
    if(time.time()-last_fetch < 120):
        print("later update bangumi db")
        return
    else:
        print("update db for bangumi-moe")
    pc = TorrentPage.page_count()
    _idx = (int(fetch_idx)%pc)+1
    page = TorrentPage.from_page_idx(_idx)
    page = TorrentPage.from_page_idx(1)
    fetch_idx += 1
    last_fetch = time.time()
@receiver
@threading_run
@on_exception_response
@command("/bangumi", opts = {})
def cmd_bangumi_search(message: Message, *args, **kwargs):
    title = " ".join(args)
    
    time_split = 10
    start_search = time.time()
    last_report = time.time()
    
    def progress_report(idx, n):
        nonlocal start_search, message, last_report, time_split
        if(idx%50 == 0):
            tm = time.time()
            if(tm>last_report+time_split):
                elapsed = tm-start_search
                remain = elapsed/(idx+1)*(n-idx)
                simple_send("正在搜索%d/%d, 预计剩余%.1f秒"%(idx, n, remain))
                last_report = tm
                time_split+=(60-time_split)/2
    torrent_ls = moe_search(title, progress_report)
    mes = []
    imgs = []
    def ex(t: Torrent, RT, style):
        lnk = link_send_content(t.magnet)
        ret = [RT.render(text=["请输入"])]
        ret.append(RT.render(text=[lnk], BG=(255, 220, 230), fill=(30,10,20)))
        ret.append(RT.render(text=["查看磁力链接"]))
        return RT.render(text=ret)
    
    for torrent in torrent_ls.torrents[:24]:
        title = torrent.title
        with print_time("illust "+title):
            img = illust_torrent(torrent, extra=ex)
        imgs.append(img)
    ret = Grid(imgs).render().convert("RGB")
    message.response_sync(ret)