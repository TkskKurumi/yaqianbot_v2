from ..backend.paths import mainpth
from ..utils.algorithms.lcs import lcs
from ..utils.game.action_point import ActionPointMgr
from ..utils.candy import simple_send, locked
from ..utils.lvldb import TypedLevelDB
from ..utils.image.process import shadow
from ..utils.image.colors import WHITE
from ..backend import receiver, CQMessage
from ..backend.requests import get_avatar
from ..backend.receiver_decos import command, threading_run, on_exception_response
from ..backend import scheduled
from easydict import EasyDict as EDict
import random
import heapq
from os import path
from pil_functional_layout.widgets import RichText
import time
from ..utils.geometry.elements import Point2d
from ..utils.make_gif import make_gif
from threading import Lock
pth = path.join(mainpth, "ecchi")


sexuality_db = ActionPointMgr.open(path.join(pth, "sexuality"), per_hour=25)
# count_db = TypedLevelDB.open(path.join(pth, "count"))
info_db = TypedLevelDB.open(path.join(pth, "info"))


def sese_avatar(uid):
    avatar = get_avatar(uid)
    w, h = avatar.size
    heart = RichText(["💕"], width=512, bg=(
        255, 255, 255, 0), fontSize=w//9).render()
    ww, hh = heart.size
    heart = shadow(heart, color=WHITE, radius=ww/14)
    ww, hh = heart.size
    frames = []
    heart_pos = [Point2d(w/2, h) for i in range(30)]

    while(True):
        nframes = len(frames)+1
        im = avatar.copy()
        ok = True
        for idx, p in enumerate(heart_pos):
            x, y = p
            x = int(x-ww/2)
            y = int(y-hh)
            if(y > 0):
                ok = False
            im.paste(heart, box=(x, y), mask=heart)
            dx = random.normalvariate(0, nframes)
            dy = -abs(random.normalvariate(0, nframes))
            heart_pos[idx] = p + Point2d(dx, dy)*h/30
        if(ok):
            break
        frames.append(im)
    gif = make_gif(frames)
    return gif


def _rnd_count(cost):
    return (abs(random.normalvariate(0, cost))+cost)/8


def get_info(uid):
    return info_db.get(uid, dict())


def pop_window(ls, interv=7*24*3600):
    ls = [tuple(i) for i in ls]
    t = time.time() - interv
    while(ls and ls[0][0] < t):
        heapq.heappop(ls)
    return ls


def push_window(ls, value):
    ls = [tuple(i) for i in ls]
    heapq.heappush(ls, value)
    return ls


def window_count(uid):
    info = get_info(uid)
    in_win = info.get("in_win", [])

    out_win = info.get("out_win", [])

    in_win = pop_window(in_win)
    out_win = pop_window(out_win)
    into = 0
    outof = 0
    for _, i in in_win:
        into += i
    for _, o in out_win:
        outof += o
    return into, outof


def incr_count(uid, into=0, outof=0):
    info = get_info(uid)
    if(into):
        info["into"] = info.get("into", 0)+into

        in_win = info.get("in_win", [])
        in_win = push_window(in_win, (time.time(), into))
        info["in_win"] = pop_window(in_win)

    if(outof):
        info["outof"] = info.get("outof", 0)+outof
        out_win = info.get("out_win", [])
        out_win = push_window(out_win, (time.time(), outof))
        info["out_win"] = pop_window(out_win)

    info_db[uid] = info


def cmd_sese_status(message: CQMessage, *args, send=True, **kwargs):
    uid = kwargs.get("uid") or message.sender.id
    name = message.get_nickname_by_id(uid)
    sex = sexuality_db[uid]
    info = get_info(uid)
    # into, outof = info.get("into", 0), info.get("outof", 0)
    into, outof = window_count(uid)
    mes = ["%s:\n星宇: %.1f%%\n七天内被注入量: %.2fml\n七天内输出量: %.2fml" %
           (name, sex, into, outof)]
    if("extra" in kwargs):
        mes.extend(kwargs["extra"])
    if(send):
        simple_send(mes)
    return mes


def cmd_sese_0721(message: CQMessage, *args, **kwargs):
    uid = message.sender.id
    sex = 35
    if(not sexuality_db.afford(uid, sex)):
        remain = sexuality_db.time_targeting_str(uid, sex)
        ex = ["\n星宇不足%.1f%%、无法色色，" % sex, "需要"+remain]
        return cmd_sese_status(message, extra=ex)
    sexuality_db.cost(uid, sex)
    outof = _rnd_count(sex)
    incr_count(uid, outof=outof)
    cmd_sese_status(message, extra=["\n出来了%.2fml"])


def cmd_sese_fuck_mate(message: CQMessage, *args, **kwargs):
    uid = message.sender.id
    uname = message.get_nickname_by_id(uid)
    sex = 35
    if(not sexuality_db.afford(uid, sex)):
        remain = sexuality_db.time_targeting_str(uid, sex)
        ex = ["\n星宇不足%.1f%%、无法色色，" % sex, "需要"+remain]
        return cmd_sese_status(message, extra=ex)

    if(message.ated):
        mate_id = str(random.choice(message.ated))
        if(mate_id == uid):
            simple_send("不可以超自己")
            return
        mate_name = message.get_nickname_by_id(mate_id)
    elif(args):
        name = " ".join(args)
        mate_list = message.get_group_member_list()
        candi = []
        for i in mate_list:
            nm = i.get("nickname", "") + i.get("card", "")
            l = lcs(nm, name).common_ratio
            candi.append((l, nm, i.get("user_id")))
        mate = max(candi)
        mate_id = str(mate[2])
        mate_name = name
    else:
        mate_list = message.get_group_member_list()
        mate = random.choice(mate_list)
        while(str(mate["user_id"]) == str(uid)):
            mate = random.choice(mate_list)
        mate_id = str(mate["user_id"])
        mate_name = mate.get("nickname") or mate.get("card")
    sexuality_db.cost(uid, sex)
    count = _rnd_count(sex)

    incr_count(mate_id, into=count)
    incr_count(uid, outof=count)

    mes = []
    mes.extend(cmd_sese_status(message, send=False, uid=uid, extra=[
               "\n%s出来了%.1fml, 星宇消退了%.1f%%" % (uname, count, sex)]))
    mes.append("\n")
    mes.extend(cmd_sese_status(message, send=False, uid=mate_id,
               extra=["\n%s被%s注入了%.1fml" % (mate_name, uname, count)]))
    mes.append(sese_avatar(mate_id))
    simple_send(mes)


def cmd_sese_rank(message: CQMessage, *args, **kargs):
    mate_list = message.get_group_member_list()
    users = []
    for user in mate_list:
        uid = str(user.get("user_id"))
        into, outof = window_count(uid)
        if((not into) and (not outof)):
            continue
        users.append((user, into, outof))
    if(not users):
        simple_send("本群没有色色")
        return
    sort_out = sorted(users, key=lambda x: -x[2])
    sort_in = sorted(users, key=lambda x: -x[1])

    mes = []
    mes.append("被注入量排行：")
    for idx, u in enumerate(sort_in[:10]):
        user, value, outof = u
        name = user.get("nickname") or user.get("card")
        s = "#%d-%s: %.2fml" % (idx+1, name, value)
        mes.append(s)

    mes.append("输出量排行：")
    for idx, u in enumerate(sort_out[:10]):
        user, into, value = u
        name = user.get("nickname") or user.get("card")
        s = "#%d-%s: %.2fml" % (idx+1, name, value)
        mes.append(s)
    mes = "\n".join(mes)
    rt = RichText([mes], width=720, bg=(255, 255, 255, 255),
                  fill=(0, 0, 0, 255)).render()
    simple_send(rt)


parties = dict()
party_lck = Lock()


@scheduled
@threading_run
def party_schedule():
    global parties, party_lck
    with locked(party_lck):
        itms = list(parties.items())
        for group, party in itms:
            st = party["starttime"]
            if(st > time.time()):
                continue

            message = party["message"]
            attendee = list(party["attendee"])

            if(len(attendee) < 2):
                simple_send("人数不足，取消银趴！")
                parties.pop(group)
                continue
            n = len(attendee)
            mes = []
            mes.append("%s的银趴开始啦~\n" % party["group_name"])

            actions = [
                lambda a, b:f"{a}超市了{b}",
                lambda a, b:f"{b}榨了{a}",
                lambda a, b:f"{b}冲了{a}",
                lambda a, b:f"{a}草了{b}",
                lambda a, b:f"{a}推倒了{b}"
            ]
            intos = {}
            outofs = {}
            for i in range(n):
                u = attendee[i]
                for j in range(n//2):
                    v = random.choice(attendee)
                    while(u == v):
                        v = random.choice(attendee)
                    uname = message.get_nickname_by_id(u)
                    vname = message.get_nickname_by_id(v)
                    count = _rnd_count(20)
                    mes.append(random.choice(actions)(uname, vname)+"\n")
                    incr_count(u, outof=count)
                    incr_count(v, into=count)
                    outofs[u] = outofs.get(u, 0) + count
                    intos[v] = intos.get(v, 0) + count
            for u in attendee:
                uname = message.get_nickname_by_id(u)
                outo = outofs.get(u, 0)
                into = intos.get(u, 0)
                m = "%s总共输出了%.2fml, 收入了%.2fml" % (uname, outo, into)
                mes.append(m+"\n")
                mes.extend(cmd_sese_status(message, uid=u, send=False))
                mes.append("\n")
            img = RichText(mes, width=720, bg=(255,)*4).render()
            simple_send(mes[:1]+[img])
            parties.pop(group)


def cmd_sese_silver_party(message: CQMessage, *args, **kwargs):
    global parties, party_lck
    with locked(party_lck):
        uid = message.sender.id
        group = message.group
        if(group in parties):
            party = parties[group]
        else:
            party = dict()
            party["group"] = group
            party["attendee"] = set()
            party["message"] = message
        party["attendee"].add(uid)
        tm = time.time()
        if(len(party["attendee"]) >= 5):
            starttime = tm + 5
        elif(len(party["attendee"]) >= 3):
            starttime = tm + 10
        else:
            starttime = tm + 120

        ginfo = message.api.get_group_info(group_id=message.raw.group_id)
        party["starttime"] = starttime
        party["group_name"] = ginfo["group_name"]
        parties[group] = party
        mes = []
        mes.append("在%s开银趴" % ginfo["group_name"])
        mes.append("\n参与者: \n")
        for i in party["attendee"]:
            name = message.get_nickname_by_id(i)
            mes.append("「%s 」" % name)
        prompt = "开始" if len(party["attendee"]) > 1 else "人数不足取消，来人加入吧"
        mes.append("\n将在%d秒后%s" % (starttime - tm, prompt))
        simple_send(mes)


@receiver
@threading_run
@on_exception_response
@command("/银趴", opts={})
def cmd_receiver_silver_party(message: CQMessage, *args, **kwargs):
    return cmd_sese_silver_party(message, *args, **kwargs)


@receiver
@threading_run
@on_exception_response
@command("/色色", opts={})
def cmd_sese(message, *args, **kwargs):
    if(not args):
        return cmd_sese_status(message, *args, **kwargs)
    elif(args[0] == "0721"):
        return cmd_sese_0721(message, *args, **kwargs)
    elif(args[0] == "超群友"):
        return cmd_sese_fuck_mate(message, *args[1:], **kwargs)
    elif(args[0].startswith('超')):
        return cmd_sese_fuck_mate(message, *args[1:], **kwargs)
    elif(args[0].startswith("排行")):
        return cmd_sese_rank(message, *args, **kwargs)
    else:
        simple_send("未知")
