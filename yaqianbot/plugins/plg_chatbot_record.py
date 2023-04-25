from os import path
from threading import Lock

from ..backend.cqhttp.message import CQMessage
from ..backend.paths import mainpth
from ..utils.lvldb import TypedLevelDB
from ..utils.candy import locked
from ..utils.lockpool import LockPool
from ..backend import receiver
from ..backend.receiver_decos import command, is_su, on_exception_response, threading_run
lckpool = LockPool()
recording_path = path.join(mainpth, "chatbot", "message_records")
db = TypedLevelDB.open(recording_path)
buffers = dict()
commited = dict()
def get_mes_record(group_id):
    with lckpool.locked(group_id):
        if(group_id not in buffers):
            buffers[group_id] = []
        if(group_id not in commited):
            if(group_id in db):
                commited[group_id] = db[group_id]
            else:
                commited[group_id] = []
        return commited[group_id], buffers[group_id]
BUFF_SIZE = 64
WINDOW_SIZE = 16384
def set_mes_record(group_id, com, buf, force_commit = False):
    with lckpool.locked(group_id):
        if(len(buf)>BUFF_SIZE or force_commit):
            concat = com+buf
            if(len(concat)>WINDOW_SIZE):
                concat = concat[-WINDOW_SIZE:]
            db[group_id] = concat
            commited[group_id] = concat
            buffers[group_id] = []
        else:
            buffers[group_id] = buf
def add_mes_record(group_id, mes):
    com, buf = get_mes_record(group_id)
    buf.append(mes)
    set_mes_record(group_id, com, buf)
@receiver
@threading_run
def receiver_mes_record(mes):
    add_mes_record(mes.group, mes.raw)
@receiver
@threading_run
@on_exception_response
@command("/回看",opts={})
def cmd_record_lookup(mes:CQMessage, *args, **kwargs):

    com, buf = get_mes_record(mes.group)
    rec = com+buf
    beg = 0
    end = 1
    if(len(args)>0):
        beg = int(args[0])
        if(len(args)>1):
            end = int(args[1])
        else:
            end = beg+1
    rets = []
    for i in range(beg, end):
        if(i>=len(rec)):
            break
        rec_mes = rec[-1-i]
        rec_mes = CQMessage.from_cq(rec_mes)
        uin = int(rec_mes.sender.id)
        uname = rec_mes.sender.name
        fmes = rec_mes.get_mseg_list()
        fmes = mes.construct_forward(fmes, uin=uin,name=uname)
        rets.append(fmes)
    if(not rets):
        mes.response_sync("超过了聊天记录的范围")
    else:
        mes.send_forward_message(rets[::-1])
