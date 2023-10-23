from ...utils.lvldb import TypedLevelDB
from ...backend.paths import mainpth
from os import path
from ...backend.kook import KHLMessage

tag_db = TypedLevelDB.open(
    path.join(mainpth, "diffusion", "user_tag")
)

def _add_user_tag(uid, fr, to):
    d = tag_db.get(uid, {})
    d[fr] = to
    tag_db[uid] = d

def _get_user_tag(uid):
    return tag_db.get(uid, {})

def get_user_tag(message: KHLMessage):
    ret = {}
    utag = {}
    for k, v in tag_db.items():
        ret.update(v)
        if (k == message.sender.id):
            utag = v
    ret.update(utag)
    return ret

