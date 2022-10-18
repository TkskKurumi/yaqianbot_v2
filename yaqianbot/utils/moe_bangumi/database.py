from ..jsondb import jsondb
from ..lvldb import TypedLevelDB
from os import path
from .paths import workpth
# torrents = jsondb(path.join(workpth, "torrents"), method=lambda x:x[:3])
torrents = TypedLevelDB(path.join(workpth, "torrents_leveldb"))