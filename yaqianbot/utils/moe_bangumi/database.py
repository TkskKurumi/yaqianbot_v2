from ..jsondb import jsondb
from os import path
from .paths import workpth
torrents = jsondb(path.join(workpth, "torrents"), method=lambda x:x[:3])
