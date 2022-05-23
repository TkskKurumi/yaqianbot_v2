from os import path
import tempfile
import os

workpth = os.environ.get("PIXIV_PATH") or path.join(tempfile.gettempdir(),'pyxyv')
cachepth = path.join(workpth, "cache")