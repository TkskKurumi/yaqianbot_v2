from os import path
import tempfile
import os
home = path.expanduser("~")
workpth = os.environ.get("BANGUMI_MOE_PATH") or path.join(home, '.bangumi_moe')
cachepth = path.join(workpth, "cache")
temppth = path.join(workpth, "tmp")
mime2ext = {
    "image/jpeg":".jpg",
    "image/png":".png",
    "application/json":".json",
    "application/javascript":".js",
    "text/html":".html",
    "text/plain":".txt",
    "image/gif":".gif"
}
def ensure_directory(pth):
    dir = path.dirname(pth)
    if(not path.exists(dir)):
        os.makedirs(dir)
    return pth