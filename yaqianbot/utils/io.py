import json
import os
def ensure_dir(filename):
    dir = os.path.dirname(filename)
    if(not os.path.exists(dir)):
        os.makedirs(dir)
    return dir
def loadtext(filename):
    with open(filename, "r", encoding='utf-8') as f:
        ret = f.read()
    return ret
def savetext(filename, content):
    ensure_dir(filename)
    with open(filename, "w", encoding='utf-8') as f:
        f.write(content)
    return filename
def loadjson(filename):
    with open(filename, "r") as f:
        j = json.load(f)
    return j
def savejson(filename, content):
    ensure_dir(filename)
    with open(filename, "w") as f:
        json.dump(content, f)
    return filename