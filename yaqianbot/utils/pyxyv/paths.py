from os import path
import tempfile
import os


workpth = os.environ.get("PIXIV_PATH") or path.join(tempfile.gettempdir(),'pyxyv')
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
def myhash(x, length = 50):
    mask = (1<<length)-1
    offset = 7
    if(isinstance(x, list) or isinstance(x, tuple)):
        ret = 0
        for i in x:
            ret = ret<<offset
            ret ^= myhash(i, length=length)
            ret = (ret>>length)^(ret&mask)
        return ret
    elif(isinstance(x, str)):
        ls = list(x.encode("utf-8"))
        return myhash(ls, length=length)
    elif(isinstance(x, int)):
        ret = x
        while(ret>>length):
            ret = (ret>>length)^(ret&mask)
        return ret
    elif(isinstance(x, dict)):
        ls = ["dict:"]
        for key in sorted(x):
            ls.extend([key, x[key]])
        return myhash(ls, length=length)
    else:
        raise TypeError(type(x))
    

def base32(x, length = 10):
    if(not isinstance(x, int)):
        x = myhash(x, length = length*5)
    chr = "0123456789abcdefghijklmnopqrstuvwxyz"
    ret = []
    for i in range(length):
        ret.append(chr[x & 0b11111])
        x>>=5
    return "".join(ret[::-1])
if(__name__=="__main__"):
    print(myhash("foo"))
    print(myhash("fop"))
    print(myhash("foobar"))
    print(myhash("foobas"))
    print(myhash({"foo":"bar"}))
    azhe = ["abc", {"foo":"bar"}, 123, list(range(3))]
    print(myhash(azhe))
    a = []
    a.append(a)
    # print(myhash(a))
