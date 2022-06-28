import numpy as np
def imghash(img, length=10):
    im = img.convert("RGB").resize((10, 10))
    return base32(np.array(im), length=length)
def myhash(x, length = 50):
    mask = (1<<length)-1
    offset = 7
    if(isinstance(x, list) or isinstance(x, tuple) or isinstance(x, np.ndarray) or isinstance(x, bytes)):
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
    elif(np.issubdtype(type(x), np.integer)):
        return myhash(int(x))
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