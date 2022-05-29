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