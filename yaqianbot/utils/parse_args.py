
def split(str):
    ret = []
    cur = []
    in_quote = False
    prev = None
    for i in str:
        if(not in_quote):
            if(i == " " or i == "\n" or i == "\r"):
                if(cur):
                    ret.append("".join(cur))
                    cur = list()
            elif((not cur) and i == '"'):
                in_quote = True
            else:
                cur.append(i)
        elif(in_quote):
            if(prev == "\\"):
                cur.append(i)
            else:
                if(i == '"'):
                    in_quote = False
                    if(cur):
                        ret.append("".join(cur))
                        cur = list()
                elif(i == "\\"):
                    pass
                else:
                    cur.append(i)
            prev = i
    if(cur):
        ret.append("".join(cur))
    return ret


def parse_args(str, options, bool_opts = None, ls_opts = None):
    if(bool_opts is None):
        bool_opts = set()
    if(ls_opts is None):
        ls_opts = set()
    splited = split(str)
    args = []
    kwargs = dict()
    kw = None
    for i in splited:
        if(i.startswith("-") and (i in options)):
            if(i in bool_opts):
                print('set true', i, bool_opts)
                kwargs[i]=True
            else:
                kw = i
        else:
            if(kw is not None):
                
                if(kw in ls_opts):
                    kwargs[kw] = kwargs.get(kw, [])
                    kwargs[kw].append(i)
                else:
                    kwargs[kw] = i
                    kw = None
            else:
                args.append(i)
    
    return args, kwargs


if(__name__ == "__main__"):
    string = "meow -ls aqua aqua(konosuba) -bool -a 0.8 -w 1.5"
    opts = {"-ls", "-bool", "-a"}
    ls_opts = {"-ls"}
    bool_opts = {"-bool"}
    print(parse_args(string, opts, bool_opts, ls_opts))
