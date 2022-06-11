
def split(str):
    ret = []
    cur = []
    in_quote = False
    prev = None
    for i in str:
        if(not in_quote):
            if(i == " "):
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


def parse_args(str, options, boolean_opt = None):
    if(boolean_opt is None):
        boolean_opt = set()
    splited = split(str)
    args = []
    kwargs = dict()
    kw = None
    for i in splited:
        if(i.startswith("-") and (i in options)):
            if(i in boolean_opt):
                kwargs[i]=True
            elif(kw is not None):
                kwargs[kw] = True
            else:
                kw = i
        else:
            if(kw is not None):
                kwargs[kw] = i
                kw = None
            else:
                args.append(i)
    if(kw is not None):
        kwargs[kw] = True
    return args, kwargs


if(__name__ == "__main__"):
    print(split(r'new foo bar "hello\"azhe" -shoot'))
    options = {"-i", "--input", "-y"}
    bool_opt = {"-y"}
    string = "   -i meow --input   meow meow -y foo bar"
    print(parse_args(string, options, bool_opt))
    # print(parse_args(r'-i "/tmp/path hase space.jpg" -y foo.jpg', boolean_args={"-y"}))
