
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
            elif((not cur)and i == '"'):
                in_quote = True
            else:
                cur.append(i)
        elif(in_quote):
            if(prev=="\\"):
                cur.append(i)
            else:
                if(i=='"'):
                    in_quote = False
                    if(cur):
                        ret.append("".join(cur))
                        cur = list()
                elif(i=="\\"):
                    pass
                else:
                    cur.append(i)
            prev = i
    if(cur):
        ret.append("".join(cur))
    return ret
if(__name__=="__main__"):
    print(split(r'new foo bar "hello\"azhe" -shoot'))