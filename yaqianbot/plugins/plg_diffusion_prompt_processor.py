from ..utils.candy import lockedmethod, simple_send, locked
from ..backend.receiver_decos import *
from ..backend import receiver
from ..backend.cqhttp.message import CQMessage
from functools import partial
import random
FLOAT_PATTERN = r"\d+\.?\d*"
def split(prompt):
    specials = r"sample\(%s|regex\(|merge\(%s|\)|%s\*\(|\("%(FLOAT_PATTERN, FLOAT_PATTERN, FLOAT_PATTERN)
    prompts = re.split(specials, prompt)
    operators = re.findall(specials, prompt)
    ls = []
    for idx, i in enumerate(prompts):
        if(idx):
            ls.append((1, operators[idx-1]))
        ls.append((0, i))
    return ls

def translated(prompt, entries):
    def _process_symbols(prompt: str) -> str:
        prompt = prompt.replace("\uff0c", ',')
        prompt = re.sub("[ ,]*,[ ,]*", ", ", prompt)
        prompt = prompt.strip(", ")
        return prompt
    ret = prompt
    if(entries):
        SEP = ", "    
        entries.sort(key=lambda x: -len(x[0]))
        for k, v in entries:
            if(k in ret):
                ret = ret.replace(k, SEP+v+SEP)
    return _process_symbols(ret)
class NoopProcessor:
    def __init__(self, prompt, entries=None):
        self.tags = []
        prompt = translated(prompt, entries)
        ops = r"/\*|\*/|,|\{|\}"
        tags = re.split(ops, prompt)
        ops = re.findall(ops, prompt)
        w = 1
        for idx, i in enumerate(tags):
            if(idx):
                op = ops[idx-1]
                if(op == "/*"):
                    w*=-1
                elif(op == "*/"):
                    w*=-1
                elif(op == "{"):
                    w*=1.1
                elif(op == "}"):
                    w/=1.1
            if(i.strip()):
                self.tags.append((w, i))
class RegexProcessor:
    def __init__(self, ls, tags, entries):
        self.tags = []
        p = [j for i, j in ls]
        p = "".join(p)
        pattern = re.compile(p)
        for k, v in entries:
            if(pattern.match(k)):
                self.tags.extend(NoopProcessor(v).tags)
        self.tags.extend(tags)
class MergeProcessor:
    def __init__(self, n, ls, tags, entries=None, shuffle=False):
        ls = ListProcessor(ls, entries=entries)
        sm = 0
        for w, t in ls.tags+tags:
            sm += abs(w)
        self.tags = []
        for w, t in ls.tags+tags:
            self.tags.append((w/sm*n, t))
        if(shuffle):
            random.shuffle(self.tags)
class LBracProcessor:
    def __init__(self, ls, tags, entries=None):
        isop, t = ls[0]
        assert(not isop)
        ls[0] = (isop, "("+t)
        isop, t = ls[-1]
        assert(not isop)
        ls[-1] = (isop, t+")")
        
        self.tags = ListProcessor(ls, entries=entries).tags+tags
class MultProcessor:
    def __init__(self, n, ls, tags, entries=None):
        ls = ListProcessor(ls, entries=entries)
        self.tags = []
        for w, t in ls.tags+tags:
            self.tags.append((w*n, t))
class ListProcessor:
    def extend(self, pro):
        self.tags.extend(pro.tags)
    def __init__(self, ls, entries=None):
        self.tags = []
        stack = []
        for isop, i in ls:
            if(isop):
                if(i!=")"):
                    if(i.startswith("merge(")):
                        num = float(re.findall(FLOAT_PATTERN, i)[0])
                        op = partial(MergeProcessor, num, entries=entries)
                        stack.append((op, [], []))
                    elif(i.startswith("sample(")):
                        num = float(re.findall(FLOAT_PATTERN, i)[0])
                        op = partial(MergeProcessor, num, entries=entries, shuffle=True)
                        stack.append((op, [], []))
                    elif(i.startswith("(")):
                        op = partial(LBracProcessor, entries=entries)
                        stack.append((op, [], []))
                    elif(i.startswith("re")):
                        op = partial(RegexProcessor, entries=entries)
                        stack.append((op, [], []))
                    else:
                        num = float(re.findall(FLOAT_PATTERN, i)[0])
                        op = partial(MergeProcessor, num, entries=entries)
                        stack.append((op, [], []))
                else:
                    if(stack):
                        # print(ls)
                        assert stack, "extra ')' in %s"%ls
                        op, cont, others = stack[-1]
                        br = op(cont, others)
                        stack.pop()
                        if(stack):
                            stack[-1][-1].extend(br.tags)
                        else:
                            self.extend(br)
            elif(stack):
                stack[-1][1].append((isop, i))
            else:
                self.extend(NoopProcessor(i, entries=entries))
        while(stack):
            op, cont, others = stack[-1]
            br = op(cont, others)
            stack.pop()
            if(stack):
                stack[-1][-1].extend(br.tags)
            else:
                self.extend(br)
class PromptProcessor:
    def __init__(self, prompt, entries=None):
        ls = split(prompt)
        lp = ListProcessor(ls, entries=entries)
        self.tags = lp.tags
    @property
    def raw(self):
        d = {}
        i = 0
        j = 0
        for w, t in self.tags:
            j+=abs(w)
            while(i<j):
                d[w] = d.get(w, list())
                d[w].append(t)
                i+=1
        ret = []
        for i, j in d.items():
            _ = ", ".join(j)
            if(i<0):
                _ = "/*"+_+"*/"
            ret.append(_)
        return ", ".join(ret)

        
            



@receiver
@threading_run
@on_exception_response
@command("/测试", opts={})
def cmd_test_pp(message: CQMessage, *args, **kwargs):
    p = " ".join(args)
    PP = PromptProcessor(p)
    ret = str(PP.tags)
    simple_send(ret)