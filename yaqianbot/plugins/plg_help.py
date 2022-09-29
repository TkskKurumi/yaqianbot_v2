from ..backend.receiver_decos import on_exception_response, threading_run, command, all_commands
from ..backend.cqhttp.message import CQMessage
from ..backend import receiver
from ..utils.candy import simple_send
from ..utils.image.background import triangles
from ..utils.image import sizefit
from pil_functional_layout.widgets import RichText, CompositeBG, AddBorder
import importlib
from PIL import Image
__all__ = ["plugin", "plugin_func", "plugin_func_option",
           "OPT_OPTIONAL", "OPT_NOARGS"]

OPT_OPTIONAL = 1 << 0
OPT_NOARGS = 1 << 1
INDENT = " "*2


def add_indent(s, n = 1):
    return '\n'.join([INDENT*n+i for i in s.split('\n')])


class plugin_func_option:
    def __init__(self, name, desc, arg_desc = None, type=0):
        self.type = type
        self.name = name
        self.desc = desc
        self.arg_desc = arg_desc
    def __str__(self):
        if(self.type & OPT_NOARGS):
            arg_desc = ""
        else:
            if(self.arg_desc is None):
                arg_desc = " [参数]"
            else:
                arg_desc = " "+self.arg_desc
        # ret ="f"{self.name}{"" if self.type&OPT_NOARGS else " 参数"}\n{INDENT}{self.desc}""
        ret = f"""{self.name}{arg_desc}\n{INDENT}{self.desc}"""
        return ret


class plugin_func:
    def __init__(self, name, desc = None):
        self.name = name
        self.opts = []
        self.desc = desc
    def __str__(self):
        ret = [self.name]
        if(self.desc):
            
            ret.append(INDENT + "说明:")
            ret.append(add_indent(self.desc, 2))
        if(self.opts):
            ret.append(INDENT+"选项:")
            for i in self.opts:
                ret.append(add_indent(str(i), 2))
        return "\n".join(ret)

    def append(self, x):
        self.opts.append(x)


plugins = {}


def rand_img(message: CQMessage):
    if("pixiv" in plugins):
        return Image.open(plugins["pixiv"].module.rand_img(message))
    elif("PIXIV" in plugins):
        return Image.open(plugins["PIXIV"].module.rand_img(message))
    else:
        return triangles(512, 288)


class plugin:
    def __init__(self, name, friendly_name=None):
        self.name = name
        if(friendly_name is None):
            friendly_name = name
        self.friendly_name = friendly_name
        self.funcs = []
        plugins[self.friendly_name.lower()] = self

    def append(self, x):
        self.funcs.append(x)

    @property
    def module(self):
        print(self.name)
        # ret = __import__(self.name)
        ret = importlib.import_module(self.name)
        print(ret)
        return ret

    def __str__(self):
        ret = [self.friendly_name]
        if(self.funcs):
            for i in self.funcs:
                ret.append(add_indent(str(i)))
        else:
            ret.append(INDENT+"此帮助主题下没有任何内容，奇怪咧？")
        return "\n".join(ret)


def list_insert(ls, *elements):
    ret = []
    for idx, i in enumerate(ls):
        if(idx):
            ret.extend(elements)
        ret.append(i)
    return ret

@receiver
@threading_run
@on_exception_response
@command("/指令列表", opts={})
def cmd_command_list(message: CQMessage, *args, **kwargs):
    mes = []
    for fn, i in all_commands.items():
        pattern, func = i
        mes.append(pattern)
    simple_send(", ".join(mes))
@receiver
@threading_run
@on_exception_response
@command("/help", opts={})
def cmd_help(message: CQMessage, *args, **kwargs):
    mes = []
    if(not args):
        for name, plg in plugins.items():
            nm = plg.friendly_name
            mes.append("输入/help %s查看%s帮助" % (nm, nm))
        if(not plugins):
            mes.append("没有任何帮助主题，奇怪咧")
    else:
        for name in args:
            if(name in plugins):
                mes.append(str(plugins[name]))
            else:
                found = None
                for plg_nm, plg in plugins.items():
                    if(name.lower() == plg_nm.lower()):
                        found = plg
                if(found is not None):
                    mes.append(str(found))
                else:
                    mes.append("未知帮助%s" % name)
    fs = 22
    width = 520
    bg = (255, 255, 255, 180)
    fg = (0, 0, 0, 255)
    mes = "\n".join(mes)
    
    RT = RichText([mes], fontSize=fs, width=width,
                  autoSplit=False, dontSplit=False, bg=bg, fill=fg, alignX=0)
    RT = RT.render()
    RT = AddBorder(RT, borderWidth=fs, borderColor=bg).render()
    w, h = RT.size
    if(h < w):
        RT = sizefit.fit_expand(RT, w, w, bg=(0, 0, 0, 0))

    RT = CompositeBG(RT, rand_img(message))
    message.response_sync(RT.render())
__all__ = ["plugin_func_option", "plugin", "plugin_func", "OPT_NOARGS"]