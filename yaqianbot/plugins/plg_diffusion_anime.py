from ..backend.cqhttp.message import CQMessage
from ..backend.receiver_decos import *
from ..backend import receiver
from . import plg_diffusion_v3
from .plg_diffusion_v3 import PPv2, HOST
from .plg_diffusion_v3_ld import do as do_ld
from .plg_diffusion_v3_ld import get_upload_id 
from ..utils.parse_args import parse_args
from ..utils.make_gif import make_gif
from ..utils.candy import simple_send
import numpy as np
import tqdm
from PIL import Image
def PP(message, t):
    s, o = plg_diffusion_v3.get_user_entries(message.sender.id)
    return PPv2(t, s+o)

def gen_noise(w=512, h=512, seed = None):
    shape = (h//8, w//8, 4)
    if (seed is not None):
        np.random.seed(seed)
    return np.random.normal(size=shape)

def noise_as_im(arr):
    def scale_min_max(arr: np.ndarray, lo=0, hi=1):
        mn = arr.min()
        mx = arr.max()
        return (arr-mn)/(mx-mn)*(hi-lo)+lo
    arr = scale_min_max(arr, 0, 255).astype(np.uint8)
    img = Image.fromarray(arr)
    return img

# def fast_slow_fast(x):
#     if (x<=0): return 0
#     if (x>=1): return 1
#     if (x<=0.5):
#         return (0.25-(x-0.5)**2)**0.5
#     return 1 - (0.25-(x-0.5)**2)**0.5
def fast_slow_fast(x, n=2):
    if (x<=0): return 0
    if (x>=1): return 1
    if (x==0.5): return 0.5
    x = (x-0.5)*2
    y = abs(x)**(n-1) * x
    return (y+1)/2
for i in range(6):
    print(i/5, fast_slow_fast(i/5))
class PolyScheduler:
    def __init__(self, xys, interp=lambda x:fast_slow_fast(x)):
        self.xys = sorted(xys)
        self.interp = interp
    def __call__(self, x):
        x0, y0 = self.xys[0]
        if (x<=x0):
            return y0
        xend, yend = self.xys[-1]
        if (x>=xend):
            return yend
        for idx, i in enumerate(self.xys):
            x0, y0 = i
            x1, y1 = self.xys[idx+1]
            if (x0<=x and x<=x1):
                x2 = (x-x0)/(x1-x0)
                ratio = self.interp(x2)
                return y0*(1-ratio) + y1*ratio
        assert False
Layer = lambda **kwargs: kwargs
class Frame:
    def __init__(self, message, args, kwargs, noise=None, w=512, h=512, mix = False, mix_w=1):
        self._args = args
        self._kwargs= kwargs
        self._mes = message
        self.mix = mix
        self.mix_w = mix_w

        self.pp = PP(message, " ".join(args))
        self.beta = kwargs.get("-beta", 1)
        if (noise is None):
            self.noise = gen_noise(w, h, kwargs.get("-seed", None))
        else:
            self.noise = noise
    def __mul__(self, other):
        if (isinstance(other, float) or isinstance(other, int)):
            return Frame(self._mes, self._args, self._kwargs, mix=True, mix_w=self.mix_w*other, noise=self.noise*other)
        return NotImplemented
    def __add__(self, other):
        if (isinstance(other, Frame)):
            return [self, other]
        if (isinstance(other, list)):
            ls = list(other) # copy
            ls.append(self)
            return ls
        return NotImplemented
    def as_layers(self):
        return [Layer(prompt=self.pp.raw, beta=self.beta*self.mix_w, noise=self.noise)]
def as_layers(ls):
    if (isinstance(ls, Frame)):
        return ls.as_layers()
    ret = []
    for i in ls:
        ret.extend(i.as_layers())
    return ret
@receiver
@threading_run
@on_exception_response
@command("/test_grad", opts={})
def cmd_diffusion_grad_img2img(message: CQMessage, *args, **kwargs):
    segments = []
    for i in args:
        if (i == "--"):
            segments.append([])
        else:
            if (not segments):
                segments.append([])
            segments[-1].append(i)

    

    opts = {"-common", "-frame", "-fpk", "-loop"}
    ls_opts = {"-common"}
    bool_opts = {"-loop"}

    common_prompts = ["best quality, /*lowres, */"]
    n_frames = None
    loop = False
    for i in segments:
        args, kwargs = parse_args(i, opts, ls_opts=ls_opts, bool_opts=bool_opts)
        print("DEBUG", args, kwargs)
        if ("-common" in kwargs):
            common_prompts.append(" ".join(kwargs["-common"]))
        if ("-frame" in kwargs):
            n_frames = int(kwargs["-frame"])
        if ("-fpk" in kwargs):
            n_frames = int(kwargs["-fpk"]) * (len(segments)-1) + 1
        if ('-loop' in kwargs):
            loop = True
    common_prompts = ", ".join(common_prompts)

    w, h = 512, 640

    keyframes = []
    for i in segments:
        args, kwargs = parse_args(i, opts, ls_opts=ls_opts, bool_opts=bool_opts)
        if (common_prompts):
            args.append(", "+common_prompts)
        print("DEBUG added common", args, kwargs)
        keyframes.append(Frame(message, args, kwargs, w=w, h=h))
    
    if (len(keyframes)<=1):
        simple_send("不够")
        return
    
    sched = PolyScheduler([(idx, i) for idx, i in enumerate(keyframes)])
    
    frames = []
    if (n_frames is None):
        n_frames = 7*(len(keyframes)-1)

    if (loop):
        sched = PolyScheduler([(idx, i) for idx, i in enumerate(keyframes)] + [(len(keyframes), keyframes[0])])
        n_frames = round(n_frames/(len(keyframes)-1)*len(keyframes))

    fps = 3
    for i in tqdm.trange(n_frames):
        progress = i/(n_frames-1)
        x = progress*sched.xys[-1][0]

        layers = sched(x)
        layers = as_layers(layers)
        noise = 0
        for i in layers:
            noise += i.get("noise", 0)
            i.pop("noise")
        noise_im = noise_as_im(noise)
        noise_im = get_upload_id(noise_im, HOST=HOST)
        print("DEBUG", "layers:", layers, "noise:", noise_im)
        frames.append(do_ld(width=512, height=640, layers=layers, raw_beta=False, use_noise=noise_im))
    gif = make_gif(frames, fps=fps, frame_area_sum=384*384*15)
    simple_send(gif)
