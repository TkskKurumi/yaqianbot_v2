import numpy as np
import random, os
from PIL import Image
from PIL import ImageDraw
from ..backend.receiver_decos import on_exception_response, command
from ..backend import receiver, startswith
from ..backend import threading_run
from ..backend.cqhttp import CQMessage
from ..utils.make_gif import make_gif, make_gif_size
from ..utils.candy import simple_send


class Node:
    def __init__(self, up, lo, le, ri, pic):
        self.up = up
        self.lo = lo
        self.le = le
        self.ri = ri
        self.pic = pic
        self.child = [[None, None], [None, None]]
        # print("DEBUG:", "created", self)
    def add_leaf_to(self, ls):
        is_leaf = True
        for i in self.child:
            for j in i:
                if (j is not None):
                    is_leaf = False
                    j.add_leaf_to(ls)
        if (is_leaf and (self.width==self.height)):
            ls.append(self)
    def get_color(self):
        return tuple([int(i) for i in self.pic.get_avg(self.up, self.lo, self.le, self.ri)])
    def __repr__(self):
        c = self.get_color()
        return "<Node %dx%d@%d,%d, color=%s>"%(self.ri-self.le, self.lo-self.up, self.le, self.up, c)
    @property
    def is_leaf(self):
        for i in self.child:
            for j in i:
                if (j is not None):
                    return False
        return True
    @property
    def ax(self):
        return (self.up, self.lo, self.le, self.ri)
    @property
    def width(self):
        return self.ri-self.le
    @property
    def height(self):
        return self.lo-self.up
    @property
    def spl(self):
        width = self.width
        height = self.height
        if (width==height and width==1):
            return 0
        spl = max_2pow(min(width, height))
        if (width==height and spl==width):
            spl = spl//2
        return spl
    def elim_square_recur(self):
        if (self.is_leaf and (self.width!=self.height)):
            self.build_child()
        for i in self.child:
            for j in i:
                if (j is not None):
                    j.elim_square_recur()
    def build_child(self):
        def try_build_node(up1, lo1, le1, ri1):
            heigth1 = lo1-up1
            width1 = ri1-le1
            if (heigth1>0 and width1>0):
                return Node(up1, lo1, le1, ri1, self.pic)
            return None
        spl = self.spl
        if (spl==0):
            return self
        self.child[0][0] = try_build_node(self.up, self.up+spl, self.le, self.le+spl)
        self.child[0][1] = try_build_node(self.up, self.up+spl, self.le+spl, self.ri)
        self.child[1][0] = try_build_node(self.up+spl, self.lo, self.le, self.le+spl)
        self.child[1][1] = try_build_node(self.up+spl, self.lo, self.le+spl, self.ri)
        return self
    def get_child_by_xy(self, x, y):
        spl = self.spl
        if (spl==0):
            return self
        ydx = 0 if y<self.up+spl else 1
        xdx = 0 if x<self.le+spl else 1
        return self.child[ydx][xdx]
    def build_child_by_xy(self, x, y, max_new=0):
        c = self.get_child_by_xy(x, y)
        if (c is self):
            return self
        if (c is None):
            if (max_new<=0):
                return self
            else:
                self.build_child()
                self.elim_square_recur()
                c = self.get_child_by_xy(x, y)
                assert c is not None, "%s, %s"%(self, (x, y))
                return c.build_child_by_xy(x, y, max_new-1)
        else:
            return c.build_child_by_xy(x, y, max_new)

        


def max_2pow(n):
    ret = 1
    while ((ret<<1) <= n):
        ret = ret<<1
    return ret

def split(up, lo, le, ri):
    w = ri-le
    h = lo-up
    if (w==1 and h==1):
        return 0
    
    spl = max_2pow(min(w, h))
    if (spl==w and spl==h):
        return spl//2
    return spl


    


class DotPic:
    def __init__(self, image):
        if (isinstance(image, Image.Image)):
            arr = np.array(image.convert("RGBA"))
        else:
            arr = image
        arr = arr.astype(np.float64)
        presum = arr.copy()
        h, w = arr.shape[:2]
        for y in range(h):
            for x in range(w):
                if (x):
                    presum[y, x] = presum[y, x-1] + presum[y, x]
        for x in range(w):
            for y in range(h):
                if (y):
                    presum[y, x] = presum[y-1, x] + presum[y, x]
        self.arr = arr
        self.presum = presum
        self.root = Node(0, h, 0, w, self)
        self.root.elim_square_recur()
        self.w = w
        self.h = h
    def get_leaf(self):
        ret = []
        self.root.add_leaf_to(ret)
        return ret
    def get_presum(self, lo, ri):
        if (lo<0):
            return 0
        if (ri<0):
            return 0
        return self.presum[lo][ri]
    def get_sum(self, up, lo, le, ri):
        return self.get_presum(lo-1, ri-1)-self.get_presum(lo-1, le-1) - self.get_presum(up-1, ri-1) + self.get_presum(up-1, le-1)
    def get_avg(self, up, lo, le, ri):
        area = (lo-up) * (ri-le)
        return self.get_sum(up, lo, le, ri)/area
        
    def render(self):
        ret = Image.new("RGBA", (self.w, self.h), (0, 0, 0, 0))
        dr = ImageDraw.Draw(ret)
        leafs = self.get_leaf()
        print("Num Leafs", len(leafs))
        for node in leafs:
            up, lo, le, ri = node.up, node.lo, node.le, node.ri
            c = node.get_color()
            dr.ellipse((le, up, ri, lo), fill=c)
        return len(leafs), ret
    
@receiver
@threading_run
@on_exception_response
@command("/dotpic", opts={})
def cmd_dotpic(message: CQMessage, *args, **kwargs):
    if(message.get_reply_image()):
        img = message.get_reply_image()
    else:
        _, img = message.get_sent_images()[0]

    w, h = img.size
    w = w - (w%32)
    h = h - (h%32)
    img = img.resize((w, h))

    dotpic = DotPic(img)
    frames = [dotpic.render()[1]]

    npix    = min(18000, img.width*img.height/8)
    alpha = 1.5
    i = 0
    while (True):
        i += 1
        for j in range(int(alpha**i)):
            x, y = random.randrange(dotpic.w), random.randrange(dotpic.h)
            dotpic.root.build_child_by_xy(x, y, 1)
        n_leaf, pic = dotpic.render()
        frames.append(pic)
        if (n_leaf>npix):
            break
    fps = 8
    frames.extend(frames[-1:] * round(fps*0.5))
    gif = make_gif_size(frames, fps=fps)
    simple_send(gif)
    # simple_send("n_leaf = %d, img_area = %d, npix = %d"%(n_leaf, img.width*img.height, npix))