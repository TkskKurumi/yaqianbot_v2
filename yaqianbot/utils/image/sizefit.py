from PIL import Image


def _wh_fit_area(w, h, area):
    rate = (area/w/h)**0.5
    return int(w*rate), int(h*rate)


def area(img: Image.Image, area):
    w, h = img.size
    a = w*h
    rate = (area/a)**0.5
    w, h = _rate(w, h, rate)
    return img.resize((w, h), Image.BILINEAR)


def _rate(w, h, rate, rateh=None):
    if(rateh is None):
        rateh = rate
    return int(w*rate), int(h*rateh)


def resize_ratio(img, rw=1, rh=None):
    if(rh is None):
        rh = rw
    w, h = img.size
    w, h = _rate(w, h, rw, rh)
    return img.resize((w, h), Image.Resampling.BILINEAR)


def fix_width(img, width=None):
    w, h = img.size
    w, h = _rate(w, h, rate=width/w)
    return img.resize((w, h), Image.Resampling.BILINEAR)


def fix_height(img, height=None):
    w, h = img.size
    w, h = _rate(w, h, rate=height/h)
    return img.resize((w, h), Image.Resampling.BILINEAR)


def fit_shrink(img, width=None, height=None):
    w, h = img.size
    rate = 1
    if(width is not None):
        rate = min(width/w, rate)
    if(height is not None):
        rate = min(height/h, rate)
    w, h = _rate(w, h, rate)
    return img.resize((w, h), Image.Resampling.BILINEAR)


def fit_expand(img: Image.Image, width: int, height: int, align_x: float = 0.5, align_y: float = 0.5, bg=None):
    w, h = img.size
    if(w*height > width*h):
        rate = width/w
        w = width
        h = int(rate*h)
    else:
        rate = height/h
        w = int(rate*w)
        h = height
