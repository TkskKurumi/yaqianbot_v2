from PIL import Image
from . import colors

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


def fit_crop(img: Image.Image, width, height, align_x=0.5, align_y=0.5):
    w, h = img.size
    if(w*height > width*h):
        # too wide
        img = resize_ratio(img, height/h)
    else:
        img = resize_ratio(img, width/w)
    w, h = img.size
    left = int((w-width)*align_x)
    top = int((h-height)*align_y)
    return img.crop((left, top, left+width, top+height))


def fit_expand(img: Image.Image, width: int, height: int, align_x: float = 0.5, align_y: float = 0.5, bg=None):
    w, h = img.size
    if(w*height>h*width):
        # too wide
        rate = width/w
    else:
        rate = height/h
    if(bg is None):
        bg = colors.image_border_color(img).get_rgba()
    ret = Image.new(img.mode, (width, height), bg)
    img = resize_ratio(img, rate)
    w, h=img.size
    left = int((width-w)*align_x)
    top = int((height-h)*align_y)
    ret.paste(img, (left, top))
    return ret