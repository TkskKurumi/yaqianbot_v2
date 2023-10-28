from ..backend.configure import bot_config
from ..utils.candy import simple_send
from pil_functional_layout.widgets import RichText
import requests
import qrcode
from PIL import Image, ImageFilter
import numpy as np
from io import BytesIO  
from ..utils.image.colors import image_colors
if("DIFFUSION_HOST_V3" in bot_config):
    HOST = bot_config.get("DIFFUSION_HOST_V3").strip("[/~～]")
else:
    HOST = "http://localhost:8002"
def img2bio(img):
    bio = BytesIO()
    if("A" not in img.mode):
        img.save(bio, "JPEG")
    else:
        img.save(bio, "PNG")
    bio.seek(0)
    return bio

def mask_by_light(lo, hi):
    def inner(image, beta=1):
        arr = np.array(image.convert("L")).astype(np.float16)/255
        arr = lo+arr*(hi-lo)
        arr = (arr*255*beta).astype(np.uint8)
        return Image.fromarray(arr)
    return inner
def mask_by_border(lo, hi, radratio=0.01):
    def inner(image: Image.Image, beta=1):
        w, h = image.size
        rad = ((w*w+h*h)**0.5)*radratio
        print('radius', rad)
        im_blur = image.filter(ImageFilter.GaussianBlur(rad))
        arr0 = np.array(image).astype(np.float32)
        arr1 = np.array(im_blur).astype(np.float32)
        diff = arr0-arr1
        diff = np.sqrt((diff**2).sum(axis=-1))
        diff = (diff-diff.min())/(diff.max()-diff.min())
        diff = diff*(hi-lo)+lo
        diff = diff*255*beta
        ret = Image.fromarray(diff.astype(np.uint8))
        return ret
    return inner

def make_qr_mask(width, height, w, h, b, fdist):
    box_size = width//(w+b*2)
    ret = np.zeros((height, width), np.float16)
    min_beta, max_beta = sorted([fdist(0), fdist(1)])
    def dist_center(x, y, w, h):
        xdist = (x-w/2)/(w/2)
        ydist = (y-h/2)/(h/2)
        l2 = (xdist**2+ydist**2)**0.5
        return min(l2, 1)
    def is_critical_7_1(x, y):
        x_b = x==0 or x==6
        x_c = 2<=x and x<5
        y_b = y==0 or y==6
        y_c = 2<=y and y<5
        if (x_b and y_b):
            return False
        if ((x_b or x_c) and (y_b or y_c)):
            return True
        return False
        
    def is_critical_7_0(x, y):
        x_border = x==0 or x==6
        y_border = y==0 or y==6
        x_center = x==3
        y_center = y==3
        if(x_center and y_center):
            return True
        if (x_border and y_border):
            return False
        if (x_border or y_border):
            return True
        return False

    is_critical_7 = is_critical_7_0

    def is_anchor_box(x, y, w, h):
        def is_critical_5(x, y):
            x_border = x==0 or x==4
            y_border = y==0 or y==4
            x_center = x==2
            y_center = y==2
            if(x_center and y_center):
                return True
            if (x_border or y_border):
                return True
            return False
        if (x<7 and y<7):
            box = (x, y, 7)
            return box, is_critical_7(x, y)
        elif (x<7 and y>=h-7):
            box = (x, y-(h-7), 7)
            return box, is_critical_7(x, y-(h-7))
        elif (x>=w-7 and y<7):
            box = (x-(w-7), y, 7)
            return box, is_critical_7(x-(w-7), y)
        elif (w-9<=x and x<w-4 and h-9<=y and y<h-4):
            box = x-(w-9), y-(h-9), 5
            return box, is_critical_5(x-(w-9), y-(h-9))
        return False, False
    for pix_y in range(height):
        y = pix_y//box_size-b
        for pix_x in range(width):
            x = pix_x//box_size-b
            if (x<0 or y<0 or x>=w or y>=h):
                ret[pix_y, pix_x] = min_beta
            else:
                anchor, crit = is_anchor_box(x, y, w, h)
                if (anchor and crit):
                    ret[pix_y, pix_x] = min_beta+(max_beta-min_beta)*0.78
                else:
                    dist = dist_center(pix_x%box_size, pix_y%box_size, box_size, box_size)
                    beta = fdist(dist)
                    ret[pix_y, pix_x] = beta
    return Image.fromarray((ret*255).astype(np.uint8))

            




def make_qr_with_mask(data, box_size=10, border=1, fdist=lambda x:1-x, l1=False):
    qrmake = qrcode.QRCode(
        error_correction=qrcode.constants.ERROR_CORRECT_M,
        box_size=box_size,
        border=border
    )
    qrmake.add_data(data)
    qr = qrmake.make_image().convert("L")
    width, height = qr.size
    w, h = width//box_size-border*2, height//box_size-border*2
    mask = make_qr_mask(width, height, w, h, border, fdist)
    return qr, mask

def get_upload_id(image, HOST):
    bio = img2bio(image)
    url = HOST+"/upload_image"
    r = requests.post(url, files={"data": bio})
    return r.json()["data"]["img_id"]

def recolor(qr, black, white):
    arr = np.array(qr.convert("L")).astype(np.float32)/255
    h, w = arr.shape
    arr = arr.reshape((h, w, 1))
    ret = np.zeros((h, w, 3), np.float32)
    ret = arr*white - arr*black + black
    return Image.fromarray(ret.astype(np.uint8))

def show_palette(message, img):
    colors = image_colors(img, 10, weight_by_s=False, return_type="numpy")
    mes = []
    for idx, i in enumerate(colors):
        c = tuple(int(j) for j in i)[:3]
        mes.append(Image.new("RGB", (32, 32), c))
        mes.append(str(c)+"\n")
    simple_send(RichText(mes, width=512, bg=(255,)*3, fill=(0,)*3).render())


def do_qr_ld(prompt, data, black=(0, 0, 0), white=(255, 255, 255), width=512, height=768, qrsize=None, mode=None, beta0=0.85, beta1=0.65, xpos=1, ypos=1, post_process_mask=None, preserve_mean=True, HOST="http://localhost:8000"):
    def fdist(x):
        nonlocal beta0, beta1
        st, ed = 1/3, 19/20
        if (x<st):
            return beta0
        elif(x>ed):
            return beta1
        else:
            x = (x-st)/(ed-st)
            return beta0+(beta1-beta0)*x
        


    qr, mask = make_qr_with_mask(data, fdist=fdist)
    
    if (mode is None):
        mode = "qr"
    if (mode == "light"):
        mask = mask_by_light(beta0, beta1)(qr)
    elif (mode == "border"):
        mask = mask_by_border(beta0, beta1)(qr)
    
    qr = recolor(qr, black, white)

    if (qrsize is None):
        qrsize = int(width*0.35)
    else:
        qrsize = int(min(width, height)*qrsize)
    
    

    qr = qr.resize((qrsize, qrsize), Image.LANCZOS)
    mask = mask.resize((qrsize, qrsize), Image.LANCZOS)

    if (callable(post_process_mask)):
        mask = post_process_mask(mask)

    qr_id = get_upload_id(qr, HOST)
    mask_id = get_upload_id(mask, HOST)
    
    make_dict = lambda **kwargs: kwargs
    postj = make_dict(
        width=width,
        height=height,
        layers=[
            make_dict(prompt=prompt),
            make_dict(image=qr_id, beta_mask=mask_id, xpos=xpos, ypos=ypos, blend_mode="")
        ]
    )

    r = requests.post(HOST+"/layered_diffusion", json=postj)

    j = r.json()
    img = j["data"]["image"]

    r = requests.get(HOST+"/images/"+img)
    bio = BytesIO()
    bio.write(r.content)
    bio.seek(0)
    im = Image.open(bio)
    return im