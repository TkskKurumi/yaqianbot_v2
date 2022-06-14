from typing import Dict, List
from PIL import Image, ImageDraw
from pil_functional_layout.widgets import Text, Row, Column, RichText, CompositeBG, addBorder
from ..image.background import triangles
from ..image.colors import Color
from ..image.print import print_colors
from ..candy import print_time
from ..image import process, sizefit, image_colors, adjust_L
from ...backend import requests
from .mania_difficulty import Chart as Chart4k
from .user import User


def hex_masked_image(img):
    w, h = img.size
    meow = h/2/(3**0.5)
    mask = Image.new("L", (w, h), 0)
    dr = ImageDraw.Draw(mask)
    points = [(0, h/2), (meow, 0), (w-meow, 0),
              (w, h/2), (w-meow, h), (meow, h)]
    dr.polygon(points, fill=255)
    ret = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    ret.paste(img, mask=mask, box=(0, 0))
    return ret


def comma_number(n, spl=3):
    meow = []
    de = int(10**spl)
    while(int(n)):
        meow.append(str(int(n % de)))
        n /= de
    return ",".join(meow[::-1])


def rank_badge(rank="S", font_size=28):
    if(rank == "S"):
        bgc = (0, 170, 180)
        fgc = (255, 222, 22)
        sdc = Color.from_hsl(240, 1, 0.7)
    elif(rank == "SH"):
        rank = "S"
        bgc = (0, 170, 180)
        fgc = Color.from_hsl(180, 0.9, 0.95)
        sdc = (0, 100, 100)
    elif(rank == "X"):
        rank = "SS"
        bgc = (190, 10, 140)
        fgc = Color.from_hsl(180, 0.9, 0.95)
        sdc = (0, 100, 100)
    elif(rank == "A"):
        bgc = (130, 212, 26)
        fgc = (0, 80, 30)
        sdc = Color.from_hsl(80, 1, 0.95)
    elif(rank == "B"):
        bgc = (230, 180, 58)
        fgc = (60, 30, 30)
        sdc = Color.from_hsl(43, 1, 0.95)
    else:
        bgc = (140, 30, 180)
        fgc = (0, 0, 0)
        sdc = (255, 255, 255)
    bgc = Color(*bgc)
    fgc = Color(*fgc)
    sdc = Color(*sdc)
    # print(sdc)
    t = Text(rank, fill=fgc.get_rgba(), fontSize=font_size).render()
    t = process.shadow(t, radius=font_size/20, color=sdc)
    h = t.size[1]
    bg = triangles(int(h*1.5), h, [bgc], n=5, m=5)
    w, h = bg.size
    bg = hex_masked_image(bg)

    t1 = Image.new("RGBA", bg.size, (0, 0, 0, 0))
    t1.paste(t, box=((w-t.size[0])//2, -h//10), mask=t)
    bg.alpha_composite(t1)
    return bg


def normalize_sum(ls, sm, dtype=int):
    rate = sm/sum(ls)
    return [dtype(i*rate) for i in ls]


def illust_score(score: Dict, size=1080, style="dark"):
    w = size
    h = int(size*0.15)

    # beatmap cover
    bmset = score["beatmapset"]
    bm = score["beatmap"]
    try:
        cover = requests.get_image(bmset["covers"]["slimcover"])[1]
    except Exception:
        cover = requests.get_image(bmset["covers"]["cover"])[1]
    # colors
    colors = image_colors(cover, 3)
    c = colors[0]
    # print_colors(colors)
    color_main = c.replace(S=0.8, L=0.5)
    if(style == "dark"):
        color_bg = c.replace(L=0.15)
        color_fg = c.replace(L=0.9)
        cover = adjust_L(cover, -0.5)
    else:
        color_bg = c.replace(L=0.85)
        color_fg = c.replace(L=0.1)
        cover = adjust_L(cover, 0.5)

    # rank badge
    rank = score["rank"]
    rnk_bdg = rank_badge(rank, font_size=h)
    rnk_bdg = sizefit.fix_height(rnk_bdg, h)

    # prepare font size
    fnts = [1, 0.8, 0.6, 0.5]
    fnts = normalize_sum(fnts, h)
    fnt_size_major, fnt_size_minor, fnt_size_extra, gap = fnts
    # fnt_size_major, fnt_size_minor, fnt_size_gap = normalize_sum(fnts, h)

    # widgets
    f = lambda text, **kwargs: text
    textrender_major = Text(f, fontSize=fnt_size_major,
                            fill=color_fg.get_rgba())
    textrender_minor = Text(f, fontSize=fnt_size_minor,
                            fill=color_fg.get_rgba())
    textrender_extra = Text(f, fontSize=fnt_size_extra,
                            fill=color_fg.get_rgba())
    text_title = bmset["title"]
    text_title = textrender_major.render(text=text_title)

    if(score.get("pp")):
        text_pp = "%.1f pp" % score.get("pp")
    else:
        text_pp = "UNRANKED"
    text_pp = textrender_major.render(text=text_pp)
    line1 = Row([text_title, text_pp], width=w)

    star_rating = bm["difficulty_rating"]
    text_score = comma_number(score["score"])
    acc = score["accuracy"]
    text_ver = bm["version"]
    if(score.get("mods")):
        text_ver+="["+", ".join(score["mods"])+"]"
    text_version=textrender_minor.render(text=text_ver)
    text_scores = "%.2f*, %s/%.2f%%" % (star_rating, text_score, acc*100)
    text_scores = textrender_minor.render(text=text_scores)

    text_type = "%s%d" % (score["type"].upper(), score["type_idx"])
    text_type = textrender_minor.render(text=text_type)
    line2 = [text_version, text_scores, text_type]
    line2 = Row(line2, width=w)

    line3 = []
    if(bm["mode"] == "mania" and bm["cs"] == 4):
        # mania4k
        
        chart = Chart4k.from_osu_id(bm["id"], dt="DT" in score["mods"])
        with print_time("illust_score - calc 4k difficulty"):
            UNUSED, diff_all = chart.calc_all()
        labels = list(diff_all)
        for i in ["Overall", "Jackish", "Streamish"]:
            labels.remove(i)
        ls = sorted([(diff_all[label], label) for label in labels])[-3:]
        ls = ["%s: %.1f" % (label, diff) for diff, label in ls]
        text_4kdiff = ", ".join(ls)
        text_4kdiff = textrender_extra.render(text=text_4kdiff)
        line3.append(text_4kdiff)

    lines = [line1, line2]
    if(line3):
        lines.append(Row(line3, width=w))
    lines = Column(lines, height=h)
    lines = lines.render()

    row = Row([rnk_bdg, lines], borderWidth=gap).render()
    row = process.shadow(row, radius=size/600, color=color_main.replace(A=128))
    w, h = row.size
    border = Image.new("RGBA", (w+gap*2+h//4, h+gap*2), color_fg.get_rgba())
    border = hex_masked_image(border)
    bg = sizefit.fit_crop(cover, w+gap+h//4, h+gap)
    bg = hex_masked_image(bg)
    border.paste(bg, box=(gap//2, gap//2), mask=bg)
    border.paste(row, box=(gap, gap), mask=row)
    return border


def get_flag(country_code, forbidden_cc=None, text_kwargs=None):
    if(forbidden_cc is None):
        forbidden_cc = {"TW": "Taiwan", "HK": "Hong Kong"}
    if(text_kwargs is None):
        text_kwargs = dict()
    if(country_code in forbidden_cc):
        img = Text(forbidden_cc[country_code], **text_kwargs).render()
        # return img
    else:
        img = requests.get_image(
            r"https://assets.ppy.sh/old-flags/%s.png" % (country_code,))[1]
        img = sizefit.fix_height(img, text_kwargs.get("fontSize", 72))
        # TODO: use svg like following
        # svg = https://osu.ppy.sh/assets/images/flags/1f1e8-1f1f3.svg
    return img


call_kwargs = lambda **kwargs: kwargs


def illust_user(user: User, style="dark", size=1080, mode=None):
    user.get_info(mode = mode)
    bests = user.get_scores("best", mode = mode)
    recents = user.get_scores("recent", mode = mode)
    stat = user.info.statistics
    # sizes
    avatar_size = size//5
    ls = [1, 0.8, 0.2, 0.3]
    fnt_major, fnt_minor, fnt_mini, gap = normalize_sum(ls, avatar_size)

    avatar = user.info.avatar_url
    avatar = requests.get_image(avatar)[1]
    avatar = sizefit.fix_height(avatar, avatar_size)
    cover = requests.get_image(user.info.cover_url)[1]

    with print_time("illust_user - colors", enabled=False):
        # colors
        c = image_colors(avatar, 1, weight_by_s=True)[0]
        cover_colors = image_colors(cover, 10)
        color_main = c.replace(S=0.8, L=0.5)
        print_colors([c, color_main])
        if(style == "dark"):
            color_bg = c.replace(L=0.15)
            color_fg = c.replace(L=0.9)
            cover_colors = [c.replace(L=0.15) for c in cover_colors]
            cover = adjust_L(cover, -0.5)
        else:
            color_bg = c.replace(L=0.85)
            color_fg = c.replace(L=0.1)
            cover_colors = [c.replace(L=0.85) for c in cover_colors]
            cover = adjust_L(cover, 0.5)


    # text
    ftext_content = lambda text, **kwargs: text

    def ftext(fontSize): return Text(ftext_content,
                                     fontSize=fontSize, fill=color_fg.get_rgba())
    text_major = ftext(fnt_major)
    text_minor = ftext(fnt_minor)
    text_mini = ftext(fnt_mini)

    line1 = []
    text_username = user.info.username
    text_username = text_major.render(text=text_username)
    line1.append(text_username)
    tmp = call_kwargs(fontSize=fnt_major, fill=color_fg.get_rgba(), bg=color_main.replace(S=0.2).get_rgba())  # nopep8
    flag = get_flag(user.info.country_code, text_kwargs=tmp)
    line1.append(flag)
    line1 = Row(line1, width=size)

    line2 = []
    if(stat.get("global_rank") is None):
        # no available rank
        pass
    else:
        pp = stat.pp
        world_rank = stat.global_rank
        country_rank = stat.country_rank
        tmp = call_kwargs(fontSize=fnt_minor, fill=color_fg.get_rgba(), bg=color_main.replace(S=0.2).get_rgba())  # nopep8
        tmp1 = call_kwargs(fontSize=fnt_minor, fill=color_fg.get_rgba())  # nopep8
        flag = get_flag(user.info.country_code, text_kwargs=tmp)
        t = "%.1f pp" % pp
        text_pp = text_minor.render(text=t)
        text_rank = RichText(["#%d / " % world_rank, flag, "#%d" % country_rank],
                             dontSplit=True, autoSplit=False, width=size, **tmp1)
        # t = "#%d / %s-#%d" % (world_rank, user.info.country_code, country_rank)
        # text_rank = text_minor.render(text=t)
        flag = get_flag(user.info.country_code, text_kwargs=tmp)
        line2.extend([text_pp, text_rank])
    if(line2):
        line2 = Row(line2, width=size)
    lines = [line1, line2]
    lines = [i for i in lines if i]

    col = Column(lines, height=avatar_size)
    row = Row([avatar, col], borderWidth=gap, outer_border=True)
    # row = addBorder(row, borderWidth=gap)
    row = CompositeBG(row, bg = cover)
    # print_colors(color_fg)
    # return row.render()
    columns = [row]
    for i in recents[:3]:
        
        with print_time("illust_user - illust_score"):
            im_score = illust_score(i, size=int(size*0.95))
            columns.append(im_score)
    for i in bests[:10]:
        with print_time("illust_user - illust_score"):
            im_score = illust_score(i, size=int(size*0.95))
            columns.append(im_score)
        
    col = Column(columns, alignX=0.6, borderWidth = gap, outer_border=True).render()
    # col = addBorder(col, borderWidth = gap).render()
    w, h = col.size
    img = triangles(w, h, cover_colors)
    ret = CompositeBG(col, img)
    # return col.render()
    return ret.render()

if(__name__ == "__main__"):
    from .user import User
    ids = ["TkskKurumi"]# , "[Crz]Ha0201"]
    for id in ids:
        u = User(id)
        # im = illust_score(scores[0], style="dark")
        from ..image.print import image_show_terminal
        # image_show_terminal(im,ra)
        im = illust_user(u, mode="osu", size=800)
        image_show_terminal(im, rate=1/len(ids))
