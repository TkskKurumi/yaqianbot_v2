from typing import Dict, List
import numpy as np
from functools import wraps
from PIL import ImageOps
import random
from PIL import Image, ImageDraw
from pil_functional_layout.widgets import Text, Row, Column, RichText, CompositeBG, AddBorder, Pill
from ..image.background import triangles, unicorn, frandrange, grids
from ..image.colors import Color
from ..image.print import print_colors
from ..candy import print_time, locked
from ..image import process, sizefit, image_colors, adjust_L, adjust_A
from ...backend import requests
from .mania_difficulty import Chart as Chart4k
from .user import User
from ..plot_lock import pyplot_lock
from datetime import datetime
# import requests
from . import credentials
sqrt3 = 3**0.5


def add_hex_bg(img, bg, expandw=1, border=0, offsetY=0, offsetX=0.5, debug=False):
    imw, imh = img.size
    w = imw+imh/sqrt3*expandw+border/sqrt3*4
    h = imh+border*2
    w, h = int(w), int(h)
    if(callable(bg)):
        sized_bg = bg(w, h)
    else:
        sized_bg = sizefit.fit_crop(bg, w, h)
    sized_bg = hex_masked_image(sized_bg)
    pasted = Image.new("RGBA", sized_bg.size, (0, )*4)
    left = int((w-imw)*0.5+offsetX)
    top = int((h-imh)*0.5+offsetY)
    pasted.paste(img, box=(left, top), mask=img)
    if(debug):
        from ..image.print import image_show_terminal
        image_show_terminal(sized_bg, rate=0.5)
        print("sized_bg0")
    sized_bg.alpha_composite(pasted)
    if(debug):
        from ..image.print import image_show_terminal
        image_show_terminal(sized_bg, rate=0.5)
        print("sized_bg1")
    return sized_bg


def illust_om4k(beatmap_id):
    url = "https://osu.ppy.sh/api/v2/beatmaps/%s" % beatmap_id
    r = requests.get(url, headers=credentials.get_header())
    bm = r.json()
    bms = bm["beatmapset"]
    if(bm["mode"] == "mania" and bm["cs"] == 4):
        # mania4k
        chart = Chart4k.from_osu_id(beatmap_id)
    else:
        raise Exception("beatmap is not mania 4k.")

    try:
        cover = requests.get_image(bmset["covers"]["slimcover"])[1]
    except Exception:
        cover = requests.get_image(bmset["covers"]["cover"])[1]
    raise NotImplementedError


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


def cached_func(fkey=lambda *args, **kwargs: args+tuple(kwargs.items())):
    def deco(func):
        cache = dict()

        @wraps(func)
        def inner(*args, **kwargs):
            key = fkey(*args, **kwargs)
            if(key in cache):
                return cache[key]
            else:
                ret = func(*args, **kwargs)
                cache[key] = ret
                return ret
        return inner
    return deco


def normalize_sum(ls, sm, dtype=int):
    rate = sm/sum(ls)
    return [dtype(i*rate) for i in ls]


def judgement(mode, judge, size):
    if(mode == "mania"):
        return mania_judgement(judge, size=size)
    else:
        return osu_judgement(judge, size)


@cached_func()
def osu_judgement(judge, size=48):
    # def mania_judgement(judge, skin="default", size=48):
    t_hue_range = [0, 330]
    t_s = 1
    if(judge == "geki"):
        text = "激"
        t_hue_range = [240-15, 240+15]
    elif(judge == "300"):
        text = "300"
        t_hue_range = [240-15, 240+15]
    elif(judge == "katu"):
        text = "喝"
        t_hue_range = [120-15, 120+15]
    elif(judge == "100"):
        text = "100"
        t_hue_range = [120-15, 120+15]
    elif(judge == "50"):
        text = "50"
        t_s = 0.2
    elif(judge == "miss"):
        text = "X"
        t_hue_range = [-15, 15]
    else:
        raise ValueError("Unknown osu judge %s" % judge)

    colors = []
    for i in range(5):
        le, up = t_hue_range
        h = le+(up-le)/4*i
        l = frandrange(0.4, 1)
        colors.append(Color.from_hsl(h, t_s, l))
    random.shuffle(colors)
    mask = Text(text, fontSize=size, bg=(0, )*4, fill=(255,)*4).render()
    fill = grids(*mask.size, color_h=colors, color_v=colors)
    bg = Image.new("RGBA", mask.size, (0,)*4)
    bg.paste(fill, mask=mask)
    return bg


@cached_func()
def mania_judgement(judge, size=48):
    t_hue_range = [0, 330]
    t_s = 1
    if(judge == "geki" or judge == "300g"):
        text = "300"
    elif(judge == "300"):
        text = "300"
        t_hue_range = [30, 60]
    elif(judge == "katu" or judge == "200"):
        text = "200"
        t_hue_range = [120-15, 120+15]
    elif(judge == "100"):
        text = "100"
        t_hue_range = [240-15, 240+15]
    elif(judge == "50"):
        text = "50"
        t_s = 0.2
    elif(judge == "miss"):
        text = "miss"
        t_hue_range = [-15, 15]
    else:
        raise ValueError("Unknown mania judge %s" % judge)

    colors = []
    for i in range(5):
        le, up = t_hue_range
        h = le+(up-le)/4*i
        l = frandrange(0.4, 0.8)
        colors.append(Color.from_hsl(h, t_s, l))
    random.shuffle(colors)
    mask = Text(text, fontSize=size, bg=(0, )*4, fill=(255,)*4).render()
    fill = grids(*mask.size, color_h=colors, color_v=colors)
    bg = Image.new("RGBA", mask.size, (0,)*4)
    bg.paste(fill, mask=mask)
    return bg


def illust_score_detail(score: Dict, size=1280, style="dark"):
    bmset = score["beatmapset"]
    bm = score["beatmap"]

    try:
        cover = requests.get_image(bmset["covers"]["slimcover"])[1]
    except Exception:
        cover = requests.get_image(bmset["covers"]["cover"])[1]

    # colors
    c = image_colors(cover, 1)[0]
    colors = image_colors(cover, 3)
    if(style == "dark"):
        color_bg = c.replace(L=0.15)
        color_fg = c.replace(L=0.9)
        cover = adjust_L(cover, -0.5)
        cover_dim = adjust_L(cover, -0.5)
    else:
        color_bg = c.replace(L=0.85)
        color_fg = c.replace(L=0.1)
        cover = adjust_L(cover, 0.5)
        cover_dim = adjust_L(cover, 0.5)

    @cached_func()
    def func_triangles(w, h): return triangles(w, h, colors)

    @cached_func()
    def func_triangles_bg(w, h):
        im = func_triangles(w, h)
        if(style == "dark"):
            return adjust_L(im, -0.85)
        else:
            return adjust_L(im, 0.85)

    def func_triangles_fg(w, h):
        im = func_triangles(w, h)
        if(style == "dark"):
            return adjust_L(im, 0.85)
        else:
            return adjust_L(im, -0.85)
    t_size_title = int(size/25)
    t_size_version = int(t_size_title*0.7)
    t_size_beatmap_id = int(t_size_version*0.7)
    t_size_username = int(t_size_title)
    t_size_played = int(t_size_title*0.7)
    t_size_counter = int(t_size_title*0.8)

    border_size = int(t_size_title*0.35)

    def kwget(st, default=None):
        f = lambda **kwargs: kwargs.get(st, default)
        return f
    T = Text(content=kwget("text"), fontSize=kwget(
        "fs", t_size_title), fill=color_fg.get_rgba())

    t_title = bmset.get("title_unicode") or bmset.get("title")
    t_title = T.render(text=t_title, fs=t_size_title)
    t_version = bm['version']
    if(score.get("mods")):
        mods = ["+"+i for i in score.get("mods")]
        t_version += "["+", ".join(mods)+"]"
    t_version = T.render(text=t_version, fs=t_size_version)
    t_bid = str(bm["id"])
    t_bid = T.render(text=t_bid, fs=t_size_beatmap_id)
    t_title = Column([t_title, t_version, t_bid], alignX=0.1).render()

    t_player = score["user"]["username"]
    text_type = "%s%d" % (score["type"].upper(), score["type_idx"])
    t_player += " - "+text_type
    if(score.get("pp")):
        t_player += " - %.1f pp" % score.get("pp")
    t_player = T.render(text=t_player, fs=t_size_username)
    date_played = score["created_at"]
    date_played = datetime.fromisoformat(date_played)
    date_played = date_played.strftime("%Y-%m-%d %H:%M")
    t_played = T.render(text="Played on %s" % date_played, fs=t_size_played)
    header = Row([t_title, Column([t_player, t_played])], width=size).render()
    header = AddBorder(header, borderWidth=border_size, borderColor=(0,)*4)
    header = CompositeBG(header, cover_dim).render()

    t_score = comma_number(score["score"])
    t_score = "%s %.2f%%" % (t_score, score["accuracy"]*100)
    t_score = T.render(text=t_score, fs=t_size_title)
    t_score = process.shadow(t_score, color=color_fg, radius=t_size_title/15)
    t_score = add_hex_bg(t_score, func_triangles)
    counter_width = int(t_score.size[0]*0.4)
    counter_left = list()
    counter_right = list()
    for which_column, judges in ((counter_left, "300 katu 50"), (counter_right, "geki 100 miss")):
        for judge in judges.split():
            im_judgement = judgement(score['mode'], judge, t_size_counter)
            judge_count = str(score["statistics"]["count_"+judge])
            im_judge_count = T.render(text=judge_count, fs=t_size_counter)
            # im = Pill(im_judgement, im_judge_count, colorA = color_fg.get_rgba(), colorB=color_bg.get_rgba(), colorBorder = colors[0])
            im = Row([im_judgement, im_judge_count],
                     width=counter_width).render()
            im = process.shadow(im, color=color_fg, radius=t_size_counter/15)
            im = add_hex_bg(im, func_triangles_bg, debug=False,
                            offsetY=-t_size_counter/4)
            im = add_hex_bg(im, func_triangles, expandw=0,
                            border=int(t_size_counter/6))
            which_column.append(im)

    counter_left = Column(counter_left, borderWidth=border_size)
    counter_right = Column(counter_right, borderWidth=border_size)
    counters = [counter_left, counter_right]

    # counters = Row(counters)
    score_and_count = Column([t_score, Row(counters)], borderWidth=border_size)
    score_area_contents = [score_and_count]
    if(bm["mode"] == "mania" and bm["cs"] == 4):
        # mania4k
        is_dt = "DT" in score["mods"]
        chart = Chart4k.from_osu_id(bm["id"], dt=is_dt)
        with locked(pyplot_lock):
            plot = chart.plot(int(size*0.5), int(size/2*0.7),
                              80, transparent=True)
        # plot = ImageOps.invert(plot)
        bg = func_triangles(*plot.size)
        if(style == "dark"):
            arr = np.array(plot)
            arr[:, :, :-1] = 255-arr[:, :, :-1]
            plot = Image.fromarray(arr)
            bg = adjust_L(bg, -0.5)
            bg = adjust_A(bg, -0.1)
        else:
            bg = adjust_L(bg, 0.5)
        bg.alpha_composite(plot)
        score_area_contents.append(bg)
    score_area = Row(score_area_contents, borderWidth=border_size)
    ret = Column([header, score_area], borderWidth=border_size)
    ret = AddBorder(ret, borderWidth=border_size)
    ret = CompositeBG(ret, cover)
    return ret.render()


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
        cover_dim = adjust_L(cover, -0.5)
    else:
        color_bg = c.replace(L=0.85)
        color_fg = c.replace(L=0.1)
        cover = adjust_L(cover, 0.5)
        cover_dim = adjust_L(cover, 0.5)
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
    text_title = bmset.get("title_unicode") or bmset.get("title")
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
        mods = ["+"+i for i in score.get("mods")]
        text_ver += "["+", ".join(mods)+"]"
    text_version = textrender_minor.render(text=text_ver)
    text_scores = "%.2f*, %s/%.2f%%" % (star_rating, text_score, acc*100)
    text_scores = textrender_minor.render(text=text_scores)

    text_type = "%s%d" % (score["type"].upper(), score["type_idx"])
    text_type = textrender_minor.render(text=text_type)
    line2 = [text_version, text_scores, text_type]
    line2 = Row(line2, width=w)

    line3 = []
    if(bm["mode"] == "mania" and bm["cs"] == 4):
        # mania4k
        is_dt = "DT" in score["mods"]
        chart = Chart4k.from_osu_id(bm["id"], dt=is_dt)
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
    user.get_info(mode=mode)
    bests = user.get_scores("best", mode=mode)
    recents = user.get_scores("recent", mode=mode)
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
    row = CompositeBG(row, bg=cover)
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

    col = Column(columns, alignX=0.6, borderWidth=gap,
                 outer_border=True).render()
    # col = addBorder(col, borderWidth = gap).render()
    w, h = col.size
    img = triangles(w, h, cover_colors)
    ret = CompositeBG(col, img)
    # return col.render()
    return ret.render()


if(__name__ == "__main__"):
    from .user import User
    u = User("TkskKurumi")
    scores = u.get_scores()
    score = scores[0]
    im = illust_score_detail(score)

    from ..image.print import image_show_terminal
    image_show_terminal(im, rate=0.5)
    im = illust_score(score)
    image_show_terminal(im, rate=0.5)
# if(__name__ == "__main__"):
#     from .user import User
#     ids = ["TkskKurumi"]# , "[Crz]Ha0201"]
#     for id in ids:
#         u = User(id)
#         # im = illust_score(scores[0], style="dark")
#         from ..image.print import image_show_terminal
#         # image_show_terminal(im,ra)
#         im = illust_user(u, mode="osu", size=800)
#         image_show_terminal(im, rate=1/len(ids))
