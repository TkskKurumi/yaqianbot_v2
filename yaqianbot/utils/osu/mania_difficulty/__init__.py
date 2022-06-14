from collections import defaultdict
from functools import wraps
import math
import re
import requests
from matplotlib import pyplot as plt
import numpy as np
import io
from PIL import Image
from ...jsondb import jsondb
from ...io import loadtext
from ...myhash import base32
from ..paths import pth
from os import path
INF = float("inf")
EPS = 1e-3
JACK_ALPHA = 2.5
JACK_BETA = 1.05


def mean(ls):
    if(not ls):
        return 0
    return sum(ls)/len(ls)

calc_cache = dict()



def geometric_mean(ls, weights=None):
    loged = [math.log(abs(i)+EPS) for i in ls]
    if(weights is None):

        loged_mean = mean(loged)
    else:
        loged_mean = weighted_mean(loged, weights)
    return math.exp(loged_mean)


def weighted_mean(ls, weights):

    meow = sum([i*weights[idx] for idx, i in enumerate(ls)])
    if(not meow):
        return 0
    return meow/sum(weights)


def soft_greatest(ls):
    return weighted_mean(ls, ls)


def chart_reduce(ls, method="C"):
    if(method == 'A'):
        soft = weighted_mean(ls, [i**0.5 for i in ls])
        # soft = soft_greatest(ls)
        # return soft_greatest(ls)
        mx = max(ls)
        return weighted_mean([soft, mx], [2, 1])
        # return geometric_mean([soft, mx], [5, 1])
    elif(method == "B"):
        ls = sorted(ls)[::-1]
        alpha = 0.4**(1/len(ls))
        weight = [(alpha**i)*ls[i] for i in range(len(ls))]
        return weighted_mean(ls, weight)
    else:
        ls = sorted(ls)
        n = len(ls)
        ed = int(n*0.85)
        st = int(n*0.05)
        # return soft_greatest(ls[st:ed])*1.2
        return chart_reduce(ls[st:ed], method="A")


def harmonic_mean(ls):
    sm = sum([1/i for i in ls])
    return len(ls)/sm


def lower_bound(ls, key, cmp=lambda x, key: x >= key):
    le = 0
    ri = len(ls)-1
    while(le <= ri):
        mid = (le+ri) >> 1
        if(cmp(ls[mid], key)):
            ri = mid-1
        else:
            le = mid+1

    return le

difficulty_cache = jsondb(path.join(pth, "mania_difficulty_cache"))
algorithms_fingerprint = loadtext(__file__)
algorithms_fingerprint = base32(algorithms_fingerprint)

def weight_by_rank(ls, alpha=0.5):
    ls = sorted(ls, key=lambda x:-x)
    weight = [alpha**i for i in range(len(ls))]
    return weighted_mean(ls, weight)
class Chart:
    def __init__(self, notes, title, cache_key = None):
        self.notes = notes
        self.title = title
        self.cache_key = cache_key
    def get_duration(self):
        start = min(self.notes, key=lambda x: x[1])[1]
        end = max(self.notes, key=lambda x: x[2])[2]
        return end-start

    def get_multi(self, notes):
        density = len(self.notes)/self.get_duration()
        # consider quite close notes (in dumps) as multi-press
        eps = 1/density/5
        rets = defaultdict(list)
        last = -INF
        for idx, starttime, endtime in notes:
            if(starttime > last+eps):
                last = starttime
            rets[last].append((idx, starttime, endtime))
        return rets

    def get_slice(self, start, end):
        def cmp(note, tm):
            return note[1] >= tm
        idx = lower_bound(self.notes, start, cmp=cmp)
        jdx = lower_bound(self.notes, end, cmp=cmp)
        return self.notes[idx:jdx]
    def _calc_stream_calc_uniformity(self, multi):
        notes = defaultdict(list)
        for k, v in multi.items():
            for note in v:
                notes[note[0]].append(note)
        # diversity = list()

        meow = []
        for i in notes:
            for j in notes:
                if(i == j):
                    continue
                last = None
                last_t = None
                ls = notes[i]+notes[j]
                ls = sorted(ls, key=lambda x: x[1])
                neq = 0
                tot = len(ls)
                for k in ls:
                    if(k[0] != last):
                        neq += 1
                    last = k[0]
                    last_t = k[1]
                meow.append((tot-neq)/tot)
        return mean(meow)*10

    def calc_stream_partial(self, multi):
        if(not multi):
            return 0
        ls = list(multi)
        duration = max(ls)-min(ls)
        if(duration == 0):
            return 0
        meow = sum([len(multi[i])**0.5 for i in ls])
        # meow = len(multi)
        d_stream = meow/(duration/1000)
        d_density = self._calc_stream_calc_uniformity(multi)
        return d_stream*((d_density/4)**0.1)/1.3/1.27/1.06

    def calc_jumpstream_partial(self, multi):
        ls = list(multi)
        if(not ls):
            return 0
        duration = max(ls)-min(ls)
        meow = 0
        for i in ls:
            if(len(multi[i]) >= 2):
                meow += 2
            else:
                meow += 0.5
        single = 0
        for i in ls:
            if(len(multi[i]) == 1):
                single += 2
        # d_stream = self.calc_stream(multi)
        if(duration == 0):
            return 0
        d_meow = meow/(duration/1000)
        d_single = single/(duration/1000)
        return geometric_mean([d_single, d_meow], [1, 1.5])*0.709*0.87

    def calc_jack_column(self, notes):
        if(not notes):
            return 0

        last = notes[0][-1]
        scores = dict()
        for interval in [2, 5]:
            meow = list()
            weight = list()
            for idx, i in enumerate(notes[interval:]):
                col, st, ed = i
                last = notes[idx][1]
                tmp = (st-last)/1000/(interval**1.25)
                try:
                    _ = 1/(tmp+EPS)
                    meow.append(_**JACK_BETA)
                    weight.append(_**(JACK_ALPHA+JACK_BETA))
                except ZeroDivisionError as e:
                    print(st, last, tmp)
                    print(i)
                    print(notes[idx])
                    raise e
            if(not meow):
                return 0
            return weighted_mean(meow, weight)
            # meow_idx = [(idx, i) for idx, i in enumerate(meow)]
            # sorted_meow = sorted(meow_idx, key=lambda x:-x[1])
            # sorted_weight = [(0.5**idx)*weight[idx] for idx, i in sorted_meow]
            # sorted_meow = [i for idx, i in sorted_meow]
            # # scores[interval] = weighted_mean(meow, weight)
            # scores[interval] = weighted_mean(sorted_meow, sorted_weight)
        ret = geometric_mean([scores[3], scores[5]], [3, 5])
        return ret

    def calc_jack_partial(self, multi):
        if(not multi):
            return 0
        notes = defaultdict(list)
        for k, v in multi.items():
            for note in v:
                notes[note[0]].append(note)
        d = list()
        for k, v in notes.items():
            d.append(self.calc_jack_column(v))
        # soft = soft_greatest(d)
        # mx = max(d)
        sd = sorted(d)
        weights = [2.5**i for i in range(len(sd))]
        # ret = weighted_mean([soft, mx], [1, 5])
        ret = weighted_mean(sd, weights)
        return ret*1.06

    def calc_chordjack_partial(self, multi, debug=None):

        d_jack = self.calc_jack_partial(multi)
        ls = list(multi)
        strengths = []
        scores = dict()
        for noteinterval in [1, 2]:
            meow = []
            weight = []
            for idx, i in enumerate(ls[1:]):
                last = ls[idx]
                notes = multi[i]
                last_notes = multi[last]
                cols = set([note[0] for note in notes])
                last_cols = set([note[0] for note in last_notes])
                inter = cols & last_cols
                merged = cols | last_cols
                n = len(inter)
                m = len(merged)
                complexity = n*(m-n+1)

                interval = (i-last)/1000/noteinterval
                strength = (1/(interval+EPS))
                meow.append(geometric_mean(
                    [complexity**1.2, strength], [2, JACK_ALPHA]))
                # strengths.append(geometric_mean([n,n,strength]))
                # print(interval)
            scores[noteinterval] = soft_greatest(meow)
        return weighted_mean([scores[1], scores[2]], [2, 1])*1.15

    def calc_tech_partial(self, multi):
        b = self.calc_jack_partial(multi)
        c = self.calc_stream_partial(multi)
        return geometric_mean([b, c], [2, 1])*1.02
        # print(b, c)
        mn = min(b, c)
        mx = max(b, c)
        return weighted_mean([mn, mx], [5, 1])*1.05

    def calc_overall_partial(self, multi):
        streamish = self.calc_streamish_partial(multi)
        jackish = self.calc_jackish_partial(multi)
        return max(streamish, jackish)
        # st = self.calc_stream_partial(multi)
        # js = self.calc_jumpstream_partial(multi)
        # hs = self.calc_handstream_partial(multi)
        # jack = self.calc_jack_partial(multi)
        # cj = self.calc_chordjack_partial(multi)
        # tech = self.calc_tech_partial(multi)
        # gmean = geometric_mean([st, js, hs, jack, cj, tech])
        # ls = [st, js, hs, jack, cj, tech]
        # soft = soft_greatest(ls)
        # mx = max(ls)
        # # mx = (soft_greatest(ls)+max(ls))/2
        # ret = weighted_mean([mx, soft], [5, 1])

    def calc_jackish_partial(self, multi):
        a = self.calc_jack_partial(multi)
        b = self.calc_chordjack_partial(multi)
        return max(a, b)

    def calc_streamish_partial(self, multi):
        a = self.calc_stream_partial(multi)
        b = self.calc_jumpstream_partial(multi)
        c = self.calc_handstream_partial(multi)
        return max(a, b, c)

    def all_pattern_partial(self):
        meow = lambda **kwargs: kwargs
        return meow(
            Stream=self.calc_stream_partial,
            Jumpstream=self.calc_jumpstream_partial,
            Handstream=self.calc_handstream_partial,
            Chordjack=self.calc_chordjack_partial,
            Jackspeed=self.calc_jack_partial,
            Tech=self.calc_tech_partial,
            Overall=self.calc_overall_partial,
            Streamish=self.calc_streamish_partial,
            Jackish=self.calc_jackish_partial
        )

    def calc_handstream_partial(self, multi):
        ls = list(multi)
        if(not ls):
            return 0
        duration = max(ls)-min(ls)
        meow = 0
        for i in ls:
            if(len(multi[i]) >= 3):
                meow += 2
            elif(len(multi[i]) >= 2):
                meow += 0.5

        single = 0
        for i in ls:
            if(len(multi[i]) == 1):
                single += 2
        # d_stream = self.calc_stream(multi)
        if(duration == 0):
            return 0
        d_meow = meow/(duration/1000)
        d_single = single/(duration/1000)
        return geometric_mean([d_single, d_meow], [1, 1.5])*1.25

    @classmethod
    def from_osu_string(cls, s, dt=False, rate=1, cache_key = None):
        pattern = r'(\d+,\d+,\d+,\d+,\d+,\d+):'
        """with open(pth, "r", encoding="utf-8") as f:
            s = f.read()"""
        notes = []
        for idx, notestr in enumerate(re.findall(pattern, s)):
            x, y, starttime, note_type, _0, endtime = [
                int(_) for _ in re.split('[,:]', notestr)]
            column_idx = x
            if(endtime <= starttime):
                endtime = starttime
            if(dt):
                starttime /= 1.5
                endtime /= 1.5
            elif(rate != 1):
                starttime /= rate
                endtime /= rate
            notes.append((column_idx, starttime, endtime))
        notes = sorted(notes, key=lambda x: x[1])

        title = re.findall(r"TitleUnicode:([\s\S]+?)[\r\n]", s)[0]
        version = re.findall(r"Version:([\s\S]+?)[\r\n]", s)[0]
        if(rate != 1):
            version += " x%.2f" % (rate, )
        return cls(notes, title+" - "+version, cache_key = cache_key)

    @classmethod
    def from_osu_id(cls, id, dt=False, rate=1):
        url = "https://osu.ppy.sh/osu/%s" % id
        r = requests.get(url)
        s = r.text
        cache_key = "beatmap(%d, dt=%s, rate=%s)"%(id, dt, rate)
        return cls.from_osu_string(s, dt=dt, rate=rate, cache_key = cache_key)

    @classmethod
    def from_osu(cls, pth, dt=False, rate=1):
        with open(pth, "r", encoding="utf-8") as f:
            s = f.read()
        return cls.from_osu_string(s, dt=dt, rate=rate)

    def calc_all(self):
        if(self.cache_key is not None):
            cache_key = "%s-%s"%(algorithms_fingerprint, self.cache_key)
            if(cache_key in difficulty_cache):
                print("cached difficulty calculation for %s"%cache_key)
                return difficulty_cache[cache_key]
            else:
                print("calculate difficulty for %s"%cache_key)
        else:
            print("no cache key")
            cache_key = None
        start = 0       # nopep8
        step = 5*1000  # nopep8
        interval = 10*1000  # nopep8
        ret_time = defaultdict(list)  # nopep8
        ret_all = defaultdict(float)  # nopep8

        stamina_tmp = 0
        stamina_mx = 0
        while(start < self.notes[-1][1]):
            slice = self.get_slice(start, start+interval)
            multi = self.get_multi(slice)
            for name, func in self.all_pattern_partial().items():
                score = func(multi)
                ret_time[name].append(score)
                if(name == "Overall"):
                    if(score > stamina_tmp):
                        t_rise = 90
                        target = score
                    else:
                        t_rise = 15
                        target = score
                    stamina_tmp += (target-stamina_tmp) * \
                        (0.9**(t_rise*1000/step))
                    stamina_mx = max(stamina_mx, stamina_tmp)
                    ret_time["Stamina"].append(stamina_tmp)
            ret_time["Time"].append(start/1000)
            start += step
        for name in self.all_pattern_partial():
            ret_all[name] = chart_reduce(ret_time[name])

        cur = 0
        stamina = 0
        """for i in ret_time["Overall"]:
            if(i>cur):
                cur+=(i-cur)*(0.5**(step/(120*1000)))
            else:
                cur+=(i-cur)*(0.5**(step/(30*1000)))
            stamina = max(cur, stamina)
        ret_all["Stamina"] = stamina"""
        ret_all["Stamina"] = stamina_mx

        ret = ret_time, ret_all
        if(cache_key is not None):
            difficulty_cache[cache_key]=ret
        return ret

    def plot(self):

        by_time, overall = self.calc_all()

        plt.figure(figsize=(1024//80, 768//80), dpi=80)
        # polar radar
        plt.subplot(2, 2, 1, polar=True)
        plt.title(self.title)

        labels = list(self.all_pattern_partial())+["Stamina"]
        labels.remove("Streamish")
        labels.remove("Jackish")

        length = len(labels)
        datas = [overall[i] for i in labels]
        labels = ["%s: %.1f" % (i, datas[idx]) for idx, i in enumerate(labels)]
        angles = list(np.linspace(0, 2 * np.pi, length, endpoint=False))

        plt.polar(angles + angles[:1], datas+datas[:1], ".-", linewidth=1)
        plt.fill(angles + angles[:1], datas+datas[:1], alpha=0.25)
        plt.thetagrids(np.array(angles) * 180 / np.pi, labels)

        plt.subplot(2, 2, 2)
        for label in ["Stream", "Jumpstream", "Handstream"]:
            plt.plot(by_time["Time"], by_time[label], label=label)
        plt.legend()

        plt.subplot(2, 2, 3)
        for label in ["Chordjack", "Jackspeed"]:
            plt.plot(by_time["Time"], by_time[label], label=label)
        plt.legend()

        plt.subplot(2, 2, 4)
        # for label in list(self.all_pattern_partial())+["Stamina"]:
        # for label in ["Streamish", "Jackish", "Tech"]:
        for label in ["Overall", "Stamina"]:
            plt.plot(by_time["Time"], by_time[label], label=label)

        plt.legend()

        bio = io.BytesIO()
        plt.savefig(bio)
        plt.clf()
        bio.seek(0)
        return Image.open(bio)
