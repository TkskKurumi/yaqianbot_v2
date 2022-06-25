from typing import List
from ..geometry.elements import Point2d as Point
from ..geometry.elements import Circle
from ..geometry.calculation import does_segment_intersect, float_sign, get_circle

def normalize_edge(u, v):
    return tuple(sorted([u, v]))


class Graph:  # undirected graph
    def __init__(self, neibours=None, edges=None):
        self.neibours = neibours or dict()
        self.edges = edges or set()

    def add_edge(self, u, v):
        self.neibours[u] = self.neibours.get(u, set())
        self.neibours[u].add(v)
        self.neibours[v] = self.neibours.get(v, set())
        self.neibours[v].add(u)
        edg = normalize_edge(u, v)
        self.edges.add(edg)

    def remove_edge(self, u, v):
        if(v in self.neibours[u]):
            self.neibours[u].remove(v)
            self.neibours[v].remove(u)
            self.edges.remove(normalize_edge(u, v))


class DelaunayMesh:
    def __init__(self, points: List[Point], debug=False):
        n = len(points)
        sorted_points = list(range(n))
        sorted_points.sort(key=lambda idx: tuple(points[idx]))
        G = Graph()
        if(debug):
            from PIL import Image, ImageDraw
            from .. import make_gif
            w = 0
            h = 0
            for p in points:
                x, y=p
                w = int(max(x, w))+1
                h = int(max(y, h))+1
            im = Image.new("RGB", (w, h), (255,)*3)
            dr = ImageDraw.Draw(im)
            frames = []
        def debug_rm_edge(u, v):
            u, v=normalize_edge(u, v)
            point_u = points[u]
            point_v = points[v]
            dr.line((*point_u, *point_v), fill=(255,)*3, width=2)
        def debug_draw_edge(u, v, color = (0, 0, 255), final_color=(255,80,120)):
            u, v=normalize_edge(u, v)
            point_u = points[u]
            point_v = points[v]
            dr.line((*point_u, *point_v), fill=color, width=2)
            frames.append(im.copy())
            dr.line((*point_u, *point_v), fill=final_color, width=2)
        def getp(idx):
            return points[sorted_points[idx]]

        def getidx(idx):
            return sorted_points[idx]

        def addedge(i, j):
            if(debug):
                debug_draw_edge(i, j)
            return G.add_edge(i, j)

        def recursion(l, r):
            if(r-l <= 2):
                # less equal than 3 points
                for i in range(l, r):
                    for j in range(i+1, r+1):
                        addedge(getidx(i), getidx(j))
                return
            else:
                mid = (l+r) >> 1
                recursion(l, mid)
                recursion(mid+1, r)
                current_l = getidx(l)
                current_r = getidx(r)

                # find base edge
                updated = True
                while(updated):
                    updated = False
                    point_l = points[current_l]
                    point_r = points[current_r]
                    # update L
                    for t in G.neibours[current_l]:
                        point_t = points[t]
                        LR = point_r-point_l
                        TR = point_r-point_t
                        cross = LR.cross(TR)
                        cross_sign = float_sign(cross)
                        if(cross_sign > 0):
                            is_lower = True
                        elif(cross_sign == 0):
                            _TR_ = TR.sqrlength()
                            _LR_ = LR.sqrlength()
                            is_lower = _TR_ < _LR_
                        else:
                            is_lower = False
                        if(is_lower):
                            current_l = t
                            updated = True
                            break
                    if(updated):
                        continue
                    # update R
                    for t in G.neibours[current_r]:
                        point_t = points[t]
                        LR = point_r-point_l
                        LT = point_t-point_l
                        cross = LR.cross(LT)
                        cross_sign = float_sign(cross)
                        if(cross_sign < 0):
                            is_lower = True
                        elif(cross_sign == 0):
                            _LR_ = LR.sqrlength()
                            _LT_ = LT.sqrlength()
                            is_lower = _LT_<_LR_
                        else:
                            is_lower = False
                        if(is_lower):
                            current_r = t
                            updated = True
                            break
                    
                # print("Add base l-r", point_l, point_r)
                # G.add_edge(current_l, current_r)
                addedge(current_l, current_r)
                # add all L-R edge
                while(True):
                    point_l = points[current_l]
                    point_r = points[current_r]
                    LR = point_r-point_l
                    update_side = None
                    found = None

                    for t in G.neibours[current_l]:
                        point_t = points[t]
                        LT = point_t-point_l
                        cross = LR.cross(LT)
                        cross_sgn = float_sign(cross)
                        if(cross_sgn > 0):
                            if(found is None):
                                found = t
                                update_side = "L"
                            else:
                                circle = get_circle(point_l, point_r, points[found])
                                if(point_t in circle):
                                    found = t
                                    update_side = "L"
                    for t in G.neibours[current_r]:
                        point_t = points[t]
                        RT = point_t-point_r
                        cross = LR.cross(RT)
                        cross_sgn = float_sign(cross)
                        if(cross_sgn>0):
                            if(found is None):
                                found = t
                                update_side = "R"
                            else:
                                circle = get_circle(point_l, point_r, points[found])
                                # print(points[found], circle)
                                if(point_t in circle):
                                    found = t
                                    update_side = "R"
                    if(update_side == "L"):
                        to_remove = []
                        point_found = points[found]
                        for t in G.neibours[current_l]:
                            point_t = points[t]
                            if(does_segment_intersect(point_r, point_found, point_l, point_t)):
                                to_remove.append(t)
                        for t in to_remove:
                            if(debug):
                                debug_draw_edge(current_l, t, color = (255, 100, 255))
                                debug_rm_edge(current_l, t)
                            G.remove_edge(current_l, t)
                        current_l = found
                        # print("Add l-r", point_found, point_r)
                        # G.add_edge(current_l, current_r)
                        addedge(current_l, current_r)
                    elif(update_side == "R"):
                        to_remove = []
                        point_found = points[found]
                        for t in G.neibours[current_r]:
                            point_t = points[t]
                            if(does_segment_intersect(point_l, point_found, point_r, point_t)):
                                to_remove.append(t)
                        for t in to_remove:
                            if(debug):
                                debug_draw_edge(current_r, t, color = (255, 100, 255))
                                debug_rm_edge(current_r, t)
                            G.remove_edge(current_r, t)
                        current_r = found
                        # print("Add l-r", point_l, point_found)
                        # G.add_edge(current_l, current_r)
                        addedge(current_l, current_r)
                    else:
                        break
        recursion(0, n-1)
        self.G = G
        if(debug):
            gif = make_gif.make_gif(frames, fps=3,frame_area_sum=1e7)
            print(gif)