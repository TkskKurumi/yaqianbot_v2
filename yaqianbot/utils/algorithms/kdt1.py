from __future__ import annotations
from typing import Iterable, List, Literal

import numpy as np
# from .datastructures import heap
import heapq
EPS = 1e-6
INF = float("inf")


def vec_dist(v: np.ndarray, u: np.ndarray):
    diff = v-u
    return np.sqrt(np.sum(diff**2))


def vec_dist_batch(v, vecs):
    diff = vecs-v
    sqr = diff**2
    sqrsum = np.sum(sqr, axis=-1)
    return np.sqrt(sqrsum)


node_names = ["left", "right", "ax", "ax_v", "depth", "vecs", "vec_ids"]


class KDT:
    def __init__(self):
        for name in node_names:
            setattr(self, name, list())
        self.n_nodes = 0
        self._perf_debug_knn_times = 0
        self._perf_debug_leaf_times = 0
        self._perf_debug_nodes = 0

    def new_node(self):
        ret = self.n_nodes
        for name in node_names:
            getattr(self, name).append(None)
        self.n_nodes += 1
        return ret

    def knn(self, vec, k, search_branch=10):
        self._perf_debug_knn_times += 1
        # rets = heap(fcmp=lambda x, y: x[0] > y[0])
        rets = []
        # branches = heap(fcmp=lambda x, y: x[0] < y[0])
        branches = []
        searched_branch = 0

        def pop_worst():
            nonlocal rets
            while(len(rets) > k):
                heapq.heappop(rets)

        def process_leaf(u):
            nonlocal rets
            self._perf_debug_leaf_times += 1
            vecs = self.vecs[u]
            dists = vec_dist_batch(vec, vecs)
            for idx, dist in enumerate(dists):
                vec_id = self.vec_ids[u][idx]
                # rets.push((dist, vec_id, vecs[idx]))
                heapq.heappush(rets, (-dist, vec_id, vecs[idx]))
                pop_worst()

        def process_node(u):
            nonlocal searched_branch
            searched_branch += 1
            while(True):
                self._perf_debug_nodes += 1
                vecs = self.vecs[u]
                if(vecs is not None):
                    # leaf node
                    # print("leaf node", u)
                    return process_leaf(u)
                # print("branch node", u)
                ax = self.ax[u]
                ax_v = self.ax_v[u]
                if(vec[ax] < ax_v):
                    heapq.heappush(branches, (ax_v-vec[ax], self.right[u]))
                    u = self.left[u]
                else:
                    heapq.heappush(branches, (vec[ax]-ax_v, self.left[u]))
                    u = self.right[u]
        process_node(0)
        while(branches):
            # top = branches.pop()
            top = heapq.heappop(branches)
            norm_dist, u = top
            if(len(rets) < k):
                process_node(u)
            elif(searched_branch < search_branch):

                if(norm_dist < -rets[0][0]):
                    process_node(u)

            else:
                break

        return sorted([(-dist, idx, vec) for dist, idx, vec in (rets)])

    def build(self, vecs: np.ndarray | List[np.ndarray], vec_ids: Literal[None] | List = None, stop_depth=INF, stop_cluster=1, depth=0):
        if(vec_ids is None):
            self.all_vecs = vecs
            vec_ids = list(range(len(vecs)))
        u = self.new_node()
        self.depth[u] = depth
        if(depth >= stop_depth or len(vecs) <= stop_cluster):
            # leaf node
            self.vecs[u] = vecs
            self.vec_ids[u] = vec_ids
            return u
        else:
            mean = np.mean(vecs, axis=0)
            std = np.std(vecs, axis=0)
            ax = np.argmax(std)
            ax_v = mean[ax]
            if(std[ax] == 0):
                self.vecs[u] = vecs
                self.vec_ids = vec_ids
                return u
            self.ax[u] = ax
            self.ax_v[u] = ax_v

            le_vecs = list()
            le_ids = list()
            ri_vecs = list()
            ri_ids = list()
            for idx, vec in enumerate(vecs):
                v = vec[ax]
                if(v < ax_v):
                    le_vecs.append(vec)
                    le_ids.append(vec_ids[idx])
                else:
                    ri_vecs.append(vec)
                    ri_ids.append(vec_ids[idx])
            self.left[u] = self.build(
                le_vecs, le_ids, depth=depth+1,
                stop_depth=stop_depth, stop_cluster=stop_cluster
            )
            self.right[u] = self.build(
                ri_vecs, ri_ids, depth=depth+1,
                stop_depth=stop_depth, stop_cluster=stop_cluster
            )
            return u


if(False and __name__ == "__main__"):
    a = [0, 0, 1]
    b = [1, 2, 3]
    c = [10, 10, 8]
    d = [3, 2, 1]
    vecs = np.array([a, b, c, d])
    k = KDT()
    k.build(vecs)
    for i in vecs:
        print(i)
        ls = k.knn(i, 4)
        for dist, idx, vec in ls:
            print("    dist=%.1f, vec=%s, diff=%s" % (dist, vec, i-vec))
if(__name__ == "__main__"):
    meow = 1e7
    nd = 512
    # n = int(max(1, (meow/nd)**0.5))
    n = 100
    # m = int(max(1, meow/nd/n))
    m = 100
    print("%d-dimension, search %d times in %d vecs"%(nd, m, n))
    n_knn = 10
    nvecs = np.random.normal(0, 1, (n, nd))
    mvecs = np.random.normal(0, 1, (m, nd))
    k = KDT()
    k.build(nvecs, stop_cluster=1)
    print(k.n_nodes,"nodes")
    def vanilla():
        sumdist = 0
        for i in range(m):
            vec = mvecs[i]
            dists = vec_dist_batch(vec, nvecs)
            dist = np.min(dists)
            sumdist += dist
            # sorted(dists)
        return sumdist

    def kdt():
        sumdist = 0
        for i in range(m):
            vec = mvecs[i]
            nn = k.knn(vec, n_knn, search_branch=5)
            dist = nn[0][0]
            sumdist += dist
        return sumdist
    print(vanilla())
    print(kdt())
    import timeit
    print(timeit.timeit(vanilla, number=10))
    print(timeit.timeit(kdt, number=10))
    print("%.1f leafs per 1-nn" %(k._perf_debug_leaf_times/k._perf_debug_knn_times))
    print("%.1f visited nodes per 1-nn" %(k._perf_debug_nodes/k._perf_debug_knn_times))
