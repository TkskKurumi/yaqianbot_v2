import numpy as np
import random
from collections import defaultdict


def kmeans(points, k, iter=8, weights = None):
    n = len(points)
    points = np.array(points, np.float32)
    if(weights is None):
        weights = [1 for i in range(n)]
    rets = random.sample(list(range(n)), k)
    rets = [points[i] for i in rets]
    for it in range(iter):
        pointsum = [0 for i in range(k)]
        pointn = [0 for i in range(k)]
        for i in range(n):
            p = points[i]
            dist = np.array(rets, np.float16)-p
            dist = dist**2
            dist = np.sum(dist, axis=-1)
            idx = np.argmin(dist)
            # print(pointsum, p)
            pointsum[idx] += p*weights[i]
            pointn[idx] += weights[i]
        for i in range(k):
            if(pointn[i]):
                rets[i]=pointsum[i]/pointn[i]
            else:
                rets[i]=random.choice(points)
    return rets



if(__name__=="__main__"):
    from matplotlib import pyplot as plt
    points = np.random.normal(0,1,(60, 2))
    means = kmeans(points, 20)
    plt.scatter(*points.T)
    plt.scatter(*zip(*means))
    plt.savefig("./tmp.jpg")
