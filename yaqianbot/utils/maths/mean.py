import math
EPS = 1e-6
def linear_mean(ls, weights = None):
    if(weights is None):
        weights = [1 for i in ls]
    sum_ls = sum([i*weights[idx] for idx, i in enumerate(ls)])
    sum_weight = sum(weights)
    return sum_ls/sum_weight
def geometric_mean(ls, weights = None):
    if(weights is None):
        weights = [1 for i in ls]
    loged = [math.log(i+EPS) for i in ls]
    mean_loged = linear_mean(loged, weights)
    return math.exp(mean_loged)