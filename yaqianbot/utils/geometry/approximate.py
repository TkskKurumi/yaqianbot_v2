EPS = 1e-8
def float_sign(x):
    if(x < -EPS):
        return -1
    elif(x > EPS):
        return 1
    else:
        return 0